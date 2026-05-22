import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def moore_penrose_iter_pinv(matrix: torch.Tensor, iterations: int = 6, eps: float = 1e-6) -> torch.Tensor:
    """Approximate inverse."""
    identity = torch.eye(matrix.size(-1), device=matrix.device, dtype=matrix.dtype)
    abs_matrix = matrix.abs()
    col_norm = abs_matrix.sum(dim=-2).max(dim=-1).values
    row_norm = abs_matrix.sum(dim=-1).max(dim=-1).values
    z = matrix.transpose(-1, -2) / (col_norm * row_norm).clamp_min(eps).unsqueeze(-1).unsqueeze(-1)

    for _ in range(iterations):
        matrix_z = matrix @ z
        z = 0.25 * z @ (13 * identity - matrix_z @ (15 * identity - matrix_z @ (7 * identity - matrix_z)))
    return z


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """Init attention."""
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.res_conv = (
            nn.Conv2d(heads, heads, kernel_size=(33, 1), padding=(16, 0), groups=heads, bias=False)
            if residual
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention."""
        b, n, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, h, self.dim_head).transpose(1, 2).contiguous() for t in qkv]
        q = q * self.scale

        landmarks = min(self.num_landmarks, n)
        orig_n = n
        if n % landmarks != 0:
            pad_len = landmarks - (n % landmarks)
            q = F.pad(q, (0, 0, 0, pad_len), value=0)
            k = F.pad(k, (0, 0, 0, pad_len), value=0)
            v = F.pad(v, (0, 0, 0, pad_len), value=0)

        padded_n = q.size(2)
        chunks = padded_n // landmarks
        q_landmarks = q.view(b, h, landmarks, chunks, self.dim_head).mean(dim=-2)
        k_landmarks = k.view(b, h, landmarks, chunks, self.dim_head).mean(dim=-2)

        attn1 = torch.matmul(q, k_landmarks.transpose(-1, -2)).softmax(dim=-1)
        attn2 = torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2)).softmax(dim=-1)
        attn3 = torch.matmul(q_landmarks, k.transpose(-1, -2)).softmax(dim=-1)

        attn1 = self.dropout(attn1)
        attn3 = self.dropout(attn3)
        attn2_inv = moore_penrose_iter_pinv(attn2, self.pinv_iterations)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)
        if self.res_conv is not None:
            out = out + self.res_conv(v)
        out = out[:, :, :orig_n, :]
        out = out.transpose(1, 2).contiguous().view(b, orig_n, h * self.dim_head)
        return self.to_out(out)


class WindowSparseAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        window_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        """Init attention."""
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")

        self.heads = heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    @staticmethod
    def _pad_spatial_grid(x: torch.Tensor, target_size: int) -> torch.Tensor:
        """Pad token grid."""
        b, h, g_h, g_w, d = x.shape
        if g_h == target_size and g_w == target_size:
            return x
        pad_h = target_size - g_h
        pad_w = target_size - g_w
        padded = x.permute(0, 1, 4, 2, 3).reshape(b * h, d, g_h, g_w)
        padded = F.pad(padded, (0, pad_w, 0, pad_h))
        return padded.view(b, h, d, target_size, target_size).permute(0, 1, 3, 4, 2).contiguous()

    @staticmethod
    def _pad_valid_mask(mask: torch.Tensor, target_size: int) -> torch.Tensor:
        """Pad valid mask."""
        b, h, g_h, g_w = mask.shape
        if g_h == target_size and g_w == target_size:
            return mask
        pad_h = target_size - g_h
        pad_w = target_size - g_w
        padded = F.pad(mask.float(), (0, pad_w, 0, pad_h), value=0.0)
        return padded.bool()

    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Partition windows."""
        b, h, g_h, g_w, d = x.shape
        ws = self.window_size
        x = x.view(b, h, g_h // ws, ws, g_w // ws, ws, d)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return x.view(b, h, -1, ws * ws, d)

    def _mask_partition(self, mask: torch.Tensor) -> torch.Tensor:
        """Partition masks."""
        b, h, g_h, g_w = mask.shape
        ws = self.window_size
        mask = mask.view(b, h, g_h // ws, ws, g_w // ws, ws)
        mask = mask.permute(0, 1, 2, 4, 3, 5).contiguous()
        return mask.view(b, h, -1, ws * ws)

    def _window_reverse(self, windows: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Merge windows."""
        b, h, n_windows, window_area, d = windows.shape
        ws = self.window_size
        n_side = grid_size // ws
        x = windows.view(b, h, n_side, n_side, ws, ws, d)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return x.view(b, h, grid_size, grid_size, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention."""
        b, n, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, h, self.dim_head).transpose(1, 2).contiguous() for t in qkv]

        q_cls = q[:, :, :1, :] * self.scale
        cls_attn = torch.matmul(q_cls, k.transpose(-1, -2)).softmax(dim=-1)
        cls_attn = self.dropout(cls_attn)
        out_cls = cls_attn @ v

        if n == 1:
            out = out_cls.transpose(1, 2).contiguous().view(b, 1, h * self.dim_head)
            return self.to_out(out)

        spatial_tokens = n - 1
        grid_size = int(math.isqrt(spatial_tokens))
        if grid_size * grid_size != spatial_tokens:
            raise ValueError(f"Expected square-packed tokens, got {spatial_tokens}.")

        q_spatial = (q[:, :, 1:, :] * self.scale).view(b, h, grid_size, grid_size, self.dim_head)
        k_spatial = k[:, :, 1:, :].view(b, h, grid_size, grid_size, self.dim_head)
        v_spatial = v[:, :, 1:, :].view(b, h, grid_size, grid_size, self.dim_head)
        valid_mask = torch.ones((b, 1, grid_size, grid_size), dtype=torch.bool, device=x.device)

        padded_grid = int(math.ceil(grid_size / self.window_size) * self.window_size)
        q_spatial = self._pad_spatial_grid(q_spatial, padded_grid)
        k_spatial = self._pad_spatial_grid(k_spatial, padded_grid)
        v_spatial = self._pad_spatial_grid(v_spatial, padded_grid)
        valid_mask = self._pad_valid_mask(valid_mask, padded_grid).expand(-1, h, -1, -1)

        q_windows = self._window_partition(q_spatial)
        k_windows = self._window_partition(k_spatial)
        v_windows = self._window_partition(v_spatial)
        valid_windows = self._mask_partition(valid_mask)

        n_windows = q_windows.size(2)
        window_area = self.window_size * self.window_size
        q_flat = q_windows.view(b * h * n_windows, window_area, self.dim_head)
        k_flat = k_windows.view(b * h * n_windows, window_area, self.dim_head)
        v_flat = v_windows.view(b * h * n_windows, window_area, self.dim_head)
        valid_flat = valid_windows.view(b * h * n_windows, window_area)

        cls_k = k[:, :, :1, :].unsqueeze(2).expand(-1, -1, n_windows, -1, -1).reshape(
            b * h * n_windows, 1, self.dim_head
        )
        cls_v = v[:, :, :1, :].unsqueeze(2).expand(-1, -1, n_windows, -1, -1).reshape(
            b * h * n_windows, 1, self.dim_head
        )

        cls_scores = torch.matmul(q_flat, cls_k.transpose(-1, -2))
        local_scores = torch.matmul(q_flat, k_flat.transpose(-1, -2))
        local_scores = local_scores.masked_fill(~valid_flat.unsqueeze(1), -1e9)
        scores = torch.cat([cls_scores, local_scores], dim=-1)
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)

        out_cls_part = torch.matmul(attn[:, :, :1], cls_v)
        out_local_part = torch.matmul(attn[:, :, 1:], v_flat)
        out_flat = (out_cls_part + out_local_part) * valid_flat.unsqueeze(-1).type_as(v_flat)

        out_windows = out_flat.view(b, h, n_windows, window_area, self.dim_head)
        out_spatial = self._window_reverse(out_windows, padded_grid)
        out_spatial = out_spatial[:, :, :grid_size, :grid_size, :].reshape(b, h, spatial_tokens, self.dim_head)

        out = torch.cat([out_cls, out_spatial], dim=2)
        out = out.transpose(1, 2).contiguous().view(b, n, h * self.dim_head)
        return self.to_out(out)


class TransLayer(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        dropout: float = 0.1,
        attention_type: str = "nystrom",
        sparse_window_size: int = 7,
    ) -> None:
        """Init layer."""
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if attention_type == "nystrom":
            self.attn = NystromAttention(
                dim=dim,
                dim_head=dim // heads,
                heads=heads,
                num_landmarks=num_landmarks,
                pinv_iterations=pinv_iterations,
                residual=True,
                dropout=dropout,
            )
        elif attention_type == "sparse":
            self.attn = WindowSparseAttention(
                dim=dim,
                dim_head=dim // heads,
                heads=heads,
                window_size=sparse_window_size,
                dropout=dropout,
            )
        else:
            raise ValueError("attention_type must be 'nystrom' or 'sparse'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer."""
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        """Init encoder."""
        super().__init__()
        self.proj7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.proj5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Encode positions."""
        b, _, c = x.shape
        cls_token = x[:, :1]
        feat_token = x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(b, c, height, width)
        encoded = cnn_feat + self.proj7(cnn_feat) + self.proj5(cnn_feat) + self.proj3(cnn_feat)
        encoded = encoded.flatten(2).transpose(1, 2).contiguous()
        return torch.cat((cls_token, encoded), dim=1)


class TransMIL(nn.Module):
    def __init__(
        self,
        in_features: int = 1280,
        hidden_dim: int = 512,
        out_features: int = 1,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        dropout: float = 0.1,
        attention_type: str = "nystrom",
        sparse_window_size: int = 7,
    ) -> None:
        """Init model."""
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")

        self.in_features = in_features
        self.out_features = out_features

        self.feature_proj = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())
        self.layer1 = TransLayer(
            dim=hidden_dim,
            heads=heads,
            num_landmarks=num_landmarks,
            pinv_iterations=pinv_iterations,
            dropout=dropout,
            attention_type=attention_type,
            sparse_window_size=sparse_window_size,
        )
        self.pos_layer = PPEG(hidden_dim)
        self.layer2 = TransLayer(
            dim=hidden_dim,
            heads=heads,
            num_landmarks=num_landmarks,
            pinv_iterations=pinv_iterations,
            dropout=dropout,
            attention_type=attention_type,
            sparse_window_size=sparse_window_size,
        )

        # Attention pooling for better aggregation
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, out_features)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> dict:
        """Run model."""
        if x.dim() not in (2, 3):
            raise ValueError(f"Expected 2-D or 3-D input, got {x.dim()}-D.")
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if x.size(-1) != self.in_features:
            raise ValueError(f"Expected feature dim {self.in_features}, got {x.size(-1)}.")

        h = self.feature_proj(x.float())
        n_tokens = h.size(1)
        grid_size = int(math.ceil(math.sqrt(n_tokens)))
        target_len = grid_size * grid_size
        add_len = target_len - n_tokens
        if add_len > 0:
            h = torch.cat([h, h[:, :add_len, :]], dim=1)

        # Add dummy token for PPEG compatibility (will be removed)
        dummy_token = h.new_zeros(h.size(0), 1, h.size(-1))
        h = torch.cat((dummy_token, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, grid_size, grid_size)
        h = self.layer2(h)
        h = h[:, 1:]  # Remove dummy token
        h = self.norm(h)

        # Attention pooling - learns to weight important tokens
        attn_weights = F.softmax(self.attn_pool(h), dim=1)  # [B, N, 1]
        attn_pooled = torch.sum(attn_weights * h, dim=1)    # [B, hidden_dim]

        # Mean pooling as additional global feature
        mean_pooled = h.mean(dim=1)                         # [B, hidden_dim]

        # Combine both pooling methods
        pooled = torch.cat([attn_pooled, mean_pooled], dim=-1)
        logits = self.classifier(pooled)

        out = {"logits": logits}
        if return_attention:
            out["attention"] = attn_weights.transpose(1, 2)  # [B, 1, N]
        return out
