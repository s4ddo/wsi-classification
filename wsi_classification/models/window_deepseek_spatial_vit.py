"""Windowed DeepSeekSpatialViT with local window-based attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from wsi_classification.models.pos_embeds import SpatialEncoding, RotaryEmbedding
from wsi_classification.models.deepseek_spatial_vit_rope import DeepSeekMoE


# Global-Local Windowed Attention
class WindowedMLA(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, window_size, use_rope=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.use_rope = use_rope

        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.kv_down = nn.Linear(dim, latent_dim)
        self.kv_up = nn.Linear(latent_dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, coords, cls_tkn, mask=None):
        B, N, C = x.shape
        win_size = self.window_size

        # Infer grid stride and convert coords to 0-based grid indices
        ux = torch.unique(coords[0, :, 0])
        stride = (ux[1:] - ux[:-1]).min().item() if len(ux) > 1 else 1.0

        grid_col = (coords[..., 0] / stride).round().long()
        grid_row = (coords[..., 1] / stride).round().long()
        grid_col = grid_col - grid_col.min(dim=1, keepdim=True).values
        grid_row = grid_row - grid_row.min(dim=1, keepdim=True).values

        H = int(grid_row.max().item()) + 1
        W = int(grid_col.max().item()) + 1

        b_idx = torch.arange(B, device=x.device)[:, None].expand(-1, N)

        dense = x.new_zeros(B, H, W, C)
        dense[b_idx, grid_row, grid_col] = x

        dense_coords = coords.new_zeros(B, H, W, 2).float()
        dense_coords[b_idx, grid_row, grid_col] = coords.float()

        valid = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)
        valid[b_idx, grid_row, grid_col] = True

        # 1. Pad dimensions to be divisible by window_size
        pad_h = (win_size - (H % win_size)) % win_size
        pad_w = (win_size - (W % win_size)) % win_size
        H_p, W_p = H + pad_h, W + pad_w

        dense = F.pad(dense.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        dense_coords = F.pad(dense_coords.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        valid = F.pad(valid.unsqueeze(1).float(), (0, pad_w, 0, pad_h)).squeeze(1).bool()

        # 2. Reshape to window grid
        dense = dense.view(B, H_p // win_size, win_size, W_p // win_size, win_size, C)
        dense = dense.permute(0, 1, 3, 2, 4, 5).reshape(-1, win_size * win_size, C)
        B_win, N_win, _ = dense.shape

        dense_coords = dense_coords.view(B, H_p // win_size, win_size, W_p // win_size, win_size, 2)
        dense_coords = dense_coords.permute(0, 1, 3, 2, 4, 5).reshape(B_win, N_win, 2)

        valid = valid.view(B, H_p // win_size, win_size, W_p // win_size, win_size)
        valid = valid.permute(0, 1, 3, 2, 4).reshape(B_win, N_win)

        # 3. Create attention mask for padding
        win_mask = valid.view(B_win, 1, 1, N_win)

        # 4. Detach coords
        coords_win = dense_coords / 10000.0  # [B_win, N_win, 2]

        # 5. Attention with CLS Token
        # Concatenate CLS token to each window
        cls_tkn = cls_tkn.detach().expand(B_win, -1, -1)
        x_windows = torch.cat([cls_tkn, dense], dim=1)  # (B_win, N_win + 1, C)

        q = self.q_proj(x_windows)
        kv = self.kv_up(self.kv_down(x_windows))
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B_win, N_win + 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B_win, N_win + 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B_win, N_win + 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.use_rope:
            q_rope, k_rope = self.rope(coords_win, q[:, :, 1:, :], k[:, :, 1:, :], skip_first=False)
            # Unfortunately pytorch won't let you do the inplace edit you have to concat it like this.
            q = torch.cat([q[:, :, 0:1, :], q_rope], dim=2)
            k = torch.cat([k[:, :, 0:1, :], k_rope], dim=2)

        # Apply mask: CLS has 1 mask
        full_mask = torch.cat([torch.ones(B_win, 1, 1, 1, device=x.device, dtype=torch.bool), win_mask], dim=-1)

        # Attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask)

        # 6. Reshape back
        out_cls = out[:, :, 0:1, :].transpose(1, 2).reshape(B, -1, C).mean(dim=1, keepdim=True)
        out_patches = out[:, :, 1:, :].transpose(1, 2).reshape(B, H_p, W_p, C)
        out_patches = out_patches[:, :H, :W, :][b_idx, grid_row, grid_col]

        return self.out_proj(out_patches), self.out_proj(out_cls)


class WinTransBlock(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, num_shared, num_routed, top_k, hidden_dim, window_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedMLA(dim, num_heads, latent_dim, window_size)

        self.norm2 = nn.LayerNorm(dim)
        self.moe = DeepSeekMoE(dim, num_shared, num_routed, top_k, hidden_dim)

    def forward(self, x, coords, cls_token):
        # x is (B, N, C), cls_token is (B, 1, C)

        # 1. Attention: Returns updated patches and updated global token
        patches, new_cls_token = self.attn(self.norm1(x), coords, cls_token)
        x = x + patches
        cls_token = cls_token + new_cls_token

        # 2. MoE: Only process patches
        x = x + self.moe(self.norm2(x))

        return x, cls_token


class WinDeepSeekSpatialViT(nn.Module):
    uses_coords = True

    def __init__(self, input_dim=1280, num_classes=2, dim=128, depth=4,
                 num_heads=4, latent_dim=64, num_shared=1, num_routed=4, top_k=2, window_size=7, **kwargs):
        super().__init__()

        self.feature_proj = nn.Linear(input_dim, dim)
        self.pos_embed = SpatialEncoding(dim)  # Dynamic positional embedding generator

        self.blocks = nn.ModuleList([
            WinTransBlock(dim, num_heads, latent_dim, num_shared, num_routed, top_k, dim * 2, window_size)
            for _ in range(depth)
        ])

        # Attention pooling for better aggregation
        self.attn_pool = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim * 2, num_classes)

    def forward(self, x, coords):
        """
        Args:
            x (torch.Tensor): Feature bags of shape [B, N, Input_Dim]
            coords (torch.Tensor): Coordinates of shape [B, N, 2]
        """
        B, N, _ = x.shape

        # 1. Project 1280D image features to hidden dimension
        x = self.feature_proj(x)  # -> [B, N, dim]

        # 2. Generate and add spatial positional embeddings
        spatial_tokens = self.pos_embed(coords)
        x = x + spatial_tokens

        # 3. Pass through Windowed Transformer blocks
        # Use a dummy CLS token that won't be used for final classification
        dummy_cls = x.new_zeros(B, 1, x.size(-1))
        for bi, block in enumerate(self.blocks):
            x, _ = block(x, coords, dummy_cls)

        x = self.norm(x)

        # Attention pooling - learns to weight important tokens
        attn_weights = F.softmax(self.attn_pool(x), dim=1)  # [B, N, 1]
        attn_pooled = torch.sum(attn_weights * x, dim=1)    # [B, dim]

        # Mean pooling as additional global feature
        mean_pooled = x.mean(dim=1)                         # [B, dim]

        # Combine both pooling methods
        pooled = torch.cat([attn_pooled, mean_pooled], dim=-1)

        logits = self.head(pooled)
        return {"logits": logits}
