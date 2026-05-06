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

    def forward(self, x, coords, cls_tkn, H, W, mask=None):
        B, N, C = x.shape
        win_size = self.window_size

        x = torch.cat([x, coords], dim=-1)  # [B, N, C+2]
        C_total = C + 2

        # 1. Pad dimensions to be divisible by window_size
        pad_h = (win_size - (H % win_size)) % win_size
        pad_w = (win_size - (W % win_size)) % win_size
        H_p, W_p = H + pad_h, W + pad_w

        # 2. Reshape to window grid
        x = x.view(B, H, W, C_total)
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # Pad spatially, not N

        x = x.view(B, H_p // win_size, win_size, W_p // win_size, win_size, C_total)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, win_size * win_size, C_total)
        B_win, N_win, _ = x.shape

        # 3. Create attention mask for padding
        if mask is None:
            spatial_mask = torch.zeros(B, H_p, W_p, device=x.device)
            spatial_mask[:, :H, :W] = 1

            mask_grid = spatial_mask.view(B, H_p // win_size, win_size, W_p // win_size, win_size)

            mask = mask_grid.permute(0, 1, 3, 2, 4).reshape(-1, win_size * win_size).view(B_win, 1, 1, -1)


        # 4. Detach coords
        coords_win = x[:, :, -2:] / 10000.0  # [B_win, N_win, 2]
        x = x[:, :, :-2]    # [B_win, N_win, C]

        # 5. Attention with CLS Token
        # Concatenate CLS token to each window
        cls_tkn = cls_tkn.expand(B_win, -1, -1)
        x_windows = torch.cat([cls_tkn, x], dim=1)  # (B_win, N_win + 1, C)

        q = self.q_proj(x_windows)
        kv = self.kv_up(self.kv_down(x_windows))
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B_win, N_win + 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B_win, N_win + 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B_win, N_win + 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.use_rope:
            q_rope, k_rope = self.rope(coords_win, q[:, :, 1:, :], k[:, :, 1:, :])
            # Unfortunately pytorch won't let you do the inplace edit you have to concat it like this.
            q = torch.cat([q[:, :, 0:1, :], q_rope], dim=2)
            k = torch.cat([k[:, :, 0:1, :], k_rope], dim=2)

        # Apply mask: CLS has 1 mask
        full_mask = torch.cat([torch.ones(B_win, 1, 1, 1, device=x.device), mask], dim=-1)

        # Attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask)

        # 6. Reshape back
        out_cls = out[:, :, 0:1, :].transpose(1, 2).reshape(B, -1, C).mean(dim=1, keepdim=True)
        out_patches = out[:, :, 1:, :].transpose(1, 2).reshape(B, H_p, W_p, C)
        out_patches = out_patches[:, :H, :W, :].reshape(B, N, C)

        return self.out_proj(out_patches), self.out_proj(out_cls)


class WinTransBlock(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, num_shared, num_routed, top_k, hidden_dim, window_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedMLA(dim, num_heads, latent_dim, window_size)

        self.norm2 = nn.LayerNorm(dim)
        self.moe = DeepSeekMoE(dim, num_shared, num_routed, top_k, hidden_dim)

    def forward(self, x, coords, cls_token, H, W):
        # x is (B, N, C), cls_token is (B, 1, C)

        # 1. Attention: Returns updated patches and updated global token
        patches, new_cls_token = self.attn(self.norm1(x), coords, cls_token, H, W)
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.ModuleList([
            WinTransBlock(dim, num_heads, latent_dim, num_shared, num_routed, top_k, dim * 2, window_size)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, coords=None):
        """
        Args:
            x (torch.Tensor): Feature bags of shape [B, N, Input_Dim]
            coords (torch.Tensor, optional): Coordinates of shape [B, N, 2]. Required for this model.
        """
        if coords is None:
            raise ValueError("coords are required for WinDeepSeekSpatialViT")

        B, N, _ = x.shape

        # 1. Project 1280D image features to hidden dimension
        x = self.feature_proj(x)  # -> [B, N, dim]

        # 2. Calculate dynamic grid for first slide, they have to be pre-padded (or N=1) for this to work
        unique_x = torch.unique(coords[0, :, 0])
        unique_y = torch.unique(coords[0, :, 1])
        H, W = len(unique_y), len(unique_x)

        # 3. Generate and add spatial positional embeddings
        spatial_tokens = self.pos_embed(coords)
        x = x + spatial_tokens

        # 4. Pass through Windowed Transformer blocks
        current_cls = self.cls_token.expand(B, -1, -1)
        for bi, block in enumerate(self.blocks):
            x, current_cls = block(x, coords, current_cls, H, W)

        x = torch.cat([current_cls, x], dim=1)
        x = self.norm(x)

        # Classification using only the CLS token output
        logits = self.head(x[:, 0])
        return {"logits": logits}
