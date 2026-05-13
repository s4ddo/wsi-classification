"""Deformable DETR with multi-scale deformable attention and DeepSeek MoE."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from wsi_classification.models.pos_embeds import SpatialEncoding
from wsi_classification.models.deepseek_spatial_vit_rope import DeepSeekMoE


class MultiScaleDeformableAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            num_levels=3,
            num_points=4,
            coord_stride=224.0,
            downsample_kernel=3,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.coord_stride = float(coord_stride)

        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

        self.downsamplers = nn.ModuleList([
            nn.Conv2d(
                dim, dim,
                kernel_size=downsample_kernel,
                stride=downsample_kernel // 2 + 1,
                padding=downsample_kernel // 2,
            )
            for _ in range(num_levels - 1)
        ])

        self.level_embed = nn.Parameter(torch.zeros(num_levels, dim))

        self.sampling_offsets = nn.Linear(
            dim, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            dim, num_heads * num_levels * num_points
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2 * torch.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # [H, 2]
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True).values
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for k in range(self.num_points):
            grid_init[:, :, k, :] *= (k + 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.flatten())

        # Uniform attention after softmax => zero logits.
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        nn.init.normal_(self.level_embed, std=0.02)

    @staticmethod
    def _rasterize(x_grid, grid_idx):
        """
        x_grid:   [B, N_g, D]
        grid_idx: [B, N_g, 2]  -- integer (col, row) indices in [0, W) x [0, H)
        Returns:  [B, D, H, W]
        """
        B, N_g, D = x_grid.shape
        device = x_grid.device

        W = int(grid_idx[..., 0].max().item()) + 1
        H = int(grid_idx[..., 1].max().item()) + 1

        feat = x_grid.new_zeros(B, H, W, D)

        batch_idx = torch.arange(B, device=device)[:, None].expand(-1, N_g)
        col_idx = grid_idx[..., 0].long().clamp_(0, W - 1)
        row_idx = grid_idx[..., 1].long().clamp_(0, H - 1)

        feat[batch_idx, row_idx, col_idx] = x_grid
        return feat.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

    def _sample_level(self, feat_l, sample_loc):
        """
        feat_l:     [B, D, H_l, W_l]
        sample_loc: [B, N, num_heads, num_points, 2] in [-1, 1]
        Returns:    [B, N, num_heads, num_points, head_dim]
        """
        B, _, H_l, W_l = feat_l.shape
        N = sample_loc.shape[1]
        H_, P = self.num_heads, self.num_points
        hd = self.head_dim

        # Split D into (head, head_dim) and fold (B, head) into the leading
        # dim so each head samples from its own slice of the value map
        feat_h = feat_l.view(B, H_, hd, H_l, W_l).flatten(0, 1)
        # [B*H_, hd, H_l, W_l]

        sample_loc_h = sample_loc.permute(0, 2, 1, 3, 4).flatten(0, 1)
        # [B*H_, N, P, 2]

        # grid_sample may not have bf16 kernels - use float32
        if feat_h.dtype == torch.bfloat16:
            feat_h = feat_h.float()
            sample_loc_h = sample_loc_h.float()
            sampled = F.grid_sample(
                feat_h, sample_loc_h,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.to(feat_l.dtype)
        else:
            sampled = F.grid_sample(
                feat_h, sample_loc_h,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B*H_, hd, N, P]

        sampled = sampled.view(B, H_, hd, N, P)
        return sampled.permute(0, 3, 1, 4, 2).contiguous()  # [B, N, H_, P, hd]

    def forward(self, x, coords, num_special_tokens=0):
        B, N, D = x.shape
        H_, P, L = self.num_heads, self.num_points, self.num_levels
        K = num_special_tokens

        # Convert coords to grid indices for rasterization
        grid_coords_full = coords.float() / self.coord_stride
        grid_idx = grid_coords_full[:, K:].round().long()

        # 1. Rasterize the non-special tokens into level 0
        feat0 = self._rasterize(self.value_proj(x[:, K:]), grid_idx)  # [B, D, H, W]

        # 2. Pyramid.
        feats = [feat0]
        cur = feat0
        for ds in self.downsamplers:
            cur = ds(cur)
            feats.append(cur)

        # Add level embeddings (broadcast over space)
        for l, f in enumerate(feats):
            feats[l] = f + self.level_embed[l].view(1, D, 1, 1)

        # 3. Reference points in [0, 1] grid space (col, row order)
        H0, W0 = feats[0].shape[2], feats[0].shape[3]
        wh0 = coords.new_tensor([W0, H0]).float()
        ref = (grid_coords_full + 0.5) / wh0  # [B, N, 2]

        # 4. Regress offsets / attention weights from full x (specials too)
        offsets = self.sampling_offsets(x).view(B, N, H_, L, P, 2)
        att_logits = self.attention_weights(x).view(B, N, H_, L * P)
        att_weights = F.softmax(att_logits, dim=-1).view(B, N, H_, L, P)

        # 5. Sample each level and accumulate.
        out = x.new_zeros(B, N, H_, self.head_dim)
        for l, feat_l in enumerate(feats):
            H_l, W_l = feat_l.shape[2], feat_l.shape[3]
            wh_l = coords.new_tensor([W_l, H_l]).float()

            # Offsets are in level-l pixel units => divide by (W_l, H_l) to
            # get a delta in normalized [0, 1] coords.
            sample_loc_norm = (
                    ref[:, :, None, None, :]  # [B, N, 1, 1, 2]
                    + offsets[:, :, :, l, :, :] / wh_l  # [B, N, H_, P, 2]
            )
            sample_loc_grid = 2.0 * sample_loc_norm - 1.0  # to [-1, 1]

            sampled = self._sample_level(feat_l, sample_loc_grid)
            # [B, N, H_, P, hd]
            w = att_weights[:, :, :, l, :, None]  # [B, N, H_, P, 1]
            out = out + (sampled * w).sum(dim=3)  # [B, N, H_, hd]

        out = out.reshape(B, N, D)
        return self.output_proj(out)


class DeformableTransBlock(nn.Module):
    def __init__(
            self, dim, num_heads, num_shared, num_routed, top_k_moe, mlp_hidden_dim,
            num_levels=3, num_points=4, coord_stride=224.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiScaleDeformableAttention(
            dim, num_heads=num_heads,
            num_levels=num_levels, num_points=num_points,
            coord_stride=coord_stride,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.moe = DeepSeekMoE(
            dim, num_shared, num_routed, top_k_moe, mlp_hidden_dim,
        )

    def forward(self, x, coords, num_special_tokens=0):
        x = x + self.attn(self.norm1(x), coords, num_special_tokens=num_special_tokens)
        x = x + self.moe(self.norm2(x))
        return x


class SimpleFFN(nn.Module):
    """Standard FFN replacing MoE for speed."""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DeformableTransBlockFast(nn.Module):
    """Block with standard FFN instead of MoE for speed."""
    def __init__(
            self, dim, num_heads, mlp_hidden_dim, dropout=0.1,
            num_levels=3, num_points=4, coord_stride=224.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiScaleDeformableAttention(
            dim, num_heads=num_heads,
            num_levels=num_levels, num_points=num_points,
            coord_stride=coord_stride,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SimpleFFN(dim, mlp_hidden_dim, dropout)

    def forward(self, x, coords, num_special_tokens=0):
        x = x + self.attn(self.norm1(x), coords, num_special_tokens=num_special_tokens)
        x = x + self.ffn(self.norm2(x))
        return x


class DeformableViT(nn.Module):
    uses_coords = True

    def __init__(
            self, input_dim, num_classes=2, dim=384, depth=4, num_heads=8,
            num_levels=2, num_points=4, coord_stride=224.0, dropout=0.1,
            use_moe=False, num_shared=1, num_routed=4, top_k_moe=2,
            **kwargs,
    ):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = SpatialEncoding(dim)

        if use_moe:
            self.blocks = nn.ModuleList([
                DeformableTransBlock(
                    dim, num_heads, num_shared, num_routed, top_k_moe,
                    mlp_hidden_dim=dim * 4,
                    num_levels=num_levels, num_points=num_points,
                    coord_stride=coord_stride,
                )
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                DeformableTransBlockFast(
                    dim, num_heads, mlp_hidden_dim=dim * 4, dropout=dropout,
                    num_levels=num_levels, num_points=num_points,
                    coord_stride=coord_stride,
                )
                for _ in range(depth)
            ])

        # Attention pooling for better aggregation
        self.attn_pool = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_classes)  # *2 for concat of attn_pool and mean
        )

    def forward(self, x, coords=None):
        if coords is None:
            raise ValueError("coords are required for DeformableViT")

        B, N, _ = x.shape

        x = self.feature_proj(x)
        x = x + self.pos_embed(coords)

        for block in self.blocks:
            x = block(x, coords, num_special_tokens=0)

        x = self.norm(x)

        # Attention pooling + mean pooling for global aggregation
        attn_weights = F.softmax(self.attn_pool(x), dim=1)
        attn_pooled = torch.sum(attn_weights * x, dim=1)
        mean_pooled = x.mean(dim=1)

        pooled = torch.cat([attn_pooled, mean_pooled], dim=-1)
        logits = self.head(pooled)
        return {"logits": logits}
