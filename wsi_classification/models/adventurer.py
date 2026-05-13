import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=128, d_conv=4, expand=2, headdim=64, dropout=0.1):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError:
            raise ImportError(
                "Please `pip install mamba-ssm` or do not use this model."
            )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.token_mixer = Mamba2(d_model=dim,
                                  d_state=d_state,
                                  d_conv=d_conv,
                                  expand=expand,
                                  headdim=headdim)

        self.channel_mixer = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Token mixer + residual - Mamba kernels require full precision
        x_norm = self.norm1(x)
        if x_norm.dtype != torch.float32:
            x_mixed = self.token_mixer(x_norm.float()).to(x.dtype)
        else:
            x_mixed = self.token_mixer(x_norm)
        x = x + x_mixed
        # Channel mixer + residual
        x = x + self.channel_mixer(self.norm2(x))
        return x


class SpatialEncoding(nn.Module):
    """Projects 2D coordinates (X, Y) into the hidden dimension."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, coords):
        # Scale down coordinates to prevent massive values
        normalized_coords = coords / 10000.0
        return self.proj(normalized_coords)


class Adventurer(nn.Module):
    uses_coords = True

    def __init__(self, input_dim, num_classes=2, dim=512, depth=4,
                 mamba_d_state=128, mamba_expand=2, mamba_headdim=64,
                 dropout=0.1, **kwargs):
        super().__init__()

        if dim is None:
            dim = input_dim

        self.dim = dim
        self.depth = depth

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Spatial encoding from coordinates
        self.pos_embed = SpatialEncoding(dim)

        # Aggregation token - appended at the END so it sees all tokens
        self.agg_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.blocks = nn.ModuleList([
            MambaBlock(dim=dim,
                       d_state=mamba_d_state,
                       expand=mamba_expand,
                       headdim=mamba_headdim,
                       dropout=dropout)
            for _ in range(depth)
        ])

        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, coords):
        B, N, D = x.shape

        # Transform features
        x = self.feature_proj(x)

        # Add spatial encodings from coordinates
        spatial_tokens = self.pos_embed(coords)
        x = x + spatial_tokens

        # Append aggregation token at the END
        # This is key: Mamba processes sequentially, so last token sees all
        agg_tokens = self.agg_token.expand(B, -1, -1)
        x = torch.cat([x, agg_tokens], dim=1)

        # Pass through Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Classification from the aggregation token (last position)
        x = self.norm_f(x[:, -1])
        logits = self.head(x)

        return {"logits": logits}
