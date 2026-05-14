import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=128, d_conv=4, expand=2, headdim=64, dropout=0.1, bidirectional=False):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError:
            raise ImportError(
                "Please `pip install mamba-ssm[causal-conv1d] --no-build-isolation` or do not use this model."
            )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.bidirectional = bidirectional

        self.token_mixer = Mamba2(d_model=dim,
                                  d_state=d_state,
                                  d_conv=d_conv,
                                  expand=expand,
                                  headdim=headdim)

        if bidirectional:
            self.token_mixer_rev = Mamba2(d_model=dim,
                                          d_state=d_state,
                                          d_conv=d_conv,
                                          expand=expand,
                                          headdim=headdim)
            self.norm_rev = nn.LayerNorm(dim)
            self.fusion = nn.Linear(dim * 2, dim)

        self.channel_mixer = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Token mixer + residual
        x_out = self.token_mixer(self.norm1(x))

        if self.bidirectional:
            # Reverse direction
            x_rev = self.token_mixer_rev(self.norm_rev(x.flip(1)))
            x_rev = x_rev.flip(1)
            # Fusion
            x_out = self.fusion(torch.cat([x_out, x_rev], dim=-1))

        x = x + x_out
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
                 dropout=0.1, bidirectional=True, **kwargs):
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

        self.blocks = nn.ModuleList([
            MambaBlock(dim=dim,
                       d_state=mamba_d_state,
                       expand=mamba_expand,
                       headdim=mamba_headdim,
                       dropout=dropout,
                       bidirectional=bidirectional)
            for _ in range(depth)
        ])

        # Attention pooling for better aggregation
        self.attn_pool = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_classes)  # *2 for concat of attn_pool and mean
        )

    def forward(self, x, coords):
        B, N, D = x.shape

        # Transform features
        x = self.feature_proj(x)

        # Add spatial encodings from coordinates
        spatial_tokens = self.pos_embed(coords)
        x = x + spatial_tokens

        # Pass through Mamba blocks
        for block in self.blocks:
            # Prepend global mean as direction token
            avg = x.mean(dim=1, keepdim=True)
            x = torch.cat([avg, x], dim=1)
            # Run block
            x = block(x)
            x = x[:, 1:]  # Discard heading token

        x = self.norm_f(x)

        # Attention pooling - learns to weight important tokens
        attn_weights = F.softmax(self.attn_pool(x), dim=1)  # [B, N, 1]
        attn_pooled = torch.sum(attn_weights * x, dim=1)    # [B, dim]

        # Mean pooling as additional global feature
        mean_pooled = x.mean(dim=1)                         # [B, dim]

        # Combine both pooling methods
        pooled = torch.cat([attn_pooled, mean_pooled], dim=-1)

        logits = self.head(pooled)

        return {"logits": logits}
