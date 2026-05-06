import torch
import torch.nn as nn

from wsi_classification.models.pos_embeds import SpatialEncoding


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=128, d_conv=4, expand=2, headdim=64):
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
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        # B, N, D = x.shape

        # Token mixer + residual
        x = x + self.token_mixer(self.norm1(x))
        # Channel mixer + residual
        x = x + self.channel_mixer(self.norm2(x))

        return x


class Adventurer(nn.Module):
    uses_coords = True

    def __init__(self, input_dim, num_classes=2, dim=128, depth=4,
                 mamba_d_state=128, mamba_expand=2, mamba_headdim=64, **kwargs):
        super().__init__()

        if dim is None:
            dim = input_dim

        self.feature_proj = nn.Linear(input_dim, dim)
        self.pos_embed = SpatialEncoding(dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.ModuleList([
            MambaBlock(dim=dim,
                       d_state=mamba_d_state,
                       expand=mamba_expand,
                       headdim=mamba_headdim
                       ) for _ in range(depth)
        ])

        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, coords):
        B, N, D = x.shape

        # Transform features
        x = self.feature_proj(x)
        # Spatial embedding
        spatial_tokens = self.pos_embed(coords)
        x = x + spatial_tokens
        # Append [CLS]
        x = torch.cat([x, self.cls_token.expand(B, -1, -1)], dim=1)

        for block in self.blocks:
            # 1. Heading Average
            avg = x.mean(dim=1, keepdim=True)
            x = torch.cat([avg, x], dim=1)

            # 2. Block
            x = block(x)
            x = x[:, 1:]   # Discard average

            # 3. Flip order after every block
            embeds = x[:, :-1]
            cls_token = x[:, -1:]
            x = torch.cat([embeds.flip(1), cls_token], dim=1)   # [CLS] always at the end

        # Classification based on [CLS]
        x = self.norm_f(x[:, -1])
        logits = self.head(x)

        return {"logits": logits}
