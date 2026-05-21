from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .moe import MoEFFN


class MoETransformerBlock(nn.Module):
    """Pre-norm transformer block where the FFN is a sparse MoE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_hidden: int = 2560,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MoEFFN(dim, ffn_hidden, num_experts, top_k, dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # mask: (B, N) bool, True = valid. MultiheadAttention expects key_padding_mask
        # with True = pad, so invert.
        kpm = (~mask) if mask is not None else None
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)
        x = x + self.drop(h)
        h2, aux = self.moe(self.norm2(x), mask=mask)
        x = x + self.drop(h2)
        return x, aux
