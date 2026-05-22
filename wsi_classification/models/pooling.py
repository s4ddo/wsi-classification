from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPool(nn.Module):
    """ABMIL-style gated attention pooling (Ilse et al., 2018)."""

    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.attn_v = nn.Linear(dim, hidden)
        self.attn_u = nn.Linear(dim, hidden)
        self.attn_w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v = torch.tanh(self.attn_v(x))
        u = torch.sigmoid(self.attn_u(x))
        a = self.attn_w(v * u).squeeze(-1)  # (B, N)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        a = F.softmax(a, dim=-1)
        a = self.drop(a)
        pooled = (a.unsqueeze(-1) * x).sum(dim=1)
        return pooled, a
