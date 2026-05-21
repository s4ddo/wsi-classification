from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchScorer(nn.Module):
    """Per-patch salience scorer. Returns a scalar logit per token."""

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) -> (B, N)
        return self.net(x).squeeze(-1)


def select_topk(
    x: torch.Tensor,
    mask: torch.Tensor,
    scores: torch.Tensor,
    k: int,
):
    """Hard top-k selection with score-gating so the scorer receives gradient.

    Args:
        x: (B, N, D) features.
        mask: (B, N) bool, True = valid token.
        scores: (B, N) per-token salience logit.
        k: max tokens to keep.

    Returns:
        x_sel: (B, k', D) where k' = min(k, N).
        mask_sel: (B, k') bool.
        scores_sel: (B, k') the kept logits (pre-gate).
        idx: (B, k') selection indices into the N axis.
    """
    B, N, D = x.shape
    k_eff = min(k, N)

    scores_masked = scores.masked_fill(~mask, float("-inf"))
    _, idx = scores_masked.topk(k_eff, dim=-1)  # (B, k_eff)

    idx_expand = idx.unsqueeze(-1).expand(-1, -1, D)
    x_sel = torch.gather(x, 1, idx_expand)
    mask_sel = torch.gather(mask, 1, idx)
    scores_sel = torch.gather(scores, 1, idx)

    # Gate so the scorer head receives gradient through the kept-token forward.
    gate = torch.sigmoid(scores_sel).unsqueeze(-1)
    x_sel = x_sel * gate

    return x_sel, mask_sel, scores_sel, idx
