from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEFFN(nn.Module):
    """Sparse mixture-of-experts FFN with top-k gating.

    Load-balancing auxiliary loss follows the Switch Transformer / GShard form
    generalised to top-k: aux = num_experts * sum_i (f_i / k) * P_i, where
    f_i is the fraction of routed slots assigned to expert i and P_i is the
    mean gate probability for expert i over valid tokens.
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [Expert(dim, hidden, dropout) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        flat_x = x.reshape(B * N, D)
        flat_mask = mask.reshape(B * N) if mask is not None else None

        gate_logits = self.gate(flat_x)  # (T, E)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_probs, topk_idx = gate_probs.topk(self.top_k, dim=-1)  # (T, K)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(flat_x)
        # Dispatch per expert: gather the tokens that selected expert e in any slot.
        for e in range(self.num_experts):
            slot_match = topk_idx == e  # (T, K)
            token_mask = slot_match.any(dim=-1)  # (T,)
            if not token_mask.any():
                continue
            tok_x = flat_x[token_mask]
            # Sum of routing weights for expert e for those tokens (1 contribution per slot).
            w = (topk_probs * slot_match.float()).sum(dim=-1)[token_mask].unsqueeze(-1)
            out[token_mask] = out[token_mask] + w * self.experts[e](tok_x)

        out = out.reshape(B, N, D)

        # Load-balancing auxiliary loss over valid tokens only.
        if flat_mask is not None:
            valid_gate = gate_probs[flat_mask]
            valid_topk_idx = topk_idx[flat_mask]
        else:
            valid_gate = gate_probs
            valid_topk_idx = topk_idx

        if valid_gate.numel() == 0:
            aux = x.new_zeros(())
        else:
            T = valid_topk_idx.size(0)
            # f_i: fraction of routing slots that went to expert i (sums to 1 across i).
            one_hot = F.one_hot(valid_topk_idx, num_classes=self.num_experts).float()
            f = one_hot.sum(dim=(0, 1)) / (T * self.top_k)
            P = valid_gate.mean(dim=0)
            aux = self.num_experts * (f * P).sum()

        return out, aux
