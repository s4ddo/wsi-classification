"""SlideMoE — Virchow2 features -> top-k salience -> MoE-Transformer -> attn pool -> head."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import GatedAttentionPool
from .scorer import PatchScorer, select_topk
from .transformer import MoETransformerBlock


class SlideMoE(nn.Module):
    def __init__(
        self,
        in_dim: int = 1280,
        model_dim: int = 1280,
        num_classes: int = 1,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_hidden: int = 2560,
        num_experts: int = 4,
        top_k_experts: int = 2,
        top_k_patches: int = 8000,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        scorer_hidden: int = 256,
        pool_hidden: int = 512,
        aux_scorer_head: bool = True,
        patch_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = (
            nn.Linear(in_dim, model_dim) if in_dim != model_dim else nn.Identity()
        )
        self.input_norm = nn.LayerNorm(model_dim)
        self.scorer = PatchScorer(model_dim, scorer_hidden, dropout)
        self.top_k_patches = top_k_patches
        self.patch_dropout = float(patch_dropout)

        self.blocks = nn.ModuleList(
            [
                MoETransformerBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    ffn_hidden=ffn_hidden,
                    num_experts=num_experts,
                    top_k=top_k_experts,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.pool = GatedAttentionPool(model_dim, pool_hidden, dropout)
        self.head = nn.Linear(model_dim, num_classes)

        if aux_scorer_head:
            self.aux_head = nn.Linear(model_dim, num_classes)
        else:
            self.aux_head = None

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # features: (B, N, in_dim); mask: (B, N) bool, True = valid
        x = self.in_proj(features)
        x = self.input_norm(x)

        # Instance dropout: randomly hide a fraction of patches during training.
        # Strong, cheap regulariser for MIL on few hundred slides (the model was
        # memorising training bags). Rows that would empty out are reverted.
        if self.training and self.patch_dropout > 0.0:
            drop = torch.rand(mask.shape, device=mask.device) < self.patch_dropout
            new_mask = mask & ~drop
            empty = new_mask.sum(dim=1) == 0
            if empty.any():
                new_mask[empty] = mask[empty]
            mask = new_mask

        scores = self.scorer(x)  # (B, N)

        # Optional auxiliary slide head — pool ALL tokens by softmax(scores).
        # Gives the scorer a direct slide-level gradient and a cheap regulariser
        # against router/scorer collapse. bmm avoids materialising (B, N, D) so
        # this stays cheap even for 100k-patch slides.
        aux_logits: Optional[torch.Tensor] = None
        if self.aux_head is not None:
            a = scores.masked_fill(~mask, float("-inf"))
            a = F.softmax(a, dim=-1)
            pooled_aux = torch.bmm(a.unsqueeze(1), x).squeeze(1)
            aux_logits = self.aux_head(pooled_aux)

        # Hard top-k for the transformer stack (with gated-by-sigmoid kept tokens).
        x_sel, mask_sel, _, topk_idx = select_topk(
            x, mask, scores, self.top_k_patches
        )

        total_lb = x.new_zeros(())
        for blk in self.blocks:
            x_sel, lb = blk(x_sel, mask=mask_sel)
            total_lb = total_lb + lb
        if len(self.blocks) > 0:
            total_lb = total_lb / len(self.blocks)

        x_sel = self.norm(x_sel)
        pooled, attn = self.pool(x_sel, mask=mask_sel)
        logits = self.head(pooled)

        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "load_balance_loss": total_lb,
            "pool_attn": attn,
            "topk_idx": topk_idx,
        }
