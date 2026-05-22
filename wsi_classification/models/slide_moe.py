"""SlideMoE — Adapted for external dataloaders."""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 1. POOLING ──────────────────────────────────────────────────────────────
class GatedAttentionPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.attn_v = nn.Linear(dim, hidden)
        self.attn_u = nn.Linear(dim, hidden)
        self.attn_w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        v = torch.tanh(self.attn_v(x))
        u = torch.sigmoid(self.attn_u(x))
        a = self.attn_w(v * u).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        a = F.softmax(a, dim=-1)
        a = self.drop(a)
        pooled = (a.unsqueeze(-1) * x).sum(dim=1)
        return pooled, a

# ─── 2. SCORER ───────────────────────────────────────────────────────────────
class PatchScorer(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def select_topk(x: torch.Tensor, mask: torch.Tensor, scores: torch.Tensor, k: int):
    B, N, D = x.shape
    k_eff = min(k, N)

    scores_masked = scores.masked_fill(~mask, float("-inf"))
    _, idx = scores_masked.topk(k_eff, dim=-1)

    idx_expand = idx.unsqueeze(-1).expand(-1, -1, D)
    x_sel = torch.gather(x, 1, idx_expand)
    mask_sel = torch.gather(mask, 1, idx)
    scores_sel = torch.gather(scores, 1, idx)

    gate = torch.sigmoid(scores_sel).unsqueeze(-1)
    x_sel = x_sel * gate

    return x_sel, mask_sel, scores_sel, idx

# ─── 3. MIXTURE OF EXPERTS ───────────────────────────────────────────────────
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
    def __init__(self, dim: int, hidden: int, num_experts: int = 4, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(dim, hidden, dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        flat_x = x.reshape(B * N, D)
        flat_mask = mask.reshape(B * N) if mask is not None else None

        gate_logits = self.gate(flat_x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_probs, topk_idx = gate_probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(flat_x)
        for e in range(self.num_experts):
            slot_match = topk_idx == e
            token_mask = slot_match.any(dim=-1)
            if not token_mask.any():
                continue
            tok_x = flat_x[token_mask]
            w = (topk_probs * slot_match.float()).sum(dim=-1)[token_mask].unsqueeze(-1)
            out[token_mask] = out[token_mask] + w * self.experts[e](tok_x)

        out = out.reshape(B, N, D)

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
            one_hot = F.one_hot(valid_topk_idx, num_classes=self.num_experts).float()
            f = one_hot.sum(dim=(0, 1)) / (T * self.top_k)
            P = valid_gate.mean(dim=0)
            aux = self.num_experts * (f * P).sum()

        return out, aux

# ─── 4. TRANSFORMER BLOCK ────────────────────────────────────────────────────
class MoETransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, ffn_hidden: int = 2560, num_experts: int = 4, top_k: int = 2, dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MoEFFN(dim, ffn_hidden, num_experts, top_k, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        kpm = (~mask) if mask is not None else None
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)
        x = x + self.drop(h)
        h2, aux = self.moe(self.norm2(x), mask=mask)
        x = x + self.drop(h2)
        return x, aux

# ─── 5. SLIDE MOE ARCHITECTURE ───────────────────────────────────────────────
class SlideMoE(nn.Module):
    # Framework flag to indicate model expects coords, even if unused directly here
    uses_coords = True 

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
        # Aliases for framework compatibility
        in_features: int | None = None,
        out_features: int | None = None,
    ):
        super().__init__()
        # Use aliases if provided (framework compatibility)
        if in_features is not None:
            in_dim = in_features
        if out_features is not None:
            num_classes = out_features
        # Store for external access
        self.in_features = in_dim
        self.out_features = num_classes

        self.in_proj = nn.Linear(in_dim, model_dim) if in_dim != model_dim else nn.Identity()
        self.input_norm = nn.LayerNorm(model_dim)
        self.scorer = PatchScorer(model_dim, scorer_hidden, dropout)
        self.top_k_patches = top_k_patches
        self.patch_dropout = float(patch_dropout)

        self.blocks = nn.ModuleList([
            MoETransformerBlock(
                dim=model_dim, num_heads=num_heads, ffn_hidden=ffn_hidden,
                num_experts=num_experts, top_k=top_k_experts,
                dropout=dropout, attn_dropout=attn_dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(model_dim)
        self.pool = GatedAttentionPool(model_dim, pool_hidden, dropout)
        self.head = nn.Linear(model_dim, num_classes)

        if aux_scorer_head:
            self.aux_head = nn.Linear(model_dim, num_classes)
        else:
            self.aux_head = None

    def forward(
        self, x: torch.Tensor, coords: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Adapted forward pass to match the framework's dataloader signature.
        Args:
            x (torch.Tensor): Feature bags of shape [B, N, Input_Dim]
            coords (torch.Tensor, optional): Coordinates [B, N, 2]
            mask (torch.Tensor, optional): Boolean padding mask [B, N]
        """
        # If the framework dataloader does not provide a mask (e.g., standard batch_size=1 MIL),
        # we generate an all-True mask dynamically.
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

        x = self.in_proj(x)
        x = self.input_norm(x)

        if self.training and self.patch_dropout > 0.0:
            drop = torch.rand(mask.shape, device=mask.device) < self.patch_dropout
            new_mask = mask & ~drop
            empty = new_mask.sum(dim=1) == 0
            if empty.any():
                new_mask[empty] = mask[empty]
            mask = new_mask

        scores = self.scorer(x)

        aux_logits: Optional[torch.Tensor] = None
        if self.aux_head is not None:
            a = scores.masked_fill(~mask, float("-inf"))
            a = F.softmax(a, dim=-1)
            pooled_aux = torch.bmm(a.unsqueeze(1), x).squeeze(1)
            aux_logits = self.aux_head(pooled_aux)

        x_sel, mask_sel, _, topk_idx = select_topk(x, mask, scores, self.top_k_patches)

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