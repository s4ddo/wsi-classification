"""DeepSeekSpatialViT with full Rotary Positional Embeddings (RoPE) support.

This is the original implementation with coordinates and RoPE integration.
For the simplified window-based attention variant, see deepseek_spatial_vit.py.
"""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from wsi_classification.models.pos_embeds import SpatialEncoding, RotaryEmbedding


# 1. Multi-Head Latent Attention & MoE Blocks
class SimplifiedMLA(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, use_rope=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.use_rope = use_rope

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.kv_down = nn.Linear(dim, latent_dim)
        self.kv_up = nn.Linear(latent_dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, coords=None):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        kv_latent = self.kv_down(x)
        kv = self.kv_up(kv_latent)
        k, v = kv.chunk(2, dim=-1)

        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if coords is None:
                warnings.warn('coords required to use RoPE, but none were provided. Skipping RoPE.', Warning)
            else:
                q_rope, k_rope = self.rope(coords / 10000.0, q[:, :, 1:, :], k[:, :, 1:, :])
                # Unfortunately pytorch won't let you do the inplace edit you have to concat it like this.
                q = torch.cat([q[:, :, 0:1, :], q_rope], dim=2)
                k = torch.cat([k[:, :, 0:1, :], k_rope], dim=2)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)


class DeepSeekMoE(nn.Module):
    def __init__(self, dim, num_shared, num_routed, top_k, hidden_dim):
        super().__init__()
        self.top_k = top_k
        self.shared_experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_shared)])
        self.routed_experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_routed)])
        self.router = nn.Linear(dim, num_routed, bias=False)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        x_flat = x.view(-1, C)

        shared_out = sum(expert(x_flat) for expert in self.shared_experts)
        route_logits = self.router(x_flat)
        if mask is not None:
            route_probs = F.softmax(route_logits, dim=-1) * mask.unsqueeze(-1)
        else:
            route_probs = F.softmax(route_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(route_probs, self.top_k, dim=-1)

        routed_out = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.routed_experts):
            mask = (topk_indices == i)
            if mask.any():
                idx_tokens, idx_k = torch.where(mask)
                expert_in = x_flat[idx_tokens]
                expert_out = expert(expert_in) * topk_probs[idx_tokens, idx_k].unsqueeze(-1)
                routed_out.index_add_(0, idx_tokens, expert_out)

        out = shared_out + routed_out
        return out.view(B, N, C)


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, num_shared, num_routed, top_k, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimplifiedMLA(dim, num_heads, latent_dim)

        self.norm2 = nn.LayerNorm(dim)
        self.moe = DeepSeekMoE(dim, num_shared, num_routed, top_k, hidden_dim)

    def forward(self, x, coords=None):
        x = x + self.attn(self.norm1(x), coords)
        x = x + self.moe(self.norm2(x))
        return x


class DeepSeekSpatialViTRoPE(nn.Module):
    uses_coords = True

    def __init__(self, input_dim=1280, num_classes=2, dim=128, depth=4,
                 num_heads=4, latent_dim=64, num_shared=1, num_routed=4, top_k=2, **kwargs):
        super().__init__()

        self.feature_proj = nn.Linear(input_dim, dim)
        self.pos_embed = SpatialEncoding(dim)  # Dynamic positional embedding generator
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.ModuleList([
            ViTBlock(dim, num_heads, latent_dim, num_shared, num_routed, top_k, dim * 2)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, coords=None):
        """
        Args:
            x (torch.Tensor): Feature bags of shape [B, N, Input_Dim]
            coords (torch.Tensor, optional): Coordinates of shape [B, N, 2]. Required for this model.
        """
        if coords is None:
            raise ValueError("coords are required for DeepSeekSpatialViTRoPE")

        B, N, _ = x.shape

        # 1. Project 1280D image features to hidden dimension
        x = self.feature_proj(x)  # -> [B, N, dim]

        # 2. Generate and add spatial positional embeddings
        spatial_tokens = self.pos_embed(coords)
        x = x + spatial_tokens

        # 3. Prepend CLS token for classification
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # -> [B, N+1, dim]

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, coords)

        x = self.norm(x)

        # Classification using only the CLS token output
        logits = self.head(x[:, 0])
        return {"logits": logits}
