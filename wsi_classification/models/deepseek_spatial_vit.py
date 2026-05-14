import torch
import torch.nn as nn
import torch.nn.functional as F


class _SimplifiedMLA(nn.Module):
    """Multi-Head Latent Attention with sparse local window for large sequences.

    For large WSI bags (>10K patches), uses local window attention to reduce memory while
    preserving spatial locality. For small sequences, uses full attention.
    """

    def __init__(self, dim: int, num_heads: int, latent_dim: int, window_size: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(dim, dim)
        self.kv_down = nn.Linear(dim, latent_dim)
        self.kv_up = nn.Linear(latent_dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        kv_latent = self.kv_down(x)
        kv = self.kv_up(kv_latent)
        k, v = kv.chunk(2, dim=-1)

        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if N > 10000:
            out = self._sparse_local_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        return self.out_proj(out.transpose(1, 2).reshape(B, N, C))

    def _sparse_local_attention(self, q: torch.Tensor, k: torch.Tensor,
                               v: torch.Tensor) -> torch.Tensor:
        """Local window attention using sliding window (±window_size)."""
        B, H, N, D = q.shape
        w = self.window_size

        k_pad = F.pad(k.transpose(2, 3), (w, w)).transpose(2, 3)
        v_pad = F.pad(v.transpose(2, 3), (w, w)).transpose(2, 3)

        out = torch.zeros_like(q)
        for i in range(N):
            k_win = k_pad[:, :, i:i+2*w+1, :]
            v_win = v_pad[:, :, i:i+2*w+1, :]
            scores = torch.matmul(q[:, :, i:i+1, :], k_win.transpose(-2, -1)) / (D ** 0.5)
            attn = F.softmax(scores, dim=-1)
            out[:, :, i, :] = torch.matmul(attn, v_win).squeeze(2)

        return out


class _Expert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DeepSeekMoE(nn.Module):
    """Mixture-of-Experts FFN: always-on shared experts + top-k routed experts."""

    def __init__(self, dim: int, num_shared: int, num_routed: int, top_k: int, hidden_dim: int):
        super().__init__()
        self.top_k = top_k
        self.shared_experts = nn.ModuleList([_Expert(dim, hidden_dim) for _ in range(num_shared)])
        self.routed_experts = nn.ModuleList([_Expert(dim, hidden_dim) for _ in range(num_routed)])
        self.router = nn.Linear(dim, num_routed, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_flat = x.view(-1, C)

        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        route_probs = F.softmax(self.router(x_flat), dim=-1)
        topk_probs, topk_indices = torch.topk(route_probs, self.top_k, dim=-1)

        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            mask = (topk_indices == i)
            if mask.any():
                idx_tokens, idx_k = torch.where(mask)
                expert_out = expert(x_flat[idx_tokens]) * topk_probs[idx_tokens, idx_k].unsqueeze(-1)
                routed_out.index_add_(0, idx_tokens, expert_out)

        return (shared_out + routed_out).view(B, N, C)


class _ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, latent_dim: int,
                 num_shared: int, num_routed: int, top_k: int, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SimplifiedMLA(dim, num_heads, latent_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = _DeepSeekMoE(dim, num_shared, num_routed, top_k, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class _SpatialEncoding(nn.Module):
    """Projects 2D tile coordinates into the transformer hidden dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.proj(coords / 10000.0)


class DeepSeekSpatialViT(nn.Module):
    """Spatially-aware MIL Transformer using Multi-Head Latent Attention and MoE FFN.

    Processes a bag of patch features alongside their 2D WSI coordinates.  Each
    patch position is encoded via a learned spatial MLP and added to the projected
    features before being passed through a stack of DeepSeek-style ViT blocks.

    Args:
        in_features: Dimension of each patch feature vector (e.g. 1024 for UNI).
        out_features: Number of output classes.
        dim: Internal hidden dimension of the transformer.
        depth: Number of ViT blocks.
        num_heads: Number of attention heads (must divide ``dim``).
        hidden_dim: Hidden dimension for the MoE expert networks.
        latent_dim: Bottleneck size for the Multi-Head Latent Attention K/V projection.
        num_shared: Number of always-active shared experts in each MoE layer.
        num_routed: Total number of routed experts in each MoE layer.
        top_k: How many routed experts to activate per token.
    """

    uses_coords = True

    def __init__(
        self,
        in_features: int = 1024,
        out_features: int = 2,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 256,  # Kept for compat
        latent_dim: int = 64,
        num_shared: int = 1,
        num_routed: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.feature_proj = nn.Linear(in_features, dim)
        self.pos_embed = _SpatialEncoding(dim)

        self.blocks = nn.ModuleList([
            _ViTBlock(dim, num_heads, latent_dim, num_shared, num_routed, top_k, dim * 2)
            for _ in range(depth)
        ])

        self.attn_pool = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim * 2, out_features)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> dict:
        """Run a forward pass over a feature bag.

        Args:
            x: Patch features of shape (B, N, D) or (N, D).
            coords: Patch coordinates of shape (B, N, 2) or (N, 2).  When
                provided, spatial positional embeddings are added to the patch
                tokens before the transformer blocks.

        Returns:
            Dict with key ``"logits"`` of shape (B, out_features).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            if coords is not None and coords.dim() == 2:
                coords = coords.unsqueeze(0)

        B, N, D = x.shape
        if D != self.in_features:
            raise ValueError(f"Feature dimension mismatch: expected {self.in_features}, got {D}.")

        x = self.feature_proj(x)

        if coords is not None:
            x = x + self.pos_embed(coords)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        mean_pooled = x.mean(dim=1)
        attn_weights = F.softmax(self.attn_pool(x), dim=1)  # [B, N, 1]
        attn_pooled = torch.sum(attn_weights * x, dim=1)  # [B, dim]

        pooled = torch.cat([attn_pooled, mean_pooled], dim=-1)
        logits = self.head(pooled)

        return {"logits": logits}
