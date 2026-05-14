import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from wsi_classification.models.pos_embeds import SpatialEncoding, RotaryEmbedding
from wsi_classification.models.deepseek_spatial_vit_rope import DeepSeekMoE

from flash_attn import flash_attn_varlen_func
from native_sparse_attention.ops import (
    compressed_attention,
    topk_sparse_attention,
    avgpool_compress,
    weightedpool_compress,
    linear_compress,
)


_COMPRESS_FUNC = {
    "avgpool":      avgpool_compress,
    "weightedpool": weightedpool_compress,
    "linear":       linear_compress,
}

def _make_compress_weight(compress_type: str, num_heads: int, head_dim: int, kernel_size: int):
    """Return an nn.Parameter (or None for avgpool) for K/V compression."""
    if compress_type == "avgpool":
        return None
    if compress_type == "weightedpool":
        return nn.Parameter(torch.zeros(num_heads, kernel_size))
    if compress_type == "linear":
        return nn.Parameter(torch.zeros(num_heads, head_dim * kernel_size, head_dim))
    raise ValueError(f"Unknown compress_type: {compress_type!r}")


class NativeSparseMLA(nn.Module):
    """Multi-head Latent Attention backed by Native Sparse Attention triton ops.

    The three attention branches (compressed-block, topk-sparse, sliding-window)
    are blended with a per-head learned gate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        latent_dim: int,
        kernel_size: int = 128,
        kernel_stride: int = 64,
        block_size: int = 64,
        window_size: int = 128,
        topk: int = 4,
        init_blocks: int = 1,
        local_blocks: int = 2,
        compress_type: str = "avgpool",
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads    = num_heads
        self.head_dim     = dim // num_heads
        self.kernel_size  = kernel_size
        self.kernel_stride = kernel_stride
        self.block_size   = block_size
        self.topk         = topk
        self.init_blocks  = init_blocks
        self.local_blocks = local_blocks
        self.window_size  = window_size
        self.compress_type = compress_type
        self.compress_func = _COMPRESS_FUNC[compress_type]

        # MLA projections
        self.to_q    = nn.Linear(dim, dim,         bias=False)
        self.kv_down = nn.Linear(dim, latent_dim,  bias=False)
        self.kv_up   = nn.Linear(latent_dim, dim * 2, bias=False)
        self.proj_o  = nn.Linear(dim, dim,         bias=False)

        # NSA compression weights (None for avgpool)
        self.compress_key   = _make_compress_weight(compress_type, num_heads, self.head_dim, kernel_size)
        self.compress_value = _make_compress_weight(compress_type, num_heads, self.head_dim, kernel_size)

        # Intra-block positional encoding added to compressed keys
        self.intra_block_pe = nn.Parameter(
            torch.zeros(num_heads, kernel_size, self.head_dim)
        )

        # 3-way gate: (compressed, sparse, sliding)
        self.gate = nn.Sequential(
            nn.Linear(dim, num_heads * 3, bias=False),
            nn.Sigmoid(),
        )

        # 2D rope
        self.rope = RotaryEmbedding(self.head_dim)

    @staticmethod
    def _cast(t, src_tensor):   # To get the same dtype on parameters before triton kernels
            return t.to(dtype=src_tensor.dtype, device=src_tensor.device) if t is not None else None

    def _to_rope_fmt(self, t, B, N):
        """[B*N, H, D] -> [B, H, N, D]"""
        return t.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _to_packed_fmt(self, t, B, N):
        """[B, H, N, D] -> [B*N, H, D]"""
        return t.permute(0, 2, 1, 3).reshape(B * N, self.num_heads, self.head_dim)

    def forward(self, x, coords):
        B, N, D = x.shape
        device   = x.device

        # Pack [B, N, D] -> varlen [B*N, D] needed by flash/triton ops
        # cu_seqlens: [0, N, 2N, ..., B*N]  (each slide is one independent seq)
        cu_seqlens = torch.arange(0, (B + 1) * N, N, device=device, dtype=torch.int32)
        max_seqlen = N

        # The triton kernels require fp16 or bf16 - cast if needed
        compute_dtype = torch.bfloat16 if x.dtype == torch.float32 else x.dtype
        x_flat = x.reshape(B * N, D).to(compute_dtype)

        # QKV (MLA version)
        q  = self.to_q(x_flat).view(B * N, self.num_heads, self.head_dim)

        kv = self.kv_up(self.kv_down(x_flat))
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B * N, self.num_heads, self.head_dim)
        v = v.view(B * N, self.num_heads, self.head_dim)

        # Apply RoPE
        q_r = self._to_rope_fmt(q, B, N)
        k_r = self._to_rope_fmt(k, B, N)
        q_r, k_r = self.rope(coords / 10000.0, q_r, k_r, skip_first=True)
        q = self._to_packed_fmt(q_r, B, N)
        k = self._to_packed_fmt(k_r, B, N)

        # Cast all nn.Parameters that go into triton kernels
        pe  = self._cast(self.intra_block_pe, k)
        c_k = self._cast(self.compress_key, k)    # None for avgpool, Parameter otherwise
        c_v = self._cast(self.compress_value, k)

        # Native Sparse Attention
        # 1. Compress K/V for coarse block-level attention
        compressed_k, compressed_cu = self.compress_func(
            k,
            c_k,
            cu_seqlens,
            self.kernel_size,
            self.kernel_stride,
            pe,
        )
        compressed_v, _ = self.compress_func(
            v,
            c_v,
            cu_seqlens,
            self.kernel_size,
            self.kernel_stride,
            None,
        )

        # 2. Compressed attention -> identifies which sparse blocks matter
        compressed_seqlens  = compressed_cu[1:] - compressed_cu[:-1]
        max_compressed_seqlen = int(compressed_seqlens.max().item())

        compressed_out, topk_idx = compressed_attention(
            q, compressed_k, compressed_v,
            self.kernel_size,
            self.kernel_stride,
            self.block_size,
            self.topk,
            cu_seqlens,
            compressed_cu,
            max_seqlen,
            max_compressed_seqlen,
            None,   # Causal mask
            self.init_blocks,
            self.local_blocks,
        )

        # 3. Full-resolution sparse attention on the selected blocks
        sparse_out = topk_sparse_attention(
            q, k, v,
            topk_idx,
            self.block_size,
            cu_seqlens,
            None,           # no mask
        )

        # 4. Symmetric sliding-window attention
        sliding_out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            causal=False,
            window_size=(self.window_size, self.window_size),
        )

        # 5. Per-head learned gate blending the three branches (compression, local, sparse)
        # gate: [B*N, num_heads, 3]
        gate = self.gate(x_flat).view(B * N, self.num_heads, 3)

        out = (
            gate[..., 0:1] * compressed_out
            + gate[..., 1:2] * sparse_out
            + gate[..., 2:3] * sliding_out
        )  # [B*N, num_heads, head_dim]

        out = out.reshape(B, N, D).to(x.dtype)
        return self.proj_o(out)


class NSATransBlock(nn.Module):
    def __init__(
        self, dim, num_heads, latent_dim, num_shared, num_routed, top_k_moe,
        mlp_hidden_dim,
        kernel_size_nsa,
        kernel_stride_nsa,
        block_size_nsa,
        window_size_nsa,
        top_k_nsa,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = NativeSparseMLA(dim,
                                     num_heads,
                                     latent_dim,
                                     kernel_size_nsa,
                                     kernel_stride_nsa,
                                     block_size_nsa,
                                     window_size_nsa,
                                     top_k_nsa,
                                     )

        self.norm2 = nn.LayerNorm(dim)
        self.moe   = DeepSeekMoE(dim, num_shared, num_routed, top_k_moe, mlp_hidden_dim)

    def forward(self, x, coords=None):
        x = x + self.attn(self.norm1(x), coords)
        x = x + self.moe(self.norm2(x))
        return x


class NSADeepSeekSpatialViT(nn.Module):
    uses_coords = True

    def __init__(
        self, input_dim,
        num_classes=2,
        dim=128,
        depth=4,
        num_heads=4,
        latent_dim=64,
        num_shared=1,
        num_routed=4,
        top_k_moe=2,
        kernel_size_nsa=128,
        kernel_stride_nsa=64,
        block_size_nsa=64,
        window_size_nsa=128,
        top_k_nsa=4,
        **kwargs,
    ):
        super().__init__()

        # The Triton avgpool/weightedpool kernel only accepts these kernel sizes.
        _VALID_KERNEL_SIZES = {16, 32, 64, 128}
        kernel_size_clamped = max(k for k in _VALID_KERNEL_SIZES if k <= max(kernel_size_nsa, 16))
        if kernel_size_clamped != kernel_size_nsa:
            import warnings
            warnings.warn(
                f"kernel_size_nsa={kernel_size_nsa} is not in "
                f"{_VALID_KERNEL_SIZES}. Using kernel_size_nsa={kernel_size_clamped}.",
                UserWarning,
            )
        kernel_size_nsa = kernel_size_clamped

        self.feature_proj = nn.Linear(input_dim, dim)
        self.pos_embed = SpatialEncoding(dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.ModuleList([
            NSATransBlock(
                dim, num_heads, latent_dim, num_shared, num_routed, top_k_moe,
                dim * 2,
                kernel_size_nsa,
                kernel_stride_nsa,
                block_size_nsa,
                window_size_nsa,
                top_k_nsa,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, coords):
        B, N, _ = x.shape

        x = self.feature_proj(x)
        spatial_tokens = self.pos_embed(coords)
        x = x + spatial_tokens

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)                          # [B, N+1, dim]

        cls_coords  = torch.zeros(B, 1, 2, device=coords.device)
        full_coords = torch.cat([cls_coords, coords], dim=1)

        for block in self.blocks:
            x = block(x, full_coords)

        x = self.norm(x)
        logits = self.head(x[:, 0])
        return {"logits": logits}
