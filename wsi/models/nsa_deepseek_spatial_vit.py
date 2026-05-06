import torch
import torch.nn as nn
import torch.nn.functional as F

from local_attention import LocalAttention

from wsi.models.pos_embeds import SpatialEncoding, RotaryEmbedding
from wsi.models.deepseek_spatial_vit import DeepSeekMoE


# Optional FlexAttention import - used only when fine_attn_backend='flex'.
# FlexAttention currently runs on CUDA and (partially) CPU.
try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
    from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask

    if torch.cuda.is_available():
        _flex_attention = torch.compile(_flex_attention)

    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False


class NativeSparseMLA(nn.Module):
    """
    Native Sparse Attention with MLA — no N×N materialization.

    Three branches and their attention-matrix sizes:
      1. Compression (q × compressed-block-keys):
         [B, H, N, num_blocks]  where num_blocks = N / block_size
      2. Selection   (q × top-k-block-keys):
         [B, H, N, top_k * block_size]
      3. Sliding window:
         Now handled by `local_attention.LocalAttention`,
         which tiles into windows so memory is O(N · window_size). Same as window_deepseek_spatial_vit.py but lib
    """

    def __init__(
        self,
        dim,
        num_heads,
        latent_dim,
        block_size=16,
        window_size_nsa=64,
        top_k=4,
        fine_attn_backend="gather",  # "gather" | "flex"
        use_rope=True,
    ):
        super().__init__()
        assert fine_attn_backend in ("gather", "flex"), \
            f"unknown fine_attn_backend: {fine_attn_backend!r}"
        if fine_attn_backend == "flex" and not HAS_FLEX_ATTENTION:
            raise ImportError(
                "fine_attn_backend='flex' needs PyTorch >= 2.5 with "
                "torch.nn.attention.flex_attention available."
            )
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.fine_attn_backend = fine_attn_backend

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim
        self.block_size = block_size
        self.window_size = window_size_nsa
        self.top_k = top_k
        self.use_rope = use_rope

        self.q_proj  = nn.Linear(dim, dim)
        self.kv_down = nn.Linear(dim, latent_dim)
        # Independent KV projections per branch (prevents gradient interference)
        self.kv_up_cmp = nn.Linear(latent_dim, dim * 2)
        self.kv_up_slc = nn.Linear(latent_dim, dim * 2)
        self.kv_up_win = nn.Linear(latent_dim, dim * 2)

        # Compression MLP (phi): block of tokens -> single compressed token
        self.phi_k = nn.Linear(block_size * self.head_dim, self.head_dim)
        self.phi_v = nn.Linear(block_size * self.head_dim, self.head_dim)

        # Gating: Sigmoid(MLP(x))
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 3),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Linear(dim, dim)

        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim)

        # Sliding-window attention without materializing N×N.
        self.local_attn = LocalAttention(
            window_size=max(self.window_size // 2, 1),
            causal=False,
            look_backward=1,
            look_forward=1,
            exact_windowsize=True,
            autopad=True,
            use_rotary_pos_emb=False,
        )

    def _selection_gather(self, q, k_slc_b, v_slc_b, topk_idx, B, H, N, D, scale):
        """
        Portable per-query top-k selection.
        Output: [B, H, N, D].
        """
        eff_top_k = topk_idx.shape[-1]

        # Add an N dim to the source via stride-0 broadcast view (no allocation),
        # then gather along dim=3 (num_blocks). Only gather output is materialized.
        src_k = k_slc_b.unsqueeze(2).expand(-1, -1, N, -1, -1, -1)
        src_v = v_slc_b.unsqueeze(2).expand(-1, -1, N, -1, -1, -1)
        idx   = topk_idx[..., None, None].expand(-1, -1, -1, -1, self.block_size, D)

        k_sel = torch.gather(src_k, 3, idx)            # [B, H, N, top_k, block_size, D]
        v_sel = torch.gather(src_v, 3, idx)

        J = eff_top_k * self.block_size
        k_sel = k_sel.reshape(B, H, N, J, D)
        v_sel = v_sel.reshape(B, H, N, J, D)

        scores_slc = torch.einsum("bhnd,bhnjd->bhnj", q, k_sel) * scale
        attn_slc   = F.softmax(scores_slc, dim=-1)
        out_slc    = torch.einsum("bhnj,bhnjd->bhnd", attn_slc, v_sel)
        return out_slc

    def _selection_flex(self, q, k_slc_raw, v_slc_raw, topk_idx, num_blocks, pad_len, B, H, N, D):
        """
        Per-query top-k selection via FlexAttention.
        """
        block_size = self.block_size
        N_p = N + pad_len
        device = q.device

        # one_hot[b, h, n, j] = True iff query n at (b, h) selected compressed block j.
        one_hot = q.new_zeros(B, H, N, num_blocks, dtype=torch.bool)
        one_hot.scatter_(-1, topk_idx, True)

        # Mask function: for q in padding (q_idx >= N), clamp into the valid range.
        # Outputs at those positions are discarded after the call, so it doesn't
        # matter what they attend to as long as softmax doesn't see all-False.
        def fine_mask_fn(b_idx, h_idx, q_idx, kv_idx):
            safe_q = torch.clamp(q_idx, max=N - 1)
            compressed_kv = kv_idx // block_size
            return one_hot[b_idx, h_idx, safe_q, compressed_kv]

        block_mask = _create_block_mask(
            fine_mask_fn,
            B=B, H=H, Q_LEN=N_p, KV_LEN=N_p,
            device=device,
        )

        # Pad q and the (un-blocked) k/v to a multiple of block_size for the kernel.
        q_p = F.pad(q,         (0, 0, 0, pad_len))
        k_p = F.pad(k_slc_raw, (0, 0, 0, pad_len))
        v_p = F.pad(v_slc_raw, (0, 0, 0, pad_len))

        out_p = _flex_attention(q_p, k_p, v_p, block_mask=block_mask)
        return out_p[..., :N, :]

    def forward(self, x, coords=None):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        scale = D ** -0.5

        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)        # [B, H, N, D]
        kv_latent = self.kv_down(x)                                # [B, N, latent_dim]

        # 1. Sliding window
        kv_win = self.kv_up_win(kv_latent)
        k_win, v_win = kv_win.chunk(2, dim=-1)
        k_win = k_win.view(B, N, H, D).transpose(1, 2)
        v_win = v_win.view(B, N, H, D).transpose(1, 2)

        if self.use_rope and coords is not None:
            q, k_win = self.rope(coords, q, k_win, skip_first=True)

        out_win = self.local_attn(q, k_win, v_win)  # [B, H, N, D]

        # 2. Compression
        # Attention matrix here is [B, H, N, num_blocks] -- not N×N.
        kv_cmp = self.kv_up_cmp(kv_latent)
        k_cmp_raw, v_cmp_raw = kv_cmp.chunk(2, dim=-1)

        pad_len = (self.block_size - (N % self.block_size)) % self.block_size
        k_cmp_p = F.pad(k_cmp_raw, (0, 0, 0, pad_len))
        v_cmp_p = F.pad(v_cmp_raw, (0, 0, 0, pad_len))
        N_p = N + pad_len
        num_blocks = N_p // self.block_size

        # [B, num_blocks, H, block_size, D]
        k_cmp_b = k_cmp_p.view(B, num_blocks, self.block_size, H, D).permute(0, 1, 3, 2, 4)
        v_cmp_b = v_cmp_p.view(B, num_blocks, self.block_size, H, D).permute(0, 1, 3, 2, 4)

        # phi compresses each block -> 1 token. [B, H, num_blocks, D]
        k_cmp = self.phi_k(k_cmp_b.reshape(B, num_blocks, H, -1)).transpose(1, 2)
        v_cmp = self.phi_v(v_cmp_b.reshape(B, num_blocks, H, -1)).transpose(1, 2)

        scores_cmp = torch.matmul(q, k_cmp.transpose(-2, -1)) * scale  # [B, H, N, num_blocks]
        attn_cmp   = F.softmax(scores_cmp, dim=-1)
        out_cmp    = torch.matmul(attn_cmp, v_cmp)                     # [B, H, N, D]

        # 3. Selection
        # Block-importance scores by averaging compression logits across queries.
        eff_top_k = min(self.top_k, num_blocks)     # [B, H, num_blocks]
        _, topk_idx = scores_cmp.topk(eff_top_k, dim=-1)    # [B, H, top_k]

        kv_slc = self.kv_up_slc(kv_latent)
        k_slc_raw, v_slc_raw = kv_slc.chunk(2, dim=-1)
        k_slc_raw = k_slc_raw.view(B, N, H, D).transpose(1, 2)
        v_slc_raw = v_slc_raw.view(B, N, H, D).transpose(1, 2)

        if self.fine_attn_backend == "gather":
            k_slc_p = F.pad(k_slc_raw, (0, 0, 0, pad_len))
            v_slc_p = F.pad(v_slc_raw, (0, 0, 0, pad_len))
            k_slc_b = k_slc_p.view(B, H, num_blocks, self.block_size, D)
            v_slc_b = v_slc_p.view(B, H, num_blocks, self.block_size, D)
            out_slc = self._selection_gather(q, k_slc_b, v_slc_b, topk_idx, B, H, N, D, scale)

        elif self.fine_attn_backend == "flex":
            # If you're on CUDA and have it installed, you can swap this for the
            # lucidrains triton kernel here:
            #
            #   from native_sparse_attention_pytorch.triton_native_sparse_attention \
            #       import native_sparse_attend
            #   fmask = topk_idx.new_ones(topk_idx.shape, dtype=torch.bool)
            #   out_slc = native_sparse_attend(
            #       q, k_slc_raw, v_slc_raw,
            #       block_size=self.block_size,
            #       selected_block_indices=topk_idx,
            #       fmask=fmask,
            #       include_block_causal=False,
            #   )
            out_slc = self._selection_flex(
                q, k_slc_raw, v_slc_raw, topk_idx, num_blocks, pad_len, B, H, N, D
            )
        else:
            raise RuntimeError(f"unknown backend {self.fine_attn_backend!r}")
            # [B, H, N, D]

        # 4. Aggregation
        gates = self.gate_mlp(x)                                       # [B, N, 3]
        g_win = gates[:, :, 0:1].unsqueeze(1)                          # [B, 1, N, 1]
        g_cmp = gates[:, :, 1:2].unsqueeze(1)
        g_slc = gates[:, :, 2:3].unsqueeze(1)

        out = g_win * out_win + g_cmp * out_cmp + g_slc * out_slc      # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class NSATransBlock(nn.Module):
    def __init__(
        self, dim, num_heads, latent_dim, num_shared, num_routed, top_k_moe,
        mlp_hidden_dim, block_size, window_size_nsa, top_k_nsa, fine_attn_backend="gather",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = NativeSparseMLA(dim,
                                     num_heads,
                                     latent_dim,
                                     block_size,
                                     window_size_nsa,
                                     top_k_nsa,
                                     fine_attn_backend=fine_attn_backend)

        self.norm2 = nn.LayerNorm(dim)
        self.moe   = DeepSeekMoE(dim, num_shared, num_routed, top_k_moe, mlp_hidden_dim)

    def forward(self, x, coords=None):
        x = x + self.attn(self.norm1(x), coords)
        x = x + self.moe(self.norm2(x))
        return x


class NSADeepSeekSpatialViT(nn.Module):
    def __init__(
        self, input_dim, num_classes=2, dim=128, depth=4, num_heads=4,
        latent_dim=64, num_shared=1, num_routed=4, top_k_moe=2,
        block_size=16, window_size_nsa=64, top_k_nsa=4,
        fine_attn_backend: str = "gather",
        **kwargs,
    ):
        super().__init__()

        self.feature_proj = nn.Linear(input_dim, dim)
        self.pos_embed = SpatialEncoding(dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.ModuleList([
            NSATransBlock(
                dim, num_heads, latent_dim, num_shared, num_routed, top_k_moe,
                dim * 2, block_size, window_size_nsa, top_k_nsa,
                fine_attn_backend=fine_attn_backend,
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
        return self.head(x[:, 0])
