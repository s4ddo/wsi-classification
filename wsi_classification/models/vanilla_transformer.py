import torch
import torch.nn as nn
import torch.nn.functional as F


class _Attention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, C))


class _FFN(nn.Module):
    """Standard feedforward network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ViTBlock(nn.Module):
    """Transformer block with pre-norm attention and FFN."""

    def __init__(self, dim: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
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


class VanillaTransformer(nn.Module):
    """Vanilla transformer for direct sequence classification without MIL aggregation.

    Takes a sequence of instance features and uses standard transformer self-attention
    to process them, then aggregates with a learnable CLS token for classification.
    Optionally supports spatial coordinate encoding.

    Args:
        in_features: Dimension of each instance feature vector.
        out_features: Number of output classes.
        dim: Internal hidden dimension of the transformer.
        depth: Number of ViT blocks.
        num_heads: Number of attention heads in transformer.
        hidden_dim: Hidden dimension for the transformer feedforward networks.
        pool_method: How to aggregate sequence to classification logits. Options:
            - "cls": Use a learnable [CLS] token (default, like BERT).
            - "mean": Use mean pooling over the sequence.
            - "max": Use max pooling over the sequence.
    """

    def __init__(
        self,
        in_features: int = 1280,
        out_features: int = 1,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        pool_method: str = "cls",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_method = pool_method

        if pool_method not in ("cls", "mean", "max"):
            raise ValueError(f"pool_method must be one of 'cls', 'mean', 'max', got {pool_method}.")

        # Project features to hidden dimension
        self.feature_proj = nn.Linear(in_features, dim)
        self.pos_embed = _SpatialEncoding(dim)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            _ViTBlock(dim, num_heads, hidden_dim)
            for _ in range(depth)
        ])

        # Final norm and classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_features)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None, return_attention: bool = False) -> dict:
        """Run a forward pass over a sequence of features.

        Args:
            x: Instance features of shape (B, N, D) or (N, D). A batch dimension
                is added automatically when the input is 2-D.
            coords: Optional patch coordinates of shape (B, N, 2) or (N, 2).
                When provided, spatial positional embeddings are added to the patch
                tokens before the transformer blocks.
            return_attention: If True, the returned dict will contain attention weights.
                Note: This returns a simplified attention representation for compatibility
                with the MIL interface, not full multi-head attention.

        Returns:
            A dictionary with:
                - ``"logits"`` (torch.Tensor): Predictions, shape (B, out_features).
                - ``"attention"`` (torch.Tensor, optional): Sequence importance scores,
                  shape (B, 1, N). Only present when *return_attention* is True.

        Raises:
            ValueError: If *x* is not 2-D or 3-D, or if the feature dimension does
                not match ``in_features``.
        """
        if x.dim() not in (2, 3):
            raise ValueError(f"Expected 2-D (N, D) or 3-D (B, N, D) input, got {x.dim()}-D tensor.")

        # Ensure batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)
            if coords is not None and coords.dim() == 2:
                coords = coords.unsqueeze(0)

        B, N, D = x.shape
        if D != self.in_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.in_features}, got {D}."
            )

        # Project features to hidden dimension
        x = self.feature_proj(x)

        # Add spatial encoding if provided
        if coords is not None:
            x = x + self.pos_embed(coords)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Pool based on method
        if self.pool_method == "cls":
            pooled = x[:, 0, :]
            seq_len = N + 1
        elif self.pool_method == "mean":
            pooled = x[:, 1:, :].mean(dim=1)
            seq_len = N
        elif self.pool_method == "max":
            pooled = x[:, 1:, :].max(dim=1)[0]
            seq_len = N

        logits = self.head(pooled)

        out = {"logits": logits}
        if return_attention:
            attention_scores = torch.ones(B, 1, seq_len, device=x.device) / seq_len
            out["attention"] = attention_scores
        return out
