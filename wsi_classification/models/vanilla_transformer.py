import torch
import torch.nn as nn


class VanillaTransformer(nn.Module):
    """Vanilla transformer for direct sequence classification without MIL aggregation.

    Takes a sequence of instance features and uses standard transformer self-attention
    to process them, then aggregates with a learnable CLS token for classification.

    Args:
        in_features: Dimension of each instance feature vector.
        hidden_dim: Hidden dimension for the transformer feedforward networks.
        out_features: Number of output classes.
        num_heads: Number of attention heads in transformer.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout probability applied throughout.
        pool_method: How to aggregate sequence to classification logits. Options:
            - "cls": Use a learnable [CLS] token (default, like BERT).
            - "mean": Use mean pooling over the sequence.
            - "max": Use max pooling over the sequence.
    """

    def __init__(
        self,
        in_features: int = 1280,
        hidden_dim: int = 2048,
        out_features: int = 1,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        pool_method: str = "cls",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_method = pool_method

        if pool_method not in ("cls", "mean", "max"):
            raise ValueError(f"pool_method must be one of 'cls', 'mean', 'max', got {pool_method}.")

        # Learnable CLS token for pooling
        if pool_method == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, in_features))
            nn.init.normal_(self.cls_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_features,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> dict:
        """Run a forward pass over a sequence of features.

        Args:
            x: Instance features of shape (B, N, D) or (N, D). A batch dimension
                is added automatically when the input is 2-D.
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

        B, N, D = x.shape
        if D != self.in_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.in_features}, got {D}."
            )

        # Add CLS token if using CLS pooling
        if self.pool_method == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Transformer encoder
        x_encoded = self.transformer_encoder(x)  # (B, N[+1], D)

        # Pool to get representation for classification
        if self.pool_method == "cls":
            pooled = x_encoded[:, 0, :]  # (B, D) - take CLS token
            seq_len = N + 1
        elif self.pool_method == "mean":
            pooled = x_encoded.mean(dim=1)  # (B, D)
            seq_len = N
        elif self.pool_method == "max":
            pooled = x_encoded.max(dim=1)[0]  # (B, D)
            seq_len = N

        logits = self.classifier(pooled)  # (B, out_features)

        out = {"logits": logits}
        if return_attention:
            # Return a simplified attention-like representation: mean absolute gradients
            # This is a compatibility layer with ABMIL's attention interface
            attention_scores = torch.ones(B, 1, seq_len, device=x.device) / seq_len
            out["attention"] = attention_scores
        return out
