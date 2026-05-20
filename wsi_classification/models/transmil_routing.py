import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from wsi_classification.models.transmil import TransMIL
from wsi_classification.models.routing_attention import RoutingAttention


class RoutingTransMIL(nn.Module):
    def __init__(
        self,
        in_features: int = 1280,
        hidden_dim: int = 512,
        out_features: int = 1,
        num_clusters: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.base_model = TransMIL(
            in_features=in_features,
            hidden_dim=hidden_dim,
            out_features=out_features,
            dropout=dropout,
        )

        # Replace attention layers with routing attention
        self.base_model.layer1.attn = RoutingAttention(
            dim=hidden_dim, num_clusters=num_clusters, num_heads=num_heads
        )
        self.base_model.layer2.attn = RoutingAttention(
            dim=hidden_dim, num_clusters=num_clusters, num_heads=num_heads
        )

        # CLS token for routing attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x, valid_mask=None):
        if x.dim() == 4:
            x = x.squeeze(0)
            if valid_mask is not None:
                valid_mask = valid_mask.squeeze(0)

        H, W, D = x.shape

        # Create default valid_mask if not provided
        if valid_mask is None:
            valid_mask = torch.ones(H, W, dtype=torch.bool, device=x.device)

        flat_features = x.view(-1, D)

        h_projected = self.base_model.feature_proj(flat_features.float())
        grid = h_projected.view(H, W, self.hidden_dim)

        grid_routed_1 = checkpoint(
            self.base_model.layer1.attn, grid, valid_mask, use_reentrant=False
        )

        N_total = H * W
        h_flat = grid_routed_1.view(1, N_total, self.hidden_dim)

        cls_tokens = self.cls_token.expand(1, -1, -1)
        h_with_cls = torch.cat((cls_tokens, h_flat), dim=1)

        h_positioned = self.base_model.pos_layer(h_with_cls, H, W)

        h_spatial_2 = h_positioned[:, 1:, :].view(H, W, self.hidden_dim)
        grid_routed_2 = checkpoint(
            self.base_model.layer2.attn, h_spatial_2, valid_mask, use_reentrant=False
        )

        h_final_flat = grid_routed_2.view(1, N_total, self.hidden_dim)
        valid_mask_flat = valid_mask.view(1, N_total, 1).float()

        # Apply norm to token features (before pooling, matching base TransMIL)
        h_normed = self.base_model.norm(h_final_flat)

        # Apply mask for valid tokens only
        h_masked = h_normed * valid_mask_flat

        # Attention pooling using base model's attn_pool
        attn_weights = torch.softmax(
            self.base_model.attn_pool(h_masked).masked_fill(~valid_mask_flat.bool(), float('-inf')),
            dim=1
        )
        attn_pooled = (attn_weights * h_masked).sum(dim=1)  # [B, hidden_dim]

        # Mean pooling over valid tokens
        sum_features = h_masked.sum(dim=1)
        valid_counts = valid_mask_flat.sum(dim=1).clamp(min=1.0)
        mean_pooled = sum_features / valid_counts  # [B, hidden_dim]

        # Concatenate both pooling methods (expected by classifier)
        pooled_features = torch.cat([attn_pooled, mean_pooled], dim=-1)  # [B, hidden_dim * 2]

        logits = self.base_model.classifier(pooled_features)

        return {"logits": logits}
