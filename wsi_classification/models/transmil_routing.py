import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from wsi_classification.models.transmil import TransMIL
from wsi_classification.models.routing_attention import RoutingAttention

class RoutingTransMIL(nn.Module):
    def __init__(self, in_features=1280, hidden_dim=512, num_clusters=128, num_heads=8):
        super().__init__()
        self.base_model = TransMIL(in_features=in_features, hidden_dim=hidden_dim)
        
        self.base_model.layer1.attn = RoutingAttention(
            dim=hidden_dim, num_clusters=num_clusters, num_heads=num_heads
        )
        self.base_model.layer2.attn = RoutingAttention(
            dim=hidden_dim, num_clusters=num_clusters, num_heads=num_heads
        )

    def forward(self, x, valid_mask=None):
        if x.dim() == 4:
            x = x.squeeze(0) 
            valid_mask = valid_mask.squeeze(0)
            
        H, W, D = x.shape
        flat_features = x.view(-1, D)
        
        h_projected = self.base_model.feature_proj(flat_features.float()) 
        grid_512 = h_projected.view(H, W, 512)
        
        grid_routed_1 = checkpoint(self.base_model.layer1.attn, grid_512, valid_mask, use_reentrant=False)    
        
        N_total = H * W
        h_flat = grid_routed_1.view(1, N_total, 512)
        
        cls_tokens = self.base_model.cls_token.expand(1, -1, -1)
        h_with_cls = torch.cat((cls_tokens, h_flat), dim=1)
        
        h_positioned = self.base_model.pos_layer(h_with_cls, H, W)
        
        h_spatial_2 = h_positioned[:, 1:, :].view(H, W, 512)
        grid_routed_2 = checkpoint(self.base_model.layer2.attn, h_spatial_2, valid_mask, use_reentrant=False)
        
        h_final_flat = grid_routed_2.view(1, N_total, 512)
        valid_mask_flat = valid_mask.view(1, N_total, 1).float()

        sum_features = (h_final_flat * valid_mask_flat).sum(dim=1)
        valid_counts = valid_mask_flat.sum(dim=1).clamp(min=1.0)
        pooled_features = sum_features / valid_counts
        
        logits = self.base_model.classifier(self.base_model.norm(pooled_features))
        
        return {"logits": logits}