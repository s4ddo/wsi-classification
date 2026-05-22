import torch
import torch.nn as nn
import torch.nn.functional as F

class RoutingAttention(nn.Module):
    def __init__(self, dim=128, num_clusters=8, num_heads=4, mlp_ratio=4.0, decay=0.999):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.k = num_clusters
        self.decay = decay

        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

        self.register_buffer('centroids', torch.randn(num_heads, self.k, self.head_dim))
        self.centroids.data = F.normalize(self.centroids.data, p=2, dim=-1)

    def forward(self, token_grid, valid_mask):
        H, W, D = token_grid.shape
        
        flat_grid = token_grid.view(H * W, D)
        flat_mask = valid_mask.view(H * W)
        
        valid_indices = flat_mask.nonzero(as_tuple=True)[0]
        N_valid = valid_indices.shape[0]
        
        if N_valid == 0:
            return token_grid
            
        x_valid = flat_grid[valid_indices].unsqueeze(0)
        
        pad_len = (self.k - (N_valid % self.k)) % self.k
        if pad_len > 0:
            x_valid = F.pad(x_valid, (0, 0, 0, pad_len))
            
        _, N_padded, _ = x_valid.shape
        w = N_padded // self.k

        h_in = self.norm1(x_valid)

        Q = self.q_proj(h_in).view(1, N_padded, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h_in).view(1, N_padded, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h_in).view(1, N_padded, self.num_heads, self.head_dim).transpose(1, 2)

        Q_norm = F.layer_norm(Q, (self.head_dim,))
        K_norm = F.layer_norm(K, (self.head_dim,))

        Q_prod = torch.einsum('bhnd,hkd->bhkn', Q_norm, self.centroids)
        K_prod = torch.einsum('bhnd,hkd->bhkn', K_norm, self.centroids)

        if pad_len > 0:
            pad_mask = torch.zeros(1, 1, N_padded, device=Q_prod.device, dtype=torch.bool)
            pad_mask[..., -pad_len:] = True

            mask_value = torch.finfo(Q_prod.dtype).min
            Q_prod = Q_prod.masked_fill(pad_mask.unsqueeze(2), mask_value)
            K_prod = K_prod.masked_fill(pad_mask.unsqueeze(2), mask_value)

        _, Q_idx = torch.topk(Q_prod, w, dim=-1)
        _, K_idx = torch.topk(K_prod, w, dim=-1)
        
        Q_idx, _ = torch.sort(Q_idx, dim=-1)
        K_idx, _ = torch.sort(K_idx, dim=-1)

        Q_idx_flat = Q_idx.view(1, self.num_heads, self.k * w, 1).expand(-1, -1, -1, self.head_dim)
        K_idx_flat = K_idx.view(1, self.num_heads, self.k * w, 1).expand(-1, -1, -1, self.head_dim)

        Q_routed_flat = torch.gather(Q, 2, Q_idx_flat)
        K_routed_flat = torch.gather(K, 2, K_idx_flat)
        V_routed_flat = torch.gather(V, 2, K_idx_flat)

        Q_routed = Q_routed_flat.view(1, self.num_heads, self.k, w, self.head_dim)
        K_routed = K_routed_flat.view(1, self.num_heads, self.k, w, self.head_dim)
        V_routed = V_routed_flat.view(1, self.num_heads, self.k, w, self.head_dim)

        b, h_dim, k_dim, w_dim, d_dim = Q_routed.shape
        Q_sdpa = Q_routed.reshape(b * h_dim * k_dim, w_dim, d_dim)
        K_sdpa = K_routed.reshape(b * h_dim * k_dim, w_dim, d_dim)
        V_sdpa = V_routed.reshape(b * h_dim * k_dim, w_dim, d_dim)

        out_routed = F.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa)
        out_routed = out_routed.view(b, h_dim, k_dim, w_dim, d_dim)

        out = torch.zeros_like(Q)
        out.scatter_add_(2, Q_idx_flat, out_routed.view(1, self.num_heads, self.k * w, self.head_dim))
        
        counts = torch.zeros(1, self.num_heads, N_padded, 1, device=out.device)
        counts.scatter_add_(2, Q_idx.view(1, self.num_heads, self.k * w, 1), torch.ones_like(Q_idx).view(1, self.num_heads, self.k * w, 1).float())
        counts = counts.clamp(min=1.0)
        out = out / counts

        out = out.transpose(1, 2).reshape(1, N_padded, D)
        out = self.out_proj(out)

        h_out = x_valid + out
        h_out = h_out + self.mlp(self.norm2(h_out))

        if pad_len > 0:
            h_out = h_out[:, :-pad_len, :]

        output_grid = token_grid.clone()
        output_grid.view(H * W, D)[valid_indices] = h_out.squeeze(0)

        if self.training:
            with torch.no_grad():
                _, cluster_assignments = Q_prod.max(dim=2)
                new_centroids = torch.zeros_like(self.centroids)
                for h in range(self.num_heads):
                    for c in range(self.k):
                        mask_c = (cluster_assignments[0, h] == c).unsqueeze(-1)
                        if mask_c.sum() > 0:
                            cluster_mean = (Q_norm[0, h] * mask_c).sum(dim=0) / mask_c.sum()
                            new_centroids[h, c] = cluster_mean
                        else:
                            new_centroids[h, c] = self.centroids[h, c]
                
                self.centroids.copy_(
                    F.normalize(self.decay * self.centroids + (1 - self.decay) * new_centroids, p=2, dim=-1)
                )

        return output_grid