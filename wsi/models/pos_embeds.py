import torch
import torch.nn as nn


class SpatialEncoding(nn.Module):
    """Projects 2D coordinates (X, Y) into the transformer's hidden dimension."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, coords):
        # Scale down coordinates to prevent massive values from overwhelming the network
        # (Assuming typical WSI coordinates are in the tens of thousands of pixels)
        normalized_coords = coords / 10000.0
        return self.proj(normalized_coords)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=1000000):
        super().__init__()
        self.dim = dim

        inv_freq = 1.0 / (base ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coords, q, k=None, skip_first=True):
        # 1. Prepare frequency tensors
        y, x = coords[..., 0:1].unsqueeze(1), coords[..., 1:2].unsqueeze(1)
        angles_y = y @ self.inv_freq.view(1, -1)
        angles_x = x @ self.inv_freq.view(1, -1)

        cos_y, sin_y = torch.cos(angles_y), torch.sin(angles_y)
        cos_x, sin_x = torch.cos(angles_x), torch.sin(angles_x)

        def apply_rot(t, cos, sin):
            # t: [B, N, H, d]
            d = t.shape[-1] // 2
            x1, x2 = t[..., :d], t[..., d:]

            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        # 2. Apply Y rotation to the first half of Head_Dim
        # 3. Apply X rotation to the second half of Head_Dim
        # Slicing: [B, N, Heads, Head_Dim // 2]
        q_first_original = q[:, :, 0:1, :]
        q_out = torch.cat([
            apply_rot(q[..., :self.dim // 2], cos_y, sin_y),
            apply_rot(q[..., self.dim // 2:], cos_x, sin_x)
        ], dim=-1)
        if skip_first:
            q_out[:, :, 0:1, :] = q_first_original

        if k is not None:
            k_first_original = k[:, :, 0:1, :]
            k_out = torch.cat([
                apply_rot(k[..., :self.dim // 2], cos_y, sin_y),
                apply_rot(k[..., self.dim // 2:], cos_x, sin_x)
            ], dim=-1)
            k_out[:, :, 0:1, :] = k_first_original

            return q_out, k_out
        return q_out
