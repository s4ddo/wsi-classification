# The original purpose of this was to test how powerful embeddings are, if you can just directly predict from them.
# You can also use it for any quick test of the CLI and pipeline as well I guess, so I added it anyways.

import torch
import torch.nn.functional as F
import torch.nn as nn

from wsi_classification.models.pos_embeds import SpatialEncoding


class LinearProbe(nn.Module):
    uses_coords = True

    def __init__(
        self, input_dim, num_classes=2, use_spatial=False, **kwargs):
        super().__init__()
        self.pos_embed = None

        if use_spatial:
            self.pos_embed = SpatialEncoding(input_dim)

        self.simple_attn = nn.Linear(input_dim, 1)

        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x, coords=None):
        B, N, _ = x.shape

        if self.pos_embed is not None:
            if coords is None:
                raise ValueError("coords are required when use_spatial=True")
            spatial_tokens = self.pos_embed(coords)
            x = x + spatial_tokens

        a = F.softmax(self.simple_attn(x), dim=1)
        x_pooled = torch.sum(x * a, dim=1)

        logits = self.classifier(x_pooled)
        return {"logits": logits}
