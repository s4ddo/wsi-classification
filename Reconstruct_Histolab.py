import os
import re
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image
from histolab.slide import Slide

slide = Slide("Data/normal_077.tif", processed_path="tiles2/")

tile_dir = "tiles"
tile_size = 256
scale = 16

W, H = slide.dimensions

canvas = torch.zeros(
    3,
    H // scale,
    W // scale
)

print("Reconstructing...")

for fname in os.listdir(tile_dir):
    if not fname.endswith(".png"):
        continue

    # extract coordinates
    coords = re.findall(r'\d+', fname)

    # last 4 numbers = x1, y1, x2, y2
    x1, y1, x2, y2 = map(int, coords[-4:])

    path = os.path.join(tile_dir, fname)

    patch = read_image(path)[:3].float() / 255.0
    # downscale patch
    patch = F.interpolate(
        patch.unsqueeze(0),
        scale_factor=1/scale,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    x = x1 // scale
    y = y1 // scale

    canvas[:, y:y+patch.shape[1], x:x+patch.shape[2]] = patch

save_image(canvas, "histolab_rec2.png")