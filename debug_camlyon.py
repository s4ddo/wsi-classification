from Camelyon import Camelyon

import openslide
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import tifffile

    
    
    
dataset = Camelyon("Data/normal_074.tif")
slide = dataset.slide

w, h = slide.level_dimensions[0]
tile = 512

output = tifffile.memmap(
    "masked_slide.tif",
    shape=(h, w, 3),
    dtype=np.uint8
)
for y in range(0, h, tile):
    for x in range(0, w, tile):
        w_tile = min(tile, w - x)
        h_tile = min(tile, h - y)

        patch = slide.read_region((x, y), 0, (w_tile, h_tile))
        patch = np.array(patch)[:, :, :3]

        mask = dataset.tissue_mask(patch)

        patch[mask == 0] = [0, 0, 0]

        output[y:y+h_tile, x:x+w_tile] = patch

tifffile.imwrite("masked_slide.tif", output)