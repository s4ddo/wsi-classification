import openslide
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torchvision.utils as vutils
from torchvision.utils import save_image



class Camelyon(Dataset):
    def __init__(self, path, patch_size=256, level=0):
        self.slide = openslide.OpenSlide(path)

        self.patch_size = patch_size
        self.level = level
        self.width, self.height = self.slide.level_dimensions[level]

        # Global mask
        self.mask_level = self.slide.level_count - 1
        lowres = self.slide.read_region(
            (0, 0),
            self.mask_level,
            self.slide.level_dimensions[self.mask_level]
        )
        lowres = np.array(lowres)[:, :, :3]

        self.global_mask = self.tissue_mask(lowres)
        self.scale = self.slide.level_downsamples[self.mask_level]

        # Precompute valid indices
        mask_small = self.global_mask.astype(bool)

        grid_w = self.width // self.patch_size
        grid_h = self.height // self.patch_size

        mask_resized = cv2.resize(
            mask_small.astype(np.uint8),
            (grid_w, grid_h),
            interpolation=cv2.INTER_NEAREST
        )

        valid = mask_resized > 0
        self.valid_coords = np.column_stack(np.where(valid))




    @staticmethod
    def tissue_mask(img):
        img_u8 = img.astype(np.uint8)
        img_f = img_u8.astype(np.float32)

        # Optical density
        od = -np.log((img_f + 1.0) / 255.0)
        od_sum = od.sum(axis=2)
        od_sum = np.clip(od_sum, 0, 3)
        od_norm = (od_sum / 3.0 * 255).astype(np.uint8)

        # Otsu 
        _, mask_od = cv2.threshold(
            od_norm, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # dont combine after otsu to retain fat areas 

        # expand OD a bit
        mask_od = cv2.dilate(mask_od, np.ones((3,3), np.uint8), iterations=1)

        # saturation mask 
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
        mask_sat = (hsv[:, :, 1] > 10).astype(np.uint8) * 255

        # Remove obvious white background (away from boarders)
        mask = ((mask_od > 0) | (mask_sat > 0)).astype(np.uint8) * 255


        # Morphology 
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # remove noise (questionable)
        #min_area = int(0.0001 * mask.size)  # depends on resolution
        min_area = 500
        clean_mask = np.zeros_like(mask)

        for i in range(1, num_labels):  # skip background
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                clean_mask[labels == i] = 255

        mask = clean_mask

        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)

        return mask

    
    

    def __len__(self):
        return len(self.valid_coords)

    def __getitem__(self, i):
        gy, gx = self.valid_coords[i]  # swap order
        x = gx * self.patch_size
        y = gy * self.patch_size
        patch = self.slide.read_region(
            (x, y),
            self.level,
            (self.patch_size, self.patch_size)
        )
        patch = np.array(patch)[:, :, :3]

        patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        return patch
    
    def __del__(self):
        self.slide.close()

       
    
    
if __name__ == "__main__":
    dataset = Camelyon("Data/normal_077.tif")
    print(f"Kept {len(dataset)} patches")


    out_dir = "patches2"
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, patch in enumerate(loader):
        # patch shape: [1, 3, H, W]
        save_path = os.path.join(out_dir, f"patch_{i:06d}.png")
        save_image(patch, save_path)

    print(f"Saved {len(dataset)} patches to {out_dir}")







    """os.makedirs("selected_patches", exist_ok=True)

    for i in range(len(dataset)):
        patch = dataset[i]

        # if patch is empty-ish (you already filtered so this is optional safety)
        if patch.sum() == 0:
            patch = torch.zeros_like(patch)

        vutils.save_image(patch, f"selected_patches/patch_{i}.png")

    grid_w = dataset.width // dataset.patch_size
    grid_h = dataset.height // dataset.patch_size
    ps = dataset.patch_size

    # canvas (black background)
    canvas = torch.zeros(3, grid_h * ps, grid_w * ps)

    for i in range(len(dataset)):
        gy, gx = dataset.valid_coords[i]  # same indexing as dataset

        patch = dataset[i]

        y = gy * ps
        x = gx * ps

        canvas[:, y:y+ps, x:x+ps] = patch

    vutils.save_image(canvas, "reconstructed_wsi.png")"""


    

    
    