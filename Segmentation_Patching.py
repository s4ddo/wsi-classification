import openslide
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torchvision.utils as vutils
from torchvision.utils import save_image



class SegPatching(Dataset):
    def __init__(self, path, patch_size=256, level=0, stride=1):
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

        grid_w = int(np.ceil(self.width / self.patch_size))
        grid_h = int(np.ceil(self.height / self.patch_size))

        mask_resized = cv2.resize(
            mask_small.astype(np.uint8),
            (grid_w, grid_h),
            interpolation=cv2.INTER_NEAREST
        )

        valid = mask_resized > 0
        coords = np.column_stack(np.where(valid))

        # if PATCH SKIPPING (1 out of 4)
        self.stride = stride  # 2 → keep 1/4 patches
        stride = self.stride
        
        # random offset to avoid grid bias
        offset_y = np.random.randint(0, stride)
        offset_x = np.random.randint(0, stride)

        mask = (
            ((coords[:, 0] + offset_y) % stride == 0) &
            ((coords[:, 1] + offset_x) % stride == 0)
        )

        self.valid_coords = coords[mask]

        print(f"[SegPatching] kept {len(self.valid_coords)} / {len(coords)} patches")



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
        y_idx, x_idx = self.valid_coords[i]
        x = int(x_idx * self.patch_size)
        y = int(y_idx * self.patch_size)
        patch = self.slide.read_region(
            (x, y),
            self.level,
            (self.patch_size, self.patch_size)
        )
        patch = np.array(patch)[:, :, :3]

        patch = torch.from_numpy(patch).permute(2, 0, 1).div(255.0)
        return {
            "patch": patch,
            "coord": torch.tensor([y_idx, x_idx], dtype=torch.int64)
        }
    
    def __del__(self):
        self.slide.close()

       
    
    
if __name__ == "__main__":
    dataset = SegPatching("tumors/tumor_026.tif")
    print(f"Kept {len(dataset)} patches")


    out_dir = "patches26"
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, patch in enumerate(loader):
        save_path = os.path.join(out_dir, f"patch_{i:06d}.png")
        save_image(patch["patch"], save_path)

    print(f"Saved {len(dataset)} patches to {out_dir}")







    