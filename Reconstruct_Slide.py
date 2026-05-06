import torch
from Segmentation_Patching import SegPatching
from torchvision.utils import save_image
import torch.nn.functional as F

if __name__ == "__main__":
    dataset = SegPatching("tumors/tumor_026.tif")

    ps = dataset.patch_size
    grid_w = dataset.width // ps
    grid_h = dataset.height // ps

    scale = 16  # ADJUST (8, 16, 32)

    canvas = torch.zeros(
        3,
        (grid_h * ps) // scale,
        (grid_w * ps) // scale
    )

    print("Reconstructing...")

    for i, (gy, gx) in enumerate(dataset.valid_coords):
        patch = dataset[i]["patch"]

        patch = F.interpolate(
            patch.unsqueeze(0),
            scale_factor=1/scale,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        y = (gy * ps) // scale
        x = (gx * ps) // scale

        canvas[:, y:y+patch.shape[1], x:x+patch.shape[2]] = patch

    save_image(canvas, "reconstructed_26.png")