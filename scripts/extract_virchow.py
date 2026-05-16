import json
import h5py
import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.layers import SwiGLUPacked
import argparse
from pathlib import Path

from camelyon_patch import Camelyon16PatchDataset

SUBPATCH_SIZE = 16   # Virchow2 ViT patch size in pixels
GRID_SIZE     = 14   # 224 / 16 = 14 → 196 tokens per image patch


def build_subpatch_coords(x_coords, y_coords):
    """
    Given [B] arrays of top-left (x, y) for each 224px patch,
    return [B*196, 2] sub-patch coordinates (top-left of each 16px cell).
    """
    rows = np.arange(GRID_SIZE)
    cols = np.arange(GRID_SIZE)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")   # [14, 14]
    offsets = np.stack([grid_c.ravel(), grid_r.ravel()], axis=1) * SUBPATCH_SIZE  # [196, 2]

    sub_coords = []
    for x, y in zip(x_coords, y_coords):
        base = np.array([[x, y]], dtype=np.int64)
        sub_coords.append(base + offsets)           # [196, 2]
    return np.concatenate(sub_coords, axis=0)       # [B*196, 2]


def extract(patch_json, output_dir, batch_size=32, num_workers=8, device="cuda"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)

    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    ).to(device).eval().to(memory_format=torch.channels_last)

    virchow_transforms = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(patch_json) as f:
        all_patches = json.load(f)

    slides = {}
    for p in all_patches:
        slides.setdefault(p["image_path"], []).append(p)

    with torch.inference_mode():
        for wsi_path, patches in slides.items():
            base_name = Path(wsi_path).stem
            h5_path = output_dir / f"{base_name}.h5"

            if h5_path.exists():
                print(f"Skipping {base_name}, already exists.")
                continue

            print(f"Processing {base_name} ({len(patches)} patches → {len(patches)*196} tokens)...")

            dataset = Camelyon16PatchDataset(
                patch_info=patches,
                transform=virchow_transforms,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                prefetch_factor=4 if num_workers > 0 else None,
            )

            n_total = len(patches) * 196
            slide_label = max(p.get("label", 0) for p in patches)

            with h5py.File(h5_path, "w") as hf:
                feat_ds  = hf.create_dataset("features",     shape=(n_total, 1280), dtype="float32")
                coord_ds = hf.create_dataset("coordinates",  shape=(n_total, 2),    dtype="int64")
                hf.create_dataset("label", data=slide_label, dtype="int64")

                offset = 0
                for batch in loader:
                    images   = batch["image"].to(device, non_blocking=True,
                                                 memory_format=torch.channels_last)
                    x_coords = batch["x"].numpy()
                    y_coords = batch["y"].numpy()

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(images)            # [B, 197, 1280]

                    if SUBPATCH_SIZE == 16:     # 16x16 local tokens
                        tokens = out[:, 1:, :]             # [B, 196, 1280] - drop CLS
                        coords_np = build_subpatch_coords(x_coords, y_coords)  # [B*196, 2]
                    elif SUBPATCH_SIZE == 224:  # 224x224 class tokens
                        tokens = out[:, 0:, :]
                    tokens_np = tokens.cpu().float().numpy().reshape(-1, 1280)   # [B(*196), 1280]

                    bs = tokens_np.shape[0]
                    feat_ds [offset:offset + bs] = tokens_np
                    coord_ds[offset:offset + bs] = coords_np
                    offset += bs

            print(f"  Saved {h5_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_json",  required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    extract(args.patch_json, args.output_dir, args.batch_size, args.num_workers, args.device)