#!/usr/bin/env python3
"""
Fast UNI embedding extraction - optimized for speed with large batch inference.
Pre-processes patches to tensors before GPU inference.
"""

import torch
import h5py
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def patchify_and_preprocess(image_path, transform, patch_size=224, stride=224):
    """Load image, patchify, and preprocess all patches to tensors."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    patch_tensors = []
    coords = []

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patch_tensor = transform(patch)  # Already normalized
            patch_tensors.append(patch_tensor)
            coords.append((x, y))

    if not patch_tensors:
        return None, None

    # Stack into (N_patches, 3, 224, 224)
    patch_tensors = torch.stack(patch_tensors)
    coords = np.array(coords, dtype=np.float32)

    return patch_tensors, coords


def extract_uni_embeddings_fast(
    image_paths,
    output_dir,
    csv_output,
    patch_size=224,
    batch_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Extract embeddings with maximum batching for speed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load UNI model
    logger.info("Loading UNI model...")
    login()
    model = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model = model.to(device)
    model.eval()

    results = []

    logger.info(f"Using batch_size={batch_size} for GPU inference")

    for slide_name, (image_path, label) in tqdm(image_paths.items(), desc="Processing slides"):
        try:
            # Load and preprocess all patches
            patch_tensors, coords = patchify_and_preprocess(
                image_path, transform, patch_size=patch_size
            )

            if patch_tensors is None:
                logger.warning(f"No patches extracted from {slide_name}")
                continue

            # Batch GPU inference
            embeddings = []
            with torch.inference_mode():
                for i in range(0, len(patch_tensors), batch_size):
                    batch_tensors = patch_tensors[i:i+batch_size].to(device)
                    batch_embeddings = model(batch_tensors)  # [B, 1024]
                    embeddings.append(batch_embeddings.cpu().numpy())

            # Stack embeddings
            embeddings = np.vstack(embeddings)  # (N_patches, 1024)

            # Save to h5
            h5_path = output_dir / f"{slide_name}.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("features", data=embeddings, dtype=np.float32)
                f.create_dataset("coords", data=coords, dtype=np.float32)

            results.append({
                "slidename": slide_name,
                "label": label,
                "num_patches": len(patch_tensors),
            })

            logger.info(f"Saved {slide_name}: {len(patch_tensors)} patches")

        except Exception as e:
            logger.error(f"Error processing {slide_name}: {e}")

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_output, index=False)
    logger.info(f"Saved CSV to {csv_output} with {len(results)} slides")

    return df


def main():
    digestpath_dir = Path("/home/s4ddo/Uni/wsi-classification/Datasets/DigestPath")
    output_dir = Path("/home/s4ddo/Uni/wsi-classification/Datasets/DigestPath_UNI_features")
    csv_output = Path("/home/s4ddo/Uni/wsi-classification/Datasets/DigestPath_metadata.csv")

    # Collect image paths
    image_paths = {}

    neg_dir = digestpath_dir / "tissue-train-neg"
    for img_path in neg_dir.glob("*.jpg"):
        slide_name = img_path.stem
        if "_mask" not in slide_name:
            image_paths[slide_name] = (str(img_path), 0)

    pos_dir = digestpath_dir / "tissue-train-pos-v1"
    for img_path in pos_dir.glob("*.jpg"):
        slide_name = img_path.stem
        if "_mask" not in slide_name:
            image_paths[slide_name] = (str(img_path), 1)

    logger.info(f"Found {len(image_paths)} tissue images")

    # Extract embeddings
    df = extract_uni_embeddings_fast(
        image_paths=image_paths,
        output_dir=output_dir,
        csv_output=csv_output,
        batch_size=256*2,
    )

    logger.info(f"\nDataset summary:")
    logger.info(f"Total slides: {len(df)}")
    logger.info(f"Negative samples: {(df['label'] == 0).sum()}")
    logger.info(f"Positive samples: {(df['label'] == 1).sum()}")
    logger.info(f"Average patches per slide: {df['num_patches'].mean():.1f}")


if __name__ == "__main__":
    main()
