#!/usr/bin/env python3
"""
Validate DigestPath h5 files and metadata CSVs for format correctness.
"""

import h5py
import pandas as pd
from pathlib import Path
import sys


def validate_h5_files(features_dir, max_files=10):
    """Validate h5 file format."""
    features_dir = Path(features_dir)
    h5_files = sorted(features_dir.glob("*.h5"))

    if not h5_files:
        print(f"❌ No h5 files found in {features_dir}")
        return False

    print(f"Found {len(h5_files)} h5 files. Validating first {max_files}...")

    errors = 0
    for h5_path in h5_files[:max_files]:
        try:
            with h5py.File(h5_path, "r") as f:
                if "features" not in f or "coords" not in f:
                    print(f"  ❌ {h5_path.name}: Missing 'features' or 'coords' key")
                    errors += 1
                else:
                    features = f["features"]
                    coords = f["coords"]

                    # Validate shapes
                    if len(features.shape) != 2 or features.shape[1] != 1024:
                        print(f"  ❌ {h5_path.name}: Features shape {features.shape}, expected (N, 1024)")
                        errors += 1
                    elif len(coords.shape) != 2 or coords.shape[1] != 2:
                        print(f"  ❌ {h5_path.name}: Coords shape {coords.shape}, expected (N, 2)")
                        errors += 1
                    elif features.shape[0] != coords.shape[0]:
                        print(f"  ❌ {h5_path.name}: Mismatch - {features.shape[0]} features, {coords.shape[0]} coords")
                        errors += 1
                    else:
                        print(f"  ✓ {h5_path.name}: {features.shape[0]} patches, 1024-dim embeddings")
        except Exception as e:
            print(f"  ❌ {h5_path.name}: {e}")
            errors += 1

    return errors == 0


def validate_csv(csv_path):
    """Validate metadata CSV."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
        required_cols = {"slidename", "label"}

        if not required_cols.issubset(df.columns):
            print(f"❌ CSV missing required columns. Has: {df.columns.tolist()}, needs: {required_cols}")
            return False

        # Validate labels
        valid_labels = {0, 1}
        if not set(df["label"].unique()).issubset(valid_labels):
            print(f"❌ CSV has invalid labels: {df['label'].unique()}")
            return False

        print(f"✓ CSV valid: {len(df)} slides")
        print(f"  - Negative: {(df['label'] == 0).sum()}")
        print(f"  - Positive: {(df['label'] == 1).sum()}")

        return True

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False


def main():
    features_dir = "Datasets/DigestPath_UNI_features"
    metadata_csv = "Datasets/DigestPath_metadata.csv"
    train_csv = "Datasets/DigestPath_train.csv"
    val_csv = "Datasets/DigestPath_val.csv"

    print("=" * 60)
    print("Validating DigestPath Dataset Setup")
    print("=" * 60)

    all_valid = True

    print("\n1. Checking H5 files...")
    all_valid &= validate_h5_files(features_dir)

    print("\n2. Checking metadata CSV...")
    if Path(metadata_csv).exists():
        all_valid &= validate_csv(metadata_csv)
    else:
        print(f"⚠ Metadata CSV not yet created: {metadata_csv}")

    print("\n3. Checking train/val splits...")
    if Path(train_csv).exists():
        print("  Train CSV:")
        all_valid &= validate_csv(train_csv)
    else:
        print(f"⚠ Train CSV not yet created: {train_csv}")

    if Path(val_csv).exists():
        print("  Val CSV:")
        all_valid &= validate_csv(val_csv)
    else:
        print(f"⚠ Val CSV not yet created: {val_csv}")

    print("\n" + "=" * 60)
    if all_valid:
        print("✓ Dataset validation passed!")
    else:
        print("❌ Some validation checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
