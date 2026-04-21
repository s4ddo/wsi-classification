#!/usr/bin/env python3
"""Reduce feature dimensionality for memory-constrained testing.

Creates reduced-dim versions of H5 files by keeping only the first N dimensions.
This maintains relative relationships in the data for testing purposes.
"""

import os
import h5py
from pathlib import Path
import sys

def reduce_h5_dimensions(input_h5: str, output_h5: str, target_dim: int = 256) -> None:
    """Reduce features to target dimensionality and save to new H5 file.

    Args:
        input_h5: Path to input H5 file
        output_h5: Path to output H5 file
        target_dim: Target feature dimensionality (keeps first N features)
    """
    with h5py.File(input_h5, 'r') as f_in:
        with h5py.File(output_h5, 'w') as f_out:
            # Copy and reduce features
            if 'features' in f_in:
                original_features = f_in['features'][:]
                reduced_features = original_features[:, :target_dim]
                f_out.create_dataset('features', data=reduced_features, compression='gzip')
                print(f"  Features: {original_features.shape} -> {reduced_features.shape}")

            # Copy coordinates as-is
            if 'coords' in f_in:
                coords = f_in['coords'][:]
                f_out.create_dataset('coords', data=coords, compression='gzip')
                print(f"  Coords: {coords.shape}")

            # Copy any other datasets
            for key in f_in.keys():
                if key not in ['features', 'coords']:
                    data = f_in[key][:]
                    f_out.create_dataset(key, data=data, compression='gzip')
                    print(f"  {key}: {data.shape}")


def main():
    input_dir = Path("student_debugging_dataset")
    output_dir = Path("student_debugging_dataset_reduced_256dim")
    target_dim = 256

    output_dir.mkdir(exist_ok=True)

    # Find all H5 files
    h5_files = sorted(input_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} H5 files\n")

    if not h5_files:
        print("No H5 files found!")
        sys.exit(1)

    for i, h5_path in enumerate(h5_files, 1):
        output_path = output_dir / h5_path.name
        print(f"[{i}/{len(h5_files)}] {h5_path.name}")
        reduce_h5_dimensions(str(h5_path), str(output_path), target_dim)

    print(f"\n✓ Reduced {len(h5_files)} H5 files to {target_dim} dimensions")
    print(f"✓ Output directory: {output_dir.absolute()}")
    print(f"\nTo use reduced data, update config:")
    print(f"  FEATURES_DIR = '{output_dir}'")
    print(f"  IN_FEATURES = {target_dim}")


if __name__ == "__main__":
    main()
