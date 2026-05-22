"""
Pool pre-extracted WSI patch features into spatially adjacent patches.

Pooling sizes:
  2x2  ->  ~448px effective patch size  (~17,500 patches per slide)
  3x3  ->  ~672px effective patch size  (~7,800 patches per slide)
  4x4  ->  ~896px effective patch size  (~4,400 patches per slide)
  6x6  ->  ~1344px effective patch size  (~2,000 patches per slide)
  10x10 -> ~2240px effective patch size  (~700 patches per slide)

Usage:
  python pool_features.py \
      --input_dir  /workspace/data/camelyon/h5_features/ \
      --output_dir /workspace/data/camelyon/ \
      --pool_sizes 2 3 4 6 10
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from warnings import warn


DEFAULT_PATCH_SIZE = 224    # Backup patch size in coordinate units used if it can not be inferred from coordinates


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_VARNAME = "features"
COORD_VARNAME = "coords"

def read_h5(path: Path):
    global FEATURE_VARNAME, COORD_VARNAME
    with h5py.File(path, "r") as f:
        feat_key = "features" if "features" in f else "embeddings"
        coord_key = "coords" if "coords" in f else "coordinates"
        FEATURE_VARNAME = feat_key
        COORD_VARNAME = coord_key
        features = f[feat_key][:].astype(np.float32)
        coords = f[coord_key][:].astype(np.float32)
    return features, coords


def write_h5(path: Path, features: np.ndarray, coords: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(FEATURE_VARNAME, data=features)
        f.create_dataset(COORD_VARNAME, data=coords)


# ──────────────────────────────────────────────────────────────────────────────
# Pooling
# ──────────────────────────────────────────────────────────────────────────────

def pool_slide(features: np.ndarray,
               coords: np.ndarray,
               pool: int,
               strategy: str = "max"):
    """
    Max-pool patch features on a spatial grid.

    Steps:
      1. Infer the coordinate stride (minimum non-zero gap between unique coords).
      2. Convert coordinates to 0-based integer grid indices.
      3. Assign each patch to a pooling cell: cell_row = grid_row // pool.
      4. For each populated cell, max-pool the features of its members.
      5. New coordinate = top-left corner of the pooling cell in pixel space.

    Args:
        features : (N, D) float32
        coords   : (N, 2) float32  — (x, y) pixel coords
        pool     : int — pooling window size (2, 3, or 4)

    Returns:
        pooled_features : (M, D) float32
        pooled_coords   : (M, 2) float32
    """
    N, D = features.shape

    # 1. Infer stride from minimum gap in unique x-coordinates
    ux = np.unique(coords[:, 0])
    if len(ux) > 1:
        stride = float(np.diff(ux).min())
    else:
        # Single column — try y
        uy = np.unique(coords[:, 1])
        stride = float(np.diff(uy).min()) if len(uy) > 1 else 224.0     # 224 as default for Virchow/ViT

    # 2. Convert to 0-based grid indices
    grid_col = np.round(coords[:, 0] / stride).astype(np.int64)
    grid_row = np.round(coords[:, 1] / stride).astype(np.int64)
    grid_col -= grid_col.min()
    grid_row -= grid_row.min()

    # 3. Assign each patch to a pooling cell
    cell_col = grid_col // pool
    cell_row = grid_row // pool

    # 4. Group patches by cell and max-pool
    cell_ids = cell_row * (cell_col.max() + 1) + cell_col
    order = np.argsort(cell_ids, kind="stable")

    sorted_ids = cell_ids[order]
    sorted_features = features[order]
    sorted_col = cell_col[order]
    sorted_row = cell_row[order]

    # Find boundaries of each unique cell
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    boundaries = np.concatenate([[0], boundaries, [N]])

    n_cells = len(boundaries) - 1
    pooled_features = np.empty((n_cells, D), dtype=np.float32)
    pooled_coords = np.empty((n_cells, 2), dtype=np.float32)

    for i in range(n_cells):
        lo, hi = boundaries[i], boundaries[i + 1]
        if strategy == "max":
            pooled_features[i] = sorted_features[lo:hi].max(axis=0)
        else:
            raise ValueError("unsupported pooling type: {}".format(strategy))
        # New coord = top-left corner of this pooling cell in pixel space
        pooled_coords[i, 0] = float(sorted_col[lo] * pool) * stride  # x
        pooled_coords[i, 1] = float(sorted_row[lo] * pool) * stride  # y

    return pooled_features, pooled_coords


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pool WSI patch features spatially.")
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing original .h5 feature files.")
    parser.add_argument("--output_dir", required=True,
                        help="Root output directory. Sub-dirs pool2x2/ etc. are created.")
    parser.add_argument("--pool_sizes", nargs="+", type=int, default=[2, 3, 4],
                        help="Pooling window sizes to generate (default: 2 3 4).")
    parser.add_argument("--strategy", type=str, default="max",
                        help="Pooling strategy to use (default: max).")
    args = parser.parse_args()
    print(vars(args))

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    h5_files = sorted(input_dir.glob("*.h5"))

    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")

    print(f"Found {len(h5_files)} slides. Generating pool sizes: {args.pool_sizes}")

    for pool in args.pool_sizes:
        out_subdir = output_dir / f"h5_features_{pool}x{pool}"
        out_subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n── Pool {pool}x{pool} -> {out_subdir}")

        patch_counts = []
        for h5_path in tqdm(h5_files, desc=f"pool {pool}x{pool}"):
            out_path = out_subdir / h5_path.name

            if out_path.exists():
                warn("{} already exists. Skipping.".format(out_path), UserWarning)
                continue

            try:
                features, coords = read_h5(h5_path)
                pf, pc = pool_slide(features, coords, pool, args.strategy)
                write_h5(out_path, pf, pc)
                patch_counts.append(len(pf))
            except Exception as e:
                print(f" x {h5_path.name}: {e}")

        if patch_counts:
            print(f"patches/slide  mean={np.mean(patch_counts):.0f}  "
                  f"min={np.min(patch_counts)}  max={np.max(patch_counts)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
