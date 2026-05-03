"""
Puts camelyon_slides.csv into format for H5Dataset
"""

import argparse
import h5py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides_csv",   type=str, default="camelyon_slides.csv")
    parser.add_argument("--features_dir", type=str, default="h5_features/")
    parser.add_argument("--out_dir",      type=str, default=".")
    parser.add_argument("--out_name",     type=str, default="camelyon",
                        help="Output files will be {out_name}_train.csv and {out_name}_val.csv")
    parser.add_argument("--label_col",    type=str, default="label")
    parser.add_argument("--slide_col",    type=str, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed",         type=int, default=123)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.slides_csv)
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())

    # Auto-detect slide column
    slide_col = args.slide_col
    if slide_col is None:
        for candidate in ["slidename", "slide_name", "ID", "id", "filename", "name"]:
            if candidate in df.columns:
                slide_col = candidate
                break
    if slide_col is None:
        raise ValueError(f"Could not auto-detect slide column from: {df.columns.tolist()}")

    label_col = args.label_col
    if label_col not in df.columns:
        for candidate in ["label", "Label", "status", "type", "diagnosis"]:
            if candidate in df.columns:
                label_col = candidate
                break
    if label_col not in df.columns:
        raise ValueError(f"Could not find label column from: {df.columns.tolist()}")

    df[slide_col] = df[slide_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    print(f"\nLabel distribution:\n{df[label_col].value_counts().to_string()}")

    features_dir = Path(args.features_dir)

    # Inspect the first existing H5 to find the correct feature key
    feature_key = None
    for _, row in df.iterrows():
        h5_path = features_dir / f"{row[slide_col]}.h5"
        if h5_path.exists():
            with h5py.File(h5_path, "r") as f:
                print(f"\nH5 keys in {h5_path.name}: {list(f.keys())}")
                for candidate in ["features", "imgs", "embeddings", "feat", "data"]:
                    if candidate in f:
                        feature_key = candidate
                        break
                if feature_key is None:
                    # Just take the first key that isn't coords
                    feature_key = next(k for k in f.keys() if k != "coords")
                print(f"Using feature key: '{feature_key}'")
            break
    if feature_key is None:
        raise RuntimeError("Could not find any H5 files to inspect.")

    rows = []
    missing = []
    for _, row in df.iterrows():
        h5_path = features_dir / f"{row[slide_col]}.h5"
        if h5_path.exists():
            with h5py.File(h5_path, "r") as f:
                num_patches = f[feature_key].shape[0]
            rows.append({
                "slidename":   row[slide_col],
                "label":       row[label_col],
                "num_patches": num_patches,
            })
        else:
            missing.append(row[slide_col])

    if missing:
        print(f"\nMissing H5 files ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")

    df = pd.DataFrame(rows)
    print(f"\nValid slides: {len(df)}")
    print(f"Patch count stats:\n{df['num_patches'].describe().to_string()}")

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_fraction,
        stratify=df["label"],
        random_state=args.seed,
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"\nTrain: {len(train_df)} | {train_df['label'].value_counts().to_dict()}")
    print(f"Val:   {len(val_df)}   | {val_df['label'].value_counts().to_dict()}")

    train_path = out_dir / f"{args.out_name}_seed{args.seed}_train.csv"
    val_path   = out_dir / f"{args.out_name}_seed{args.seed}_val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,   index=False)
    print(f"\nSaved:\n  {train_path}\n  {val_path}")

    print(
        f"\nNOTE: label_map order depends on first-encounter in the CSV. "
        f"Confirm tumor=1 by checking dataset.label_map after loading."
    )


if __name__ == "__main__":
    main()