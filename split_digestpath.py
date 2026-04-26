#!/usr/bin/env python3
"""
Split DigestPath metadata CSV into train (80%) and validation (20%) sets.
Ensures balanced distribution of positive and negative samples.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    metadata_csv = Path("Datasets/DigestPath_metadata.csv")

    if not metadata_csv.exists():
        print(f"Error: {metadata_csv} not found. Run extract_digestpath_uni.py first.")
        return

    # Load metadata
    df = pd.read_csv(metadata_csv)
    print(f"Total slides: {len(df)}")
    print(f"Positive: {(df['label'] == 1).sum()}, Negative: {(df['label'] == 0).sum()}")

    # Stratified split (80/20)
    train, val = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # Save
    train.to_csv("Datasets/DigestPath_train.csv", index=False)
    val.to_csv("Datasets/DigestPath_val.csv", index=False)

    print(f"\nTrain set: {len(train)} slides")
    print(f"  Positive: {(train['label'] == 1).sum()}, Negative: {(train['label'] == 0).sum()}")
    print(f"Val set: {len(val)} slides")
    print(f"  Positive: {(val['label'] == 1).sum()}, Negative: {(val['label'] == 0).sum()}")
    print(f"\nSaved to:")
    print(f"  - Datasets/DigestPath_train.csv")
    print(f"  - Datasets/DigestPath_val.csv")


if __name__ == "__main__":
    main()
