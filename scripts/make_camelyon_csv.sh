#!/bin/bash

for seed in 123 456 789; do
    python scripts/make_camelyon_csv.py \
    --slides_csv   /workspace/data/camelyon/camelyon_slides.csv \
    --features_dir /workspace/data/camelyon/h5_features/ \
    --val_fraction 0.2 \
    --seed $seed \
    --out_name camelyon \
    --out_dir /workspace/data/camelyon/
done
