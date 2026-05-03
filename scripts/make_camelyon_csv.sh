#!/bin/bash

for seed in 123 456 789; do
    python scripts/make_camelyon_csv.py \
    --slides_csv   /Volumes/Samsung\ PSSD\ T7\ Media/camelyon/camelyon_slides.csv \
    --features_dir /Volumes/Samsung\ PSSD\ T7\ Media/camelyon/h5_features/ \
    --val_fraction 0.2 \
    --seed $seed \
    --out_name camelyon \
    --out_dir /Volumes/Samsung\ PSSD\ T7\ Media/camelyon/
done
