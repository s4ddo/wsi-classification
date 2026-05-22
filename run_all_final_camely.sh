#!/bin/bash

# Script to run all final_camely configs sequentially
# Usage: ./run_all_final_camely.sh [wandb_project_name]

WANDB_PROJECT="${1:-final_camely_10}"

echo "Using WandB project: $WANDB_PROJECT"
echo ""

CONFIGS=(
    "wsi_classification/configs/final_camely_abmil.py"
    "wsi_classification/configs/final_camely_adventurer.py"
    "wsi_classification/configs/final_camely_clam_mb.py"
    "wsi_classification/configs/final_camely_deepseek_spatial_vit.py"
    "wsi_classification/configs/final_camely_deformable_vit.py"
    "wsi_classification/configs/final_camely_nsa_deepseek_spatial_vit.py"
    "wsi_classification/configs/final_camely_routing.py"
    "wsi_classification/configs/final_camely_slide_moe.py"
    "wsi_classification/configs/final_camely_transmil.py"
    "wsi_classification/configs/final_camely_transmil_sparse.py"
    "wsi_classification/configs/final_camely_window_deepseek_spatial_vit.py"
)

for config in "${CONFIGS[@]}"; do
    # Extract model name from config filename
    # e.g., final_camely_deformable_vit.py -> camely_deformable_vit
    filename=$(basename "$config" .py)
    short_name="${filename#final_}"

    echo "========================================"
    echo "Running: $config"
    echo "Run name: $short_name"
    echo "========================================"
    python -m wsi_classification.run --config "$config" "wandb.project=$WANDB_PROJECT" "name=$short_name"
    echo ""
    echo "Finished: $config"
    echo ""
done

echo "All configs completed!"
