#!/bin/bash

# Script to run VRAM benchmarks for all final_camely configs
# Usage: ./run_all_benchmarks.sh [wandb_project_name]

WANDB_PROJECT="${1:-vram_benchmark}"

# Patch counts for benchmarking
PATCH_COUNTS=(125000 250000 500000 1000000)

echo "Using WandB project: $WANDB_PROJECT"
echo "Patch counts: ${PATCH_COUNTS[@]}"
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
    filename=$(basename "$config" .py)
    short_name="${filename#final_}"

    echo "========================================"
    echo "Benchmarking: $config"
    echo "Model: $short_name"
    echo "========================================"

    python benchmark_vram.py \
        --config "$config" \
        --project "$WANDB_PROJECT" \
        --patch-counts "${PATCH_COUNTS[@]}" \
        --warmup-steps 1 \
        --measure-steps 1

    echo ""
    echo "Finished: $config"
    echo ""
done

echo "All benchmarks completed!"
