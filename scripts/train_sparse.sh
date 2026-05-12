#!/bin/bash
set -e

TRAIN_SPLIT_DIR="/workspace/data/camelyon/splits_new"
FEATURES_DIR="/workspace/data/camelyon/h5_features/"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for mode in "vanilla" "windowed" "nsa" "adventurer" "defr" "probe"; do
    for seed in 123 456 789; do
        echo "Starting run $mode with seed $seed (degrade_embeds_rate 0.0)..."
        python -m wsi.main \
            --cache \
            --train_csv "${TRAIN_SPLIT_DIR}/camelyon_nmlbl_seed${seed}_train.csv" \
            --val_csv   "${TRAIN_SPLIT_DIR}/camelyon_nmlbl_seed${seed}_val.csv" \
            --features_dir "$FEATURES_DIR" \
            --seed $seed \
            --epochs 5 \
            --accelerator cuda \
            --devices 0 \
            --grad_accum 4 \
            --val_check_interval 200 \
            --mode $mode \
            --lr 1e-4 \
            --weight_decay 0.05 \
            --warmup 2 \
            --degrade_embeds_rate 0.0 \
            --degrade_embeds_sigma 0.0 \
            --input_dim 1280 \
            --num_classes 2 \
            --dim 128 \
            --depth 6 \
            --num_heads 8 \
            --latent_dim 64 \
            --num_shared 2 \
            --num_routed 8 \
            --top_k 2 \
            --top_k_nsa 4 \
            --kernel_size_nsa 128 \
            --kernel_stride_nsa 64 \
            --block_size_nsa 64 \
            --window_size_nsa 128 \
            --window_size 33 \
            --mamba_d_state 128 \
            --mamba_expand 2 \
            --mamba_headdim 64
        
#        echo "Waiting for memory to flush..."
#        sleep 30
#        echo "Starting run $mode with seed $seed (degrade_embeds_rate 0.98, sigma=1.0)..."
#        python -m wsi.main \
#            --cache \
#            --train_csv "${TRAIN_SPLIT_DIR}/camelyon_nmlbl_seed${seed}_train.csv" \
#            --val_csv   "${TRAIN_SPLIT_DIR}/camelyon_nmlbl_seed${seed}_val.csv" \
#            --features_dir "$FEATURES_DIR" \
#            --seed $seed \
#            --epochs 5 \
#            --accelerator cuda \
#            --devices 0 \
#            --grad_accum 4 \
#            --val_check_interval 200 \
#            --mode $mode \
#            --lr 1e-4 \
#            --weight_decay 0.05 \
#            --warmup 2 \
#            --degrade_embeds_rate 0.98 \
#            --degrade_embeds_sigma 1.0 \
#            --input_dim 1280 \
#            --num_classes 2 \
#            --dim 128 \
#            --depth 6 \
#            --num_heads 8 \
#            --latent_dim 64 \
#            --num_shared 2 \
#            --num_routed 8 \
#            --top_k 2 \
#            --top_k_nsa 4 \
#            --kernel_size_nsa 128 \
#            --kernel_stride_nsa 64 \
#            --block_size_nsa 64 \
#            --window_size_nsa 128 \
#            --window_size 33 \
#            --mamba_d_state 128 \
#            --mamba_expand 2 \
#            --mamba_headdim 64
#
        echo "Waiting for memory to flush..."
        sleep 30
    done
done
