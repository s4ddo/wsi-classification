#!/bin/bash

for seed in 123; do
    python -m wsi.main \
        --train_csv "/workspace/data/camelyon/splits/camelyon_seed${seed}_train.csv" \
        --val_csv   "/workspace/data/camelyon/splits/camelyon_seed${seed}_val.csv" \
        --features_dir /workspace/data/camelyon/h5_features/ \
        --seed $seed \
        --epochs 10 \
        --accelerator cuda \
        --devices 0 \
        --grad_accum 4 \
        --val_check_interval 200 \
        --mode probe \
        --lr 1e-4 \
        --weight_decay 0.05 \
        --warmup 3 \
        --degrade_embeds_rate 0.0 \
        --input_dim 1280 \
        --num_classes 2 \
        --dim 320 \
        --depth 6 \
        --num_heads 8 \
        --latent_dim 64 \
        --num_shared 2 \
        --num_routed 8 \
        --top_k 2 \
        --top_k_nsa 8 \
        --block_size 64 \
        --window_size_nsa 1024 \
        --window_size 33 \
        --fine_attn_backend flex \
        --mamba_d_state 128 \
        --mamba_expand 2 \
        --mamba_headdim 64
done