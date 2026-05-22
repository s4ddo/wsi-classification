"""Native Sparse Attention DeepSeekSpatialViT config for DigestPath dataset.

Multi-branch sparse attention (compression, selection, sliding window).

Usage:
    python -m wsi_classification.run --config wsi_classification/configs/digestpath_nsa_deepseek_spatial_vit.py
"""

import torch

from wsi_classification.experiments.default_cfg import ExperimentConfig, SchedulerConfig, TestConfig, TrainConfig, WandbConfig
from wsi_classification.experiments.utils.lazy_config import LazyConfig

from wsi_classification.models.nsa_deepseek_spatial_vit import NSADeepSeekSpatialViT
from wsi_classification.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Data Details ──────────────────────────────────────────────
TRAIN_CSV = "splits/camely_train.csv"
VAL_CSV = "splits/camely_val.csv"
TEST_CSV = "splits/camely_test.csv"
FEATURES_DIR = "/workspace/data/h5_features"

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1
NUM_WORKERS = 4
IN_FEATURES = 1280  # UNI embeddings
OUT_FEATURES = 1
PRECISION = "bf16-mixed"

TRAINING_ITERATIONS = 1_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    # Test configuration with checkpoint path
    config.test = TestConfig(
        do=True,
        checkpoint_path="checkpoints/nsa.ckpt"
    )

    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        features_dir=FEATURES_DIR,
        label_col_name="label",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Native Sparse Attention variant
    config.net = LazyConfig(NSADeepSeekSpatialViT)(
        input_dim=IN_FEATURES,
        num_classes=OUT_FEATURES,
        dim=192,  # Scaled down for ~2.37M params (was 256)
        depth=2,  # Scaled down for ~2.37M params (was 4)
        num_heads=8,
        latent_dim=128,
        num_shared=1,
        num_routed=4,
        top_k_moe=2,
        kernel_size_nsa=64,
        kernel_stride_nsa=32,
        block_size_nsa=64,
        window_size_nsa=128,
        top_k_nsa=4,
    )

    config.lightning_wrapper_class = LazyConfig(MILWrapper)(
        use_bce_loss=(OUT_FEATURES == 1)
    )

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations=TRAINING_ITERATIONS,
        mode="max",
    )

    config.wandb = WandbConfig(
        project="final_camely_with_test_and_auroc",
        job_group="digestpath_nsa_deepseek_spatial_vit",
    )

    return config
