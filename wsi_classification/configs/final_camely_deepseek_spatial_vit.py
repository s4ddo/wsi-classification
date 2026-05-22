"""DeepSeek Spatial ViT config for Camely dataset.

Usage:
    python -m wsi_classification.run --config wsi_classification/configs/camely_deepseek_spatial_vit.py
"""

import torch
from wsi_classification.experiments.default_cfg import ExperimentConfig, SchedulerConfig, TestConfig, TrainConfig, WandbConfig
from wsi_classification.experiments.utils.lazy_config import LazyConfig

from wsi_classification.models.deepseek_spatial_vit import DeepSeekSpatialViT
from wsi_classification.experiments.lightning_wrappers.spatial_mil_wrapper import SpatialMILWrapper
from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Data Details ──────────────────────────────────────────────
TRAIN_CSV = "splits/camely_train.csv"
VAL_CSV = "splits/camely_val.csv"
TEST_CSV = "splits/camely_test.csv"
FEATURES_DIR = "/workspace/data/h5_features"

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # MIL bags have variable sequence lengths — keep at 1
NUM_WORKERS = 4
IN_FEATURES = 1280  # UNI embeddings
OUT_FEATURES = 1  # Binary task via BCEWithLogitsLoss (cancerous vs non-cancerous)
DIM = 208  # Scaled down for ~2.38M params (was 256)
DEPTH = 2  # Scaled down for ~2.38M params (was 4)
NUM_HEADS = 8
HIDDEN_DIM = 512
LATENT_DIM = 128
NUM_SHARED = 1
NUM_ROUTED = 4
TOP_K = 2
PRECISION = "bf16-mixed"

TRAINING_ITERATIONS = 1_000
WARMUP_ITERATIONS_PERCENTAGE = 0.1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 0.5


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False
    config.seed  = 42
    # Test configuration with checkpoint path
    config.test = TestConfig(
        do=True,
        checkpoint_path="checkpoints/transmil.ckpt"
    )

    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        features_dir=FEATURES_DIR,
        label_col_name="label",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    config.net = LazyConfig(DeepSeekSpatialViT)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_shared=NUM_SHARED,
        num_routed=NUM_ROUTED,
        top_k=TOP_K,
    )

    config.lightning_wrapper_class = LazyConfig(SpatialMILWrapper)(
        use_bce_loss=True,    # BCEWithLogitsLoss for binary output
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
        project="camely-sparse",
        job_group="camely_deepseek_spatial_vit",
    )

    return config
