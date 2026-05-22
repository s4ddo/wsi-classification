"""SlideMoE config for Camelyon dataset.

Usage:
    python -m wsi_classification.run --config wsi_classification/configs/final_camely_slide_moe.py
"""

import torch
from wsi_classification.experiments.default_cfg import ExperimentConfig, SchedulerConfig, TestConfig, TrainConfig, WandbConfig
from wsi_classification.experiments.utils.lazy_config import LazyConfig

from wsi_classification.models.slide_moe import SlideMoE
from wsi_classification.experiments.lightning_wrappers.spatial_mil_wrapper import SpatialMILWrapper
from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Data Details ──────────────────────────────────────────────
TRAIN_CSV = "splits/camely_train.csv"
VAL_CSV = "splits/camely_val.csv"
TEST_CSV = "splits/camely_test.csv"
FEATURES_DIR = "/workspace/data/h5_features"

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1
NUM_WORKERS = 4
IN_FEATURES = 1280  # Virchow embeddings
OUT_FEATURES = 1    # Binary task via BCEWithLogitsLoss

# SlideMoE Specifics - Scaled for ~2.3M params
MODEL_DIM = 256     # Scaled down for ~2.3M params (was 1024)
NUM_LAYERS = 2
NUM_HEADS = 8
FFN_HIDDEN = 512    # Scaled down for ~2.3M params (was 2048)
NUM_EXPERTS = 4
TOP_K_EXPERTS = 2
TOP_K_PATCHES = 8000

PRECISION = "bf16-mixed"
TRAINING_ITERATIONS = 1_000
WARMUP_ITERATIONS_PERCENTAGE = 0.1
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
        checkpoint_path=""
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

    config.net = LazyConfig(SlideMoE)(
        in_dim=IN_FEATURES,
        model_dim=MODEL_DIM,
        num_classes=OUT_FEATURES,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_hidden=FFN_HIDDEN,
        num_experts=NUM_EXPERTS,
        top_k_experts=TOP_K_EXPERTS,
        top_k_patches=TOP_K_PATCHES,
    )

    config.lightning_wrapper_class = LazyConfig(SpatialMILWrapper)(
        use_bce_loss=True,
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
        project="final_camely_10
",
        job_group="camely_slide_moe",
    )

    return config
