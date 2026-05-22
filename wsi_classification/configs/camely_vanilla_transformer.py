"""Vanilla Transformer config for Camely dataset.

Usage:
    python -m wsi_classification.run --config wsi_classification/configs/camely_vanilla_transformer.py
"""

import torch
from pathlib import Path

from wsi_classification.experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig, PLACEHOLDER
from wsi_classification.experiments.utils.lazy_config import LazyConfig

from wsi_classification.models.vanilla_transformer import VanillaTransformer
from wsi_classification.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Data Details ──────────────────────────────────────────────
TRAIN_CSV = "splits/camely_train.csv"
VAL_CSV = "splits/camely_val.csv"
TEST_CSV = "splits/camely_test.csv"
FEATURES_DIR = "/workspace/data/h5_features"


# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # Standard for MIL bags
NUM_WORKERS = 0  # No multiprocessing for single batch testing
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary task
DIM = 256
HIDDEN_DIM = 1024
NUM_HEADS = 8
DEPTH = 4
PRECISION = "bf16-mixed"

TRAINING_ITERATIONS = = 1_000  # Test with enough iterations for scheduler
WARMUP_ITERATIONS_PERCENTAGE = 0.0  # No warmup for quick test
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False  # Debug mode: single batch, no W&B
    config.seed = 42
    config.test.do = Truie

    # Dataset: H5 feature bags from debugging dataset
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        features_dir=FEATURES_DIR,
        label_col_name="label",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Network: Vanilla Transformer
    config.net = LazyConfig(VanillaTransformer)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
    )

    # Lightning wrapper
    config.lightning_wrapper_class = LazyConfig(MILWrapper)(
        use_bce_loss=(OUT_FEATURES == 1)
    )

    # Optimizer
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Training: Just 1 iteration for quick test
    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    # Scheduler - disable for quick test
    config.scheduler = SchedulerConfig(
        name=None,
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations=TRAINING_ITERATIONS,
        mode="max",
    )

    config.wandb = WandbConfig(
        project="camely-sparse",
        job_group="camely_vanilla_transformer",
    )

    return config
