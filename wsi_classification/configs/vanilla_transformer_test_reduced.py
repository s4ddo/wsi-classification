"""Vanilla Transformer test config with reduced feature dimensions (256-dim).

Reduced from 1280-dim to 256-dim by taking first N features.
Maintains relative relationships for testing on memory-constrained devices.
Scale back to 1280-dim by using full-dimensional data files.

Usage:
    python -m wsi_classification.run --config wsi_classification/configs/vanilla_transformer_test_reduced.py
"""

import torch
from pathlib import Path

from wsi_classification.experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from wsi_classification.experiments.utils.lazy_config import LazyConfig

from wsi_classification.models.vanilla_transformer import VanillaTransformer
from wsi_classification.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Data Details ──────────────────────────────────────────────
TRAIN_CSV = "student_debugging_dataset_train.csv"
VAL_CSV = "student_debugging_dataset_val.csv"
# Using 256-dim reduced features (from 1280-dim original)
FEATURES_DIR = "student_debugging_dataset_reduced_256dim"

# ─── Hyperparameters ───────────────────────────────────────────── 
BATCH_SIZE = 1  # Standard for MIL bags
NUM_WORKERS = 0  # No multiprocessing for testing
IN_FEATURES = 256  # Reduced from 1280
OUT_FEATURES = 1  # Binary task
HIDDEN_DIM = 128  # Reduced from 256
NUM_HEADS = 4  # Reduced from 8
NUM_LAYERS = 2
PRECISION = "bf16-mixed"

TRAINING_ITERATIONS = 100
WARMUP_ITERATIONS_PERCENTAGE = 0.0  # No warmup
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = True  # Debug mode: single batch, offline W&B
    config.seed = 42

    # Dataset: H5 feature bags from reduced-dimension dataset
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        features_dir=FEATURES_DIR,
        label_col_name="tmb_binary",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Network: Vanilla Transformer with reduced dims
    config.net = LazyConfig(VanillaTransformer)(
        in_features=IN_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        out_features=OUT_FEATURES,
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

    # Training
    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    # Scheduler - disabled for testing
    config.scheduler = SchedulerConfig(
        name=None,
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations=TRAINING_ITERATIONS,
        mode="max",
    )

    # W&B Logging (will run offline in debug mode)
    config.wandb = WandbConfig(
        project="wsi-classification-test",
        job_group="vanilla_transformer_reduced",
    )

    return config
