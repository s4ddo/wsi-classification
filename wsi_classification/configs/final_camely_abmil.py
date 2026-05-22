"""AB-MIL classification config.

Usage:
    python -m wsi_classification.experiments.run --config configs/baseline_abmil.py
"""

import torch
from pathlib import Path

from wsi_classification.experiments.default_cfg import ExperimentConfig, SchedulerConfig, TestConfig, TrainConfig, WandbConfig
from wsi_classification.experiments.utils.lazy_config import LazyConfig

from wsi_classification.models.abmil import ABMIL
from wsi_classification.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Data Details ──────────────────────────────────────────────
TRAIN_CSV = "splits/camely_train.csv"
VAL_CSV = "splits/camely_val.csv"
TEST_CSV = "splits/camely_test.csv"
FEATURES_DIR = "/workspace/data/h5_features"

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1 # Standard for MIL bags
NUM_WORKERS = 4
IN_FEATURES = 1280
OUT_FEATURES = 1 # Binary tasks
PRECISION = "bf16-mixed"

TRAINING_ITERATIONS = 1_000
WARMUP_ITERATIONS_PERCENTAGE = 0.1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 0.5


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False # set to False to actually train
    config.seed = 42
    # Test configuration with checkpoint path
    config.test = TestConfig(
        do=True,

checkpoint_path="checkpoints/abmil.ckpt"
    )

    # Dataset: Connects to your H5 extraction
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        features_dir=FEATURES_DIR,
        label_col_name="label",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        subsample_patches=1024,  # Randomly sample 1024 patches per slide per epoch
    )

    # Network: The Standard AB-MIL baseline written natively for 1280-dim CLS tokens
    config.net = LazyConfig(ABMIL)(
        in_features=IN_FEATURES,
        hidden_dim=576,  # Scaled up for ~2.3M params (was 256)
        out_features=OUT_FEATURES,
        num_branches=1
    )

    # Lightning wrapper mappings
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

    # Scheduler
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations=TRAINING_ITERATIONS,
        mode="max",
    )

    # W&B Logging
    config.wandb = WandbConfig(
        project="final_camely_with_test_and_auroc",
        job_group="baseline_abmil",
    )

    return config