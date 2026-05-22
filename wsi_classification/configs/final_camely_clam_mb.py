"""CLAM_MB (Multi-Branch) classification config.

Multi-branch variant with class-specific attention mechanisms.
Follows: https://github.com/mahmoodlab/clam

Usage:
    python -m wsi_classification.run --config wsi_classification/configs/baseline_clam_mb.py
"""

import torch
from wsi_classification.experiments.default_cfg import ExperimentConfig, SchedulerConfig, TestConfig, TrainConfig, WandbConfig
from wsi_classification.experiments.utils.lazy_config import LazyConfig
from wsi_classification.models.clam import CLAM_MB
from wsi_classification.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Dataset ───────────────────────────────────────────────────
TRAIN_CSV = "splits/camely_train.csv"
VAL_CSV = "splits/camely_val.csv"
TEST_CSV = "splits/camely_test.csv"
FEATURES_DIR = "/workspace/data/h5_features"

# ─── Model Hyperparameters ────────────────────────────────────
# CLAM_MB Parameters
GATE = True  # Gated attention mechanism
SIZE_ARG = "small"  # "small" or "big"
DROPOUT = 0.25
K_SAMPLE = 8  # Number of positive/negative instances to sample
N_CLASSES = 2  # Binary classification
EMBED_DIM = 1280  # Input feature dimension

# ─── Training Hyperparameters ─────────────────────────────────
BATCH_SIZE = 1  # MIL standard: one slide per batch
NUM_WORKERS = 4
TRAINING_ITERATIONS = 1_000
WARMUP_ITERATIONS_PERCENTAGE = 0.1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 0.5
PRECISION = "bf16-mixed"


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    # Test configuration with checkpoint path
    config.test = TestConfig(
        do=True,
        checkpoint_path="checkpoints/clam.ckpt"
    )

    # Dataset module
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        features_dir=FEATURES_DIR,
        label_col_name="label",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # CLAM_MB model (Multi-Branch with class-specific attention)
    config.net = LazyConfig(CLAM_MB)(
        gate=GATE,
        size_arg="big",  # Scaled up for ~1.45M params (best achievable without model changes)
        dropout=DROPOUT,
        k_sample=K_SAMPLE,
        n_classes=N_CLASSES,
        instance_loss_fn=torch.nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=EMBED_DIM,
    )

    # Lightning wrapper for MIL tasks
    config.lightning_wrapper_class = LazyConfig(MILWrapper)(
        use_bce_loss=(N_CLASSES == 1),
    )

    # Optimizer (Adam as in official CLAM)
    config.optimizer = LazyConfig(torch.optim.Adam)(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Training configuration
    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    # Learning rate scheduler
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations=TRAINING_ITERATIONS,
        mode="max",
    )

    # Weights & Biases logging
    config.wandb = WandbConfig(
        project="final_camely_with_test_and_auroc",
        job_group="clam_mb",
    )

    return config
