"""CLAM_SB (Single-Branch) classification config.

Follows: https://github.com/mahmoodlab/clam

Usage:
    python -m wsi_classification.run --config wsi_classification/configs/baseline_clam.py
"""

import torch
from wsi_classification.experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from wsi_classification.experiments.utils.lazy_config import LazyConfig
from wsi_classification.models.clam import CLAM_SB
from wsi_classification.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Dataset ───────────────────────────────────────────────────
TRAIN_CSV = "student_debugging_dataset_train.csv"
VAL_CSV = "student_debugging_dataset_val.csv"
FEATURES_DIR = "student_debugging_dataset"
LABEL_COL = "tmb_binary"

# ─── Model Hyperparameters ────────────────────────────────────
# CLAM Parameters (from official implementation)
GATE = True  # Gated attention mechanism
SIZE_ARG = "small"  # "small" or "big"
DROPOUT = 0.25
K_SAMPLE = 8  # Number of positive/negative instances to sample
N_CLASSES = 2  # Binary classification
EMBED_DIM = 1280  # Input feature dimension (from Virchow2 features)

# ─── Training Hyperparameters ─────────────────────────────────
BATCH_SIZE = 1  # MIL standard: one slide per batch
NUM_WORKERS = 4
TRAINING_ITERATIONS = 1000
WARMUP_ITERATIONS_PERCENTAGE = 0.1
LEARNING_RATE = 1e-4  # Official CLAM default
WEIGHT_DECAY = 1e-5  # Official CLAM default
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False
    config.seed = 1

    # Dataset module
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        features_dir=FEATURES_DIR,
        label_col_name=LABEL_COL,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # CLAM_SB model (Single-Branch)
    config.net = LazyConfig(CLAM_SB)(
        gate=GATE,
        size_arg=SIZE_ARG,
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

    # Learning rate scheduler (cosine annealing with warmup)
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations=TRAINING_ITERATIONS,
        mode="max",
    )

    # Weights & Biases logging
    config.wandb = WandbConfig(
        project="wsi-classification",
        job_group="clam_sb",
    )

    return config
