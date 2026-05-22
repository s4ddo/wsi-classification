"""TransMIL (Nystrom attention) config for DigestPath dataset.

Usage:
    python -m wsi_classification.run --config wsi_classification/configs/baseline_transmil.py
"""

import torch
from pytorch_lightning.callbacks import EarlyStopping

from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule
from wsi_classification.experiments.default_cfg import ExperimentConfig, SchedulerConfig, TestConfig, TrainConfig, WandbConfig
from wsi_classification.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from wsi_classification.experiments.utils.lazy_config import LazyConfig
from wsi_classification.models.transmil import TransMIL

# ─── Data Details ──────────────────────────────────────────────
TRAIN_CSV = "splits/camely_train.csv"
VAL_CSV = "splits/camely_val.csv"
TEST_CSV = "splits/camely_test.csv"
FEATURES_DIR = "/workspace/data/h5_features"


# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # MIL bags have variable sequence lengths — keep at 1
NUM_WORKERS = 4
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary task (cancerous vs non-cancerous)
HIDDEN_DIM = 256  # REDUCED: Drastically smaller to prevent overfitting (was 512)
NUM_HEADS = 4  # REDUCED: Fewer attention heads (was 8)
NUM_LANDMARKS = 128  # REDUCED: Fewer landmarks (was 256)
DROPOUT = 0.5  # INCREASED: More dropout (was 0.4)
PRECISION = "bf16-mixed"

TRAINING_ITERATIONS = 5_000  # REDUCED: Less training to prevent overfitting
WARMUP_ITERATIONS_PERCENTAGE = 0.1
LEARNING_RATE = 5e-5  # REDUCED: Slower learning (was 1e-4)
WEIGHT_DECAY = 1e-3  # INCREASED: Stronger L2 regularization (was 5e-4)
GRAD_CLIP = 0.5


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    # Test configuration with checkpoint path
    config.test = TestConfig(
        do=True,

checkpoint_path="checkpoints/transmil.ckpt"
    )

    # Dataset
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        features_dir=FEATURES_DIR,
        label_col_name="label",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        subsample_patches=512,  # REDUCED: Sample fewer patches per slide for stronger augmentation
    )

    # Network
    config.net = LazyConfig(TransMIL)(
        in_features=IN_FEATURES,
        hidden_dim=HIDDEN_DIM,  # Using reduced hidden dim to prevent overfitting
        out_features=OUT_FEATURES,
        heads=NUM_HEADS,
        num_landmarks=NUM_LANDMARKS,
        dropout=DROPOUT,
        attention_type="nystrom",
    )

    # Lightning wrapper
    config.lightning_wrapper_class = LazyConfig(MILWrapper)(
        use_bce_loss=(OUT_FEATURES == 1),
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

    # Wandb
    config.wandb = WandbConfig(
        project="final_camely_with_test_and_auroc",
        job_group="digestpath_transmil_nystrom",
    )

    # Early stopping to prevent overfitting
    config.callbacks = [
        LazyConfig(EarlyStopping)(
            monitor="val/loss",
            patience=20,  # Stop if val loss doesn't improve for 20 epochs
            mode="min",
            verbose=True,
        )
    ]

    return config
