"""DeepSeekSpatialViT with RoPE classification config for Camelyon16 dataset.

Usage:
    python -m wsi_classification.run --config wsi_classification/configs/camely_deepseek_spatial_vit_rope.py
"""

import torch

from wsi_classification.experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from wsi_classification.experiments.utils.lazy_config import LazyConfig

from wsi_classification.models.deepseek_spatial_vit_rope import DeepSeekSpatialViTRoPE
from wsi_classification.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

TRAIN_CSV = "Datasets/camelyon16_train.csv"
VAL_CSV = "Datasets/camelyon16_val.csv"
FEATURES_DIR = "Datasets/camelyon16_features"

BATCH_SIZE = 1
NUM_WORKERS = 4
IN_FEATURES = 1024
OUT_FEATURES = 1
PRECISION = "bf16-mixed"

TRAINING_ITERATIONS = 3_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.test.do = False

    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        features_dir=FEATURES_DIR,
        label_col_name="label",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    config.net = LazyConfig(DeepSeekSpatialViTRoPE)(
        input_dim=IN_FEATURES,
        num_classes=OUT_FEATURES if OUT_FEATURES > 1 else 2,
        dim=256,
        depth=4,
        num_heads=8,
        latent_dim=128,
        num_shared=1,
        num_routed=4,
        top_k=2,
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
        project="wsi-classification-test",
        job_group="camely_deepseek_spatial_vit_rope",
    )

    return config
