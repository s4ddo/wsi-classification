"""Run inference/testing on a checkpoint without full training.

Usage:
    python -m wsi_classification.test_checkpoint \
        --config configs/camely_abmil.py \
        --checkpoint runs/experiment_name/checkpoints/best.ckpt \
        [--disable-wandb]
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import wandb
from wsi_classification.experiments.trainer import construct_trainer
from wsi_classification.experiments.utils.cli import (
    apply_config_overrides,
    config_to_dict_for_rich,
    verify_no_interpolator_overwrites,
    load_config_from_file,
)
from wsi_classification.experiments.utils.lazy_config import instantiate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test on a checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load config
    config = load_config_from_file(args.config)
    verify_no_interpolator_overwrites(config, args.overrides)
    config = apply_config_overrides(config, args.overrides)

    # Setup
    pl.seed_everything(config.seed, workers=True)
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cudnn.benchmark = not config.deterministic
    torch.set_float32_matmul_precision("high")

    # Datamodule
    datamodule = instantiate(config.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    # Model
    network = instantiate(config.net, in_features=datamodule.input_channels, out_features=datamodule.output_channels)
    if config.compile:
        network = torch.compile(network)
    model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)

    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}")

    # W&B Logger
    experiment_dir = Path("runs") / checkpoint_path.parent.parent.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if args.disable_wandb:
        wandb_logger = None
        offline = True
    else:
        run_id = wandb.util.generate_id()
        wandb_logger = WandbLogger(
            project=config.wandb.project,
            entity=config.wandb.entity,
            save_dir=experiment_dir,
            id=run_id,
            name=f"{checkpoint_path.parent.parent.name}-test",
            offline=offline := False,
            save_code=False,
            group=config.wandb.job_group,
        )

    # Trainer (test only)
    trainer, _ = construct_trainer(config, wandb_logger, "test", experiment_dir, num_nodes=1)

    # Run test
    print("Running test...")
    trainer.test(model, datamodule=datamodule)

    if wandb_logger and not args.disable_wandb:
        wandb.finish()
    print("Test complete!")


if __name__ == "__main__":
    main()
