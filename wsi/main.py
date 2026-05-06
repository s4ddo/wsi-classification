import time
import torch
import argparse
import io
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    RichProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryConfusionMatrix

from wsi.h5_dataset import H5FeatureBagDataset
from wsi.utils import apply_embedding_dropout
from wsi.models.deepseek_spatial_vit import DeepSeekSpatialViT
from wsi.models.window_deepseek_spatial_vit import WinDeepSeekSpatialViT
from wsi.models.adventurer import Adventurer
from wsi.models.nsa_deepseek_spatial_vit import NSADeepSeekSpatialViT
from wsi.models.linear_probe import LinearProbe
from wsi.models.deformable_detr import DeformableViT


class GeneralModelPL(pl.LightningModule):
    def __init__(self, mode="vanilla", lr=1e-4, weight_decay=1e-6, degrade_embeds_rate=0.0, warmup=3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.degrade_embeds_rate = degrade_embeds_rate
        self.warmup_epochs = warmup

        if mode == "vanilla":   # Full attention
            # input_dim, num_classes, dim, depth, num_heads, latent_dim, num_shared, num_routed, top_k
            self.model = DeepSeekSpatialViT(**kwargs)
        elif mode == "windowed":    # Local-Global windowed attention
            # input_dim, num_classes, dim, depth, num_heads, latent_dim, num_shared, num_routed, top_k, window_size
            self.model = WinDeepSeekSpatialViT(**kwargs)
        elif mode == "adventurer":  # Mamba SSM token mixing
            # input_dim, num_classes, dim, depth
            self.model = Adventurer(**kwargs)
        elif mode == "nsa": # Native Sparse Attention
            self.model = NSADeepSeekSpatialViT(**kwargs)
        elif mode == "probe":   # Linear probe
            self.model = LinearProbe(**kwargs)
        elif mode == "defr":
            self.model = DeformableViT(**kwargs)

        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.val_cm = BinaryConfusionMatrix()

        self._train_seq_lens = []


    def forward(self, x, coords):
        if self.degrade_embeds_rate > 0.01:
            x = apply_embedding_dropout(x, self.degrade_embeds_rate)
        return self.model(x, coords)

    def training_step(self, batch, batch_idx):
        inputs, coords, labels = batch["input"], batch["coords"], batch["label"]

        seq_len = inputs.shape[1]
        self._train_seq_lens.append(seq_len)
        self.log("train/seq_len", float(seq_len), prog_bar=False, on_step=True, on_epoch=False)

        logits = self(inputs, coords)
        loss = F.cross_entropy(logits, labels)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self._train_seq_lens:
            lens = np.array(self._train_seq_lens)
            self.log_dict({
                "train/seq_len_max": float(lens.max()),
                "train/seq_len_min": float(lens.min()),
                "train/seq_len_mean": float(lens.mean()),
            })
        self._train_seq_lens = []


    def validation_step(self, batch, batch_idx):
        inputs, coords, labels = batch["input"], batch["coords"], batch["label"]
        logits = self(inputs, coords)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        probs_pos = torch.softmax(logits, dim=1)[:, 1]

        self.val_auroc.update(probs_pos, labels)
        self.val_f1.update(preds, labels)
        self.val_cm.update(preds, labels)

        acc = (preds == labels).float().mean()

        self.log_dict({
            "val/loss": loss,
            "val/acc": acc,
        }, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        auroc = self.val_auroc.compute()
        f1 = self.val_f1.compute()
        cm = self.val_cm.compute()

        self.log_dict({
            "val/auroc": auroc,
            "val/f1": f1,
        }, prog_bar=True)

        fig = self._make_cm_figure(cm)
        if self.logger and hasattr(self.logger.experiment, "add_figure"):
            self.logger.experiment.add_figure(
                "val/confusion_matrix", fig, global_step=self.current_epoch
            )
        plt.close(fig)

        self.val_auroc.reset()
        self.val_f1.reset()
        self.val_cm.reset()

    def configure_optimizers(self):
        # Separate weight decay: do NOT apply it to biases and LayerNorm params
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                # LayerNorm weights are 1-D, biases are also excluded
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        # Linear warmup is important with batch_size=1 and large models
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @staticmethod
    def _make_cm_figure(cm: torch.Tensor) -> plt.Figure:
        cm_np = cm.cpu().numpy()
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm_np, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        classes = ["Normal", "Tumor"]
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Confusion Matrix")
        thresh = cm_np.max() / 2.0
        for i in range(cm_np.shape[0]):
            for j in range(cm_np.shape[1]):
                ax.text(j, i, str(int(cm_np[i, j])),
                        ha="center", va="center",
                        color="white" if cm_np[i, j] > thresh else "black")
        fig.tight_layout()
        return fig


class ElapsedTimer(pl.Callback):
    """Log wall-clock time per epoch. Useful for estimating remaining compute cost."""

    def __init__(self):
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start_time
        pl_module.log("perf/epoch_duration_seconds", elapsed)
        n_slides = len(trainer.train_dataloader.dataset)
        pl_module.log("perf/slides_per_second", n_slides / elapsed)


class GradientNormLogger(pl.Callback):
    """Log gradient norm per step."""
    def on_after_backward(self, trainer, pl_module):
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        pl_module.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)



if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Path to train csv.")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Path to val csv.")
    parser.add_argument("--features_dir", type=str, default=None,
                        help="Path to feature dir.")

    # Run control
    parser.add_argument("--seed", type=int, default=123,
                        help="Global random seed.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--val_check_interval", type=int, default=200,
                        help="Validate every N training steps.")
    parser.add_argument("--accelerator", type=str, default=None,
                        help="Pass specific accelerator to pytorch-lightning, e.g., --accelerator 'gpu'.")
    parser.add_argument("--devices", type=int, nargs="+", default=None,
                        help="Pass specific device indices, e.g., --devices 0 1.")
    parser.add_argument("--degrade_embeds_rate", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps. Effective batch = grad_accum * 1.")
    # Model selection
    parser.add_argument("--mode", type=str, default="vanilla",
                        choices=["vanilla", "windowed", "nsa", "adventurer", "probe", "defr"])

    # General hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--input_dim", type=int, default=1280)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--dim", type=int, default=320)
    parser.add_argument("--depth", type=int, default=6)

    # Expert/MoE hyperparameters (shared by DeepSeek inspired models)
    group_moe = parser.add_argument_group('MoE Settings')
    group_moe.add_argument("--num_heads", type=int, default=8)
    group_moe.add_argument("--latent_dim", type=int, default=64)
    group_moe.add_argument("--num_shared", type=int, default=2)
    group_moe.add_argument("--num_routed", type=int, default=8)
    group_moe.add_argument("--top_k", type=int, default=2)

    # Special hyperparameters
    # Windowed Deepseek hyperparameters
    parser.add_argument("--window_size", type=int, default=33)

    # Adventurer hyperparameters
    parser.add_argument("--mamba_d_state", type=int, default=128)
    parser.add_argument("--mamba_expand", type=int, default=4)
    parser.add_argument("--mamba_headdim", type=int, default=64)

    # NSA hyperparameters
    parser.add_argument("--top_k_nsa", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--window_size_nsa", type=int, default=1024)
    parser.add_argument("--fine_attn_backend", type=str, default="gather", choices=["gather", "flex"])

    args = parser.parse_args()
    args_dict = vars(args)


    # --- Setup data ---
    pl.seed_everything(args.seed, workers=True)

    train_csv_path = args_dict.pop("train_csv")
    val_csv_path   = args_dict.pop("val_csv")
    features_dir   = args_dict.pop("features_dir")

    train_dataset = H5FeatureBagDataset(
        csv_path=train_csv_path,
        features_dir=features_dir,
        label_col_name="label"
    )
    val_dataset = H5FeatureBagDataset(
        csv_path=val_csv_path,
        features_dir=features_dir,
        label_col_name="label"
    )

    if len(train_dataset) == 0 or len(val_dataset) == 1:
        raise Exception("No valid data found. Check your paths and CSV file.")
    print(f"Train size: {len(train_dataset)}, Test size: {len(val_dataset)}")

    num_classes = len(train_dataset.label_map) if hasattr(train_dataset, 'label_map') else 2

    # IMPORTANT: batch_size is set to 1 to handle variable sequence lengths (N) # TODO: create padding
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=8,
                              persistent_workers=True,
                              pin_memory=True,
                              prefetch_factor=2,
                              )
    test_loader = DataLoader(val_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8,
                             persistent_workers=True,
                             pin_memory = True,
                             prefetch_factor = 2,
    )


    # --- Run model ---
    epochs = args_dict.pop("epochs")
    accel = args_dict.pop("accelerator")
    devices = args_dict.pop("devices")
    grad_accum = args_dict.pop("grad_accum")
    val_check_interval = args_dict.pop("val_check_interval")
    seed = args_dict.pop("seed")
    chosen_model = args_dict.pop("mode")

    print(f"Initializing '{chosen_model}' model")
    model = GeneralModelPL(mode=chosen_model, **args_dict)

    # Create Pytorch-Lightning logging
    logger = TensorBoardLogger(
        save_dir=f"./lightning_logs/",
        name=f"{chosen_model}",
        version=f"seed_{seed}",
    )
    # Callbacks
    checkpoint_auroc = ModelCheckpoint(
        monitor="val/auroc",
        mode="max",
        save_top_k=2,
        filename="epoch{epoch:02d}-auroc{val/auroc:.4f}",
        auto_insert_metric_name=False,
    )
    checkpoint_last = ModelCheckpoint(
        save_last=True,
        filename="last",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accel if accel is not None else "auto",
        devices=devices if devices is not None else "auto",
        precision="16-mixed",  # FP16
        accumulate_grad_batches=grad_accum,
        val_check_interval=val_check_interval,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=[
            RichProgressBar(),
            DeviceStatsMonitor(),
            ElapsedTimer(),
            GradientNormLogger(),
            checkpoint_auroc,
            checkpoint_last,
            lr_monitor,
        ],
        log_every_n_steps=1
    )

    trainer.fit(model, train_loader, test_loader)

    print(f"\nBest checkpoint (by val/auroc): {checkpoint_auroc.best_model_path}")
    print(f"Best val/auroc:                 {checkpoint_auroc.best_model_score:.4f}")
