import time
import torch
import argparse
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import DeviceStatsMonitor, RichProgressBar
from torch.utils.data import DataLoader

from wsi.h5_dataset import H5FeatureBagDataset
from wsi.utils import apply_embedding_dropout
from wsi.models.deepseek_spatial_vit import DeepSeekSpatialViT
from wsi.models.window_deepseek_spatial_vit import WinDeepSeekSpatialViT
from wsi.models.adventurer import Adventurer
from wsi.models.nsa_deepseek_spatial_vit import NSADeepSeekSpatialViT
from wsi.models.linear_probe import LinearProbe


class GeneralModelPL(pl.LightningModule):
    def __init__(self, mode="vanilla", lr=1e-4, weight_decay=1e-6, degrade_embeds_rate=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.degrade_embeds_rate = degrade_embeds_rate

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

    def forward(self, x, coords):
        if self.degrade_embeds_rate > 0.01:
            x = apply_embedding_dropout(x, self.degrade_embeds_rate)

        return self.model(x, coords)

    def training_step(self, batch, batch_idx):
        inputs, coords, labels = batch["input"], batch["coords"], batch["label"]
        logits = self(inputs, coords)
        loss = F.cross_entropy(logits, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, coords, labels = batch["input"], batch["coords"], batch["label"]
        logits = self(inputs, coords)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class ElapsedTimer(pl.Callback):
    def __init__(self):
        self.epoch_start_time = None

    def on_train_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log("epoch_duration_seconds", time.time() - self.epoch_start_time)


if __name__ == "__main__":
    # --- Constant paths --- # TODO: consider using pathlib + passing as args from whatever computer
    train_csv_path = "/Users/ab/Documents/MSAI/DL2/wsi-classification/data/DigestPath_train.csv"
    val_csv_path = "/Users/ab/Documents/MSAI/DL2/wsi-classification/data/DigestPath_val.csv"
    features_dir = "/Users/ab/Documents/MSAI/DL2/wsi-classification/data/DigestPath_UNI_features/"

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--accelerator", type=str, default=None,
                        help="Pass specific accelerator to pytorch-lightning, e.g., --accelerator 'gpu'")
    parser.add_argument("--devices", type=int, nargs="+", default=None,
                        help="Pass specific device indices, e.g., --devices 0 1")
    parser.add_argument("--degrade_embeds_rate", type=float, default=0.0)
    parser.add_argument("--mode", type=str, default="vanilla",
                        choices=["vanilla", "windowed", "nsa", "adventurer", "probe"])
    # General hyperparameters
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--input_dim", type=int, default=1024)  # TODO: change for final datasets
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    # Expert/MoE hyperparameters (shared by DeepSeek inspired models)
    group_moe = parser.add_argument_group('MoE Settings')
    group_moe.add_argument("--num_heads", type=int, default=4)
    group_moe.add_argument("--latent_dim", type=int, default=64)
    group_moe.add_argument("--num_shared", type=int, default=1)
    group_moe.add_argument("--num_routed", type=int, default=4)
    group_moe.add_argument("--top_k", type=int, default=2)
    # Special hyperparameters
    # Windowed Deepseek hyperparameters
    parser.add_argument("--window_size", type=int, default=7)
    # Adventurer hyperparameters
    parser.add_argument("--mamba_d_state", type=int, default=128)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--mamba_headdim", type=int, default=64)

    args = parser.parse_args()
    args_dict = vars(args)


    # --- Setup data ---

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

    # # Split dataset into 80% Train, 20% Test (for the complete dataset)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # IMPORTANT: batch_size is set to 1 to handle variable sequence lengths (N) # TODO: create padding
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=7,
                              persistent_workers=True,
                              )
    test_loader = DataLoader(val_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=7,
                             persistent_workers=True
                             )


    # --- Run model ---
    epochs = args_dict.pop("epochs")
    accel = args_dict.pop("accelerator")
    devices = args_dict.pop("devices")
    chosen_model = args_dict.pop("mode")
    print(f"Initializing '{chosen_model}' model")
    model = GeneralModelPL(mode=chosen_model, **args_dict)

    # Create Pytorch-Lightning logging
    logger = TensorBoardLogger(
        save_dir=f"./lightning_logs/",
        name=f"{chosen_model}",
        version=None,
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accel if accel is not None else "auto",
        val_check_interval=0.2,
        logger=logger,
        callbacks=[RichProgressBar(), DeviceStatsMonitor(), ElapsedTimer()],
        devices=devices if devices is not None else "auto",     # None and "auto" are probably the same but just careful
    )

    trainer.fit(model, train_loader, test_loader)
