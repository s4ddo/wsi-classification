import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
from pathlib import Path
import pandas as pd

from wsi_classification.datasets.h5_slidedataset.h5_dataset import H5FeatureBagDataset
from wsi_classification.datasets.synthetic_dataset import SyntheticDataset


def mil_collate_fn(batch: list[dict]) -> dict:
    """Collate variable-size MIL bags into a batch.

    Standard AB-MIL is trained with batch_size=1. When bags happen to have the
    same number of patches they are stacked into a (B, N, D) tensor; otherwise
    a list is returned so callers can handle padding themselves.

    Args:
        batch: List of sample dicts produced by :class:`H5FeatureBagDataset`.

    Returns:
        Dict with keys ``"input"``, ``"label"``, ``"slide_name"``, ``"coords"``.
    """
    inputs = [b["input"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    slide_names = [b["slide_name"] for b in batch]
    # Handle coords: some datasets return [tensor], others return tensor
    coords_raw = [b["coords"] for b in batch]
    coords = [c[0] if isinstance(c, (list, tuple)) else c for c in coords_raw]

    inputs = torch.stack(inputs, dim=0) if len(inputs) == 1 else inputs
    coords = torch.stack(coords, dim=0) if len(coords) == 1 else coords

    return {
        "input": inputs,
        "label": labels,
        "slide_name": slide_names,
        "coords": coords,
    }


class H5FeatureBagDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for HDF5 feature-bag MIL datasets.

    Args:
        train_csv: Path to CSV with training slide metadata.
        val_csv: Path to CSV with validation slide metadata.
        features_dir: Directory containing ``{slide_name}.h5`` feature files.
        label_col_name: Column name in the CSV files for the target label.
        batch_size: Number of slides per batch (typically 1 for AB-MIL).
        num_workers: DataLoader worker processes.
        use_synthetic: If True, use synthetic data for VRAM testing instead of H5 files.
        synthetic_num_patches: Number of patches N per synthetic WSI. Default 70000.
        synthetic_feature_dim: Feature dimension D for synthetic data. Default 1280.
    """

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        features_dir: str = "",
        label_col_name: str = "label",
        batch_size: int = 1,
        num_workers: int = 4,
        test_csv: str | None = None,
        use_synthetic: bool = False,
        synthetic_num_patches: int = 70000,
        synthetic_feature_dim: int = 1280,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.features_dir = features_dir
        self.label_col_name = label_col_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_synthetic = use_synthetic
        self.synthetic_num_patches = synthetic_num_patches
        self.synthetic_feature_dim = synthetic_feature_dim
        # Will be set in prepare_data() by inferring from actual H5 files
        self.input_channels = None
        self.output_channels = 1

    def prepare_data(self) -> None:
        """Infer feature dimension from first H5 file or synthetic config."""
        if self.input_channels is None:
            if self.use_synthetic:
                self.input_channels = self.synthetic_feature_dim
            else:
                features_dir = Path(self.features_dir)
                h5_files = list(features_dir.glob("*.h5"))
                if h5_files:
                    with h5py.File(h5_files[0], "r") as f:
                        if "features" in f:
                            feature_shape = f["features"].shape
                            self.input_channels = feature_shape[1]
                if self.input_channels is None:
                    self.input_channels = 1280

    def _build_label_map_from_csv(self, csv_path: str) -> dict:
        """Build a consistent label mapping from a CSV file.

        Maps string labels to integers based on first occurrence in the CSV.
        This ensures consistent label mapping across train/val/test datasets.

        Args:
            csv_path: Path to the CSV file containing labels.

        Returns:
            Dictionary mapping string labels to integer indices.
        """
        df = pd.read_csv(csv_path)
        label_map = {}
        for raw_label in df[self.label_col_name]:
            if isinstance(raw_label, str) and raw_label not in label_map:
                label_map[raw_label] = len(label_map)
        return label_map

    def setup(self, stage: str | None = None) -> None:
        """Instantiate train, validation, and test datasets.

        Uses a consistent label mapping derived from the training CSV to ensure
        label alignment across all splits (critical for correct test evaluation).
        When use_synthetic=True, creates synthetic datasets instead.

        Args:
            stage: Either ``"fit"``, ``"test"``, or ``None``.
        """
        if self.use_synthetic:
            # Use synthetic data for VRAM testing
            if stage in ("fit", None):
                self.train_dataset = SyntheticDataset(
                    num_patches_in_wsi=self.synthetic_num_patches,
                    feature_dim=self.synthetic_feature_dim,
                )
                self.val_dataset = SyntheticDataset(
                    num_patches_in_wsi=self.synthetic_num_patches,
                    feature_dim=self.synthetic_feature_dim,
                )
            if stage in ("test", None) and self.test_csv is not None:
                self.test_dataset = SyntheticDataset(
                    num_patches_in_wsi=self.synthetic_num_patches,
                    feature_dim=self.synthetic_feature_dim,
                )
            return

        # Build consistent label map from train CSV
        label_map = self._build_label_map_from_csv(self.train_csv)

        if stage in ("fit", None):
            self.train_dataset = H5FeatureBagDataset(
                csv_path=self.train_csv,
                features_dir=self.features_dir,
                label_col_name=self.label_col_name,
                label_map=label_map,
            )
            self.val_dataset = H5FeatureBagDataset(
                csv_path=self.val_csv,
                features_dir=self.features_dir,
                label_col_name=self.label_col_name,
                label_map=label_map,
            )
        if stage in ("test", None) and self.test_csv is not None:
            self.test_dataset = H5FeatureBagDataset(
                csv_path=self.test_csv,
                features_dir=self.features_dir,
                label_col_name=self.label_col_name,
                label_map=label_map,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=mil_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=mil_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        if not hasattr(self, "test_dataset"):
            raise RuntimeError("Test dataset not initialized. Ensure test_csv is provided in config.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=mil_collate_fn,
            pin_memory=True,
        )
