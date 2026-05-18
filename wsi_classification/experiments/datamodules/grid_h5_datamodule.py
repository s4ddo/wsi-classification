import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from datasets.h5_slidedataset.grid_h5_dataset import GridH5FeatureBagDataset

class GridH5DataModule(pl.LightningDataModule):
    def __init__(self, csv_path, features_dir, batch_size=1, num_workers=4, patch_size=224):
        super().__init__()
        self.csv_path = csv_path
        self.features_dir = features_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        
        self.input_channels = 1280 
        self.output_channels = 2   

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        full_dataset = GridH5FeatureBagDataset(
            csv_path=self.csv_path,
            features_dir=self.features_dir,
            patch_size=self.patch_size
        )
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)