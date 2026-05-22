import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self,
                 num_patches_in_wsi: int = 70000,
                 feature_dim: int = 1280,
                 data_type = torch.float32,
                 **_):
        """
        Args:
            num_patches_in_wsi (int): Desired number of patches N. Default is 70000.
            feature_dim (int): Desired feature dimension D. Default is 1280.
            data_type (torch.dtype): Desired data type. Default is torch.float32.
        """
        super().__init__()
        self.N = num_patches_in_wsi
        self.D = feature_dim
        self.dtype = data_type

    def __len__(self):
        return 1

    def __getitem__(self, idx: int) -> dict:
        features_t = torch.rand((self.N, self.D), dtype=self.dtype)
        coords_t = torch.randint(0, 50000, (self.N, 2), dtype=torch.float32)

        return {
            "input": features_t,
            "label": torch.tensor(0, dtype=torch.long),
            "slide_name": f"synthetic_N{self.N}_D{self.D}",
            "coords": [coords_t]
        }
