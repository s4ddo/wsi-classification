import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import h5py


class H5FeatureBagDataset(Dataset):
    """
    A PyTorch Dataset to read slide-level feature bags (N x 1280) from individual 
    HDF5 files produced by the FastPathology extraction pipeline.
    Intended for MIL (Multiple Instance Learning) classification.
    """
    def __init__(self, csv_path, features_dir, label_col_name="label", transform=None, cache=False):
        """
        Args:
            csv_path (str): Path to CSV containing 'slidename' and label.
            features_dir (str): Directory containing the extracted {slide_name}.h5 files.
            label_col_name (str): Column name in the CSV for the target label.
            transform (callable, optional): Optional transform applied to the bag of features.
        """
        super().__init__()
        self.features_dir = Path(features_dir)
        self.transform = transform
        self.label_col_name = label_col_name
        
        # Load slide-level metadata
        df = pd.read_csv(csv_path)

        # Load slide-level metadata, limiting to 5 slides for testing
        # df = pd.read_csv(csv_path).head(5)


        # Mapping for string to int labels
        self.label_map = {}
        
        # Keep only slides for which the feature .h5 file actually exist
        valid_slides = []
        for idx, row in df.iterrows():
            slide_name = str(row.get('slidename', row.get('ID')))
            h5_path = self.features_dir / f"{slide_name}.h5"
            
            raw_label = row.get(label_col_name)
            if h5_path.exists() and pd.notna(raw_label):

                # Dynamically map strings to integers if required
                # if isinstance(raw_label, str):
                #     if raw_label not in self.label_map:
                #         self.label_map[raw_label] = len(self.label_map)
                #     mapped_label = self.label_map[raw_label]
                # else:
                #     mapped_label = int(raw_label)

                # Safely map ALL labels (strings, ints, or floats) to 0-indexed integers
                # Convert to string to ensure consistent dictionary mapping
                # solves the error CUDA error: device-side assert triggered
                str_label = str(raw_label)
                if str_label not in self.label_map:
                    self.label_map[str_label] = len(self.label_map)
                
                mapped_label = self.label_map[str_label]

                valid_slides.append({
                    "slide_name": slide_name,
                    "label": mapped_label,
                    "h5_path": h5_path
                })
        
        self.slides = valid_slides
        
        print(f"Loaded {len(self.slides)} valid WSI feature bags from {features_dir}")

        # LOAD INTO RAM (do this only if you can fit)
        self._cache = {}
        if cache:
            print("Caching dataset into RAM, this can take a few minutes...")
            for item in self.slides:
                with h5py.File(item["h5_path"], "r") as f:
                    feature_key = next(
                        (k for k in ["features", "embeddings", "imgs", "feat", "data"] if k in f),
                        None
                    )
                    if feature_key is None:
                        feature_key = next(k for k in f.keys() if k not in ("coords", "coordinates"))
                    coord_key = "coordinates" if "coordinates" in f else "coords"
                    self._cache[item["slide_name"]] = (
                        torch.from_numpy(f[feature_key][:]).float(),
                        torch.from_numpy(f[coord_key][:]).float(),
                    )
            print("Dataset cached.")

    def __len__(self) -> int:
        """Return the number of valid slides in the dataset."""
        return len(self.slides)

    def __getitem__(self, idx):
        item = self.slides[idx]

        if self._cache:
            features_t, coords_t = self._cache[item["slide_name"]]
        else:
            with h5py.File(item["h5_path"], "r") as f:
                feature_key = next(
                    (k for k in ["features", "embeddings", "imgs", "feat", "data"] if k in f),
                    None
                )
                if feature_key is None:
                    feature_key = next(k for k in f.keys() if k not in ("coords", "coordinates"))
                coord_key = "coordinates" if "coordinates" in f else "coords"
                features_t = torch.from_numpy(f[feature_key][:]).float()
                coords_t = torch.from_numpy(f[coord_key][:]).float()

        if self.transform is not None:
            features_t = self.transform(features_t)

        return {
            "input": features_t,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "slide_name": item["slide_name"],
            "coords": coords_t,
        }
