import torch
from torch.utils.data import DataLoader

# Import the custom dataset class from your provided file
from h5_dataset import H5FeatureBagDataset


csv_path = "/home/tapotheker/dp2/data/combined_tcga_amc.csv"
features_dir = "/home/tapotheker/dp2/data"

print("Initializing Dataset...")
dataset = H5FeatureBagDataset(
    csv_path=csv_path,
    features_dir=features_dir,
    label_col_name="tmb_tcga"
)

if len(dataset) == 0:
    print("No valid data found. Check your paths and CSV file.")

# 3. Create the DataLoader
# IMPORTANT: batch_size is set to 1. 
# Because each slide has a different number of patches (N), the feature 
# matrices are different sizes. Standard PyTorch dataloaders cannot stack 
# different-sized tensors into a single batch without a custom collate_fn.
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

print("\nIterating through the DataLoader...")
# 4. Iterate through the data
for batch_idx, batch in enumerate(dataloader):
    features = batch["input"]
    labels = batch["label"]
    slide_names = batch["slide_name"]
    coords = batch["coords"]

    print(f"\n--- Slide {batch_idx + 1} ---")
    print(f"Slide Name: {slide_names[0]}")
    print(f"Label: {labels.item()}")
    
    # Expected shape: (1, N_patches, 1280) or (1, N_patches, 2560) if you used concat_tokens
    print(f"Features shape: {features.shape} -> (Batch Size, Number of Patches, Feature Dimension)")
    print(f"Coords shape: {coords.shape} -> (Batch Size, Number of Patches, 2)")