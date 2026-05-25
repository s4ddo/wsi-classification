# WSI Classification

Sparse-attention ViT for Whole-Slide Image (WSI) classification, benchmarked against MIL (Multiple Instance Learning) methods.

## Overview

This repository contains implementations of various attention-based models for WSI classification:

- **SlideMoE** - Sparse mixture-of-experts model with top-k patch selection
- **DeepSeek Spatial ViT** - Hierarchical sparse attention with spatial awareness
- **NSA DeepSeek Spatial ViT** - Native Sparse Attention variant
- **Window DeepSeek Spatial ViT** - Windowed attention variant
- **Deformable ViT** - Deformable attention mechanism
- **ABMIL** - Attention-based Multiple Instance Learning
- **TransMIL** - Transformer-based MIL
- **CLAM** - Clustering-constrained Attention MIL
- **Adventurer** - Spatial MIL model
- **Routing Transformer** - Sparse routing attention

## Installation

### Prerequisites

- Python >= 3.11
- CUDA 12.x

### Install Package

```bash
# Install the package
pip install /workspace/wsi-classification/
```

### Additional Dependencies

```bash
# Mamba/State-space models
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Flash Attention
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl

# Native Sparse Attention (Triton)
pip install git+https://github.com/XunhaoLai/native-sparse-attention-triton.git
```

### Optional Dependencies

```bash
# For slide-level processing (not needed if using pre-extracted features)
pip install wsi-classification[slide]

# Development dependencies
pip install wsi-classification[dev]
```

## Datasets

This repository supports benchmarking on two WSI datasets:

- **DigestPath** - Colorectal cancer detection dataset (UNI features, 1024-dim)
- **Camelyon16** - Breast cancer metastasis detection dataset (Virchow2 features, 1280-dim)

## Data Preparation

The models expect pre-extracted features in H5 format:

- **DigestPath**: UNI features (1024-dim)
- **Camelyon16**: Virchow2 features (1280-dim)

CSV files should have columns: `slide_id`, `label`, and optionally `feature_path`

### Dataset Structure

**DigestPath:**
```
Datasets/
├── DigestPath_train.csv
├── DigestPath_val.csv
└── DigestPath_UNI_features/
    ├── slide_1.h5
    └── ...
```

**Camelyon16:**
```
splits/
├── camely_train.csv
├── camely_val.csv
└── camely_test.csv

/workspace/data/h5_features/  # Virchow2 features
├── slide_1.h5
└── ...
```

### CSV Format

```csv
slide_id,label
slide_1,0
slide_2,1
```

## Running Experiments

### Basic Usage

```bash
cd wsi_classification

# Run with a config file
python run.py --config configs/digestpath_slide_moe.py

# Override specific parameters
python run.py --config configs/digestpath_slide_moe.py dataset.batch_size=2 train.iterations=5000

# Debug mode (offline, no wandb logging)
python run.py --config configs/digestpath_slide_moe.py debug=True
```

### Available Configs

**DigestPath Dataset:**
- `digestpath_slide_moe.py` - SlideMoE
- `digestpath_deepseek_spatial_vit.py` - DeepSeek Spatial ViT
- `digestpath_nsa_deepseek_spatial_vit.py` - NSA variant
- `digestpath_window_deepseek_spatial_vit.py` - Windowed variant
- `digestpath_deformable_vit.py` - Deformable ViT
- `digestpath_routing.py` - Routing Transformer
- `digestpath_adventurer.py` - Adventurer
- `digestpath_abmil.py` - ABMIL
- `digestpath_transmil.py` - TransMIL
- `digestpath_clam_mb.py` - CLAM (multi-branch)

**Camelyon16 Dataset:**
- `final_camely_slide_moe.py`
- `final_camely_deepseek_spatial_vit.py`
- `final_camely_nsa_deepseek_spatial_vit.py`
- And others (prefix: `final_camely_`)

### Resume Training

```bash
# Enable autoresume in config or via override
python run.py --config configs/digestpath_slide_moe.py autoresume.enabled=True
```

### Testing Only

```bash
# Test with a specific checkpoint
python run.py --config configs/digestpath_slide_moe.py \
    test.do=True \
    train.do=False \
    test.checkpoint_path=/path/to/checkpoint.ckpt
```

## Project Structure

```
wsi_classification/
├── configs/              # Experiment configuration files
├── datasets/             # Dataset implementations
├── experiments/          # Training infrastructure
│   ├── datamodules/      # PyTorch Lightning data modules
│   ├── lightning_wrappers/  # Lightning module wrappers
│   └── utils/            # CLI and config utilities
├── models/               # Model implementations
│   ├── slide_moe.py      # SlideMoE model
│   ├── deepseek_spatial_vit.py
│   ├── abmil.py
│   ├── transmil.py
│   ├── clam.py
│   └── ...
└── run.py                # Entry point
```

## Weights & Biases Logging

Experiments are logged to W&B by default. Configure in your config file:

```python
config.wandb = WandbConfig(
    project="your-project-name",
    job_group="experiment-group",
    entity="your-entity",  # optional
)
```

To run offline:
```python
config.debug = True  # Sets wandb offline mode
```

## Benchmarking

VRAM benchmarking utilities are provided:

```bash
# Run VRAM benchmark
python benchmark_vram.py

# Run all benchmarks
bash run_all_benchmarks.sh
```

## Testing

```bash
# Run tests
pytest tests/

# Run without slow tests
pytest tests/ -m "not slow"
```

## Citation

If you use this code, please cite the relevant papers for the models used.
