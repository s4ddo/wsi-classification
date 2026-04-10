# GROUP 8 LINKS
MEETINGS: https://meet.google.com/nti-yvpk-hep 

GIT: https://github.com/s4ddo/wsi-classification

PROPOSAL DOC: https://www.overleaf.com/3688327964zrmjmtvdqmbr#0ab1c7

PROPOSAL SLIDES: https://docs.google.com/presentation/d/1ZMjIyQuakQ28kwaN7wx0sG7TXUeGcOklpSFN4oDrTnM/edit?slide=id.p#slide=id.p

PAPERS: https://docs.google.com/spreadsheets/d/15vAzjMTW5EqeDnXdtYJdurr6HRA8BZmWQh5ADRQ5NMk/edit?usp=sharing

# wsi-classification

A robust, PyTorch Lightning-based framework for Whole-Slide Image (WSI) classification. This repository focuses on fast feature extraction using FastPathology, state-of-the-art foundation models (like Virchow2 via HuggingFace), and benchmarking Multiple Instance Learning (MIL) methods such as AB-MIL.

## Features

- **Fast Feature Extraction**: Utilizes the C++ streaming capabilities of FastPathology (`pyfast`) to quickly tile and extract features from WSIs into `.h5` feature bags.
- **Foundation Models**: Seamlessly integrates with Hugging Face models (e.g., `paige-ai/Virchow2`) via `timm` using mixed-precision (`fp16`) for rapid inference. Outputs standard 1280-dim `[CLS]` tokens or optionally 2560-dim concatenated representations.
- **Multiple Instance Learning (MIL)**: Includes a clean PyTorch Lightning baseline for AB-MIL (`Attention-Based MIL`) and handles dynamically-sized bag inputs (N patches × 1280-dim features).
- **Flexible Configuration**: Powered by `LazyConfig` for dynamic and explicit experiment specifications.

## Installation

We recommend using an environment with GPU and CUDA support (e.g., `conda` with PyTorch 2.0+). 

```bash
# Install the core local package in editable mode
pip install -e .

# Additional libraries frequently used in this project:
# pip install torch torchvision pytorch-lightning wandb h5py pandas pyfast timm
```

## Extracting Features

You can extract standard MIL compatible features (e.g. `[CLS]` tokens natively yielding 1280-dim vectors) rapidly from your slides using FastPathology.

```bash
python wsi_classification/datasets/h5_slidedataset/extract_cellvit_virchow_fast.py \
    --csv /path/to/slides.csv \
    --output_dir /path/to/output_features \
    --hf_token "YOUR_HF_TOKEN"
```

*Note: Use `--concat_tokens` if you want a 2560-dim output that includes mean patch tokens.*

## Running an Experiment

Training models is managed by the `wsi_classification/run.py` entrypoint.

```bash
# Run with a config file
python -m wsi_classification.run --config wsi_classification/configs/baseline_abmil.py

# Override config values from CLI
python -m wsi_classification.run --config wsi_classification/configs/baseline_abmil.py train.iterations=10000 dataset.batch_size=1
```

## Running Tests

The test suite mimics the package structure to ensure stable dataloader and model dynamics.

```bash
pytest tests/
```

## Project Structure

```text
wsi-classification/
├── pyproject.toml
├── paper/                           # Research proposal and papers
├── tests/                           # Test suite mirroring package structure
│   ├── datasets/                    # Tests for H5FeatureBagDataset
│   ├── experiments/                 # Tests for Lightning loss wrappers
│   └── models/                      # Tests for models (e.g. AB-MIL)
└── wsi_classification/              # Core package
    ├── run.py                       # Main experiment entrypoint
    ├── configs/                     # Experiment configuration files
    ├── scripts/                     # Utility and legacy scripts
    ├── datasets/                    # Data loading (H5 feature bags, etc.)
    │   └── h5_slidedataset/         # pyfast extraction & H5 dataloader
    ├── models/                      # Model architectures (ABMIL, etc.)
    └── experiments/                 # Training infrastructure
        ├── callbacks/               # PL callbacks (W&B cleanup, etc.)
        ├── datamodules/             # LightningDataModules mapping to H5 Bags
        ├── lightning_wrappers/      # PL modules containing loss/metrics logic
        └── utils/                   # CLI, checkpointing, and LazyConfig
```

## TODO

- [ ] Extract a second dataset using the `--concat_tokens` flag (2560-dim: CLS + mean patch tokens) with the FAST Virchow2 extractor (`wsi_classification/datasets/h5_slidedataset/extract_cellvit_virchow_fast.py`) to compare spatial aggregation dynamics against the standard 1280-dim CLS baseline.
