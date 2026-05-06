# Non-Baseline Pipeline

Training pipeline for WSI classification using various architectures.

## Quick Start

To train the model, run the following command from the **project root directory**:

```bash
python -m wsi.main --mode vanilla
```

## Available Architectures (`--mode`)

You can switch between different model backends using the `--mode` argument:

* **`vanilla`**: Uses a full-attention `DeepSeekSpatialViT` model.
* **`windowed`**: Uses `WinDeepSeekSpatialViT` for local-global windowed attention.
* **`adventurer`**: Uses the `Adventurer` architecture with Mamba SSM token mixing.

## Customizing Hyperparameters

The script automatically passes command-line arguments to the selected model. You can override defaults as follows:

```bash
# Example: Running a windowed model with specific settings
python -m wsi.main --mode windowed --depth 8 --num_heads 8 --window_size 16
```

### Argument Configuration Reference

| Argument          |  Default  | Type  | Description                                                                                          |
|:------------------|:---------:|:-----:|:-----------------------------------------------------------------------------------------------------|
| `--mode`          | `vanilla` | `str` | Selects model: `vanilla` (Full Attention), `windowed` (Local/Global), or `adventurer` (Mamba).       |
| `--input_dim`     |  `1024`   | `int` | Feature dimensionality of the input tiles.                                                           |
| `--num_classes`   |    `2`    | `int` | Number of target classes for classification.                                                         |
| `--dim`           |   `128`   | `int` | Internal embedding dimension of the model.                                                           |
| `--depth`         |    `4`    | `int` | Number of transformer blocks/layers.                                                                 |
| `--num_heads`     |    `4`    | `int` | Number of attention heads (MoE-based models).                                                        |
| `--latent_dim`    |   `64`    | `int` | Dimensionality of the latent routing space in MoE layers.                                            |
| `--num_shared`    |    `1`    | `int` | Number of expert layers shared across all tokens in MoE.                                             |
| `--num_routed`    |    `4`    | `int` | Total number of available expert layers in MoE.                                                      |
| `--top_k`         |    `2`    | `int` | Number of experts each token is routed to during the forward pass.                                   |
| `--window_size`   |    `7`    | `int` | Size of the local attention window (used only in `windowed` mode).                                   |
| `--mamba_d_state` |   `128`   | `int` | State dimension of the Mamba SSM (used only in `adventurer` mode).                                   |
| `--mamba_expand`  |    `2`    | `int` | Expansion factor for the Mamba SSM internal projection (used only in `adventurer` mode).             |
| `--mamba_headdim` |   `64`    | `int` | Splits the embedding dimensions to be processed by this many heads (used only in `adventurer` mode). |

* **Architecture-Specific**: The `window_size`, `mamba_d_state`, and `mamba_expand` arguments are **ignored** if the mode they belong to is not selected. 
* **Dataset Matching**: Ensure `--input_dim` matches the feature extractor used


## Logging
Logs are automatically saved to `./lightning_logs/` using the format `lightning_logs/[mode]/version_X/`. 
You can visualize the logs using TensorBoard:

```bash
tensorboard --logdir=./lightning_logs/
```

> **Note:** All data paths currently require local configuration in `main.py`.
