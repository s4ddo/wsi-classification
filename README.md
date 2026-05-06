# wsi-segmentation-patching

Tissue-aware patch extraction pipeline for whole-slide histopathology images, with reconstruction utilities.

## Motivation

Standard patching libraries such as histolab rely on simple thresholding for tissue detection, which tends to discard adipose (fatty) tissue and small or lightly-stained regions. This project implements a more robust tissue segmentation approach that retains significantly more tissue, including adipose tissue, small isolated regions, and low-contrast areas that histolab usually misses entirely.

Histolab's default mask produces gaps across adipose regions and drops small tissue fragments, leading to incomplete patch sets. The custom pipeline here captures the full tissue extent.


**Reconstructed slides from CAMELYON16**

| tumor_026 | normal_077 |
|---|---|
| ![](reconstructed_26.png) | ![](reconstructed_77.png) |


---

## How it works

Tissue detection combines two complementary signals:

- **Optical density (OD):** Converts RGB to OD space and applies Otsu thresholding on the summed OD channels. Effective for haematoxylin and eosin stained tissue.
- **Saturation:** Captures lightly stained and fat regions that have low OD but visible colour.

The union of both masks is then cleaned with morphological closing and opening, and small connected components below a minimum area are removed. Masking runs at the lowest available pyramid level for speed, then is upsampled to the patch grid.

---

## Files

| File | Description |
|---|---|
| `Segmentation_Patching.py` | Custom tissue masking and strided patch extraction |
| `Histolab_Patching.py` | Equivalent extraction using histolab (for comparison) |
| `Reconstruct_Slide.py` | Reconstructs a downscaled overview from the custom pipeline |
| `Reconstruct_Histolab.py` | Reconstructs a downscaled overview from histolab tiles |

---

## Requirements

```
openslide-python
torch
torchvision
opencv-python
histolab
numpy
```

```bash
conda env create -f environment.yml
conda activate wsi
```

> OpenSlide also requires the system library. On Ubuntu: `sudo apt install openslide-tools`. On macOS: `brew install openslide`.

---

## Usage

### Custom pipeline

```python
from Segmentation_Patching import SegPatching
from torch.utils.data import DataLoader

dataset = SegPatching("path/to/slide.tif", patch_size=256, level=0, stride=2)
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

Each item returned:

```python
{
    "patch": torch.Tensor,   # (3, 256, 256), float32 in [0, 1]
    "coord": torch.Tensor    # (2,) grid indices [y_idx, x_idx]
}
```

Reconstruct a downscaled overview for visual QC:

```bash
python Segmentation_Patching.py
```

```bash
python Reconstruct_Slide.py
```

### Histolab pipeline (comparison baseline)

```bash
python Histolab_Patching.py
python Reconstruct_Histolab.py
```


---

## Patch striding

`stride=2` keeps roughly 1 in 4 tissue patches (every other patch in x and y), with a random offset to avoid grid bias. Adjust to trade coverage for speed:

```python
dataset = Camelyon("slide.tif", stride=1)  # all patches
dataset = Camelyon("slide.tif", stride=4)  # ~1/16 patches
```

---


## License

MIT. See [LICENSE](LICENSE) for details.
