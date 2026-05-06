from histolab.slide import Slide
from histolab.tiler import GridTiler
from histolab.masks import TissueMask
import os

os.makedirs("tiles2/", exist_ok=True)

slide = Slide("Data/normal_077.tif", processed_path="tiles2/")

tiler = GridTiler(
    tile_size=(256, 256),
    level=0,
    check_tissue=True,
    tissue_mask=TissueMask()
)

print("Running...")
tiler.extract(slide)