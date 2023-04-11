from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window
import rasterio
from contextlib import contextmanager
import glob
import numpy as np
import os
import pandas as pd
import copy
import argparse
from scipy.ndimage.filters import gaussian_filter

# Inputs
datapath = 'data/068'
format = "JAX_068_df1_*.tif"
padding = 50
H, W = 852, 852

parser = argparse.ArgumentParser(
    description='data augmentation configurations.')
parser.add_argument('--gauss', type=bool, default=False)
parser.add_argument('--sigma', type=float, default=0.1)
args = parser.parse_args()

upsample_scale = (H - 2 * padding) / H
img_list = []
for img in glob.glob(os.path.join(datapath, format)):
    if "_dsm." in img:
        continue
    img_list.append(img)


def calculate_focal(x): return x / upsample_scale
def populate_id(x): return x + 100


if not os.path.exists("./data/cropped"):
    os.makedirs("./data/cropped")
if not os.path.exists(datapath + "_augmented"):
    os.makedirs(datapath + "_augmented")

img_numbers = []
for dat in img_list:
    with rasterio.open(dat) as src:
        window = Window(
            padding,
            padding,
            src.width - 2 * padding,
            src.height - 2 * padding
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)})
        filename = dat.split("/")[-1].split(".tif")[0]
        filenumber = str(int(filename.split("_")[-1]) + 100)
        newfilename = "_".join(filename.split("_")[:3]) + "_" + filenumber
        img_numbers.append(int(filenumber))

        cropped_dat = f"./data/cropped/{newfilename}.tif"
        with rasterio.open(cropped_dat, 'w', **kwargs) as dst:
            dst.write(src.read(window=window))

    @contextmanager
    def resample_raster(raster, scale):
        t = raster.transform

        transform = Affine(
            t.a / scale,
            t.b, t.c, t.d,
            t.e / scale,
            t.f
        )
        height = raster.height / scale
        width = raster.width / scale

        profile = src.profile
        profile.update(transform=transform, driver='GTiff',
                       height=height, width=width, crs=src.crs)

        # Note changed order of indexes, arrays are band, row, col order not row, col, band
        data = raster.read(out_shape=(int(raster.count), int(
            height), int(width)), resampling=Resampling.cubic)

        if args.gauss:
            print("Applying gaussian blur...")
            data = gaussian_filter(data, sigma=args.sigma)

        with rasterio.open(f"./{datapath + '_augmented'}/{newfilename}.tif", 'w', **profile) as dst:
            dst.write(data)
            yield data

    with rasterio.open(cropped_dat) as src:
        with resample_raster(src, upsample_scale) as resampled:
            print('Original dims: {}, New dims: {}'.format(
                src.shape, resampled.shape))

md_filename = format.replace('*', 'md')
md_filename = md_filename.replace('.tif', '.txt')
df1 = pd.read_csv(f'{datapath}/{md_filename}', sep=" ")
df2 = copy.deepcopy(df1)
focals = df1['Focal'].values
ids = df1['ID'].values
new_focals = np.array([calculate_focal(focal) for focal in focals])
new_ids = np.array([populate_id(id) for id in ids])
df2['Focal'] = new_focals
df2['ID'] = new_ids
df2.to_csv(f"./{datapath + '_augmented'}/{md_filename}", index=None, sep=' ')

img_numbers.sort()
print(f"Image numbers created (to be added to config): {img_numbers}")
print("Don't forget to also modify the focal lengths in the md.txt with the augmented ones.")
