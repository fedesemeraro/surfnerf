import osgeo.gdal as gdal
import matplotlib.pyplot as plt
from skimage.transform import resize


def read_depth_map(depth_map_path, df=1):
    """

    Open one-band depth map .tif file.

    Parameters:
    depth_map_path (string): Path to depth map
    df (int): Downscaling factor for depth map (optional)

    Outputs:
    img (array[N,M]): Depth map of size NxM
    """
    data_source = gdal.Open(depth_map_path)
    img = data_source.ReadAsArray()
    if df > 1:
        img = resize(
            img, (img.shape[0]//df, img.shape[1]//df), anti_aliasing=True
        )
    return img


def plot_depth_map(depth_map):
    """

    Plot depth map with colorbar. 

    Parameters:
    depth_map ((Tensor[H,W])

    """
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(depth_map, cmap='gray')
    plt.title('Depth map')
    plt.colorbar()
    plt.show()


img_name = "sat68_spiral_100000__view_1_depth.tiff"
depth_map = read_depth_map(f"../logs/sat68/{img_name}")
plot_depth_map(depth_map)
