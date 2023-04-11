import pyvista as pv
import numpy as np
from skimage import io


dsm = io.imread("data/068/JAX_068_df1_dsm.tif")

grid = pv.UniformGrid()
grid.dimensions = np.array([dsm.shape[0] + 1, dsm.shape[1] + 1, 1])
grid.spacing = (1,1,1)
grid.cell_data["points"] = dsm.flatten(order="F")
surface = grid.cell_data_to_point_data().warp_by_scalar()

p = pv.Plotter()
p.add_mesh(surface, cmap='jet', show_scalar_bar=False)
# p.show_grid()
# p.show_axes()
p.background_color = (1,1,1)  # white background
p.show()
