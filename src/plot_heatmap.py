import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

def plot_heatmap(x, y, z, nbins = 100, size = 5, export = None, x_label = None, y_label = None, colormap = 'CMRmap_r'):
        dx, dy = x[0] - x[1], y[0] - y[1]

        xlen, ylen = len(x), len(y)
        # Remap x,y,z into 2d grids with len(x) columns and len(y) rows
        x, y, z = [x for _ in y], [(xlen * [y]) for y in y], [[z[xlen * j + i] for i in range(xlen)] for j in range(ylen)]

        # x and y are bounds, so z should be the value *inside* those bounds.
        # Therefore, add a new last point to x and y
        zmin, zmax = min([min(arr) for arr in z]), max([max(arr) for arr in z])
        levels = MaxNLocator(nbins=nbins).tick_values(zmin, zmax)

        # pick the desired colormap, sensible levels, and define a normalization
        # instance which takes data values and translates those into levels
        cmap = plt.get_cmap(colormap)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        plt.figure(figsize=(size, size*ylen/xlen))
        cf = plt.contourf(x, y, z, levels=levels, cmap=cmap)
        cbar = plt.colorbar(cf)

        if x_label is not None: plt.xlabel(x_label)
        if y_label is not None: plt.ylabel(y_label)
        if export is not None: plt.savefig(export, dpi=350)

        plt.show()
