from pathlib import Path
import pickle
import numpy as np
import json
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from matplotlib.colors import ListedColormap


def transparent_cmap(base_cmap=plt.cm.cool):
    colors = base_cmap(np.linspace(0, 1, 256))
    colors[:, 0:3] = 0  # Set RGB values to 0 (black)
    colors[:, -1] = np.linspace(0, 1, 256)  # Set alpha values (0.2 to 1)
    # Create a new colormap with transparency
    return ListedColormap(colors)


def constant_cmap():
    colors = np.zeros((256, 4))
    colors[:, 0:3] = (1, 0, 0)  # Set RGB values to blue
    # colors[:, 0:2] = 0  # Set RGB values to 0 (black)
    # colors[:, 2] = 1  # Set blue channel to 1
    colors[10:, -1] = 1  # Set alpha values to 1 (fully opaque)
    return ListedColormap(colors)


TRANSPARENT_CMAP = transparent_cmap()


def calcHeatmap(data_dir, data_range=None, normalize=True):
    heatmap = None

    file_list = list(data_dir.iterdir())
    if data_range is None:
        data_range = range(len(file_list))

    val_cnt = 0
    for i in data_range:
        p = file_list[i]
        print(f"Processing {i}")
        with open(p, "rb") as f:
            try:
                pickled_path = pickle.load(f)
                fdata = json.loads(f.read().decode('utf-8'))
            except EOFError:
                continue
            w, h = fdata["cfg"]["width"], fdata["cfg"]["height"]
            if heatmap is None:
                heatmap = np.zeros((w, h))
            elif not np.all(heatmap.shape == (w, h)):
                raise ValueError("Heatmap shape mismatch")
            nodes = np.array(fdata["data"]["nodes"])
            if len(nodes.shape) == 3:
                x_vals = np.round(nodes[:, :, 0].flatten()).astype(np.int32)
                y_vals = np.round(nodes[:, :, 1].flatten()).astype(np.int32)
            else:
                x_vals = np.round(nodes[:, 0]).astype(np.int32)
                y_vals = np.round(nodes[:, 1]).astype(np.int32)
            val_cnt += len(x_vals)
            heatmap[x_vals, y_vals] += 1
    if normalize:
        heatmap = heatmap / val_cnt
    print(f"Total values: {val_cnt}")
    return heatmap


def smoothHeatmap(heatmap, sigma=1):
    smoothed_heatmap = gaussian_filter(heatmap, sigma=sigma)
    return smoothed_heatmap


def overlayHeatmap(savePath: Path, heatmap, img, alpha=1, cmp=TRANSPARENT_CMAP, show=False, reached=None):
    height, width = img.shape[:2]  # Get the dimensions of the input image
    assert heatmap.T.shape == (
        height, width), "Heatmap shape does not match image shape"
    print(heatmap.T.shape, img.shape)
    dpi = 100  # Set the DPI (dots per inch)
    figsize = (width / dpi, height / dpi)  # Calculate figure size in inches

    # Create the figure with the calculated size and DPI

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.imshow(heatmap.T, alpha=0.4, cmap=constant_cmap())
    ax.imshow(heatmap.T, alpha=alpha, cmap=cmp)

    if reached is not None:
        emoji = '✔' if reached else '✘'
        color = 'green' if reached else 'red'
        ax.text(1500, 400, emoji, fontsize=300, ha='center',
                va='center', color=color, alpha=0.6)
    ax.axis('off')
    plt.savefig(savePath, dpi=dpi, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()
