"""Stitch a block of B8 cells straight from a layer array, to see that they tile.

Usage: python figure_b8_mosaic.py building_binary [k]
Draws the k x k cells around Ben Thanh (default 4) from b8_256/layers/<layer>.npy,
with cell edges marked, to discussion/layer_cards_fig/b8_<layer>_mosaic.png.
"""
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
B8 = REPO / "data-processing/data/hcmc/processed/b8_256"
layer = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
lon, lat = 106.6983, 10.7769
x0 = int((lon + 180) / 360 * 2 ** 15)
y0 = int((1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * 2 ** 15)

cells = pd.read_parquet(B8 / "cells.parquet")
a = np.load(B8 / "layers" / f"{layer}.npy", mmap_mode="r")
n = a.shape[1]
idx = {(x, y): i for i, (x, y) in enumerate(zip(cells.x, cells.y))}
xs, ys = range(x0 - k // 2, x0 - k // 2 + k), range(y0 - k // 2, y0 - k // 2 + k)
m = np.full((k * n, k * n), np.nan, np.float32)
for r, y in enumerate(ys):
    for c, x in enumerate(xs):
        if (x, y) in idx:
            m[r * n:(r + 1) * n, c * n:(c + 1) * n] = a[idx[(x, y)]]

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(m, cmap="gray_r", interpolation="nearest")
for t in range(1, k):
    ax.axhline(t * n - .5, color="#e85d04", lw=.6, alpha=.6); ax.axvline(t * n - .5, color="#e85d04", lw=.6, alpha=.6)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"B8 {layer} · {k}×{k} cells around Bến Thành, stitched from the array · orange = cell edges", loc="left")
out = REPO / f"discussion/layer_cards_fig/b8_{layer}_mosaic.png"
fig.tight_layout(); fig.savefig(out, dpi=110)
print("wrote", out, "| cells found", int(np.isfinite(m[::n, ::n]).sum()), "of", k * k)
