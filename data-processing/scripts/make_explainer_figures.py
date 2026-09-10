#!/usr/bin/env python3
"""Two figures for explaining the method to a reader, from real data.

Both exist because a term was being used faster than it was being explained.
They are kept as a script rather than made once by hand so that they can be
redrawn when the channels or the data change.

  what_a_cell_is   B9's question, answered rather than deferred: what does one
                   multi-channel cell actually look like. Six greyscale grids
                   of the same square, one probe pixel with its six numbers,
                   and a three-channel false-colour composite beside them so
                   the "you cannot see six at once" limit is visible instead
                   of asserted.

  how_fmem_works   The memorization fraction, drawn in two dimensions where
                   the ratio is obvious. Third panel is the confound the
                   noise-floor run turned up: samples far outside the data
                   score "not a copy" for reasons of scale alone.

Usage:
    python make_explainer_figures.py
    python make_explainer_figures.py --what how_fmem_works
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "hcmc" / "processed"
REPORTS_DIR = PROC_DIR / "reports"

_DDPM_ROOT = ROOT.parent / "ddpm-win-master"
if str(_DDPM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DDPM_ROOT))

# What each channel means, said the way it would be said out loud, plus the
# unit its number is in. The table in the paper can be precise; a reader
# meeting the channel for the first time needs this instead.
PLAIN = {
    "building_binary":       ("is there a building\non this pixel?", "0 or 1"),
    "building_density":      ("how much building\nwithin 100 m", "share, 0 to 1"),
    "accessibility_gravity": ("how easy to reach\nshops by road", "score"),
    "poi_count":             ("how many shops etc.\nwithin 300 m", "shops"),
    "poi_entropy":           ("how mixed those\nshops are", "bits"),
    "road_density":          ("how much street\nwithin 100 m", "m per m2"),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def what_a_cell_is(store_path, cell_id, probe, out_path):
    from datasets.cell_tensors import CellTensorStore

    store = CellTensorStore(str(store_path))
    if cell_id not in set(store.cell_ids()):
        cell_id = sorted(store.cell_ids())[0]
    cell = store.read_cell(cell_id).astype(float)
    names = store.channel_names
    py, px = probe

    n = len(names)
    fig = plt.figure(figsize=(17.5, 8.8))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.25], hspace=.42, wspace=.16)

    for i, name in enumerate(names[:6]):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        grid = cell[i]
        ax.imshow(grid, cmap="gray_r", interpolation="nearest")
        ax.plot(px, py, "o", mfc="none", mec="#D9534F", mew=2.2, ms=13)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"grid {i + 1} — {name}", fontsize=10, fontweight="bold")
        plain, unit = PLAIN.get(name, (name.replace("_", " "), ""))
        ax.text(.5, -.09, plain, transform=ax.transAxes, ha="center", va="top", fontsize=9.5)
        ax.text(.5, -.29, f"at the circle:  {grid[py, px]:.2f}  ({unit})",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color="#D9534F", fontweight="bold")

    ax = fig.add_subplot(gs[:, 3:])

    def norm(a):
        top = np.percentile(a, 99.0)
        return np.clip(a / top, 0, 1) if top > 0 else np.zeros_like(a)

    # Any three channels can go to red, green and blue. Which three is a
    # viewing choice with no meaning in the data, which is the point.
    rgb_idx = [names.index(c) for c in
               ("accessibility_gravity", "poi_entropy", "building_density")
               if c in names][:3]
    while len(rgb_idx) < 3:
        rgb_idx.append(rgb_idx[-1])
    ax.imshow(np.dstack([norm(cell[i]) for i in rgb_idx]), interpolation="nearest")
    ax.plot(px, py, "o", mfc="none", mec="white", mew=2.5, ms=15)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("all six at once? you can't.\n"
                 "this is three of them forced into red / green / blue",
                 fontsize=11, fontweight="bold")
    ax.text(.5, -.045,
            "   ".join(f"{c} = {names[i]}" for c, i in zip("red green blue".split(), rgb_idx))
            + "\nthe colours are a trick for your eyes. the model never sees colour.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5)

    values = "  ".join(f"{cell[i][py, px]:.2f}" for i in range(min(n, 6)))
    fig.suptitle(
        f"What one cell actually is — tile {cell_id}, Ho Chi Minh City\n"
        f"{min(n, 6)} separate grey pictures of the same 1.2 km square, stacked. "
        "That stack is what we call 'the tensor'.\n"
        "The circle marks one pixel — one 1.17 m patch of ground. "
        f"Its {min(n, 6)} numbers, in order:   {values}", fontsize=12.5)
    fig.subplots_adjust(top=.845, bottom=.10, left=.03, right=.985)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info(f"Wrote {out_path}")


def how_fmem_works(out_path, seed=4, n_train=26):
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    train = np.column_stack([rng.normal(0, 1, n_train), rng.normal(0, 1, n_train)])

    # The "copy" star has to sit on a training cell that is itself isolated.
    # On a crowded one the second-nearest is close too and the ratio is not
    # small -- which is a real property of the metric, not a drawing problem,
    # and is why B1 forbids overlapping tiles.
    spacing, _ = cKDTree(train).query(train, k=2)
    copy_star = train[int(np.argmax(spacing[:, 1]))] + np.array([.02, .015])

    def ratio(point):
        d = np.sort(np.linalg.norm(train - point, axis=1))
        return d[0] / d[1]

    assert ratio(copy_star) < 1 / 3, f"copy star is not a copy: {ratio(copy_star):.3f}"

    def panel(ax, gen, title, verdict, colour, note, zoom=None):
        dist = np.linalg.norm(train - gen, axis=1)
        order = np.argsort(dist)
        near, second = train[order[0]], train[order[1]]
        d1, d2 = dist[order[0]], dist[order[1]]
        lim = zoom or 3.4

        ax.scatter(*train.T, s=46, c="#9AA8AA", edgecolors="white", linewidths=.7, zorder=2)
        for point, dd, label, col, side in (
            (near, d1, "closest", "#1F6F8B", 1),
            (second, d2, "2nd closest", "#B7791F", -1),
        ):
            ax.plot([gen[0], point[0]], [gen[1], point[1]], color=col, lw=2.2, zorder=3)
            if dd < .12 * lim:
                # The star is sitting on it, so a midpoint label would be
                # hidden by the marker. Anchor it out with a leader.
                ax.annotate(f"{label}: {dd:.2f}", xy=gen,
                            xytext=gen + np.array([.62, .30]) * lim * side,
                            fontsize=9.5, color=col, ha="center", fontweight="bold",
                            zorder=5,
                            arrowprops=dict(arrowstyle="-", color=col, lw=1, ls=":"),
                            bbox=dict(fc="white", ec="none", alpha=.9, pad=1.6))
            else:
                ax.annotate(f"{label}: {dd:.2f}",
                            (gen + point) / 2 + np.array([0, .10 * lim * side]),
                            fontsize=9.5, color=col, ha="center", fontweight="bold",
                            zorder=5, bbox=dict(fc="white", ec="none", alpha=.9, pad=1.6))
        ax.scatter(*gen, s=230, marker="*", c="#D9534F",
                   edgecolors="white", linewidths=1.1, zorder=6)

        cx, cy = ((gen + train.mean(0)) / 2 if zoom else (0, 0))
        ax.set_xlim(cx - lim, cx + lim); ax.set_ylim(cy - lim, cy + lim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.text(.5, -.055, f"closest ÷ 2nd closest  =  {d1 / d2:.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=11.5)
        ax.text(.5, -.135, verdict, transform=ax.transAxes, ha="center", va="top",
                fontsize=12, fontweight="bold", color=colour)
        ax.text(.5, -.215, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color="#5C6B6D")

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 6.4))
    panel(axes[0], copy_star, "a copy", "COPY   (ratio is below 1/3)", "#B22222",
          "it landed almost exactly on one training cell,\nso 'closest' is nearly zero")
    panel(axes[1], np.array([.15, -1.55]), "something new", "not a copy", "#2E7D32",
          "its two closest training cells are about\nequally far — that is what new looks like")
    panel(axes[2], np.array([26., 20.]), "what an untrained model gave us",
          "not a copy — but the answer is empty", "#B7791F",
          "it is so far outside the data that every training cell\n"
          "is about equally far. the ratio is 1 because of scale,\n"
          "not because the model invented anything", zoom=19)

    fig.suptitle("How we decide whether one generated cell is a copy\n"
                 "Grey dots = the training cells.   Red star = one cell the model produced.\n"
                 "Measure how far it is from the closest training cell, and from the "
                 "second closest, then divide.\n"
                 "F_mem is just: out of all the cells the model produced, what share "
                 "scored below 1/3.", fontsize=13)
    fig.subplots_adjust(top=.735, bottom=.21, wspace=.14)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", choices=["what_a_cell_is", "how_fmem_works", "all"],
                        default="all")
    parser.add_argument("--store", default=str(PROC_DIR / "cells_z15_1024"))
    parser.add_argument("--cell_id", default="15_26097_15392")
    parser.add_argument("--probe", type=int, nargs=2, default=[470, 300],
                        help="Row and column of the pixel to read out.")
    parser.add_argument("--out_dir", default=str(REPORTS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.what in ("what_a_cell_is", "all"):
        what_a_cell_is(args.store, args.cell_id, tuple(args.probe),
                       out_dir / "what_a_cell_is.png")
    if args.what in ("how_fmem_works", "all"):
        how_fmem_works(out_dir / "how_fmem_works.png")


if __name__ == "__main__":
    main()
