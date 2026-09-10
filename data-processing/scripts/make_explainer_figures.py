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

NEWLINE = chr(10)  # keeps literal newlines out of source strings

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



def what_independent_means(store_path, cell_id, out_path):
    """What it means for two grids to be arranged independently of each other.

    The phrase is easy to say and hard to picture, and the natural guess ---
    that it is something about the two histograms overlapping — is wrong in
    an instructive way. A histogram has no addresses in it. Two grids can have
    exactly the same pair of histograms and be laid out on top of each other,
    laid out avoiding each other, or laid out with no relation at all.

    So: take the two real grids, and move one of them around. Turning a grid
    cannot change a single one of its values, so the histograms below are the
    same picture three times over, while the arrangement above goes from
    together, to unrelated, to apart.
    """
    from analysis.spatial import bivariate_morans_i
    from datasets.cell_tensors import CellTensorStore

    store = CellTensorStore(str(store_path))
    if cell_id not in set(store.cell_ids()):
        cell_id = sorted(store.cell_ids())[0]
    cell = store.read_cell(cell_id).astype(float)
    names = store.channel_names
    A = cell[names.index("building_density")]
    B = cell[names.index("accessibility_gravity")]

    def move(k, flip, sy, sx):
        out = np.rot90(A, k)
        if flip:
            out = np.fliplr(out)
        return np.roll(np.roll(out, sy, 0), sx, 1)

    # Found by sweeping turns, mirrors and shifts for the three cases.
    ARRANGEMENTS = [
        ("they sit on top of each other", (2, 0, 0, 0),
         "the built-up parts and the reachable@parts are the same parts"),
        ("no relation — 'independent'", (2, 0, 256, 0),
         "knowing where one is bright tells you@nothing about where the other is"),
        ("they avoid each other", (3, 1, 640, 768),
         "where one is bright the other@tends to be dark"),
    ]

    fig = plt.figure(figsize=(15.6, 8.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.7, 1], hspace=.42, wspace=.13)
    a_hi, b_hi = np.percentile(A, 75), np.percentile(B, 75)

    for col, (title, params, note) in enumerate(ARRANGEMENTS):
        moved = move(*params)
        bi = bivariate_morans_i(moved, B)

        ax = fig.add_subplot(gs[0, col])
        # Paint only the brightest quarter of each grid. Overlap is then
        # something you can see rather than something you infer.
        canvas = np.ones((*A.shape, 3))
        canvas[moved >= a_hi] = [.16, .45, .58]
        canvas[B >= b_hi] = [.85, .55, .12]
        canvas[(moved >= a_hi) & (B >= b_hi)] = [.15, .15, .18]
        ax.imshow(canvas, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(.5, -.045, note.replace("@", NEWLINE), transform=ax.transAxes,
                ha="center", va="top", fontsize=10)
        ax.text(.5, -.175, f"bivariate Moran's I = {bi:+.2f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=12.5, fontweight="bold",
                color="#B22222" if bi < -.2 else ("#2E7D32" if bi > .2 else "#B7791F"))

        ax = fig.add_subplot(gs[1, col])
        ax.hist(moved.ravel(), bins=60, color="#2A7392", alpha=.85,
                label="building density")
        ax.hist(B.ravel() / B.max() * moved.max(), bins=60, color="#D98C1F",
                alpha=.6, label="accessibility (rescaled to fit)")
        ax.set_yticks([]); ax.tick_params(labelsize=8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        if col == 0:
            ax.legend(fontsize=8.5, frameon=False, loc="upper right")
        ax.set_title("the two histograms", fontsize=9.5, color="#5C6B6D")

    fig.suptitle(
        "What 'arranged independently' means — one real cell, tile "
        + cell_id + NEWLINE
        + "Blue = the most built-up quarter of the cell.   Orange = the most "
          "reachable quarter.   Black = both at once." + NEWLINE
        + "All three pictures use the SAME two grids. One of them has simply "
          "been turned and shifted, which cannot change any of its values —"
        + NEWLINE
        + "so the histograms along the bottom are identical in all three. "
          "A histogram has no addresses in it, and cannot tell these apart.",
        fontsize=12.5)
    fig.subplots_adjust(top=.80, bottom=.07, left=.03, right=.985)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info(f"Wrote {out_path}")


def what_the_descriptor_loses(store_path, cell_id, out_path):
    """Why a descriptor match is not a copy.

    Turning a cell a quarter turn cannot change any per-channel average, any
    spread, any zero count, any clumpiness or any cross-channel correlation:
    every one of those is blind to where things are. So the descriptor comes
    back the same to sixteen decimal places, while the cell has moved as far
    away, pixel for pixel, as an unrelated tile in another district.

    That is the whole reason the study needs a second measurement.
    """
    from analysis.cell_descriptor import describe_cell
    from datasets.cell_tensors import CellTensorStore

    store = CellTensorStore(str(store_path))
    ids = [c for c in sorted(store.cell_ids()) if store.read_cell(c)[0].std() > 0]
    if cell_id not in ids:
        cell_id = ids[0]
    other_id = next(c for c in ids if c != cell_id)

    names = store.channel_names
    cell = store.read_cell(cell_id).astype(float)
    turned = np.rot90(cell, 1, axes=(1, 2)).copy()
    other = store.read_cell(other_id).astype(float)

    # A handful of the descriptor's numbers, shown as numbers. An earlier
    # version plotted all 45 as scaled lines, but with only three cells the
    # scaling pinned every value to 0 or 1 and the chart became a square wave
    # that showed nothing.
    SHOWN = [
        ("building_binary_mean", "how much of the cell is built"),
        ("building_binary_zero_fraction", "share of pixels with no building"),
        ("building_binary_morans_i", "how clumped the buildings are"),
        ("building_density_mean", "average building density"),
        ("building_density_std", "how much that density varies"),
        ("accessibility_gravity_mean", "average reachability"),
        ("poi_count_mean", "average shops within 300 m"),
        ("corr_building_binary_building_density", "do the two building grids agree"),
    ]
    keys = [k for k, _ in SHOWN if k in describe_cell(cell, names)]
    described = [describe_cell(c, names) for c in (cell, turned, other)]

    def rms(x, y):
        return float(np.sqrt(((x - y) ** 2).mean()))

    fig = plt.figure(figsize=(15.6, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=.30, wspace=.12)
    for col, (grid, label, sub) in enumerate([
        (cell, "cell A — " + cell_id, "a real tile"),
        (turned, "cell A, given a quarter turn", "the same place, rotated"),
        (other, "cell B — " + other_id, "a different tile"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(grid[0], cmap="gray_r", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=11.5, fontweight="bold")
        ax.text(.5, -.05, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color="#5C6B6D")

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    label_of = dict(SHOWN)
    head = ["what the number means", "cell A", "A turned", "cell B"]
    xs = [.02, .52, .68, .86]
    for x, h in zip(xs, head):
        ax.text(x, .96, h, transform=ax.transAxes, fontsize=10.5,
                fontweight="bold", color="#5C6B6D",
                ha="left" if x < .5 else "center")
    for row, key in enumerate(keys):
        y = .84 - row * .105
        ax.text(xs[0], y, label_of[key], transform=ax.transAxes, fontsize=10.5)
        same = abs(described[0][key] - described[1][key]) < 1e-9
        for i, x in enumerate(xs[1:]):
            ax.text(x, y, f"{described[i][key]:.4f}", transform=ax.transAxes,
                    fontsize=10.5, ha="center", family="monospace",
                    color=("#2A7392" if i < 2 and same else
                           ("#B7791F" if i == 2 else "#1B2426")))
        ax.plot([.02, .95], [y - .028, y - .028], transform=ax.transAxes,
                color="#D8DEDC", lw=.8)

    biggest = max(abs(described[0][k] - described[1][k])
                  for k in described[0])
    fig.suptitle(
        "Why matching descriptors does not mean a copy" + NEWLINE
        + "A quarter turn cannot change an average, a spread, a zero count, a "
          "clumpiness or a correlation — none of them looks at where anything "
          "is." + NEWLINE
        + f"So every one of the descriptor's numbers stays the same. Across all "
          f"of them the largest change is {biggest:.0e}, which is rounding."
        + NEWLINE
        + f"But pixel by pixel, cell A and cell A turned are "
          f"{rms(cell, turned):.1f} apart — further than cell A is from an "
          f"unrelated tile in another district ({rms(cell, other):.1f}).",
        fontsize=12.5)
    fig.subplots_adjust(top=.775, bottom=.04, left=.03, right=.985)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", default="all", choices=[
        "what_a_cell_is", "how_fmem_works", "what_independent_means",
        "what_the_descriptor_loses", "all"])
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
    if args.what in ("what_independent_means", "all"):
        what_independent_means(args.store, args.cell_id,
                               out_dir / "what_independent_means.png")
    if args.what in ("what_the_descriptor_loses", "all"):
        what_the_descriptor_loses(args.store, args.cell_id,
                                  out_dir / "what_the_descriptor_loses.png")


if __name__ == "__main__":
    main()
