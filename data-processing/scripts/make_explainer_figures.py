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

    # The first panel must be the data as it actually is. An earlier version
    # showed three rearrangements and none of them was the real tile, which
    # made the figure a demonstration of a definition rather than a statement
    # about Ho Chi Minh City. The other two are the same two grids with one of
    # them moved, found by sweeping turns, mirrors and shifts.
    ARRANGEMENTS = [
        ("as they really are", (0, 0, 0, 0), "nothing has been moved",
         "knowing where one is bright tells you@almost nothing about the other"),
        ("one grid turned upside down", (2, 0, 0, 0), "accessibility rotated 180°",
         "now the built-up parts and the@reachable parts are the same parts"),
        ("one grid turned and shifted", (3, 1, 640, 768),
         "accessibility rotated 270°, mirrored, shifted",
         "now where one is bright@the other tends to be dark"),
    ]

    fig = plt.figure(figsize=(15.6, 8.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.7, 1], hspace=.42, wspace=.13)
    a_hi, b_hi = np.percentile(A, 75), np.percentile(B, 75)

    for col, (title, params, what_moved, note) in enumerate(ARRANGEMENTS):
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
        ax.set_title(title, fontsize=12, fontweight="bold", pad=20,
                     color="#1B2426" if col == 0 else "#5C6B6D")
        ax.text(.5, 1.012, what_moved, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9, style="italic", color="#5C6B6D")
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
        + "The FIRST panel is the real data, untouched. It is already the "
          "independent case." + NEWLINE
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



def untrained_vs_trained(npz_path, out_path):
    """What "untrained" means, what comes out of it, and what 1000 steps does.

    Three things at once, because they only make sense together: the network's
    weights before and after, four samples from each, and the real cells for
    scale.  The point that is hard to say and easy to show is that the reverse
    process runs regardless --- it does not check whether the network knows
    anything --- so an untrained network produces output the same way a trained
    one does.  It is just that the output is nonsense, and nonsense at the
    wrong scale, which is what makes its F_mem unreadable.
    """
    d = np.load(npz_path, allow_pickle=True)
    w0, w1, s0, s1, real = d["w0"], d["w1"], d["s0"], d["s1"], d["real"]
    losses = d["losses"]

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, (2, s0.shape[-2], s0.shape[-1]))

    fig = plt.figure(figsize=(16.4, 10.6))
    gs = fig.add_gridspec(3, 6, width_ratios=[1.15, .2, 1, 1, 1, 1],
                          hspace=.46, wspace=.10, top=.845, bottom=.035,
                          left=.045, right=.99)

    ROWS = [
        (0, s0, "the network with its random starting weights",
         "four tries, four different nonsenses"),
        (1, s1, "the same network after 1000 training steps",
         "still noise to look at — but the numbers have pulled in eightfold"),
        (2, real, "four real cells, for scale", "what it is trying to reach"),
    ]

    for row, block, title, note in ROWS:
        if row < 2:
            ax = fig.add_subplot(gs[row, 1])
            ax.imshow(noise[row], cmap="gray_r", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("goes in:\npure noise", fontsize=8.5, color="#5C6B6D")

        first = None
        for k in range(4):
            ax = fig.add_subplot(gs[row, 2 + k])
            ax.imshow(block[k][0], cmap="gray_r", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            first = first or ax

        # Place the row heading against the row's own axes rather than at a
        # guessed figure coordinate, so it cannot drift into the title.
        box = first.get_position()
        lo, hi = float(block.min()), float(block.max())
        fig.text(box.x0, box.y1 + .038, title, fontsize=12.5, fontweight="bold")
        fig.text(box.x0, box.y1 + .018, note, fontsize=10, color="#5C6B6D")
        fig.text(.985, box.y1 + .028,
                 f"its numbers run {lo:,.0f} to {hi:,.0f}", fontsize=10.5,
                 ha="right", fontweight="bold",
                 color="#B22222" if hi > 50 else "#2E7D32")

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-.35, .35, 90)
    ax.hist(w0, bins=bins, color="#B22222", alpha=.55, label="before training")
    ax.hist(w1, bins=bins, color="#2A7392", alpha=.55, label="after 1000 steps")
    ax.set_yticks([]); ax.tick_params(labelsize=8)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("the network's weights", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("PyTorch fills these with small random numbers\n"
                  "when it builds the network. Training nudges them.",
                  fontsize=8.5, color="#5C6B6D")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(np.convolve(losses, np.ones(25) / 25, mode="valid"), color="#2A7392")
    ax.set_title("how wrong its guesses are", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("training step", fontsize=8.5, color="#5C6B6D")
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = fig.add_subplot(gs[2, 0])
    ax.axis("off")
    ax.text(0, 1.0,
            "'Untrained' does not mean trained" + NEWLINE
            + "on something else. It means the" + NEWLINE
            + "weights are still the random numbers" + NEWLINE
            + "PyTorch created the network with." + NEWLINE * 2
            + "Sampling works regardless. The loop" + NEWLINE
            + "asks the network what noise it sees" + NEWLINE
            + "and subtracts the answer, a thousand" + NEWLINE
            + "times over. It never checks whether" + NEWLINE
            + "the answer is any good — so you get" + NEWLINE
            + "an output either way.",
            transform=ax.transAxes, va="top", fontsize=10, linespacing=1.55)

    fig.suptitle(
        "What an untrained model is, and what comes out of it" + NEWLINE
        + "Every row runs the same generation loop from pure random noise, four "
          "times over. The only thing that differs is the network in the middle.",
        fontsize=13.5, y=.975)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info(f"Wrote {out_path}")



def data_in_data_out(npz_path, out_path, traced="poi_count"):
    """What the data looks like going into the model and coming back out.

    The confusion this answers is a fair one: the word "unpack" sounds like it
    recovers something, and it is easy to think a generated cell can be turned
    back into the buildings and shops it was made from.  It cannot.  What
    unpacking does is smaller and completely mechanical --- it puts the numbers
    back into the units they were measured in.  The stack of grids is the same
    stack of grids the whole way through; only the units on the numbers change.

    The traced channel is followed through every stage so one number can be
    watched from a shop count, to a standardized value, to noise, and back.
    """
    d = np.load(npz_path, allow_pickle=True)
    names = list(d["names"])
    ch = names.index(traced) if traced in names else 0
    real_p, real_s = d["real_physical"], d["real_standardized"]
    noise, gen_s, gen_p = d["start_noise"], d["gen_standardized"], d["gen_unpacked"]
    shape = tuple(real_p.shape[1:])
    shape_text = f"{shape[0]} x {shape[1]} x {shape[2]}"

    STAGES = [
        (0, real_p[0][ch], "a real cell", "in the units it was measured in",
         "shops within 300 m", "#2E7D32"),
        (0, real_s[0][ch], "the same cell, standardized",
         "same picture, different numbers", "no units any more", "#2A7392"),
        (1, noise[0][ch], "pure random noise", "where every sample starts",
         "no units", "#5C6B6D"),
        (1, gen_s[0][ch], "what the model produced",
         "after 1000 denoising steps", "no units", "#2A7392"),
        (1, gen_p[0][ch], "the same output, unpacked",
         "the standardizing undone", "shops within 300 m", "#B22222"),
    ]

    fig = plt.figure(figsize=(17.0, 11.0))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, .82], hspace=1.02, wspace=.26,
                          top=.83, bottom=.05, left=.05, right=.985)

    row_anchor = {}
    for row in (0, 1):
        stages = [st for st in STAGES if st[0] == row]
        for col, (_, grid, title, note, unit, colour) in enumerate(stages):
            ax = fig.add_subplot(gs[row, col * 2])
            ax.imshow(grid, cmap="gray_r", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=12, fontweight="bold", pad=24)
            ax.text(.5, 1.02, note, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=9.5, style="italic", color="#5C6B6D")
            ax.text(.5, -.05, shape_text, transform=ax.transAxes, ha="center",
                    va="top", fontsize=9.5, family="monospace", color="#5C6B6D")
            ax.text(.5, -.135, f"{grid.min():,.1f} to {grid.max():,.1f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=11, fontweight="bold", color=colour)
            ax.text(.5, -.225, unit, transform=ax.transAxes, ha="center",
                    va="top", fontsize=9.5, color=colour)
            if col == 0:
                row_anchor[row] = ax.get_position()

    for row, col, label, sub in [
        (0, 1, "standardize", "take each grid's average off,@divide by its spread"),
        (1, 1, "1000 denoising steps", "the model subtracts the noise@it thinks it sees"),
        (1, 3, "unpack", "multiply the spread back on,@add the average back"),
    ]:
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")
        ax.annotate("", xy=(.95, .55), xytext=(.05, .55), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#1F6F8B"))
        ax.text(.5, .63, label, transform=ax.transAxes, ha="center", fontsize=11,
                fontweight="bold", color="#1F6F8B")
        ax.text(.5, .47, sub.replace("@", NEWLINE), transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="#5C6B6D")

    for row, text in ((0, "GOING IN — training"), (1, "COMING OUT — sampling")):
        fig.text(.05, row_anchor[row].y1 + .085, text, fontsize=11.5,
                 fontweight="bold", color="#1F6F8B")

    # All six grids of the unpacked output, on their own strip: "back to grids"
    # is the question the whole figure exists to answer.
    strip = gs[2, :].subgridspec(1, len(names[:6]), wspace=.16)
    first_strip = None
    for k, name in enumerate(names[:6]):
        ax = fig.add_subplot(strip[0, k])
        ax.imshow(gen_p[0][k], cmap="gray_r", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=9.5, pad=6)
        ax.text(.5, -.07, f"{gen_p[0][k].min():,.1f} to {gen_p[0][k].max():,.1f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                family="monospace", color="#5C6B6D")
        first_strip = first_strip or ax

    box = first_strip.get_position()
    fig.text(.05, box.y1 + .062, "AND THAT OUTPUT, SPLIT BACK INTO ITS GRIDS",
             fontsize=11.5, fontweight="bold", color="#1F6F8B")
    fig.text(.05, box.y1 + .040,
             "this is the step you were describing. it is just the stack, "
             "separated and renamed — no buildings or shops come back, only "
             "grids of indicator values.",
             fontsize=9.5, color="#5C6B6D")

    bad = float(gen_p[0][ch].min())
    fig.suptitle(
        "What goes into the model, and what comes out" + NEWLINE
        + f"One channel ({traced}) followed the whole way. The stack is "
          f"{shape_text} at every stage — nothing is ever reshaped." + NEWLINE
        + "Standardizing and unpacking are the same arithmetic in opposite "
          "directions, and neither checks whether the numbers make sense:"
        + NEWLINE
        + f"this deliberately undertrained demo unpacks to {bad:,.0f} shops.",
        fontsize=13, y=.982)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info(f"Wrote {out_path}")



def cell_in_3d(npz_path, out_path, n=64):
    """One cell drawn as what it is: six sheets stacked into a block.

    The flat small-multiples view is honest but it hides the thing that makes a
    cell one object rather than six pictures --- every sheet is the same square
    of ground, and a unit is a column straight down through all of them.  Drawn
    as a stack that becomes obvious.

    Real and generated are drawn on one shared colour scale, so the comparison
    is a comparison and not two separately-stretched pictures.
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize

    d = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    real, gen = d["real_std"][0], d["gen_std"][0]

    both = np.concatenate([real.ravel(), gen.ravel()])
    norm = Normalize(vmin=float(np.percentile(both, .5)),
                     vmax=float(np.percentile(both, 99.5)))
    cmap = cm.magma
    yy, xx = np.mgrid[0:n, 0:n]
    GAP = 26

    fig = plt.figure(figsize=(16.4, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 1], width_ratios=[1, 1, .05],
                          hspace=.24, wspace=.06, top=.855, bottom=.07,
                          left=.02, right=.93)

    for col, (block, label, sub) in enumerate([
        (real, "a real cell", "tile 15_26097_15392, pooled to 64 x 64"),
        (gen, "a cell the model invented", "after a 10 000-step demo run"),
    ]):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        for k in range(len(names)):
            z = np.full_like(xx, k * GAP, dtype=float)
            ax.plot_surface(xx, yy, z, rstride=1, cstride=1, shade=False,
                            facecolors=cmap(norm(block[k])), linewidth=0,
                            antialiased=False)
            ax.text(n + 13, n * .52, k * GAP, names[k], fontsize=8.5,
                    color="#1B2426", zdir="y", ha="left", va="center")
        # A unit is a column through every sheet, so draw one.
        ux, uy = 22, 40
        ax.plot([ux, ux], [uy, uy], [-GAP * .4, (len(names) - 1) * GAP + GAP * .4],
                color="#D9534F", lw=2.2, zorder=10)
        for k in range(len(names)):
            ax.scatter([ux], [uy], [k * GAP], color="#D9534F", s=26,
                       edgecolors="white", linewidths=.6, zorder=11)

        ax.set_box_aspect((1, 1, .85))
        ax.view_init(elev=20, azim=-62)
        ax.set_axis_off()
        # Room on the +x side for the channel labels, which otherwise clip.
        ax.set_xlim(-2, n + 34)
        ax.set_ylim(-2, n + 2)
        ax.text2D(.5, 1.00, label, transform=ax.transAxes, ha="center",
                  fontsize=13, fontweight="bold")
        ax.text2D(.5, .955, sub, transform=ax.transAxes, ha="center",
                  fontsize=9.5, color="#5C6B6D")
        ax.text2D(.5, .04,
                  f"numbers run {block.min():.1f} to {block.max():.1f}",
                  transform=ax.transAxes, ha="center", fontsize=10.5,
                  fontweight="bold", color="#2A7392")

    cax = fig.add_subplot(gs[0, 2])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cax.set_title("standardized\nvalue", fontsize=9, color="#5C6B6D", pad=10)
    cax.tick_params(labelsize=8)

    ax = fig.add_subplot(gs[1, :2])
    width = .38
    pos = np.arange(len(names))
    for offset, block, colour, label in ((-width / 2, real, "#2A7392", "real"),
                                         (width / 2, gen, "#B7791F", "generated")):
        parts = ax.violinplot([block[k].ravel() for k in range(len(names))],
                              positions=pos + offset, widths=width,
                              showextrema=False, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(colour); body.set_alpha(.7)
        parts["cmedians"].set_color("#1B2426")
        ax.plot([], [], color=colour, lw=6, alpha=.7, label=label)
    ax.set_xticks(pos); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("standardized value", fontsize=9.5)
    ax.legend(fontsize=10, frameon=False, ncol=2, loc="upper right")
    ax.tick_params(labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("the same numbers as distributions, channel by channel",
                 fontsize=10.5, color="#5C6B6D")

    fig.suptitle(
        "One cell, drawn as the stack it is" + NEWLINE
        + "Six sheets, one per indicator, all covering the same 1.2 km square. "
          "The red line is one unit — a column straight down through all six." + NEWLINE
        + "Both blocks share one colour scale, so a colour means the same "
          "number on the left and on the right.",
        fontsize=13, y=.975)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info(f"Wrote {out_path}")



def how_the_unet_reads(npz_path, out_path, n=16):
    """How a convolution reads the stack, and whether channel order means anything.

    The question this answers came up directly: does the model know that the
    first number in a unit's vector and the sixth are different things, and
    does their order matter?

    The answer is visible in the first layer's own weights.  Every filter is
    3 x 3 x C --- three by three across the ground, and all the way through the
    stack --- so a filter reads a patch of every channel at once and adds the
    results into one number.  Two consequences follow, and they pull in
    opposite directions:

      * The model absolutely distinguishes the channels, because each one gets
        its own weights, and training drives those weights apart.
      * The model has no notion of channel *order*.  The convolution slides
        across width and height; it does not slide along the channel axis, it
        sums over it.  Channel 6 is not "after" channel 1 in any sense the
        arithmetic can see.  Relabel them consistently and retrain and you get
        an equivalent model.

    Which is the opposite of the two spatial axes, where "next to" is exactly
    what the sliding is for.
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize, TwoSlopeNorm

    d = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    block = d["real_std"][0]
    w_init, w_trained = d["w_init"], d["w_trained"]
    n_out, n_ch = w_trained.shape[0], w_trained.shape[1]

    coarse = block[:, ::block.shape[-1] // n, ::block.shape[-1] // n][:, :n, :n]
    yy, xx = np.mgrid[0:n, 0:n]
    GAP, WIN = 7, 3
    wx, wy = 6, 7

    fig = plt.figure(figsize=(16.6, 9.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.62, 1], width_ratios=[1.25, 1, 1],
                          hspace=.46, wspace=.24, top=.845, bottom=.06,
                          left=.035, right=.975)

    # --- the stack, with one 3x3 window lit through every sheet -------------
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    norm = Normalize(*np.percentile(coarse, [1, 99]))
    for k in range(len(names)):
        z = np.full_like(xx, k * GAP, dtype=float)
        faces = cm.magma(norm(coarse[k]))
        faces[wy:wy + WIN, wx:wx + WIN] = [.0, .90, .95, 1.0]
        ax.plot_surface(xx, yy, z, rstride=1, cstride=1, shade=False,
                        facecolors=faces, linewidth=0, antialiased=False)
        ax.text(n + 3, n * .5, k * GAP, names[k], fontsize=8, zdir="y",
                ha="left", va="center")
    ax.set_box_aspect((1, 1, .8)); ax.view_init(elev=20, azim=-62)
    ax.set_axis_off(); ax.set_xlim(-1, n + 22); ax.set_ylim(-1, n + 1)
    ax.text2D(.5, 1.0, "one filter's field of view", transform=ax.transAxes,
              ha="center", fontsize=12.5, fontweight="bold")
    ax.text2D(.5, .945,
              f"3 x 3 across the ground, all {n_ch} channels deep",
              transform=ax.transAxes, ha="center", fontsize=9.5, color="#5C6B6D")
    ax.text2D(.5, .02,
              f"{WIN} x {WIN} x {n_ch} = {WIN*WIN*n_ch} numbers in,  1 number out",
              transform=ax.transAxes, ha="center", fontsize=11,
              fontweight="bold", color="#0E8C99")

    # --- one real filter, channel by channel -------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    pick = int(np.argmax(np.abs(w_trained).sum(axis=(1, 2, 3))))
    filt = w_trained[pick]
    wnorm = TwoSlopeNorm(vcenter=0, vmin=filt.min(), vmax=filt.max())
    for k in range(n_ch):
        sub = ax.inset_axes([.06 + (k % 3) * .32, .58 - (k // 3) * .40, .27, .34])
        sub.imshow(filt[k], cmap="RdBu_r", norm=wnorm, interpolation="nearest")
        sub.set_xticks([]); sub.set_yticks([])
        sub.set_title(names[k], fontsize=7.5, pad=3)
        for r in range(3):
            for c in range(3):
                sub.text(c, r, f"{filt[k, r, c]:+.2f}", ha="center", va="center",
                         fontsize=6.2,
                         color="white" if abs(filt[k, r, c]) > abs(filt).max() * .55
                         else "#1B2426")
    ax.set_title(f"the weights of one real filter (#{pick} of {n_out})",
                 fontsize=12.5, fontweight="bold", pad=12)
    ax.text(.5, -.02, "every channel gets its own nine numbers.\n"
                      "the filter multiplies, then adds all 54 together.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
            color="#5C6B6D")

    # --- does it treat the channels differently? ---------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0, 1.0,
            "Does the model know channel 1" + NEWLINE
            + "from channel 6?" + NEWLINE * 2
            + "Yes. Each channel has its own" + NEWLINE
            + "weights, and training pulls them" + NEWLINE
            + "apart — see the chart below." + NEWLINE * 2
            + "Does the ORDER mean anything?" + NEWLINE * 2
            + "No. The filter slides across the" + NEWLINE
            + "ground, left-right and up-down." + NEWLINE
            + "It does not slide down through the" + NEWLINE
            + "stack — it adds the whole depth up" + NEWLINE
            + "at once." + NEWLINE * 2
            + "So 'channel 6' is a name, not a" + NEWLINE
            + "position. Relabel them all the same" + NEWLINE
            + "way and retrain, and you get the" + NEWLINE
            + "same model." + NEWLINE * 2
            + "The opposite of the two map" + NEWLINE
            + "directions, where being next to" + NEWLINE
            + "something is the entire point.",
            transform=ax.transAxes, va="top", fontsize=10.2, linespacing=1.52)

    # --- weight strength per channel, before and after ---------------------
    ax = fig.add_subplot(gs[1, :])
    before = np.abs(w_init).mean(axis=(0, 2, 3))
    after = np.abs(w_trained).mean(axis=(0, 2, 3))
    pos = np.arange(n_ch); width = .38
    ax.bar(pos - width / 2, before, width, color="#B22222", alpha=.55,
           label="before training")
    ax.bar(pos + width / 2, after, width, color="#2A7392", alpha=.85,
           label="after 10 000 steps")
    ax.set_xticks(pos); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("average weight size", fontsize=9.5)
    ax.legend(fontsize=10, frameon=False, ncol=2)
    ax.tick_params(labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    spread = after.max() / after.min()
    ax.set_title(
        f"how hard the first layer leans on each channel, averaged over all "
        f"{n_out} filters — even at the start, {spread:.1f}x apart after "
        f"training (the per-filter patterns above differ far more than these "
        f"averages do)",
        fontsize=10, color="#5C6B6D", pad=10)

    fig.suptitle(
        "How the network reads the stack" + NEWLINE
        + "A convolution is a small window that slides over the picture. Here "
          "the window is 3 x 3 on the ground — but it goes all the way through "
          "every channel," + NEWLINE
        + "so at each position it reads 54 numbers and writes one. That is what "
          "'the first convolution absorbs any channel count' means.",
        fontsize=13, y=.975)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", default="all", choices=[
        "what_a_cell_is", "how_fmem_works", "what_independent_means",
        "what_the_descriptor_loses", "untrained_vs_trained", "data_in_data_out", "cell_in_3d", "how_the_unet_reads", "all"])
    parser.add_argument("--store", default=str(PROC_DIR / "cells_z15_1024"))
    parser.add_argument("--cell_id", default="15_26097_15392")
    parser.add_argument("--probe", type=int, nargs=2, default=[470, 300],
                        help="Row and column of the pixel to read out.")
    parser.add_argument("--out_dir", default=str(REPORTS_DIR))
    parser.add_argument("--demo_npz", default=None,
                        help="Output of the 10k-step demo run.")
    parser.add_argument("--inout_npz", default=None,
                        help="Output of the in/out demo run.")
    parser.add_argument("--train_demo_npz", default=None,
                        help="Output of the 1000-step demo run, for "
                             "untrained_vs_trained.")
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
    if args.what == "cell_in_3d":
        if not args.demo_npz:
            raise SystemExit("cell_in_3d needs --demo_npz")
        cell_in_3d(args.demo_npz, out_dir / "cell_in_3d.png")
    if args.what == "how_the_unet_reads":
        if not args.demo_npz:
            raise SystemExit("how_the_unet_reads needs --demo_npz")
        how_the_unet_reads(args.demo_npz, out_dir / "how_the_unet_reads.png")
    if args.what == "data_in_data_out":
        if not args.inout_npz:
            raise SystemExit("data_in_data_out needs --inout_npz")
        data_in_data_out(args.inout_npz, out_dir / "data_in_data_out.png")
    if args.what == "untrained_vs_trained":
        if not args.train_demo_npz:
            raise SystemExit("untrained_vs_trained needs --train_demo_npz")
        untrained_vs_trained(args.train_demo_npz,
                             out_dir / "untrained_vs_trained.png")


if __name__ == "__main__":
    main()
