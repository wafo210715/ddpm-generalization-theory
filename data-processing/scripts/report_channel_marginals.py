#!/usr/bin/env python3
"""Per-channel marginals report (B10 of the v4 pipeline map).

    Checkpoint: per-channel marginals, zero fraction, skew, tails.  Chooses
    the transform for each channel.  Gaussianity is not required.  The
    mixtures in the reference papers are solvable toy models, not data
    assumptions.

The second half of that is the important half. This report is not a normality
test and does not grade a channel against a Gaussian. It reports the shape of
each channel's marginal so that B11's transform can be chosen from evidence,
and so that a channel with no signal is visible before it becomes a training
cost rather than after.

What each column is for:

    zero fraction   How much of the channel is the empty atom.  Above roughly
                    0.9 the channel is mostly one value, and no transform will
                    change that: a monotone map moves an atom, it does not
                    spread one (see tests/test_channel_transforms.py).  The
                    lever there is the indicator's ring scale, not B11.
    skew, kurtosis  Whether a tail transform is worth applying at all.
    p99 / p50       A crude tail ratio, in physical units, which says whether
                    the channel's dynamic range is a float16 risk.
    Moran's I       Whether the channel has spatial structure.  The pilot's
                    building channel came out near 0 here, which is what a
                    flattened indicator looks like, and it is the number that
                    B3 exists to fix.
    between-cell    Variance of per-cell means over total variance.  A channel
                    that barely varies within a tile is a cell-level condition
                    rather than a grid --- the companion page's three-way sort
                    --- and this is the number that decides it.

The transform column is a *suggestion* keyed to those numbers, not a decision.
It is printed so the reasoning is visible; B11 takes the list from whoever
decides A1.

Usage:
    python report_channel_marginals.py --store data/hcmc/processed/cells_z15_1024
    python report_channel_marginals.py --store ... --max_cells 200 --plot
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "data" / "hcmc" / "processed" / "reports"

_DDPM_ROOT = ROOT.parent / "ddpm-win-master"
if str(_DDPM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DDPM_ROOT))

# Above this share of exactly-zero units, no monotone transform helps.
SPARSITY_ALARM = 0.90
# Below this, a channel has essentially no spatial pattern to learn.
MORANS_I_ALARM = 0.05
# Above this share of variance living between cells rather than within them,
# the indicator is closer to a cell-level condition than to a grid.
BETWEEN_CELL_ALARM = 0.90

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def suggest_transform(stats: dict) -> str:
    """A suggestion, from the numbers, with the reason attached.

    Deliberately simple and deliberately not authoritative.  B11's list is a
    decision for whoever settles A1; this exists so that the numbers in the
    table can be read as pointing somewhere.
    """
    # Atom first: an all-zero channel is also technically "binary", and saying
    # so would hide that it carries nothing.
    if stats["zero_fraction"] >= SPARSITY_ALARM:
        return "none helps (atom)"
    if stats["n_distinct"] <= 2:
        # A mask has nothing for a transform to reshape; its skew is just its
        # fill rate wearing a statistic's name.
        return "identity (binary)"
    if stats["skewness"] > 2.0 and stats["min"] >= 0.0:
        return "log1p or quantile"
    if abs(stats["skewness"]) > 1.0:
        return "quantile"
    return "identity"


def channel_stats(values: np.ndarray, morans: float, between: float) -> dict:
    from scipy.stats import kurtosis, skew

    finite = values[np.isfinite(values)]
    quantiles = np.percentile(finite, [1, 50, 99]) if finite.size else [0, 0, 0]
    p50, p99 = float(quantiles[1]), float(quantiles[2])

    # scipy returns nan for a constant array, which is true but useless in a
    # table.  A channel with no spread has no shape, so report 0.
    has_spread = finite.size > 0 and finite.std() > 0
    # Only the "is this a mask" question matters, so stop counting at three.
    n_distinct = int(np.unique(finite[: min(finite.size, 100_000)]).size) if finite.size else 0

    return {
        "n_values": int(finite.size),
        "n_distinct": n_distinct,
        "min": float(finite.min()) if finite.size else 0.0,
        "max": float(finite.max()) if finite.size else 0.0,
        "mean": float(finite.mean()) if finite.size else 0.0,
        "std": float(finite.std()) if finite.size else 0.0,
        "zero_fraction": float(np.mean(finite == 0.0)) if finite.size else 1.0,
        "p1": float(quantiles[0]),
        "p50": p50,
        "p99": p99,
        "tail_ratio_p99_p50": float(p99 / p50) if p50 > 0 else np.inf,
        "skewness": float(skew(finite)) if has_spread else 0.0,
        "kurtosis": float(kurtosis(finite)) if has_spread else 0.0,
        "morans_i": morans,
        "between_cell_variance_share": between,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="CellTensorStore directory.")
    parser.add_argument("--max_cells", type=int, default=200,
                        help="Marginals converge long before every cell is read.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--plot", action="store_true", help="Write histograms.")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    from analysis.spatial import morans_i
    from datasets.cell_tensors import CellTensorStore

    store = CellTensorStore(args.store)
    cell_ids = store.cell_ids()
    if not cell_ids:
        raise SystemExit(f"{args.store} holds no cells")
    if len(cell_ids) > args.max_cells:
        rng = np.random.default_rng(args.seed)
        cell_ids = [
            cell_ids[i]
            for i in sorted(rng.choice(len(cell_ids), args.max_cells, replace=False))
        ]

    names = store.channel_names
    log.info(f"{args.store}: {len(store)} cells, {len(names)} channels, "
             f"reading {len(cell_ids)}")

    per_channel_values: list[list[np.ndarray]] = [[] for _ in names]
    per_channel_morans: list[list[float]] = [[] for _ in names]
    per_cell_means: list[list[float]] = [[] for _ in names]

    for cell_id in cell_ids:
        cell = store.read_cell(cell_id).astype(np.float64)
        for i in range(len(names)):
            grid = cell[i]
            per_channel_values[i].append(grid.ravel())
            per_channel_morans[i].append(morans_i(grid, args.connectivity))
            per_cell_means[i].append(float(grid.mean()))

    rows = []
    histograms = {}
    for i, name in enumerate(names):
        values = np.concatenate(per_channel_values[i])
        morans = float(np.mean(per_channel_morans[i]))

        # Between-cell share: how much of the total variance is differences
        # between cells rather than differences within one.
        total_variance = float(values.var())
        between = (
            float(np.var(per_cell_means[i]) / total_variance)
            if total_variance > 0 else 1.0
        )

        stats = channel_stats(values, morans, min(between, 1.0))
        stats["channel"] = name
        stats["suggested_transform"] = suggest_transform(stats)
        rows.append(stats)
        if args.plot:
            histograms[name] = values

    frame = pd.DataFrame(rows).set_index("channel")

    print()
    print(f"Per-channel marginals over {len(cell_ids)} cells "
          f"({store.img_size} px, {store.img_channels} channels)")
    print()
    header = (f"{'channel':>22} | {'zero':>6} | {'skew':>7} | {'kurt':>8} | "
              f"{'p99/p50':>8} | {'Moran I':>8} | {'between':>7} | suggestion")
    print(header)
    print("-" * len(header))
    # Bracket access throughout: "kurtosis" and "skew" are also DataFrame
    # methods, and row.kurtosis would hand back the method, not the number.
    for name, row in frame.iterrows():
        tail_value = row["tail_ratio_p99_p50"]
        tail = "inf" if not np.isfinite(tail_value) else f"{tail_value:.1f}"
        print(f"{name:>22} | {row['zero_fraction']:>5.1%} | {row['skewness']:>7.2f} | "
              f"{row['kurtosis']:>8.1f} | {tail:>8} | {row['morans_i']:>8.3f} | "
              f"{row['between_cell_variance_share']:>6.1%} | "
              f"{row['suggested_transform']}")

    print()
    for name, row in frame.iterrows():
        if row["zero_fraction"] >= SPARSITY_ALARM:
            print(f"  [!] {name}: {row['zero_fraction']:.1%} exact zeros. No "
                  f"monotone transform spreads an atom;\n      the lever is the "
                  f"indicator's ring scale, not B11.")
        if abs(row["morans_i"]) < MORANS_I_ALARM:
            print(f"  [!] {name}: Moran's I {row['morans_i']:.3f}. Effectively no "
                  f"spatial pattern --\n      this is what the pilot's flattened "
                  f"building channel looked like.")
        if row["between_cell_variance_share"] >= BETWEEN_CELL_ALARM:
            print(f"  [!] {name}: {row['between_cell_variance_share']:.1%} of "
                  f"variance is between cells, not within.\n      Closer to a "
                  f"cell-level condition than to a grid.")

    out_dir = Path(args.out) if args.out else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "channel_marginals.csv")
    log.info(f"Wrote {out_dir}/channel_marginals.csv")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(names)
        fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6))
        axes = np.atleast_2d(axes)
        for i, name in enumerate(names):
            values = histograms[name]
            axes[0, i].hist(values, bins=80, color="#444")
            axes[0, i].set_title(name, fontsize=9)
            axes[0, i].set_yscale("log")
            non_zero = values[values != 0]
            if non_zero.size:
                axes[1, i].hist(non_zero, bins=80, color="#b46")
            axes[1, i].set_yscale("log")
            axes[1, i].set_xlabel("non-zero units only", fontsize=8)
        axes[0, 0].set_ylabel("all units (log count)")
        axes[1, 0].set_ylabel("non-zero (log count)")
        fig.tight_layout()
        fig.savefig(out_dir / "channel_marginals.png", dpi=130)
        log.info(f"Wrote {out_dir}/channel_marginals.png")


if __name__ == "__main__":
    main()
