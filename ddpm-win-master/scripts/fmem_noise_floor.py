"""F_mem on an untrained model: the metric's noise floor (D2).

    Score F_mem on an untrained model to get the metric's noise floor.
    Random weights produce nonsense cells.  Whatever memorization fraction
    they score is the floor; only values above it mean the trained model
    copied.  Independent of encoding, so it survives the redesign.  Feeds
    the onset threshold in E4.

The floor is not zero and it is not a constant.  A gap ratio is a ratio of two
distances among N training cells, so even pure noise scores something, and how
much depends on how many training cells there are and how they sit relative to
each other.  That means the floor has to be measured per mutant, since C7's
mutants differ in exactly those two things --- which is why this script sweeps
``--n_train`` rather than printing one number.

E4 reads memorization onset as the first checkpoint whose F_mem rises above the
floor for that mutant, so the number this produces is a threshold in the
paper's headline result and its confidence interval belongs in the disclosure.

Usage:
    python scripts/fmem_noise_floor.py --config configs/hcmc_v4_1024.yaml
    python scripts/fmem_noise_floor.py --config ... --n_train 256 512 1024 \
        --n_generated 256 --sampler ddim --ddim_steps 50
    python scripts/fmem_noise_floor.py --config ... --sampler gaussian  # no U-Net
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

from config_loader import TrainingConfig
from datasets.dataset import get_dataset
from evaluation.gap_ratio import (
    DEFAULT_GAP_THRESHOLD,
    STD_TOLERANCE,
    check_standardized,
    compute_fmem,
)
from model.ddpm import DDPM
from model.diffusion_config import DiffusionConfig
from model.unet import DDPMUnet
from sampling.ddim import sample_ddim
from sampling.ddpm_sample import sample_ddpm


def load_training_tensors(config: TrainingConfig, n_train: int, seed: int):
    """Standardized training cells, and the store's channel names."""
    dataset = get_dataset(
        config.dataset_name,
        img_size=config.img_size,
        data_dir=config.data_root,
        split="train",
        center=config.center,
        normalize=config.normalize,
        channels=config.channels,
    )
    if n_train > len(dataset):
        raise ValueError(
            f"asked for {n_train} training cells, the split has {len(dataset)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), n_train, replace=False)

    cells = []
    for i in indices:
        item = dataset[int(i)]
        if isinstance(item, dict):
            item = item.get("tensor", item.get("image"))
        cells.append(np.asarray(item))
    return np.stack(cells)


def untrained_samples(
    config: TrainingConfig,
    n_generated: int,
    sampler: str,
    ddim_steps: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """Cells from a model with random, never-trained weights.

    ``sampler="gaussian"`` skips the U-Net entirely and draws standard normal
    tensors.  That is the floor of the floor: it says what the gap ratio reads
    on samples with no structure at all, and it costs no GPU time.

    It is also the one that can be read at face value.  An untrained U-Net's
    reverse process is not a contraction towards the data --- measured on a
    six-channel 16 px store its samples came back with a per-channel std around
    240 --- so its cells sit far outside the training cloud, both nearest
    neighbours are about equally far, and the gap ratio collapses to 1 for
    scale reasons rather than for reasons about structure.  The script prints
    the generated spread so that a floor of 0.000 can be told apart from a
    floor of 0.000 that means nothing.
    """
    torch.manual_seed(seed)
    shape = (n_generated, config.img_channels, config.img_size, config.img_size)

    if sampler == "gaussian":
        return torch.randn(*shape).numpy()

    diffusion_config = DiffusionConfig(
        n_steps=config.timesteps,
        schedule_type=config.schedule_type,
        device=config.device,
        shift_schedule=config.shift_schedule,
        img_size=config.img_size,
        ref_img_size=config.ref_img_size,
    )
    unet = DDPMUnet(config.img_channels, config.model_channels)
    model = DDPM(config.timesteps, unet, diffusion_config=diffusion_config)
    model = model.to(config.device).eval()

    batches = []
    remaining = n_generated
    while remaining > 0:
        take = min(batch_size, remaining)
        if sampler == "ddim":
            generated, _ = sample_ddim(
                model, n_images=take, config=config,
                diffusion_config=diffusion_config, device=config.device,
                ddim_steps=ddim_steps,
            )
        else:
            generated, _ = sample_ddpm(
                model, n_images=take, config=config,
                diffusion_config=diffusion_config, device=config.device,
            )
        batches.append(generated.detach().cpu().numpy())
        remaining -= take
    return np.concatenate(batches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training config YAML.")
    parser.add_argument("--n_train", type=int, nargs="+", default=[256],
                        help="Training-set sizes to measure the floor at. The "
                             "floor moves with n, so pass every mutant size.")
    parser.add_argument("--n_generated", type=int, default=256)
    parser.add_argument("--sampler", choices=["ddim", "ddpm", "gaussian"],
                        default="ddim")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--gap_threshold", type=float, default=DEFAULT_GAP_THRESHOLD)
    parser.add_argument("--sample_batch_size", type=int, default=16)
    parser.add_argument("--distance_batch_size", type=int, default=64)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_check_standardized", action="store_true",
                        help="Skip the E2 guard on the training cells. The "
                             "generated side is reported rather than checked, "
                             "since an untrained model has no reason to "
                             "produce standardized output.")
    parser.add_argument("--out", type=str, default=None, help="Write JSON here.")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)
    config.print_summary()

    print(f"\nSampler: {args.sampler} (untrained weights)")
    generated = untrained_samples(
        config, args.n_generated, args.sampler, args.ddim_steps, args.seed,
        args.sample_batch_size,
    )
    print(f"Generated {generated.shape[0]} cells of shape {generated.shape[1:]}")

    # Report the generated spread rather than enforcing it.  An untrained model
    # is under no obligation to emit standardized cells, and how far off it is
    # decides whether the floor below means anything.
    generated_std = generated.std(axis=(0, 2, 3))
    print("Generated per-channel std: "
          + ", ".join(f"{s:.3g}" for s in generated_std))
    off_scale = np.abs(generated_std - 1.0) > STD_TOLERANCE
    if np.any(off_scale):
        print(
            f"  [!] {int(off_scale.sum())} of {len(generated_std)} channels are "
            f"off the standardized scale by more than {STD_TOLERANCE}.\n"
            f"      Cells this far outside the training cloud make both nearest\n"
            f"      neighbours about equally distant, so the gap ratio tends to 1\n"
            f"      and the floor reads near zero for reasons of scale, not\n"
            f"      structure. Compare against --sampler gaussian before using\n"
            f"      this number as the E4 threshold."
        )

    device = config.device if torch.cuda.is_available() else "cpu"
    results = []
    print(f"\n{'n_train':>8} | {'F_mem floor':>11} | {'95% CI':>18} | "
          f"{'median d1/d2':>12}")
    print("-" * 60)

    for n_train in args.n_train:
        train = load_training_tensors(config, n_train, args.seed)
        if not args.no_check_standardized:
            check_standardized(train, "train_tensors")
        result = compute_fmem(
            train, generated,
            gap_threshold=args.gap_threshold,
            device=device,
            batch_size=args.distance_batch_size,
            n_bootstrap=args.n_bootstrap,
            random_state=args.seed,
            check=False,
        )
        ratios = result.pop("gap_ratios")
        result["n_train"] = n_train
        result["median_gap_ratio"] = float(np.median(ratios))
        result["generated_channel_std"] = [float(v) for v in generated_std]
        results.append(result)
        print(f"{n_train:>8} | {result['fraction_mem']:>11.4f} | "
              f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}] | "
              f"{result['median_gap_ratio']:>12.4f}")

    print(
        "\nRead this as the floor for E4: a trained checkpoint has copied only "
        "\nwhen its F_mem rises above the upper end of the interval for the "
        "\nsame training-set size."
    )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"config": args.config, "args": vars(args),
                       "results": results}, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
