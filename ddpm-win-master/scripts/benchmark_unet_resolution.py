"""Benchmark model/unet.py across resolutions for the v4 pipeline.

The v4 map (B2) fixes the calculation unit at the pixel and puts the
experiments at 1024 px per zoom-15 tile, with 512 px as the calibration tier
and 128/256 px for debugging.  The companion page's cost table extrapolates
the pilot's measured 10.98 it/s at 128 px by pixel count alone.  This script
measures the real numbers instead.

Two things the companion page asks for are built into the architecture presets:

  * "the U-Net needs more levels so that its attention block sees a small
    bottleneck".  Every preset here adds levels until the MiddleBlock runs at
    16x16, so the attention cost is identical across resolutions and the
    numbers isolate the cost of the extra pixels.

  * six input channels rather than the pilot's three, since B8 stores cells as
    multi-channel tensors.

Note on ``channel_mults`` in this U-Net: the width is multiplied *cumulatively*
(``nxt = cur * channel_mults[i]``), not ``model_channels * channel_mults[i]``
as in the reference DDPM implementations.  So the pilot's [1, 2, 2, 2] means
widths 1x, 2x, 4x, 8x, and a trailing ``1`` adds a level at constant width.
The presets below use that convention, giving the usual 1-2-4-8-8-8-8 ladder.

Usage:
    python scripts/benchmark_unet_resolution.py
    python scripts/benchmark_unet_resolution.py --img_sizes 1024 --batch_sizes 2 4
    python scripts/benchmark_unet_resolution.py --amp bf16
"""

import argparse
import gc
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch import nn

from model.ddpm import DDPM
from model.unet import DDPMUnet

# Target spatial size of the MiddleBlock, in pixels.  Small enough that the
# attention block sees the whole cell at once.
BOTTLENECK_PX = 16


def channel_mults_for(img_size: int, bottleneck_px: int = BOTTLENECK_PX) -> list[int]:
    """Levels needed to bring *img_size* down to *bottleneck_px*.

    Returns the cumulative-multiplier list this U-Net expects: widths ramp
    1-2-4-8 over the first four levels, then stay flat.

    >>> channel_mults_for(128)
    [1, 2, 2, 2]
    >>> channel_mults_for(1024)
    [1, 2, 2, 2, 1, 1, 1]
    """
    n_levels = 1
    size = img_size
    while size > bottleneck_px:
        size //= 2
        n_levels += 1
    ramp = [1, 2, 2, 2][:n_levels]
    return ramp + [1] * (n_levels - len(ramp))


def build_model(
    img_size: int,
    img_channels: int,
    model_channels: int,
    attn_at_bottleneck: bool,
) -> DDPMUnet:
    mults = channel_mults_for(img_size)
    is_attn = [False] * len(mults)
    if attn_at_bottleneck:
        # The coarsest level, on top of the MiddleBlock's own attention.
        is_attn[-1] = True
    return DDPMUnet(
        img_channels=img_channels,
        model_channels=model_channels,
        channel_mults=mults,
        is_attn=is_attn,
    )


def time_training_steps(
    ddpm: DDPM,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch_size: int,
    img_channels: int,
    img_size: int,
    device: torch.device,
    amp: str,
    n_warmup: int,
    n_measure: int,
) -> float:
    """Seconds per training step, averaged over *n_measure* steps.

    One step is what ``train.py`` does: sample t, q_sample, predict eps, MSE,
    backward, optimizer step.
    """
    amp_dtype = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}[amp]
    x_0 = torch.randn(batch_size, img_channels, img_size, img_size, device=device)

    def one_step() -> None:
        t = torch.randint(0, ddpm.timesteps, (batch_size,), device=device)
        eps = torch.randn_like(x_0)
        x_t = ddpm.q_sample(x_0, t, eps)
        if amp_dtype is None:
            eps_theta = ddpm.eps_model(x_t, t)
            loss = loss_fn(eps_theta, eps)
        else:
            with torch.autocast("cuda", dtype=amp_dtype):
                eps_theta = ddpm.eps_model(x_t, t)
                loss = loss_fn(eps_theta, eps.to(eps_theta.dtype))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for _ in range(n_warmup):
        one_step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_measure):
        one_step()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_measure


def benchmark_one(
    img_size: int,
    batch_size: int,
    img_channels: int,
    model_channels: int,
    timesteps: int,
    device: torch.device,
    amp: str,
    attn_at_bottleneck: bool,
    n_warmup: int,
    n_measure: int,
) -> dict:
    """One (img_size, batch_size) cell of the table.  Never raises on OOM."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = {
        "img_size": img_size,
        "batch_size": batch_size,
        "img_channels": img_channels,
        "model_channels": model_channels,
        "channel_mults": channel_mults_for(img_size),
        "bottleneck_px": BOTTLENECK_PX,
        "attn_at_bottleneck": attn_at_bottleneck,
        "amp": amp,
    }
    unet = ddpm = optimizer = None
    try:
        unet = build_model(img_size, img_channels, model_channels, attn_at_bottleneck)
        result["params_m"] = sum(p.numel() for p in unet.parameters()) / 1e6
        ddpm = DDPM(timesteps, unet).to(device)
        optimizer = torch.optim.AdamW(ddpm.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()

        sec = time_training_steps(
            ddpm, optimizer, loss_fn, batch_size, img_channels, img_size,
            device, amp, n_warmup, n_measure,
        )
        result["it_per_s"] = 1.0 / sec
        result["samples_per_s"] = batch_size / sec
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        result["reserved_vram_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
    except RuntimeError as exc:  # e.g. cuDNN workspace failures at 1024 px
        result["status"] = f"error: {type(exc).__name__}"
        result["error"] = str(exc)[:200]
    finally:
        unet = ddpm = optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img_sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None,
                        help="Default: a per-resolution sweep that keeps the "
                             "sample count roughly constant.")
    parser.add_argument("--img_channels", type=int, default=6,
                        help="B8: cells are multi-channel tensors; 6 is the "
                             "working assumption until A1 fixes the row count.")
    parser.add_argument("--model_channels", type=int, default=128,
                        help="Matches the pilot's ch=128 so numbers compare.")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off",
                        help="off matches train.py today.")
    parser.add_argument("--attn_at_bottleneck", action="store_true",
                        help="Add an attention block at the coarsest level on "
                             "top of the MiddleBlock's own.")
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_measure", type=int, default=10)
    parser.add_argument("--out", type=str, default=None, help="Write results as JSON.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device; this benchmark is about VRAM and it/s.")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)

    default_batches = {128: [8, 16, 32], 256: [4, 8, 16], 512: [2, 4, 8], 1024: [1, 2, 4]}

    print("=" * 92)
    print(f"U-Net resolution benchmark - {props.name}, {props.total_memory / 1024**3:.0f} GiB")
    print(f"torch {torch.__version__}, ch={args.model_channels}, "
          f"C={args.img_channels}, amp={args.amp}, bottleneck={BOTTLENECK_PX}px, "
          f"attn_at_bottleneck={args.attn_at_bottleneck}")
    print("=" * 92)
    header = (f"{'px':>5} | {'levels':>6} | {'params':>8} | {'bs':>4} | {'it/s':>8} | "
              f"{'smp/s':>7} | {'peak GB':>8} | {'500K':>9} | status")
    print(header)
    print("-" * len(header))

    results = []
    for img_size in args.img_sizes:
        batches = args.batch_sizes or default_batches.get(img_size, [4])
        for bs in batches:
            r = benchmark_one(
                img_size, bs, args.img_channels, args.model_channels,
                args.timesteps, device, args.amp, args.attn_at_bottleneck,
                args.n_warmup, args.n_measure,
            )
            results.append(r)
            n_levels = len(r["channel_mults"])
            if r["status"] == "ok":
                days_500k = 500_000 / r["it_per_s"] / 86400
                print(f"{img_size:>5} | {n_levels:>6} | {r['params_m']:>7.1f}M | {bs:>4} | "
                      f"{r['it_per_s']:>8.3f} | {r['samples_per_s']:>7.2f} | "
                      f"{r['peak_vram_gb']:>8.1f} | {days_500k:>8.1f}d | ok")
            else:
                params = f"{r['params_m']:>7.1f}M" if "params_m" in r else " " * 8
                print(f"{img_size:>5} | {n_levels:>6} | {params} | {bs:>4} | "
                      f"{'-':>8} | {'-':>7} | {'-':>8} | {'-':>9} | {r['status']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"device": props.name, "torch": torch.__version__,
                       "args": vars(args), "results": results}, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
