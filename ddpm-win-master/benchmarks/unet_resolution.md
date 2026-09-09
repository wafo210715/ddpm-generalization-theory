# U-Net throughput and VRAM by resolution

Measured on the Linux box (RTX PRO 6000 Blackwell Max-Q, 95 GiB, torch
2.14.0+cu130) with `scripts/benchmark_unet_resolution.py`. One "iteration" is a
full `train.py` step: sample t, `q_sample`, predict eps, MSE, backward, AdamW
step. Every configuration uses **6 input channels** (B8 stores cells as
multi-channel tensors) and enough levels that the MiddleBlock's attention runs
on a **16x16 bottleneck** at every resolution, so the numbers isolate the cost
of the extra pixels rather than mixing in a changing attention cost.

Levels per resolution, in this U-Net's cumulative-multiplier convention:

| px | levels | `channel_mults` | widths at ch=128 |
|---:|---:|---|---|
| 128 | 4 | `[1, 2, 2, 2]` | 128, 256, 512, 1024 |
| 256 | 5 | `[1, 2, 2, 2, 1]` | + 1024 |
| 512 | 6 | `[1, 2, 2, 2, 1, 1]` | + 1024 |
| 1024 | 7 | `[1, 2, 2, 2, 1, 1, 1]` | + 1024 |

`model_channels=128` matches the pilot, so the 128 px row is directly
comparable to the 10.98 it/s recorded in `configs/hcmc_vitality_v3_pro6k.yaml`.

## fp32, ch=128 (what `train.py` does today)

| px | params | bs | it/s | samples/s | peak VRAM | 500K steps |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 157.8M | 8 | 11.855 | 94.8 | 6.5 GB | 0.5 d |
| 128 | 157.8M | 16 | 6.123 | 98.0 | 10.5 GB | 0.9 d |
| 128 | 157.8M | 32 | 3.077 | 98.5 | 19.0 GB | 1.9 d |
| 256 | 284.8M | 4 | 5.629 | 22.5 | 12.3 GB | 1.0 d |
| 256 | 284.8M | 8 | 2.938 | 23.5 | 20.3 GB | 2.0 d |
| 256 | 284.8M | 16 | 1.534 | 24.6 | 37.3 GB | 3.8 d |
| 512 | 411.7M | 2 | 2.786 | 5.6 | 22.1 GB | 2.1 d |
| 512 | 411.7M | 4 | 1.473 | 5.9 | 38.6 GB | 3.9 d |
| 512 | 411.7M | 8 | 0.754 | 6.0 | 72.5 GB | 7.7 d |
| 1024 | 538.6M | 1 | 1.373 | 1.37 | 40.0 GB | 4.2 d |
| 1024 | 538.6M | 2 | 0.690 | 1.38 | 73.9 GB | 8.4 d |
| 1024 | 538.6M | 4 | — | — | OOM | — |

## fp32, ch=64

| px | params | bs | it/s | samples/s | peak VRAM | 500K steps |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 71.2M | 4 | 15.220 | 60.9 | 5.1 GB | 0.4 d |
| 256 | 71.2M | 8 | 8.182 | 65.5 | 9.1 GB | 0.7 d |
| 256 | 71.2M | 16 | 4.089 | 65.4 | 17.3 GB | 1.4 d |
| 512 | 103.0M | 4 | 3.874 | 15.5 | 17.6 GB | 1.5 d |
| 512 | 103.0M | 8 | 1.966 | 15.7 | 34.0 GB | 2.9 d |
| 512 | 103.0M | 16 | 0.987 | 15.8 | 66.8 GB | 5.9 d |
| 1024 | 134.7M | 4 | 0.931 | 3.72 | 67.2 GB | 6.2 d |
| 1024 | 134.7M | 8 | — | — | OOM | — |

## bf16 autocast, ch=128

| px | bs | it/s | samples/s | peak VRAM | 500K steps |
|---:|---:|---:|---:|---:|---:|
| 512 | 2 | 3.996 | 8.0 | 21.6 GB | 1.4 d |
| 512 | 4 | 2.135 | 8.5 | 36.3 GB | 2.7 d |
| 512 | 8 | 1.107 | 8.9 | 65.5 GB | 5.2 d |
| 1024 | 2 | 1.025 | 2.05 | 67.6 GB | 5.6 d |
| 1024 | 4 | — | — | OOM | — |

## What the numbers say

**The companion page's ×4-per-doubling estimate is right.** Throughput in
samples/s at ch=128 is 94.8 at 128 px, 22.5 at 256, 5.6 at 512, 1.37 at 1024 —
each step down is a factor of 4.1 to 4.2. The page's extrapolated ≈0.17 it/s at
1024 px assumed the pilot's bs=8, which does not fit; at the batch sizes that do
fit, 1024 px runs at 0.69 to 1.37 it/s. A 500K-step run at 1024 px is 4 to 8
days on this card, not the 34 days the naive table suggested, because the fit
forces a smaller batch and each step is correspondingly cheaper.

**Throughput is compute-bound above bs=2.** samples/s is flat across batch size
within a resolution (98.5, 98.0, 94.8 at 128 px), so batch size trades gradient
noise against wall clock, not against efficiency. The pilot's reasoning —
smallest batch gives the most updates per hour — still holds.

**VRAM, not compute, is what 1024 px costs.** At ch=128 the largest batch that
fits is 2 (73.9 GB of 95). Two levers each buy about a factor of 2.7 in
samples/s: dropping to ch=64 (3.72 samples/s at bs=4, 67 GB) or accepting bs=1.
bf16 autocast is the weakest of the three, worth about 1.5x with the memory
saving mostly eaten by the fp32 master weights.

**Model size is dominated by the ramp, not the depth.** Levels added past the
fourth are flat-width, so 1024 px costs 538.6M parameters against 157.8M at
128 px — 3.4x the parameters for 64x the pixels.

## Waiting on the user

Batch size, `model_channels`, and whether to enable autocast are training-run
decisions tied to the mutant sizing in C7, and are left open here.
