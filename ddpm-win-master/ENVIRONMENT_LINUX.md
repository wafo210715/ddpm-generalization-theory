# Environment — Linux box, RTX PRO 6000 Blackwell

This is the working machine for the v4 pipeline. The Windows dual-RTX-3090 setup
described in `ENVIRONMENT.md` is the previous machine and is kept for reference.

## Hardware / driver

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| Compute capability | sm_120 |
| VRAM | 95.0 GiB usable (97 887 MiB reported) |
| SMs | 188 |
| Driver | 595.84, CUDA 13.2 |
| Host RAM | 136 GB |
| CPU threads | 32 |

Measured on this box with torch 2.14.0+cu130:

| kernel | throughput |
|---|---|
| 8192³ matmul, fp32 (TF32 paths on) | 49.7 TFLOP/s |
| 8192³ matmul, fp16 | 285.5 TFLOP/s |

## Setup

```bash
cd ddpm-win-master
uv venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -r requirements-linux.txt
.venv/bin/python -m pytest tests/ -q
```

`requirements-linux.txt` is the high-level spec; `uv-linux.lock` is the exact
resolution that produced the benchmark numbers (`uv pip freeze` output, install
it with `VIRTUAL_ENV=.venv uv pip install -r uv-linux.lock`).

## A note on the CUDA build

The plan called for CUDA 12.8. The default PyPI wheel for torch 2.14 is built
against CUDA 13.0, and the 595.84 driver (CUDA 13.2) on this box serves it, so
that is what resolved and what every benchmark here used. A cu128 wheel also
supports sm_120 if a 12.8 runtime is ever needed:

```bash
VIRTUAL_ENV=.venv uv pip install --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

## Verify the GPU

```bash
.venv/bin/python tests/check_gpu.py
```

## Differences from the Windows environment

- Single GPU, so `use_dual_gpu` in the configs stays `false`; 95 GiB on one card
  is more than the two 24 GB cards together and needs no DataParallel.
- Paths are POSIX; activate with `source .venv/bin/activate`, or call
  `.venv/bin/python` directly as the commands in this repo do.
- The geospatial stack (geopandas, rasterio, osmnx, pyproj) installs from wheels
  without a conda environment, which it did not on Windows.
