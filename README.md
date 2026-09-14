# DDPM Urban Vitality Research

Diffusion models trained on urban geospatial data function as structure-preserving compressors that internalize the low-dimensional manifold of urban life. This repository investigates whether a "generalization window" exists during DDPM training, a period where structural concepts of urban vitality are learned before memorization sets in. By tracking memorization dynamics across logarithmic checkpoints with metrics derived from Bonnaire et al. (2025) and Wang et al. (2024), the project aims to isolate the irreducible structure of urban vitality from idiosyncratic noise in the data. The working dataset covers Ho Chi Minh City at 1km resolution (4,182 non-empty cells encoded as 128x128x3 RGB images), and the core tool is a discrete-time DDPM with a 157.8M parameter U-Net (v3 architecture).

## Repository Structure

```
ddpm-generalization-theory/
├── ddpm-win-master/            # Our DDPM foundation with config system, dual GPU, modular architecture
│   ├── model/                  #   DDPM core, U-Net, diffusion config
│   ├── datasets/               #   MNIST loader, HCMC geospatial PNG loader
│   ├── evaluation/             #   Memorization (k-NN gap), FID
│   ├── sampling/               #   DDIM sampling
│   ├── analysis/               #   Channel statistics, synergy analysis
│   ├── configs/                #   YAML configs (default, hcmc_vitality_v2, pilot)
│   ├── scripts/                #   Pilot GATE orchestration, batch F_mem
│   └── train.py                #   Main training script
├── Why-Diffusion-Models-Don-t-Memorize-main/   # Bonnaire et al. reference implementation
├── Diffusion-Model-Generalizability-main/      # Wang et al. reference + NVIDIA EDM
├── data-processing/            # HCMC geospatial data pipeline (boundary, grid, encoding)
├── discussion/                 # Research proposals, experiment frameworks
├── references/                 # Academic reference materials
├── TODO.md                     # Authoritative task list
└── archive/                    # Superseded planning docs and dated progress reports
```

## Research Roadmap

### Stage 1: HCMC RGB Encoding and Generalization Window Detection

Prove the core hypothesis on real urban data. Train a DDPM on HCMC geospatial cell images, track memorization via k-NN gap ratio and SSCD reproducibility, and identify the checkpoint where structural learning plateaus before memorization begins.

| Task | Status |
|------|--------|
| HCMC boundary + 1km fishnet grid (8,436 cells) | Done |
| Data pipeline v2 (4,182 cells, 128x128x3, global p95 normalization) | Done |
| Pre-encoding analysis (intrinsic dim ~40, k=6 typologies, MI=0.485) | Done |
| DDPM config + HCMC dataset loader (3263/407/409 split) | Done |
| Pilot GATE validation (3 resolutions, F_mem=0.0000) | Done |
| F_mem batch computation pipeline | Done |
| DDIM sampling | Done |
| Full 500K-step training (v3, 157.8M params, pro6k) | Done |
| Channel statistics on 19 checkpoints | Done |
| Synergy analysis on 19 checkpoints | Done |
| F_mem evaluation (50/100/1000 samples) | Done |

### Stage 2: Vector Embedding and Vitality Mapping

Replace the lossy RGB encoding with structured vector representations. Embed buildings, roads, and POIs into a unified 512-dimensional vector, train diffusion on those embeddings, and decode back through structured decoders. This enables direct mapping from physical environment to human-centric vitality indicators, and supports intervention experiments to detect non-linear urban vitality thresholds.

### Stage 3+: Validation and Extension

Validate findings across other Vietnamese cities. Test whether the generalization window is consistent across different urban morphologies. Investigate transfer learning from HCMC to other cities as a measure of how universal the learned vitality structure is.

## Quick Start

### Environment

```bash
cd ddpm-win-master
uv venv
.venv\Scripts\activate.bat
uv pip sync uv.lock
```

Or with plain venv:

```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Data Pipeline

The data processing scripts live in `data-processing/scripts/`. They download OSM data (buildings, POIs, street networks) for HCMC, compute three channels, and export 128x128x3 PNGs per grid cell.

```bash
cd data-processing/scripts
python build_boundary_grid.py      # HCMC boundary + 1km fishnet grid
python build_hcmc_pipeline_v2.py   # Full pipeline: channels, normalize, export PNGs
python pre_encoding_analysis.py    # Intrinsic dimension + k-means clustering
```

### Training

```bash
cd ddpm-win-master

# HCMC vitality (dual GPU, 50K steps, logarithmic checkpointing)
python train.py --config configs/hcmc_vitality_v2.yaml

# MNIST baseline
python train.py --config configs/default.yaml
```

### Evaluation

```bash
cd ddpm-win-master

# Memorization fraction across all checkpoints
python evaluation/compute_fmem_batch.py --config configs/hcmc_vitality_v2.yaml

# DDIM sampling from a checkpoint
python sampling/ddim.py --checkpoint model_ckpts/hcmc_vitality_v2/step_10000.pth --n_samples 64
```

## Channel Encoding

Each 1km grid cell is encoded as a 128x128x3 RGB image. Channels are normalized independently using global 95th percentile values.

| Channel | Source | Encoding |
|---------|--------|----------|
| B | OSM building footprints | Binary raster (built/unbuilt) |
| G | OSM POIs | Shannon entropy of POI categories |
| R | OSM street network | Exponential accessibility decay |

## Metrics Strategy

| Metric | Role | Status |
|--------|------|--------|
| k-NN gap ratio (F_mem) | Primary memorization tracker | Implemented |
| SSCD reproducibility | Primary reproducibility tracker | Planned |
| FID | Removed | InceptionV3 meaningless for urban rasters |

## Hardware

- Windows 10, dual RTX 3090 (24GB each, NVLink)
- Python 3.10, PyTorch 2.7+cu118
- Pilot validated on RTX 4060 Laptop (8.6GB VRAM) at batch_size=4
- Memory test: 4.3GB peak at 128x128x3, batch_size=16 on RTX 3090

## Key References

1. Bonnaire, L. et al. (2025). "Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training."
2. Wang, T. et al. (2024). "Diffusion Models Learn Low-Dimensional Distributions via Subspace Clustering."
3. Karras, T. et al. (2022). "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS.
4. Zhang, L. et al. (2024). "The Emergence of Reproducibility and Consistency in Diffusion Models." ICML.
