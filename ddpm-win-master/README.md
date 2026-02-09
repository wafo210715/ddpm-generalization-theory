# DDPM Foundation for Urban Vitality Research

A discrete-time DDPM implementation with configurable noise schedules, dual GPU support, and logarithmic checkpointing for analyzing the "generalization window" in diffusion model training.

## Overview

This project implements a foundation DDPM framework with multiple noise schedule types (linear, VP, VE) to study how diffusion models learn structural concepts of urban vitality before memorizing noise. The implementation emphasizes:

- **Discrete timesteps** for precise tracking of training progress
- **Logarithmic checkpointing** to identify the "generalization window"
- **Multiple noise schedules** for comparing memorization dynamics
- **Dual GPU support** via PyTorch DataParallel

## Features

### Configuration System
- YAML-based configuration for easy experiment setup
- Multiple schedule types: linear, fixed, VP (variance preserving), VE (variance exploding)
- Step-based training with configurable checkpoint schedules

### Noise Schedules

| Schedule | Parameters | Source | Use Case |
|----------|-----------|--------|----------|
| Linear | `b0`, `bT` | DDPM (Ho et al., 2020) | Baseline (used in Why-Diffusion-Models-Don-t-Memorize) |
| VP | `beta_d`, `beta_min` | Song et al. SDE | Theoretical analysis |
| VE | `sigma_min`, `sigma_max` | Song et al. SDE | Theoretical analysis |
| Fixed | `beta` | - | Constant noise rate |

**Note**: Cosine schedule (Nichol & Dhariwal, 2021) was previously implemented but has been **removed from the codebase**. Neither of our primary research repos (Why-Diffusion-Models-Don-t-Memorize, Diffusion-Model-Generalizability) use cosine schedule, and it exhibited numerical instability issues. Our comparison focuses on Linear, VP, and VE schedules.

### Dual GPU Training

**Checkpoint Format Notice**:
- **Single GPU**: Saves full DDPM state dict (`beta`, `alpha`, `eps_model`)
- **Dual GPU**: Saves U-Net weights only (`eps_model.module`)

This is a PyTorch DataParallel implementation detail. The sampling script automatically detects and handles both formats.

**Performance**:
- Dual GPU: ~2x faster per training session
- Parallel single-GPU: Faster overall when comparing multiple schedules

## Quick Start

### Environment Setup
```bash
cd ddpm-win-master
uv venv
.venv\Scripts\activate.bat  # Windows
uv pip sync uv.lock
```

### Training

#### Single schedule
```bash
python train.py --config configs/default.yaml
```

#### Parallel schedule comparison (uses both GPUs simultaneously)
```bash
python train_parallel_schedules.py --schedule1 linear --schedule2 vp --steps 10000
```

### Sampling
```bash
python sample.py model_ckpts/mnist_32x32_steps50000_step10000.pth
```

## Configuration

Edit `configs/default.yaml` to change schedule type:

```yaml
# Diffusion schedule configuration
schedule_type: "linear"  # Options: linear, fixed, vp, ve

# Schedule parameters (all defined, only used by active schedule)
b0: 0.0001      # Linear: starting beta
bT: 0.02        # Linear: ending beta
beta: 0.006     # Fixed: constant beta
beta_d: 19.9    # VP: noise level extent
beta_min: 0.1   # VP: initial slope
sigma_min: 0.02 # VE: min noise level
sigma_max: 100.0 # VE: max noise level
```

## Project Structure

```
ddpm-win-master/
├── configs/                    # YAML configuration files
│   ├── default.yaml            # MNIST configuration
│   └── geospatial.yaml         # Geo-spatial urban vitality config
├── model/                      # Model implementations
│   ├── ddpm.py                 # Core DDPM with DiffusionConfig support
│   ├── unet.py                 # U-Net with time embeddings
│   └── diffusion_config.py     # Schedule configuration (linear, fixed, VP, VE)
├── datasets/                   # Data loading
│   └── dataset.py              # MNIST loader
├── scripts/                    # Utility scripts
│   └── convert_checkpoint.py   # .pth ↔ .safetensors converter
├── tests/                      # Test suite
│   ├── test_all.py             # Run all tests
│   ├── test_schedules.py       # Schedule validation
│   └── ...
├── train.py                    # Main training script
├── train_parallel_schedules.py # Parallel schedule comparison
├── sample.py                   # Sampling script
├── config_loader.py            # TrainingConfig with YAML loading
└── monitor_gpu.py              # GPU monitoring utility
```

## Hardware

**Tested on**:
- Windows 10
- Python 3.10.11
- PyTorch 2.7.1+cu118
- 2x NVIDIA RTX 3090 (24GB each, NVLink connected)

## Parameter Naming Convention

**Important**: When adding features from research repositories, we maintain the original parameter names from those papers/repos:

| Schedule | Parameter Names | Source |
|----------|---------------|--------|
| Linear | `b0`, `bT` | DDPM standard |
| Fixed | `beta` | - |
| VP | `beta_d`, `beta_min` | Song et al. SDE |
| VE | `sigma_min`, `sigma_max` | Song et al. SDE |

NO prefixes like `vp_beta_d` - use the original names directly.

## Roadmap

### Phase 1: Foundation ✓
- Environment setup and GPU validation
- Basic DDPM implementation with config system
- Dual GPU support
- Test suite

### Phase 2: Schedule Comparison (In Progress)
- ✅ Linear, Fixed, VP, VE schedules implemented
- ✅ Parallel training utility
- ✅ Schedule comparison experiments (Linear, VP, VE)
- ⏳ Memorization tracking (k-NN gap, FID)
- ⏳ Manifold analysis (Jacobian rank)

### Phase 3: Geo-Spatial Data (Planned)
- Urban vitality data loading (ArcGIS TIF → JPG)
- High-resolution support (512x512, 1024x1024)
- Geo-location metadata preservation

## Research Context

This project is part of a broader research effort to understand diffusion model generalization and memorization:

- **Why Diffusion Models Don't Memorize** - Bonnaire et al., 2025
- **Diffusion Models Learn Low-Dimensional Distributions** - Wang et al., 2024
- **EDM: Elucidating Design Space** - Karras et al., 2022

## License

This project incorporates code from the above research repositories. See individual repository licenses for details.
