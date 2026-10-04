# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace rules (set 30 Sep 2026; these come first)

1. Answer first, briefly, in plain words. Pipeline codes (A1, B8, E3…) only in brackets.
2. Ask before creating any file in `discussion/` or `references/`. No handoffs, cards, maps, templates or plans unless asked.
3. To explain something, draw it inline in the chat, not as a new file.
4. Annotated bibliography: one file, `references/annotated_bibliography.md`. Each paper is a full citation plus one paragraph of at most 300 words (changed from 200 on 2 Oct 2026; for Chinese, about 600 characters).
5. Measure every number on the data on this machine; name the script and the tile subset. When a number contradicts a document, fix the document in place and quote the old text.
6. Don't commit or push without asking.
7. `discussion/` holds only the decisions in force, the data viewers, the proposal, the literature extraction and the diffusion reading note. Everything else goes to `archive/`.

How we work, and an index of the working files (`vitality_stance.md`, `reading_method.md`, the data ledger), is in `AGENTS.md` at the root (from 4 Oct 2026).

## Code Style Guidelines

### Parameter Naming Consistency

**CRITICAL**: When adding features from research repositories, MAINTAIN the original parameter names from those repos. This ensures code consistency and makes cross-referencing easier.

**Research Repositories and Their Naming Conventions**:

1. **DDPM (Ho et al., 2020)** - `Why-Diffusion-Models-Don-t-Memorize-main/`
   - Linear schedule: `b0`, `bT` (starting/ending beta)

2. **Improved DDPM (Nichol & Dhariwal, 2021)** - referenced in both repos
   - Cosine schedule: `s` (offset parameter)

3. **Score-Based SDEs (Song et al., 2021)** - `Diffusion-Model-Generalizability-main/edm/`
   - VP schedule: `beta_d`, `beta_min` (NOT `vp_beta_d`, `vp_beta_min`)
   - VE schedule: `sigma_min`, `sigma_max` (NOT `ve_sigma_min`, `ve_sigma_max`)

4. **EDM (Karras et al., 2022)** - `Diffusion-Model-Generalizability-main/edm/`
   - EDM loss: `P_mean`, `P_std`, `sigma_data`

**Examples**:
```python
# CORRECT - matches research repo
def vp_schedule(self, beta_d: float = 19.9, beta_min: float = 0.1):
    ...

# WRONG - adds unnecessary prefix
def vp_schedule(self, vp_beta_d: float = 19.9, vp_beta_min: float = 0.1):
    ...
```

**Rationale**: We are building upon their work. Keeping naming consistent:
- Makes it easier to cross-reference with research papers
- Simplifies code migration between repos
- Reduces cognitive load when switching between implementations

## Repository Overview

This repository contains research code for understanding diffusion model generalization and memorization, combining three main research efforts:

1. **Why Diffusion Models Don't Memorize** (`Why-Diffusion-Models-Don-t-Memorize-main/`) - Theoretical analysis and numerical experiments on implicit dynamical regularization in diffusion training
2. **Diffusion Model Generalizability** (`Diffusion-Model-Generalizability-main/`) - Research on low-dimensional distribution learning via subspace clustering
3. **EDM (Elucidating Design Space of Diffusion Models)** (`Diffusion-Model-Generalizability-main/edm/`) - NVIDIA's official implementation of diffusion model training/sampling
4. **ddpm-win-master** (`ddpm-win-master/`) - Integrated DDPM foundation with config system, dual GPU support, and modular architecture for urban vitality research

## Environment Setup

### Memorization Experiments
```bash
cd Why-Diffusion-Models-Don-t-Memorize-main/Experiments
conda env create -f environment.yml  # GPU (CUDA 11.8)
conda env create -f environment_cpu.yml  # CPU-only
conda activate memorization
```

Or using pip:
```bash
pip install -r Experiments/requirements.txt
```

### Theory Experiments
```bash
cd Why-Diffusion-Models-Don-t-Memorize-main/Theory
pip install -r requirements.txt
```

### Generalizability Experiments
```bash
cd Diffusion-Model-Generalizability-main
conda env create -f edm/environment.yml -n generalizability
conda activate generalizability
```

### ddpm-win-master (Integrated Foundation)
```bash
cd ddpm-win-master
# Using uv (recommended)
uv venv
.venv\Scripts\activate.bat  # Windows
uv pip sync uv.lock

# Or using venv
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Platform**: Windows 10 with dual RTX 3090 (24GB each, NVLink connected)

## Common Commands

### Training Models

**GMM (Gaussian Mixture Model) Training:**
```bash
cd Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Training
python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1
```
Parameters: `-n` (num samples), `-d` (dimension), `-s` (seed), `-de` (network width), `-O` (Adam/SGD_Momentum), `-B` (batch size), `-t` (timestep, -1 for normal mode)

**CelebA U-Net Training:**
```bash
cd Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Training
python run_Unet.py -n 1024 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -t -1
```
Parameters: `-n` (num images), `-i` (dataset index), `-s` (image size), `-LR` (learning rate), `-W` (base filters), `-t` (timestep)

**EDM Training:**
```bash
cd Diffusion-Model-Generalizability-main/edm
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
```

**ddpm-win-master Training:**
```bash
cd ddpm-win-master
# Single GPU (default config)
python train.py

# Custom config
python train.py --config configs/default.yaml

# Geo-spatial config
python train.py --config configs/geospatial.yaml
```

**GPU Monitoring (ddpm-win-master):**
```bash
# Run in separate terminal while training
python monitor_gpu.py

# Or use nvidia-smi directly
nvidia-smi -l 1  # Refresh every 1 second
```

### Generation and Evaluation

**Generate samples (DDIM):**
```bash
cd Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Generation
python generate.py -D CelebA -n 1024 -i 0 -s 32 -B 512 -LR 0.0001 -O Adam -W 32 -Ns 100
```

**Compute Memorization Fraction:**
```bash
cd Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Evaluation
python compute_fmem.py -D CelebA -n 1024 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B 512 --gap_threshold 0.333
```

**Compute FID:**
```bash
cd Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Evaluation
python compute_FID.py -D CelebA -n 1024 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B 512 -istat 1
```

**EDM Generation:**
```bash
cd Diffusion-Model-Generalizability-main/edm
python generate.py --outdir=out --seeds=0-63 --batch=64 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
```

### Theory Experiments

**Plot spectral density:**
```bash
cd Why-Diffusion-Models-Don-t-Memorize-main/Theory/src
python plot_spectrum_U.py
```

**Train random features network:**
```bash
cd Why-Diffusion-Models-Don-t-Memorize-main/Theory/src
python Training_random_features.py
```

## Architecture

### Key Components

**Diffusion Framework** (`Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Utils/Diffusion.py`):
- `TrainingConfig`: Configuration for training parameters (dataset, batch size, learning rate, timesteps)
- `DiffusionConfig`: Diffusion process configuration (beta schedules, alpha values)
- `forward_diffusion()`: Adds noise to images at timestep t
- `sample_diffusion_from_noise()`: Standard DDPM sampling
- `sample_diffusion_from_noise_DDIM()`: DDIM sampling for faster generation
- `train()`: Main training loop with model checkpointing

**Model Architectures**:
- `Unet.py`: U-Net implementation for image diffusion
- `TinyModels.py`: Simple time model for GMM experiments

**Data Handling**:
- `loader.py`: Dataset loading utilities
- `GMM_data_scores.py`: Gaussian mixture model data generation

**Evaluation**:
- `compute_fmem.py`: Memorization fraction via k-NN gap analysis
- `compute_FID.py`: Fréchet Inception Distance computation

### ddpm-win-master Framework

**Config System** (`config_loader.py`, `model/diffusion_config.py`):
- `TrainingConfig`: YAML-based training configuration (step-based, not epoch-based)
- `DiffusionConfig`: Beta schedule configuration (linear, cosine, fixed)
- Support for logarithmic checkpointing for "generalization window" analysis

**Model** (`model/ddpm.py`, `model/unet.py`):
- `DDPM`: Core DDPM implementation with q_sample and p_sample
- `DDPMUnet`: U-Net with time embeddings for image diffusion

**Data** (`datasets/dataset.py`):
- MNIST loader (60,000 training images, 28x28 grayscale)
- Extensible for geo-spatial urban vitality data

**Training** (`train.py`):
- Config-based training with argparse
- Dual GPU support via DataParallel (`use_dual_gpu: true` in config)
- Step-based training with logarithmic checkpointing

**Configs** (`configs/`):
- `default.yaml`: MNIST configuration
- `geospatial.yaml`: Geo-spatial urban vitality configuration (512x512 RGB)

**Tests** (`tests/`):
- `test_all.py`: Run all tests
- `test_config_yaml.py`: Config system validation
- `test_train_config.py`: Training tests (single/dual GPU)

### EDM Framework

Built on NVIDIA's EDM codebase with key modifications:
- `train.py`: Main training script with click CLI
- `training/training_loop.py`: Core training logic
- `training/networks.py`: Network architectures (DDPM++, NCSN++, ADM)
- `trainMoLRG.py`: Modified training for Mixture of Low-Rank Gaussians
- `jacobian.py`: Jacobian analysis for low-rank property evaluation

## File Structure Patterns

**Research repos**:
- Scripts use relative imports - always run from the subdirectory (e.g., `Experiments/src/Training/`)
- Model checkpoints saved to pattern: `Saves/{DATASET}{config}/Models/Model_{step}`
- Generated samples go to: `Saves/{config}/Samples/`
- Configuration classes (`TrainingConfig`, `DiffusionConfig`) centralize hyperparameters

**ddpm-win-master**:
- Run scripts from `ddpm-win-master/` root
- Configs in `configs/` directory (YAML format)
- Model checkpoints: `model_ckpts/{run_name}.pth` or `model_ckpts/{run_name}_step{step}.pth`
- Logs: `logs/` directory
- Data: `data/` directory (currently contains MNIST)

## Training Modes

**Research repos**:
Two training modes available via `-t` parameter:
- `-t -1`: Normal mode (random timesteps each batch)
- `-t {0-999}`: Fixed-time training (train only at specified timestep)

**ddpm-win-master**:
- Step-based training (configured via `n_steps` in YAML)
- Default: 50,000 steps (~3.5 hours on dual RTX 3090s for MNIST)
- Logarithmic checkpointing at: 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000...
- Dual GPU: Set `use_dual_gpu: true` in config YAML

## Key Papers

1. "Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training" (Bonnaire et al., 2025)
2. "The Emergence of Reproducibility and Consistency in Diffusion Models" (Zhang et al., ICML 2024)
3. "Diffusion Models Learn Low-Dimensional Distributions via Subspace Clustering" (Wang et al., 2024)
4. "Elucidating the Design Space of Diffusion-Based Generative Models" (Karras et al., NeurIPS 2022)
