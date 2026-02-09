# Plan: Incorporating Diffusion Research Features into DDPM Foundation
## For Urban Vitality Research via Manifold Learning

## Executive Summary

**Objective**: Integrate advanced diffusion model research features into basic ddpm-win-master foundation to study urban vitality through manifold learning.

**Key Research Goal**: Identify the "generalization window" - the training phase where structural concepts of urban vitality are learned before memorization occurs. Position diffusion models as "epistemic probes" that internalize the low-dimensional manifold of urban life.

**Integration Approach**: Incremental, testing each feature as it's integrated

**Critical Insight**: Find the "minimal dataset" that captures intrinsic dimensions of urban vitality, avoiding unnecessary data while preserving generalization.

**Implementation Phases**:
1. Environment setup & foundation verification (1-2 days)
2. Configuration & checkpointing system (2-3 days)
3. Geo-spatial data pipeline (3-5 days)
4. Memorization tracking - **Priority 1** (5-7 days)
5. DDIM sampling & latent space analysis (3-4 days)
6. Low-dimensional manifold analysis - **Priority 2** (7-10 days)
7. Subset training & minimal dataset experiments (5-7 days)
8. Intrinsic dimensionality analysis (4-5 days, parallel)
9. High-resolution scaling (7-10 days)

**Fallback Plan**: SD1.5 fine-tuning if training from scratch proves insufficient

---

## Context

This plan integrates advanced research functionalities from two diffusion model research repositories into the user's basic ddpm-win-master foundation. The goal is to verify theoretical findings about memorization and generalization in diffusion models on **geo-spatial urban vitality datasets**.

**Research Abstract**: Despite the influx of big data, urban vitality research remains constrained by a 'correlationist' paradigm, reducing complex urban synergies to linear, additive feature lists. This proposal advances a generative epistemology, repositioning diffusion models (DDPM with U-Net as an example) from tools of automated design to epistemic probes for manifold learning. We argue that diffusion models function as structure-preserving compressors, capable of internalizing the low-dimensional manifold of urban life. Central to this inquiry is the **generalization window** - a training phase where structural concepts are learned before memorization occurs. By curating minimal, intrinsic-dimension-aligned datasets, this research aims to isolate the robust, irreducible 'essence' of vitality from idiosyncratic noise.

**Foundation**: ddpm-win-master (basic DDPM implementation with U-Net)
**Research Sources**:
- Why-Diffusion-Models-Don-t-Memorize-main (memorization vs generalization dynamics)
- Diffusion-Model-Generalizability-main (low-dimensional subspace learning)

**Dataset**: Geo-spatial data (ArcGIS projections → TIF → JPG → 512x512 or 1024x1024)

---

## Phase 1: Environment Setup & Foundation Preparation

### Step 1.1: Create Virtual Environment
```bash
# Create conda environment for Windows with dual 3090 GPUs
conda create -n ddpm-research python=3.10 -y
conda activate ddpm-research

# Install PyTorch with CUDA support for RTX 3090
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install tqdm pillow matplotlib numpy scipy pytorch-fid natsort
```

### Step 1.2: Fix Foundation Codebase Issues
**Missing dependency**: Create `datasets/dataset.py` module
**Multi-GPU support**: Add DataParallel to train.py
**Checkpoint directory**: Create `model_ckpts/` folder

### Step 1.3: Foundation Code Structure Documentation
```
ddpm-win-master/
├── model/
│   ├── ddpm.py          # Core DDPM algorithm (1000 timesteps, linear schedule)
│   └── unet.py          # U-Net with time embeddings, attention blocks
├── util/
│   └── img_utils.py     # Image display/save utilities
├── train.py             # Basic training loop (single dataset, no checkpointing)
└── sample.py            # Basic sampling (no DDIM support)
```

---

## Phase 2: Functionality Catalog

This section lists ALL available functionalities from research repos for reference.

### From Why-Diffusion-Models-Don-t-Memorize-main:

#### Training Features:
| Feature | File | Description | Priority for Urban Vitality |
|---------|------|-------------|---------------------------|
| GMM Training | `run_GMM.py` | Train on synthetic Gaussian mixture data | Low (validation only) |
| CelebA Training | `run_Unet.py` | Train U-Net on image subsets with checkpointing | **HIGH** (pattern for subset training) |
| Fixed-time Training | `Diffusion.py` | Train at specific timestep (-1 for normal mode) | Medium (analysis) |
| Logarithmic Checkpointing | `cfg.py` | Save models at log-spaced intervals | **HIGH** (find generalization window) |

#### Generation Features:
| Feature | File | Description | Priority |
|---------|------|-------------|----------|
| DDIM Sampling | `Diffusion.py` | Fast deterministic sampling | **HIGH** (latent space analysis) |
| Multi-checkpoint Generation | `generate.py` | Generate from multiple training stages | **HIGH** (track changes) |

#### Evaluation Features:
| Feature | File | Description | Priority |
|---------|------|-------------|----------|
| Memorization Fraction | `compute_fmem.py` | k-NN gap ratio analysis | **CRITICAL** (find generalization window) |
| FID Computation | `compute_FID.py` | Fréchet Inception Distance | **HIGH** (quality tracking) |
| Bootstrap Confidence | Both | Statistical confidence intervals | Medium (validation) |

#### Utility Features:
| Feature | File | Description | Priority |
|---------|------|-------------|----------|
| TrainingConfig | `Diffusion.py` | Centralized configuration class | **HIGH** (code organization) |
| DiffusionConfig | `Diffusion.py` | Schedule parameters | **HIGH** (flexible schedules) |
| Empirical Score | `Diffusion.py` | Ground truth score computation | Medium (theoretical) |
| CelebA Loader | `loader.py` | Dataset subset loading | **HIGH** (pattern for geo-spatial) |

#### Theory Features:
| Feature | File | Description | Priority |
|---------|------|-------------|----------|
| Random Features Training | `Training_random_features.py` | Theoretical model validation | Low (validation) |
| Spectral Density Analysis | `plot_spectrum_U.py` | Eigenvalue density computation | Medium (manifold analysis) |

### From Diffusion-Model-Generalizability-main:

#### Training Features:
| Feature | File | Description | Priority |
|---------|------|-------------|----------|
| EDM Training | `train.py` | Multi-architecture (ddpmpp/ncsnpp/adm) | Low (future enhancement) |
| Multi-preconditioning | `train.py` | VP/VE/EDM preconditioning | Medium (comparison) |
| MoLRG Training | `trainMoLRG.py` | Train on low-rank Gaussian mixtures | Low (validation) |
| Augmentation | `augment.py` | Data augmentation during training | Low (future) |
| EMA | `training_loop.py` | Exponential moving average | Medium (stability) |

#### Generation Features:
| Feature | File | Description | Priority |
|---------|------|-------------|----------|
| EDM Sampler | `generate.py` | Configurable sampling parameters | Medium (flexibility) |
| Optimal Denoiser | `generate.py` | Theoretical baseline comparison | **HIGH** (theoretical validation) |
| Multi-schedule Support | `generate.py` | VP/VE/EDM schedules | Medium (comparison) |

#### Evaluation Features:
| Feature | File | Description | Priority |
|---------|------|-------------|----------|
| SSCD Features | `sscd.py` | Self-supervised descriptors | **HIGH** (reproducibility) |
| Reproducibility Score | `sscd.py` | RP score between sample sets | **HIGH** (consistency) |
| Memorization Score | `sscd.py` | MS score vs training data | **HIGH** (memorization) |
| Jacobian Rank Analysis | `jacobian.py` | Low-rank property evaluation | **CRITICAL** (manifold proof) |
| FID Pipeline | `fid.py` | Complete FID computation | **HIGH** (quality) |

#### Dataset Features:
| Feature | File | Description | Priority |
|---------|------|-------------|----------|
| Dataset Tool | `dataset_tool.py` | ZIP/PNG dataset creation | **HIGH** (geo-spatial prep) |
| MoLRG Dataset | `dataset.py` | Synthetic low-rank data | Low (validation) |
| Teacher-Student Setup | `generate.py` + `train.py` | Train on generated data | Medium (analysis) |

---

## Phase 3: Integration Roadmap

### Stage 1: Core Infrastructure (Foundation Enhancement)
1. **Configuration Management** - Port `TrainingConfig` and `DiffusionConfig` classes
2. **Multi-schedule Support** - Add VP/VE/EDM beta schedules
3. **Checkpoint System** - Logarithmic checkpoint saving
4. **Multi-GPU Training** - Proper DataParallel/DistributedDataParallel

### Stage 2: Advanced Sampling
1. **DDIM Sampling** - Fast deterministic generation
2. **Optimal Denoiser** - Theoretical baseline implementation
3. **Configurable Sampler** - EDM-style parameter controls

### Stage 3: Evaluation Metrics
1. **Memorization Metrics**:
   - k-NN gap ratio analysis (fraction collapsed)
   - SSCD-based memorization score
2. **Quality Metrics**:
   - FID computation pipeline
   - Reproducibility score
3. **Theoretical Metrics**:
   - Jacobian rank analysis
   - Spectral density computation

### Stage 4: Synthetic Data & Experiments
1. **GMM Module** - Gaussian mixture data generation
2. **MoLRG Module** - Low-rank Gaussian mixtures
3. **Teacher-Student Setup** - Synthetic dataset pipeline

### Stage 5: Real Dataset Integration
1. **Custom Dataset Loader** - For user's real-world data
2. **Subset Training** - Train on data subsets
3. **Comparative Analysis** - Memorization vs generalization tracking

---

## Phase 4: Implementation Priority (For Urban Vitality Research)

### Priority 1: Finding the Generalization Window (Memorization Tracking)
**Goal**: Identify when the model switches from learning structural concepts to memorizing noise

**Required Features**:
- Logarithmic checkpointing system (save models at log-spaced intervals)
- Memorization fraction computation (k-NN gap ratio analysis)
- FID computation pipeline (quality tracking)
- Reproducibility score (SSCD features)
- Subset training capability (train on "minimal, intrinsic-dimension-aligned datasets")

**Research Validation**: Plot memorization fraction vs training steps to identify the "generalization window"

### Priority 2: Proving Low-Dimensional Manifold Learning
**Goal**: Demonstrate the diffusion model is learning the "low-dimensional manifold of urban life"

**Required Features**:
- Jacobian rank analysis (prove denoising autoencoder is low-rank)
- Spectral density analysis (analyze learned matrix properties)
- Empirical score computation (compare learned vs ground truth)
- Multi-schedule support (VP/VE/EDM for comparison)

**Research Validation**: Show Jacobian rank remains low throughout training, confirming manifold learning

### Priority 3: Latent Space as "Computational Laboratory"
**Goal**: Use the latent space to uncover the "hidden generative grammar of the city"

**Required Features**:
- DDIM sampling (deterministic trajectories through latent space)
- Optimal denoiser baseline (theoretical comparison)
- Interpolation in latent space
- Time-step specific analysis (what's learned at each diffusion step?)

**Research Validation**: Navigate latent space to understand urban vitality structure

### Priority 4: High-Resolution Geo-Spatial Support
**Goal**: Handle 512x512 or 1024x1024 urban geo-spatial images

**Required Features**:
- High-resolution U-Net (modifications for larger images)
- Geo-spatial data loader (ArcGIS TIF → JPG pipeline)
- Multi-GPU training optimizations (for larger images)
- Memory-efficient sampling

**Implementation**: Incremental - start with 256x256, scale up

---

## Phase 5: Intrinsic Dimensionality Analysis (Critical Addition)

**Goal**: Determine the minimal dataset size that captures the intrinsic dimension of urban vitality

### Step: Intrinsic Dimension Detection
**Duration**: 4-5 days (can be done in parallel with other steps)

**Files to Create**:
- `analysis/intrinsic_dim.py` - Intrinsic dimensionality estimation
- `analysis/subspace.py` - Subspace clustering analysis
- `analysis/minimal_dataset.py` - Tools to find minimal sufficient dataset

**Key Features**:
- **Intrinsic Dimension Estimation**:
  - Use correlation dimension, MLE estimators
  - Track how dimension grows with dataset size
  - Identify saturation point (intrinsic dimension)

- **Subspace Clustering**:
  - Apply subspace clustering to training data
  - Identify low-dimensional structures
  - Determine minimal samples needed per subspace

- **Minimal Dataset Construction**:
  - Curate dataset that captures intrinsic dimensions
  - Subset strategies: geometric coverage, feature diversity, vitality range
  - Validate: model on minimal dataset should generalize to full dataset

**Deliverable**: Recommended minimal dataset size + curated subset

---

## Phase 6: SD1.5 Fine-tuning Fallback (Backup Plan)

**Goal**: Prepare for potential SD1.5 fine-tuning if training from scratch fails

### Considerations:
- **Latent Space Impact**: How does SD1.5's pre-trained latent space affect "essence" of vitality?
- **Approach**: Compare scratch-trained vs fine-tuned models
- **Analysis**: Use same memorization/manifold metrics on both

**Files to Create** (if needed):
- `model/sd15_finetune.py` - SD1.5 fine-tuning script
- `analysis/latent_compare.py` - Compare scratch vs fine-tuned latent spaces

**Deliverable**: Framework for comparing scratch vs fine-tuned approaches

---

## Phase 7: Incremental Integration Steps (Urban Vitality Research)

### Step 1: Foundation Verification (Immediate)
**Duration**: 1-2 days
**Goal**: Ensure basic ddpm-win-master runs on dual 3090 setup

1. Create conda environment with CUDA 12.1 support
2. Fix missing `datasets/dataset.py` module
3. Add DataParallel to train.py
4. Test with small MNIST subset (verify training works)
5. Verify sampling produces images

**Deliverable**: Working foundation codebase that trains and generates

---

### Step 2: Configuration & Checkpointing System
**Duration**: 2-3 days
**Goal**: Enable systematic training with checkpoint tracking

**Files to Create**:
- `configs.py` - Port `TrainingConfig` and `DiffusionConfig` from research repo
- `schedules.py` - Beta schedule variants (linear, cosine, EDM)

**Files to Modify**:
- `train.py` - Add logarithmic checkpoint saving based on `cfg.py` pattern
- `model/ddpm.py` - Support multiple schedules

**Deliverable**: Training that saves checkpoints at log-spaced intervals (e.g., steps: 100, 250, 500, 1000, 2500, 5000, 10000...)

---

### Step 3: Geo-Spatial Data Pipeline
**Duration**: 3-5 days
**Goal**: Load and preprocess ArcGIS TIF images

**Files to Create**:
- `data/geospatial.py` - Custom dataset loader for TIF/JPG geo-spatial images
- `data/preprocessing.py` - TIF → JPG conversion, cropping, normalization

**Key Features**:
- Support for 512x512 and 1024x1024 images
- Geo-location metadata preservation
- Proper normalization for urban imagery

**Deliverable**: Working data pipeline that loads geo-spatial images

---

### Step 4: Memorization Tracking (Priority 1)
**Duration**: 5-7 days
**Goal**: Implement metrics to find the "generalization window"

**Files to Create**:
- `evaluation/memorization.py` - Port `compute_fmem.py` (k-NN gap ratio)
- `evaluation/quality.py` - Port `compute_FID.py` and `sscd.py` features
- `evaluation/config.py` - Evaluation configuration

**Key Features**:
- Memorization fraction computation at each checkpoint
- FID tracking across training
- Reproducibility score (compare samples at different checkpoints)

**Deliverable**: Plot showing memorization vs training steps → identifies "generalization window"

---

### Step 5: DDIM Sampling & Latent Space Analysis (Priority 3)
**Duration**: 3-4 days
**Goal**: Enable deterministic sampling for latent space exploration

**Files to Create**:
- `sampling/ddim.py` - Port DDIM sampling from `Diffusion.py`
- `sampling/optimal.py` - Port optimal denoiser

**Files to Modify**:
- `sample.py` - Add DDIM option
- `model/ddpm.py` - Support deterministic sampling

**Deliverable**: DDIM sampling script with configurable steps and latent space interpolation

---

### Step 6: Low-Dimensional Manifold Analysis (Priority 2)
**Duration**: 7-10 days
**Goal**: Prove model learns low-dimensional structure

**Files to Create**:
- `evaluation/jacobian.py` - Port Jacobian rank analysis
- `evaluation/spectral.py` - Port spectral density analysis
- `theory/random_features.py` - Port random features model for validation

**Key Features**:
- Compute Jacobian rank at various checkpoints
- Spectral analysis of learned parameters
- Compare with theoretical predictions

**Deliverable**: Evidence that diffusion model learns low-dimensional manifold

---

### Step 7: Subset Training & "Minimal Dataset" Experiments
**Duration**: 5-7 days
**Goal**: Train on curated minimal datasets to isolate "essence" of vitality

**Files to Create**:
- `data/subset.py` - Tools for creating curated subsets
- `experiments/subset_training.py` - Training script for subset experiments

**Key Features**:
- Subset by geographic region
- Subset by urban feature type
- Subset by vitality metric (high vs low vitality areas)

**Deliverable**: Comparison of models trained on different subsets → insights into urban vitality

---

### Step 8: High-Resolution Scaling
**Duration**: 7-10 days
**Goal**: Scale to 512x512 or 1024x1024

**Files to Modify**:
- `model/unet.py` - Add higher-resolution support
- `train.py` - Memory optimizations for larger images
- `sampling/` - Memory-efficient sampling

**Key Features**:
- Gradient checkpointing
- Mixed precision training
- Multi-GPU optimizations

**Deliverable**: Working high-resolution model on geo-spatial data

---

## Verification

1. Test foundation code runs on dual 3090 setup
2. Verify each integrated feature works independently
3. End-to-end test: train → generate → evaluate
4. Compare results with research paper findings

---

## Key Files to Modify/Create

### New Files to Create:
```
ddpm-win-master/
├── configs.py                    # Configuration management
├── schedules.py                  # Beta schedule variants
├── data/
│   ├── __init__.py
│   ├── geospatial.py            # TIF/JPG geo-spatial loader
│   ├── subset.py                # Subset creation tools
│   └── preprocessing.py         # Image preprocessing
├── sampling/
│   ├── __init__.py
│   ├── ddim.py                  # DDIM sampling
│   └── optimal.py               # Optimal denoiser
├── evaluation/
│   ├── __init__.py
│   ├── memorization.py          # k-NN gap analysis
│   ├── quality.py               # FID, reproducibility
│   ├── jacobian.py              # Jacobian rank
│   └── spectral.py              # Spectral density
├── analysis/
│   ├── __init__.py
│   ├── intrinsic_dim.py         # Intrinsic dimensionality
│   ├── subspace.py              # Subspace clustering
│   └── minimal_dataset.py       # Minimal dataset tools
├── experiments/
│   ├── __init__.py
│   └── subset_training.py       # Subset experiments
└── train_enhanced.py            # Enhanced training script
```

### Files to Modify:
- `model/ddpm.py` - Add schedule variants, DDIM support
- `model/unet.py` - Configurable resolution
- `train.py` - Checkpointing, multi-GPU
- `sample.py` - DDIM, optimal denoiser
- `util/img_utils.py` - Additional visualization

---

## Quick Start Path (For Immediate Results)

If you want to start seeing results quickly, here's a minimal path:

**Week 1**: Steps 1-2 (Foundation + Checkpointing)
**Week 2**: Step 3 (Geo-spatial pipeline) + Step 4 (Basic memorization tracking)
**Week 3**: Initial training run + analysis of "generalization window"

This gives you:
- Working training on your data
- Identification of when memorization occurs
- Foundation for deeper analysis

---

## Implementation Order (Step-by-Step)

### First Sprint: Foundation & Data (Weeks 1-2)
1. ✅ Create conda environment with CUDA 12.1
2. ✅ Fix ddpm-win-master foundation (dataset.py, DataParallel)
3. ✅ Port TrainingConfig/DiffusionConfig classes
4. ✅ Add logarithmic checkpointing
5. ✅ Create geo-spatial data loader (TIF/JPG support)
6. ✅ Test training on small subset

### Second Sprint: Memorization Tracking (Weeks 3-4)
7. ✅ Port memorization fraction computation (k-NN gap)
8. ✅ Port FID computation
9. ✅ Port reproducibility score (SSCD)
10. ✅ Run training with all checkpoints
11. ✅ Analyze "generalization window" results

### Third Sprint: Manifold Analysis (Weeks 5-6)
12. ✅ Port Jacobian rank analysis
13. ✅ Port spectral density analysis
14. ✅ Run manifold analysis on checkpoints
15. ✅ Validate low-dimensional learning

### Fourth Sprint: Advanced Features (Weeks 7-8)
16. ✅ Add DDIM sampling
17. ✅ Add intrinsic dimensionality analysis
18. ✅ Create minimal dataset
19. ✅ Compare: full vs minimal dataset results

### Fifth Sprint: Scaling (Weeks 9-10)
20. ✅ Scale to higher resolutions (if needed)
21. ✅ Optimize for dual 3090 GPUs
22. ✅ Final experiments and analysis

---

## Session Recovery Information

**Plan Location**: `C:\sd\ddpm-generalization-theory\INTEGRATION_PLAN.md`
**CLAUDE.md**: `C:\sd\ddpm-generalization-theory\CLAUDE.md`
**Foundation Code**: `C:\sd\ddpm-generalization-theory\ddpm-win-master\`

**Current Status**: Planning phase complete, ready to begin implementation
**Next Step**: Phase 1 - Environment Setup & Foundation Preparation

**Research Context**:
- Urban vitality research via manifold learning
- Finding "generalization window" in diffusion training
- Geo-spatial data (ArcGIS → TIF → JPG)
- Minimal dataset approach via intrinsic dimensionality
- Dual RTX 3090 GPUs, Windows environment
