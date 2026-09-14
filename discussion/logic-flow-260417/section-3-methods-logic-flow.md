# Section 3: Methods — Logic Flow (Revised)

## Overall Logic
```
3.1 Research Framework [C4][C1][C6] — Pipeline overview with epistemic probe annotations
3.2 Data Preparation [C5] — From geospatial vectors to RGB tensors
    3.2.1 Rasterization
    3.2.2 RGB Fusion
    3.2.3 Pre-encoding Analysis
3.3 Model Architecture [C2] — DDPM formulation and U-Net specification
    3.3.1 Denoising Diffusion Probabilistic Model
    3.3.2 U-Net Noise Prediction Network
3.4 Noise Schedules — Implemented schedules and rationale for baseline choice
3.5 Training Procedure and Checkpointing Strategy [C3] — Protocol for manifold learning
3.6 Sample Generation — DDPM and DDIM sampling
3.7 Evaluation Metrics [C6] — How we probe what the model learns
    3.7.1 Memorization Fraction
    3.7.2 Distributional Fidelity
    3.7.3 Fréchet Inception Distance (FID)
    3.7.4 Inter-Channel Synergy Analysis
3.8 Irreducible Characteristics Detection [C6] — Identifying learned structural signals
```

**Core principle**: Methods must be reproducible and honest. Every parameter from codebase, every design choice justified by theory or pragmatism.

---

### 3.1 Research Framework [C4][C1][C6]

**Core argument**: High-level pipeline overview — what each module does and which epistemic insight it produces.

**Logic**:
- S1: The methodology comprises four modules: (1) data preparation, (2) model architecture, (3) training with structured checkpointing, (4) evaluation via sampling and analysis
- S2: Data preparation encodes three supply-side BE dimensions [C4] — density, accessibility, diversity — as RGB channels without composite scoring [C1]
- S3: The DDPM is trained as a structure-preserving compressor [C2] — not for generation quality but for manifold learning
- S4: Logarithmic checkpointing captures the temporal evolution of learning, enabling generalization window identification [C3]
- S5: Evaluation probes the model through four complementary lenses [C6]:
  - **Memorization fraction** → tells us WHEN the model transitions from structural learning to memorization
  - **Distributional fidelity** → tells us HOW WELL each channel's marginal distribution is learned
  - **Inter-channel synergy** → tells us WHETHER the model captures joint distribution (not just marginals) — the core epistemic probe
  - **Irreducible characteristics** → tells us WHAT the model considers essential urban structure

**Visual**: [V4] Pipeline diagram — end-to-end flowchart with annotations showing which module produces which insight

**Callbacks**:
- [C4] supply-demand ← 2.1 → 3.1 (scope: supply-side only)
- [C1] correlationist paradigm → 3.1 (no composite score)
- [C6] epistemic probe ← 2.4 → 3.1 (framework overview) → 3.7/3.8 (operationalized)

---

### 3.2 Data Preparation [C5]

**Core argument**: Convert heterogeneous geospatial vectors into normalized RGB tensors that preserve raw inter-channel relationships.

#### 3.2.1 Rasterization

**Logic**:
- S1: Input data: polygon vectors (buildings), polyline vectors (roads), point vectors (POIs) — standard shapefile format
- S2: Key distinction: **cell/grid size** vs **calculation unit**
  - Cell size = physical distance of each image tile = 1km
  - Calculation unit = resolution of attribute computation = 50m
  - This distinction matters: too-large calculation units blur spatial detail; too-small units produce sparse attributes
- S3: Per-channel computation at 50m resolution:
  - **B (Building density)**: continuous building footprint coverage within 150m buffer → scalar density value
  - **G (POI diversity)**: Shannon entropy index of POI categories within 150m buffer → scalar diversity value
  - **R (Road accessibility)**: gravity-model distance decay to road network (β=0.3/km, cutoff=3000m) → scalar accessibility value
- S4: Result: three sets of 20×20 raster grids per 1km cell (1km / 50m = 20)

**Design choices**:
- 150m buffer (density/diversity): captures local walkable neighborhood
- 3000m cutoff (accessibility): gravity model with β=0.3 reflects road network reach
- Shannon index over simple counts: captures category-level heterogeneity

#### 3.2.2 RGB Fusion [C5]

**Logic**:
- S1: Three 20×20 raster grids → upsampled to 128×128 via scipy.ndimage.zoom (bilinear interpolation)
- S2: Per-channel normalization: global 95th percentile → clip [0,1] → scale to [0,255]
  - Global (not per-cell) normalization ensures inter-cell comparability
  - 95th percentile (not max) mitigates outlier distortion
  - Convention: white = denser / more accessible / more diverse; black = absence
- S3: Three channels stacked into single 128×128 RGB image
  - R = accessibility, G = POI diversity, B = building density
- S4: Each channel is lossy compression — but [C5 callback] synergies between channels matter, not per-channel fidelity
- S5: Epistemological choice: model must discover mixing logic, not receive researcher-imposed weights

**Why 128×128**: Balances spatial resolution with computational feasibility. CIFAR-10 (32×32) too coarse; 256×256+ requires exponentially more capacity.

**Callbacks**: [C5] RGB synthesis ← 2.4 P2 → 3.2.2 → Experiments 4.3

#### 3.2.3 Pre-encoding Analysis

**Logic**:
- S1: Study area: HCMC (Ho Chi Minh City), 1km fishnet grid = 8,436 total cells
- S2: After filtering empty cells (no BE data): **731 non-empty cells**
- S3: Deterministic split (seed=42): 584 train / 73 validation / 74 test (80/10/10)
- S4: Image normalization for DDPM: [0,255] → [-1,1] (standard DDPM preprocessing)
- S5: **No data augmentation** — deliberate choice with trade-off:
  - Per Yoon et al. (2023), adding more data (even dummy data) can push the model past its memorization capacity and induce generalization. Augmentation (flip/rotate) would increase effective dataset size, potentially widening the generalization window.
  - However, augmentation adds no new structural information — only rotated/flipped views of the same data. It would artificially modify the natural learning dynamics we aim to observe.
  - Without augmentation, the natural generalization window may be shorter (as noted in 0412 methodology notes: "a small, no augmentation dataset would reach memorization stage earlier, and the generalization window would theoretically, shorter, hence harder for us to detect and analyze"). We accept this difficulty to preserve observational honesty.
- S6: **Pre-encoding analysis** (not yet performed, but necessary part of the methodology):
  - PCA / intrinsic dimension estimation
  - Typology classification using 3D channel statistics
  - Purpose: understand dataset composition before model training
  - Required before comparative experiments to characterize what the training data contains
- S7: 731 cells is small by ML standards but appropriate for feasibility study — Bonnaire et al. demonstrate generalization window on small datasets

**Evidence**: [Bonnaire et al. - 2025] for small-dataset justification; [Yoon et al. - 2023] for memorization-generalization dichotomy and data-size effects

**Visual**: [V5] Dataset distribution (PCA, channel histograms) — PRODUCIBLE
**Visual**: [V6] Sample RGB images from dataset — PRODUCIBLE

---

### 3.3 Model Architecture [C2]

#### 3.3.1 Denoising Diffusion Probabilistic Model

**Core argument**: Standard DDPM formulation — discrete timestep, pixel-space, purpose-built for manifold learning rather than generation quality.

**Logic**:
- S1: **Forward process**: gradually add Gaussian noise over T=1000 discrete timesteps
  - q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
  - β_t follows linear schedule: β₁ = 1e-4 → β_T = 2e-2 (Section 3.4)
- S2: **Reverse process**: learn to denoise — parameterized by neural network ε_θ
  - p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
  - Training objective: L_simple = E_{t,x_0,ε} [||ε - ε_θ(x_t, t)||²]
  - MSE between true noise ε and predicted noise ε_θ
- S3: **Why pixel-space DDPM** (not latent diffusion / LDM): purpose is learning the data manifold at full resolution, not efficient generation. The structure we want to analyze exists in pixel space.
- S4: **Why discrete timestep** (not continuous SDE): discrete timestep is an implementation alignment with key references rather than a methodological necessity for the epistemic probe. The probe observes what the model learns (the converged manifold), not how it gets there — discrete vs continuous changes the optimization path, not the destination. We use discrete time because our primary memorization metric (F_mem, Section 3.7.1) is inherited from Bonnaire et al. (2025), who operate in discrete DDPM; similarly, Yoon et al. (2023) use discrete time. Using the same framework is a form of controlled variable, ensuring our results are directly comparable with theirs.

**Evidence**: [Ho et al. - 2020] DDPM foundational formulation

#### 3.3.2 U-Net Noise Prediction Network

**Core argument**: U-Net backbone scaled for urban data complexity — 157.8M parameters determined by convergence testing.

**Logic**:
- S1: Architecture: DDPMUnet with time-conditioned residual blocks
- S2: Exact specification:
  - Base channels: 128 → total: ~157.8M parameters
  - Channel multipliers: [1, 2, 2, 2] at four resolution levels
  - Attention: disabled (is_attn=[F,F,F,F]) — 128×128 doesn't warrant self-attention; reduces cost
  - Residual blocks: 2 per resolution level
  - Group normalization: 32 groups
  - Activation: SiLU
  - Time embedding: 512-dim positional embedding
  - Dropout: 0.1
- S3: **Capacity justification**: v2 (ch=32, 9.88M params, 50K steps) failed to converge — generated samples showed multi-colored noise, not matching the dark-red-dominated training distribution. v3 (ch=128) converged at ~86K steps, confirming that urban RGB data requires significantly more capacity than benchmark datasets (MNIST, CIFAR-10).

**Visual**: [V7] Architecture diagram (U-Net with time embedding, resolution levels)

**Callbacks**: [C2] structure-preserving compressor ← 2.3 → 3.3 (operationalized)

---

### 3.4 Noise Schedules

**Core argument**: Noise schedule selection shapes how the model traverses the data manifold during training. We implement four discrete schedules inherited from our primary research references; linear serves as baseline.

**Logic**:
- S1: The noise schedule β_t controls how quickly signal is destroyed in the forward process — directly affects what the model learns and when memorization begins
- S2: **Implemented schedules** (all discretized for T=1000 timesteps in `diffusion_config.py`):

  | Schedule | Source | Formulation | Status |
  |----------|--------|-------------|--------|
  | **Linear** | Ho et al., 2020 (DDPM) | β_t = β₀ + (β_T − β₀) · t/T, β₀=1e-4, β_T=2e-2 | **Active (v3 baseline)** |
  | **VP** | Song et al., 2021 (Score SDEs) | σ(t) = √(exp(0.5·β_d·t² + β_min·t) − 1), β_d=19.9, β_min=0.1 | Implemented, not yet trained |
  | **VE** | Song et al., 2021 (Score SDEs) | σ(t) = σ_min·(σ_max/σ_min)^t, σ_min=0.02, σ_max=100.0 | Implemented, not yet trained |
  | **Fixed** | — | constant β=0.006 | Implemented |

- S3: **Schedules not used** (with rationale):
  - **Cosine** (Nichol & Dhariwal, 2021): initially implemented, **removed** due to numerical instability — beta values approaching 1.0 at late timesteps produce black images during sampling. Additionally, neither of our primary research repos (Bonnaire et al.; Wang et al.) uses cosine, so removing it maintains alignment with the theoretical foundations we build upon.
  - **EDM** (Karras et al., 2022): natively continuous-time formulation — **consciously skipped** because it would add complexity to our discrete framework without clear benefit for generalization window analysis.

- S4: **Why linear as baseline**: three converging reasons:
  1. Most standard DDPM schedule — largest body of comparative literature
  2. Directly comparable with Bonnaire et al.'s generalization window findings (their primary experiments use linear)
  3. Simplest noise progression to interpret when analyzing training dynamics
- S5: **Why VP/VE matter for future work**: Song et al.'s VP and VE schedules represent fundamentally different noise geometries — VP preserves variance throughout the forward process while VE allows it to explode. Whether these geometric differences affect the timing and nature of the generalization window is an open question. Infrastructure for parallel schedule comparison exists (`train_parallel_schedules.py`, `compare_schedules.py`).

**Evidence**:
- ✅ [Ho et al. - 2020]: linear schedule
- ✅ [Song et al. - 2021]: VP and VE SDE formulations
- ✅ [Nichol & Dhariwal - 2021]: cosine schedule (removed)
- ✅ [Karras et al. - 2022]: EDM (skipped)
- ✅ Codebase: `diffusion_config.py` (implementations), `20260209.md` (removal rationale), `RESEARCH_METHODOLOGY_DESIGN.md:199-213` (schedule vs time formulation table)

---

### 3.5 Training Procedure and Checkpointing Strategy [C3]

**Core argument**: Training protocol designed to capture the generalization window through structured, logarithmic checkpointing.

**Logic**:
- S1: **Optimizer**: AdamW, lr=1e-4
- S2: **Loss**: MSE between predicted and true noise (standard DDPM objective)
- S3: **Batch size**: 8 (4 per GPU) — **trade-off documented**:
  - bs=8 is benchmarked as fastest throughput on dual GPU with DataParallel (gloo backend; NCCL unavailable on Windows), yielding ~22h for 500K steps on RTX 5090.
  - However, small bs means fewer samples per gradient update, producing noisier gradients and potentially less stable convergence. Larger bs (e.g., 16, 32) would provide smoother training but significantly longer runtimes (bs=16 ~44h benchmarked, bs=32 ~73h).
  - The 22h quick training comes at the sacrifice of convergence stability — an acceptable trade-off for a feasibility study, but comparative experiments with larger bs should be planned.
- S4: **Gradient clipping**: max_norm=1.0
- S5: **Total steps**: 500,000 (~22 hours on RTX 5090 with bs=8). Extended from v2's 50K after v2 demonstrated no convergence.
- S6: **Logarithmic checkpointing** [C3 callback]:
  - Dense early (rapid learning phase): 100, 250, 500, 1K, 2.5K, 5K, 10K, 20K
  - Sparse late (convergence/memorization phase): 50K, 100K, 200K, 300K, 400K, 500K
  - Total: 14 checkpoints for temporal analysis of learning dynamics
- S7: **Validation**: loss computed every 5,000 steps over 50 validation batches
- S8: **Hardware**: v3 trained on single RTX 5090 (24GB), Windows 11. v2 was trained on dual RTX 3090 (24GB each, NVLink). Eventual target: migrate v3 back to dual 3090 for reproducibility.
- S9: **Resume support**: checkpoint saves include optimizer state and step count, enabling seamless recovery from interruptions
- S10: **Planned comparative experiments** (not yet executed):
  - **U-Net capacity**: ch=32 (v2, 9.88M, failed) / ch=128 (v3, 157.8M, converged) / ch=256 (future — tests whether further capacity changes the generalization window). Yoon et al. (2023) experiment with UNet-64/128/256, providing precedent.
  - **Batch size**: bs=8 (current) / bs=16 / bs=32 — tests convergence stability vs training time trade-off.
  - **Schedule**: linear (baseline) / VP / VE — tests whether noise geometry affects generalization window dynamics.
  - Not a full factorial — designed around the research question: what affects the generalization window?

**Callbacks**: [C3] generalization window ← 2.3 → 3.5 (checkpointing strategy) → Experiments 4.2

---

### 3.6 Sample Generation

**Core argument**: Two sampling methods serve different analysis purposes.

**Logic**:
- S1: **DDPM sampling** (1000-step): full stochastic denoising from pure Gaussian noise. Highest quality samples, used for memorization fraction and visual quality assessment.
- S2: **DDIM sampling** (50-step): deterministic denoising trajectory. Faster generation, useful for large-scale distribution analysis where sample quantity matters more than per-sample quality.
- S3: **Sampling protocol**: generate samples from each checkpoint during the generalization window → temporal evolution of learned structure
- S4: **Sample size**: 30-100 samples per checkpoint — balances statistical reliability with computational cost

**Evidence**: [Ho et al. - 2020] DDPM sampling; [Song et al. - 2021] DDIM

---

### 3.7 Evaluation Metrics [C6]

**Core argument**: Four complementary metrics probe different facets of what the model learns — together they operationalize the epistemic probe.

#### 3.7.1 Memorization Fraction (F_mem)

**Logic**:
- S1: Measures whether the model generates novel samples or copies training data
- S2: Method (Bonnaire et al. 2025): for each generated sample x_gen:
  - Find nearest neighbor in training set: NN_train
  - Find nearest neighbor among other generated samples: NN_gen
  - Gap ratio = d(x_gen, NN_train) / d(x_gen, NN_gen)
  - If gap ratio < threshold (0.333), sample is considered "memorized"
- S3: F_mem = fraction of memorized samples across all generated
- S4: **Interpretation**: F_mem ≈ 0 → generalization (model learned distribution, not individual samples); F_mem → 1 → memorization
- S5: **Epistemic significance**: identifies the generalization window boundary — the transition from structural learning to memorization

**Evidence**: [Bonnaire et al. - 2025]

#### 3.7.2 Distributional Fidelity

**Logic**:
- S1: Measures how well generated samples reproduce the **marginal** distribution of each RGB channel independently — the per-channel histogram. Does the model's B-channel distribution (building density) match the training B-channel distribution? Same for G (diversity) and R (accessibility). Each channel evaluated in isolation.
- S2: Metric: Wasserstein-1 distance (Earth Mover's Distance) between generated and training channel distributions
- S3: Computed per-channel (B, G, R separately) → reveals which dimensions are learned faster or more accurately
- S4: Spatial autocorrelation (Moran's I) per channel as a coarse spatial coherence check:
  - Moran's I measures whether neighboring pixels in a generated image are more similar than random pairs — i.e., does the image contain spatial clusters or is it random noise?
  - High Moran's I = spatial structure exists; Low Moran's I = noise
  - **Important caveat**: Moran's I operates at intra-cell pixel level (128×128), measuring whether pixels within a single 1km cell cluster together. This is a coarse proxy — the urban spatial structure we actually care about (inter-cell relationships, street patterns, neighborhood form) operates at a larger scale. Our model treats each cell independently, so intra-cell pixel clustering is the only spatial metric available. Treat as supplementary signal, not primary evidence.
- S5: **Note on training vs evaluation**: DDPM training does NOT generate samples and compare them to training data. Training optimizes noise prediction (MSE between true noise ε and predicted noise ε_θ). Distributional alignment EMERGES from this process — it is mathematically designed (equivalent to maximizing data log-likelihood) but not directly engineered as a training objective. The epistemic probe philosophy: we let distributional alignment emerge naturally, then PROBE whether it matches reality.
- S6: **Epistemic significance**: tracks convergence of individual channel marginals — prerequisite for meaningful synergy analysis. A model that hasn't learned each marginal cannot meaningfully learn the joint.

#### 3.7.3 Fréchet Inception Distance (FID)

**Logic**:
- S1: Standard metric for generative model quality — measures distributional distance between generated and real images in Inception-v3 feature space
- S2: In our context, FID serves as a supplementary quality indicator rather than primary metric (FID is designed for natural images, not urban rasters)
- S3: **Caveat**: FID's applicability to RGB-encoded urban data (not natural images) is not well-established. We report it for comparability but do not weight it heavily.
- S4: **Epistemic significance**: secondary quality measure; primary insights come from memorization fraction and synergy analysis

**Evidence**: [Heusel et al. - 2017] FID original; [Zhou et al. - 2026] FID applied to urban layouts

#### 3.7.4 Inter-Channel Synergy Analysis

**Logic**:
- S1: The core epistemic probe — does the model learn the JOINT distribution P(density, diversity, accessibility) or just the marginals independently?
- S2: Method: for each generated image, compute spatial mean per channel → (B_mean, G_mean, R_mean)
- S3: Compute pairwise inter-channel correlations:
  - **Pearson r**: linear relationship
  - **Spearman ρ**: monotonic relationship
  - **Mutual Information**: non-linear dependency (histogram-based estimation)
- S4: **Regime classification** based on pairwise |r| thresholds:
  - **Coupled**: all pairs |r| > 0.5 → model learned full joint distribution
  - **{X}-driven**: one channel pair dominates → partial joint learning (e.g., diversity-driven = G-R strong, others weak)
  - **Fragmented**: all pairs |r| < 0.3 → channels learned independently (no synergy)
- S5: Track regime evolution across training checkpoints → WHEN does synergy emerge
- S6: Compare generated synergy structure with training data synergy structure → does the model's learned joint distribution match the real one?
- S7: **Epistemic significance**: directly tests whether DDPM captures the inter-dimensional synergies that correlationist methods miss [C1 callback]

**Evidence**: Synergy analysis framework from pilot reports; regime classification from analysis code

**Callbacks**: [C6] epistemic probe ← 2.4 → 3.7.4 (operationalized) → Experiments 4.3; [C1] correlationist → 3.7.4 (tests what correlationist misses)

---

### 3.8 Irreducible Characteristics Detection [C6]

**Core argument**: Features stable across the generalization window represent what the model considers essential urban structure — the "learned generative grammar."

**Logic**:
- S1: For each checkpoint within the generalization window, compute per-channel statistics (mean, std) and inter-channel correlations
- S2: Track coefficient of variation (CV) of each statistic across checkpoints
- S3: **Stable characteristics** (low CV): features the model has converged on — considered irreducible structural properties
- S4: **Evolving characteristics** (high CV): features still changing — not yet settled into learned structure
- S5: **Epistemic significance**: irreducible characteristics answer the core question — what does the model internalize as the essential structure of urban vitality? These are the features that persist even as training continues, suggesting they represent fundamental properties of the data manifold rather than noise.

**Evidence**: Pilot analysis framework (minimum_viable_report.md)

**Callbacks**: [C6] epistemic probe → 3.8 (core output) → Discussion 5.1

---

## Callback Map for Section 3

```
[C1] Correlationist paradigm
     ← 2.2 (critique)
     → 3.1 (why no composite score)
     → 3.7.4 (synergy analysis tests what correlationist misses)
     → Discussion 5.1

[C2] Structure-preserving compressor
     ← 2.3 (theory)
     → 3.3 (operationalized as DDPM)
     → Experiments 4.3

[C3] Generalization window
     ← 2.3 (Bonnaire theory)
     → 3.5 (checkpointing strategy)
     → 3.7.1 (identification via F_mem)
     → Experiments 4.2

[C4] Supply-demand
     ← 2.1 (framework)
     → 3.1 (scope: supply-side only)
     → Discussion 5.2 (limitation)

[C5] RGB channel synthesis
     ← 2.4 P2 (theoretical basis)
     → 3.2.2 (implementation)
     → Experiments 4.3

[C6] Epistemic probe
     ← 2.4 P3 (theoretical positioning)
     → 3.1 (pipeline overview)
     → 3.7 (evaluation metrics)
     → 3.8 (irreducible characteristics — core output)
     → Experiments 4.3
```

---

## Evidence Requirements

### References
| # | Reference | Used in | Status |
|---|-----------|---------|--------|
| 1 | [Ho et al. - 2020] DDPM | 3.3.1, 3.6 | Need to add (foundational) |
| 2 | [Song et al. - 2021] Score SDEs | 3.4 (VP/VE), 3.6 (DDIM) | Need to add |
| 3 | [Bonnaire et al. - 2025] | 3.5, 3.7.1 | Already cited |
| 4 | [Zhang et al. - 2024] | 3.7.1 | Already cited |
| 5 | [Heusel et al. - 2017] FID | 3.7.3 | Need to add |
| 6 | [Zhou et al. - 2026] | 3.7.3 (FID on urban data) | Already cited |
| 7 | [Yoon et al. - 2023] | 3.2.3 (augmentation/data-size), 3.5 (capacity experiments) | Need to add |

### From Codebase (No Citation Needed)
- Architecture: DDPMUnet, ch=128, 157.8M params, channel_mults=[1,2,2,2]
- Training: bs=8, lr=1e-4, AdamW, 500K steps, T=1000, linear schedule
- Data: 128×128 RGB, 731 cells (584/73/74), [-1,1] normalization
- RGB: R=accessibility (gravity β=0.3, 3km cutoff), G=diversity (Shannon 150m buffer), B=density (footprint 150m buffer)
- Hardware: v3 = single RTX 5090, Windows 11 (~22h for 500K); v2 = dual RTX 3090, Windows 11
- Schedules: linear (active), VP/VE/fixed (implemented, not yet trained)
- v2 failure: ch=32, 9.88M, 50K steps, no convergence
- v3 convergence: ~86K steps

---

## Key Design Decisions

| Decision | Justification |
|----------|---------------|
| 128×128 resolution | Balance spatial detail vs computational cost |
| ch=128 (157.8M params) | v2 (ch=32, 9.88M) failed to converge |
| bs=8 | Fastest throughput; but sacrifices convergence stability. Trade-off accepted for feasibility study. |
| No augmentation | Observe natural learning dynamics; augmentation adds no new structural info (Yoon et al. 2023) |
| Linear schedule baseline | Standard DDPM; most comparable with Bonnaire et al. findings |
| Pixel-space DDPM | Manifold learning at full resolution, not efficient generation |
| 95th percentile normalization | Mitigates outlier distortion |
| 150m buffer (density/diversity) | Local walkable neighborhood |
| 3000m cutoff (accessibility) | Gravity model with β=0.3 reflects road network reach |
| No self-attention | 128×128 doesn't warrant it; reduces computational cost |
