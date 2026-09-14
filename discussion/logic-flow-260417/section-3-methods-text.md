## 3. Methods

### 3.1 Research Framework

This section presents the methodology for operationalizing the generative epistemology proposed in Section 2.4. The pipeline comprises four modules: data preparation, model architecture, training with structured checkpointing, and evaluation via sampling and analysis. Before detailing each, we outline how these modules connect to the epistemic probe framework.

The data preparation module (Section 3.2) encodes three supply-side built environment dimensions — density, accessibility, and diversity — as RGB image channels. This encoding deliberately avoids composite scoring or pre-computed vitality indices, which would embed the very correlationist assumptions critiqued in Section 2.2 into the input data. Instead, each dimension is preserved as a raw, unweighted channel, requiring the model to discover any relationships between them independently.

The model architecture module (Section 3.3) specifies a Denoising Diffusion Probabilistic Model (DDPM) trained on these RGB images as a structure-preserving compressor — learning the joint distribution of the three built environment dimensions rather than optimizing for generation quality. Section 3.4 details the noise schedule options, and Section 3.5 describes the training protocol designed to capture the generalization window identified in Section 2.3.

The evaluation module (Sections 3.6–3.8) probes what the model has learned through four complementary lenses, each answering a distinct question: *memorization fraction* identifies **when** the model transitions from structural learning to memorization; *distributional fidelity* measures **how well** each channel's marginal distribution is reproduced; *inter-channel synergy analysis* tests **whether** the model captures the joint distribution across channels — the core of the epistemic probe; and *irreducible characteristics detection* identifies **what** the model internalizes as essential urban structure.

---

### 3.2 Data Preparation

#### 3.2.1 Rasterization

The input data comprises three geospatial vector layers in shapefile format: building footprints (polygons), road networks (polylines), and points of interest (points). Each layer is converted from vector to raster format through a process informed by a critical distinction between two spatial scales.

The **cell size** (1 km) defines the physical extent of each image tile — the unit the model processes. The **calculation unit** (50 m) defines the resolution at which urban attributes are computed within each cell. This distinction is consequential: a calculation unit that is too large blurs spatial detail, while one that is too small produces sparse attribute values. At 50 m resolution, each 1 km cell contains a 20 × 20 grid of calculation units (1 km ÷ 50 m = 20).

For each calculation unit, a scalar attribute is computed from the corresponding geospatial layer:

- **Building density (Blue channel)**: Continuous building footprint coverage within a 150 m buffer around each unit. This captures the physical mass concentration of the local walkable neighborhood.
- **POI diversity (Green channel)**: Shannon entropy index of POI categories within a 150 m buffer. The Shannon index is chosen over simple density counts because it captures category-level heterogeneity — a diverse mix of restaurants, shops, and services scores higher than many instances of a single category.
- **Road accessibility (Red channel)**: Gravity-model distance decay to the road network, with decay parameter β = 0.3/km and a 3,000 m cutoff. The gravity model reflects connectivity infrastructure reach: a unit close to a major road network scores higher than one at the urban periphery.

The result is three sets of 20 × 20 raster grids per 1 km cell, each representing one built environment dimension.

#### 3.2.2 RGB Fusion

The three 20 × 20 raster grids are upsampled to 128 × 128 pixels via bilinear interpolation (scipy.ndimage.zoom) and combined into a single RGB image. The choice of 128 × 128 resolution balances spatial granularity against computational feasibility: 32 × 32 (CIFAR-10 standard) is too coarse to capture urban spatial patterns, while 256 × 256 and above would require substantially greater model capacity and training time.

Each channel is normalized independently using a global 95th percentile clipping strategy: the 95th percentile value across all cells is mapped to 1.0, values are clipped to [0, 1], then scaled to [0, 255]. Global (not per-cell) normalization ensures inter-cell comparability. The 95th percentile, rather than the maximum, mitigates outlier distortion from extreme values in dense urban cores. The convention is consistent across channels: white indicates denser, more accessible, or more diverse; black indicates absence.

The channel assignment is: R = accessibility, G = POI diversity, B = building density. Three channels are then stacked into a single 128 × 128 × 3 tensor.

Each channel is a lossy compression of complex geospatial data. Building density loses vertical information and building-level geometry. POI diversity (Shannon index) loses category-level detail and spatial distribution within the buffer. Road accessibility (gravity model) loses network topology. However, as argued in Section 2.4, the epistemologically significant question is not per-channel fidelity but whether the model captures the synergies between channels — the joint distribution P(density, diversity, accessibility). The RGB encoding withholds any researcher-imposed weighting or mixing logic, requiring the model to discover these relationships independently.

#### 3.2.3 Pre-encoding Analysis

The study area is Ho Chi Minh City (HCMC), covered by a 1 km fishnet grid producing 8,436 total cells. After filtering cells with no built environment data (empty water bodies, undeveloped periphery), 731 non-empty cells remain. These are split deterministically (seed = 42) into 584 training, 73 validation, and 74 test images (80/10/10). Image values are normalized from [0, 255] to [−1, 1] following standard DDPM preprocessing.

**No data augmentation** (horizontal flip, rotation) is applied. This is a deliberate choice involving a recognized trade-off. Yoon et al. (2023) demonstrate that increasing effective dataset size — even with dummy data carrying no meaningful content — can push a diffusion model past its memorization capacity and induce generalization. Augmentation would similarly increase the effective dataset, potentially widening the generalization window. However, augmentation adds no new structural information about urban form, only transformed views of existing data. Without augmentation, the model's natural learning dynamics are preserved, even though the generalization window may be narrower and harder to detect. This trade-off is accepted for observational honesty.

A **pre-encoding analysis** of the dataset — including principal component analysis, intrinsic dimension estimation, and typology classification using the three-dimensional channel statistics — is a necessary part of the methodology but has not yet been performed. This analysis would characterize the dataset's distributional composition, identify dominant urban typologies, and inform decisions about dataset curation and augmentation for future comparative experiments.

---

### 3.3 Model Architecture

#### 3.3.1 Denoising Diffusion Probabilistic Model

We adopt the standard DDPM formulation (Ho et al., 2020). The forward process progressively corrupts a data point x₀ by adding Gaussian noise over T = 1,000 discrete timesteps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \, \beta_t \, \mathbf{I})$$

where β_t is the noise schedule (Section 3.4). Over the full sequence, this process gradually transforms the structured urban image into pure Gaussian noise.

The reverse process learns to denoise — a neural network ε_θ is trained to predict the noise ε that was added at each timestep:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

This is a mean-squared error between the true noise and the predicted noise. The training process does not generate complete images and compare them to the training data; instead, it learns to reverse individual noise steps. Distributional alignment with the training data emerges from this process — Ho et al. (2020) proved that the noise prediction objective is equivalent to maximizing a variational lower bound on the data log-likelihood. What the model prioritizes during learning — which structural features it captures first, and when memorization begins — is not directly engineered but emerges from the training dynamics. This is precisely what the epistemic probe aims to observe.

Two architectural choices distinguish our setup. First, we use a **pixel-space** DDPM rather than a latent diffusion model (LDM). LDMs compress images into a lower-dimensional latent space before applying the diffusion process, which improves computational efficiency for generation tasks. Our objective, however, is not efficient generation but manifold learning — analyzing what the model internalizes about the data at full resolution.

Second, we use **discrete timesteps** (t ∈ {0, 1, ..., 999}) rather than a continuous-time SDE formulation. This choice is an implementation alignment with key references rather than a methodological necessity for the epistemic probe. The probe observes what the model learns — the converged manifold — not the optimization path used to reach it. Whether noise is added in 1,000 discrete steps or along a continuous SDE trajectory does not change the data distribution the model must ultimately learn. We adopt discrete time because our primary memorization metric (F_mem, Section 3.7.1) is inherited from Bonnaire et al. (2025), whose framework operates in discrete DDPM; Yoon et al. (2023) similarly use discrete time. Using the same formulation is a form of controlled variable, ensuring our results are directly comparable with the theoretical foundations we build upon.

#### 3.3.2 U-Net Noise Prediction Network

The noise prediction network ε_θ is a U-Net (DDPMUnet) with time-conditioned residual blocks. The exact specification is:

| Parameter | Value |
|-----------|-------|
| Base channels | 128 |
| Total parameters | ~157.8M |
| Channel multipliers | [1, 2, 2, 2] at four resolution levels |
| Self-attention | Disabled at all levels |
| Residual blocks per level | 2 |
| Group normalization | 32 groups |
| Activation | SiLU |
| Time embedding dimension | 512 |
| Dropout | 0.1 |

Self-attention is disabled at all resolution levels — at 128 × 128, the resolution does not warrant the computational cost of attention layers. Group normalization (32 groups) and dropout (0.1) provide regularization.

The model capacity of 157.8M parameters was determined empirically. An earlier configuration (v2) with base channels = 32 (9.88M parameters, 50K training steps) failed to converge: after 50,000 steps, training loss was still decreasing, and generated samples showed multi-colored noise rather than matching the dark-red-dominated training distribution. The current configuration (v3, base channels = 128) converged at approximately 86,000 steps, confirming that urban RGB data requires significantly more model capacity than benchmark datasets such as MNIST or CIFAR-10.

---

### 3.4 Noise Schedules

The noise schedule β_t determines how quickly signal is destroyed during the forward process — and consequently shapes what the model learns and when memorization begins. Four discrete schedules are implemented in the codebase:

| Schedule | Source | Status |
|----------|--------|--------|
| **Linear** | Ho et al. (2020) | Active (v3 baseline) |
| **VP** (Variance Preserving) | Song et al. (2021) | Implemented, not yet trained |
| **VE** (Variance Exploding) | Song et al. (2021) | Implemented, not yet trained |
| **Fixed** | — | Implemented |

The linear schedule interpolates β uniformly from 1e-4 to 2e-2 over 1,000 timesteps, producing a constant rate of noise addition. The VP schedule (discretized from Song et al.'s continuous SDE) preserves total variance throughout the forward process, with signal-to-noise ratio changing geometrically. The VE schedule allows variance to grow without bound through an exponential progression.

Two schedules were considered but not adopted. The cosine schedule (Nichol & Dhariwal, 2021) was initially implemented but removed due to numerical instability: beta values approaching 1.0 at late timesteps produce black images during sampling, and neither of the primary research references (Bonnaire et al., 2025; Wang et al., 2025) uses cosine. The EDM schedule (Karras et al., 2022) was consciously skipped as a natively continuous-time formulation that would add complexity to the discrete framework without clear benefit for generalization window analysis.

The linear schedule serves as the baseline for three reasons: it is the most standard DDPM schedule with the largest body of comparative literature; it is directly comparable with Bonnaire et al.'s generalization window findings; and its uniform noise progression is the simplest to interpret when analyzing training dynamics.

VP and VE schedules are retained for future comparison experiments. They represent fundamentally different noise geometries — VP preserves variance while VE allows explosion — and whether these geometric differences affect the timing and nature of the generalization window remains an open question.

---

### 3.5 Training Procedure and Checkpointing Strategy

The training protocol is designed to capture the temporal evolution of learning — specifically, to identify the generalization window.

**Optimization.** AdamW optimizer with learning rate 1e-4, gradient clipping at max norm 1.0. The loss function is the standard DDPM noise prediction objective (MSE between true and predicted noise).

**Batch size and convergence trade-off.** Batch size is set to 8. On the available hardware (single NVIDIA RTX 5090, 24 GB VRAM, Windows 11), this configuration completes 500,000 steps in approximately 22 hours. This represents a deliberate trade-off: smaller batch sizes yield more gradient updates per epoch (408 updates per epoch with bs=8 versus 102 with bs=32), but produce noisier gradients and potentially less stable convergence. Larger batch sizes (bs=16 or 32) would provide smoother training at the cost of significantly longer runtimes. The 22-hour configuration prioritizes iteration speed for a feasibility study, with the recognition that convergence dynamics may differ at larger batch sizes.

**Logarithmic checkpointing.** Model snapshots are saved at 14 points during training: dense in the early phase (steps 100, 250, 500, 1,000, 2,500, 5,000, 10,000, 20,000) where learning dynamics change most rapidly, and sparse in the later phase (50,000, 100,000, 200,000, 300,000, 400,000, 500,000) where changes are more incremental. This logarithmic spacing reflects the nature of diffusion model training: the most structurally significant learning — the transition from generating noise to capturing urban patterns — occurs in the early steps, analogous to how the first years of a new urban district see dramatic transformation while later decades bring incremental change. The generalization window is most likely to fall within this early phase, making dense early checkpointing essential for its detection.

**Validation.** Validation loss is computed every 5,000 steps over 50 validation batches (73 images). Divergence between training and validation loss serves as a secondary signal for memorization onset.

**Hardware.** The v3 training run uses a single RTX 5090 on Windows 11. The earlier v2 run used dual RTX 3090 (24 GB each, NVLink interconnect) with PyTorch DataParallel. The eventual target is to migrate the full training pipeline back to dual RTX 3090 for reproducibility.

**Planned comparative experiments.** Three experimental dimensions are identified for future investigation, designed around the core research question of what affects the generalization window:

1. **U-Net capacity**: ch = 32 (v2, 9.88M, failed to converge) / ch = 128 (v3, 157.8M, converged) / ch = 256 (future). Yoon et al. (2023) experiment with UNet-64/128/256, providing precedent for how model capacity interacts with memorization dynamics.
2. **Batch size**: bs = 8 (current, fastest) / bs = 16 / bs = 32 — tests the convergence stability versus training time trade-off.
3. **Noise schedule**: linear (baseline) / VP / VE — tests whether noise geometry affects generalization window dynamics.

These are not intended as a full factorial design but as targeted comparisons addressing specific hypotheses about the generalization window.

---

### 3.6 Sample Generation

Two sampling methods are employed for different analysis purposes.

**DDPM sampling** (1,000-step) performs the full stochastic denoising process, starting from pure Gaussian noise and reversing all 1,000 diffusion timesteps. This produces the highest-quality samples and is used for memorization fraction computation and visual quality assessment.

**DDIM sampling** (50-step) follows a deterministic denoising trajectory (Song et al., 2021), generating samples approximately 20 times faster. This is used for large-scale distribution analysis where sample quantity matters more than per-sample fidelity.

For each checkpoint within the generalization window, 30–100 samples are generated to track the temporal evolution of the model's learned structure. This sample range balances statistical reliability against computational cost.

---

### 3.7 Evaluation Metrics

Four complementary metrics probe different facets of what the model learns. Together, they operationalize the epistemic probe framework.

#### 3.7.1 Memorization Fraction (F_mem)

The memorization fraction quantifies whether the model generates novel samples that reflect the learned distribution, or copies of individual training images. Following Bonnaire et al. (2025), the metric operates as follows: for each generated sample, its nearest neighbor in the training set (NN_train) and its nearest neighbor among other generated samples (NN_gen) are identified using L2 distance. The gap ratio is computed as d(sample, NN_train) / d(sample, NN_gen). If this ratio falls below a threshold (0.333), the sample is classified as "memorized" — it is closer to a specific training image than to other generated samples, indicating replication rather than generalization.

F_mem is the fraction of memorized samples across all generated images at a given checkpoint. F_mem ≈ 0 indicates generalization: the model has learned the data distribution rather than individual data points. F_mem → 1 indicates memorization. The transition from low to high F_mem identifies the generalization window boundary.

#### 3.7.2 Distributional Fidelity

Distributional fidelity measures how well generated samples reproduce the marginal distribution of each RGB channel — the per-channel histogram. For each channel (B = building density, G = POI diversity, R = road accessibility), the Wasserstein-1 distance (Earth Mover's Distance) is computed between the generated and training channel distributions. Lower distance indicates better fidelity. Computed per-channel, this metric reveals which dimensions are learned faster or more accurately.

It is important to clarify what this metric does and does not capture. It evaluates each channel independently — does the model's building density distribution match the real building density distribution? This is a marginal analysis. A model could reproduce each marginal accurately while failing to capture how the channels co-vary — for instance, generating images with correct building density and correct POI diversity distributions independently, but without the correlation that in real cities, dense areas tend to have diverse POIs. Capturing this co-variation is the role of inter-channel synergy analysis (Section 3.7.4).

Spatial autocorrelation (Moran's I) is computed per channel as a supplementary check on whether generated images contain spatially coherent patterns rather than random noise. Moran's I measures whether neighboring pixels in an image are more similar than distant pixel pairs — values near 1 indicate spatial clustering, values near 0 indicate randomness. However, an important caveat applies: Moran's I operates at the intra-cell pixel level (128 × 128), measuring clustering within a single 1 km cell. The urban spatial structure that ultimately matters — inter-cell relationships, street patterns, neighborhood form — operates at a larger scale. Our model treats each cell as an independent input, so intra-cell pixel clustering is the only spatial metric available. Moran's I should be treated as a supplementary coherence signal rather than primary evidence of urban structure learning.

A conceptual note on training versus evaluation: DDPM training optimizes the noise prediction loss — the MSE between true and predicted noise at each timestep. It does not generate complete images and compare them to training data (unlike fine-tuning approaches such as DreamBooth). Distributional alignment with the training data emerges from the noise prediction objective, which is mathematically equivalent to maximizing a variational lower bound on the data log-likelihood (Ho et al., 2020). The design choice is deliberate: by not engineering distributional alignment as an explicit training target, we allow the model's natural learning priorities to emerge, and then probe what those priorities reveal.

#### 3.7.3 Fréchet Inception Distance (FID)

FID measures the distributional distance between generated and real images in the feature space of a pretrained Inception-v3 network (Heusel et al., 2017). It is the standard quality metric for generative models.

In our context, FID serves as a supplementary quality indicator rather than a primary metric. FID is designed for natural images and its applicability to RGB-encoded urban rasters — where pixel values represent computed geospatial attributes rather than visual appearance — is not well established. Zhou et al. (2026) report FID evaluations on black-and-white urban layouts, suggesting some applicability, but our RGB fusion results differ in kind. We report FID for comparability with the diffusion model literature but do not weight it as heavily as memorization fraction and synergy analysis.

#### 3.7.4 Inter-Channel Synergy Analysis

The inter-channel synergy analysis is the core epistemic probe. While distributional fidelity (Section 3.7.2) evaluates each channel's marginal distribution independently, synergy analysis evaluates the joint distribution — whether the model learns that building density, POI diversity, and road accessibility co-vary in structured ways, as they do in real cities.

The method operates as follows. For each generated image, the spatial mean of each channel is computed after converting from [−1, 1] to [0, 1], yielding a three-dimensional point (B_mean, G_mean, R_mean). Across all generated images at a checkpoint, pairwise correlations between these spatial means are computed using Pearson's r (linear relationship), Spearman's ρ (monotonic relationship), and mutual information (non-linear dependency).

Based on pairwise |r| thresholds, the model's learning regime is classified:

- **Coupled**: all channel pairs exhibit |r| > 0.5, indicating the model has learned the full joint distribution.
- **{X}-driven**: one channel pair dominates (e.g., diversity-driven indicates strong G–R correlation but weak B–R and B–G), indicating partial joint learning.
- **Fragmented**: all pairs exhibit |r| < 0.3, indicating channels learned independently with no synergy.

Tracking regime evolution across training checkpoints reveals when synergy emerges. Comparing the generated synergy structure with the training data's synergy structure reveals whether the model's learned joint distribution matches reality.

This metric directly tests whether the DDPM captures the inter-dimensional synergies that correlationist methods miss. As argued in Section 2.2, the correlationist paradigm treats density, accessibility, and diversity as independent variables, losing the relational structure of how they jointly distribute across the city. The synergy analysis probes whether the model recovers this relational structure from the data alone.

---

### 3.8 Irreducible Characteristics Detection

Features that remain stable across the generalization window represent what the model internalizes as essential urban structure — the irreducible properties of the data manifold that the model cannot ignore if it is to generalize.

The detection method operates as follows. For each checkpoint within the generalization window, per-channel statistics (mean, standard deviation) and inter-channel correlations are computed across generated samples. The coefficient of variation (CV = std/mean) of each statistic is tracked across checkpoints. Characteristics with low CV are classified as **stable** — the model has converged on these features as irreducible structural properties. Characteristics with high CV are classified as **evolving** — the model has not yet settled on these aspects.

The epistemic significance of irreducible characteristics is that they answer the core research question: what does the model consider essential about urban form? These are the features that persist even as training continues, suggesting they represent fundamental properties of the data manifold rather than noise or incidental particulars. If the model consistently treats certain density-accessibility-diversity relationships as irreducible across the generalization window, this constitutes evidence that these relationships are structural features of the urban data distribution — features that a correlationist analysis might overlook or flatten into linear coefficients.
