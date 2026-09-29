# TODO — DDPM Urban Vitality Research

## CRITICAL PRINCIPLE: Parallelize Wherever Possible

> **Do NOT run tasks sequentially when they can overlap.** Example: once intrinsic dimension is identified and training starts, begin dataset pre-classification and analysis in parallel. GPU trains while CPU analyzes.

---

## Step 1: HCMC Data Pipeline & DDPM Demo (RGB Image Encoding)

> **Format**: Each 1km cell = one **256×256×3** RGB PNG. 609 square cells = 609 training images.
> **Resolution**: 256×256 (~3.9m/pixel). Urban rasters (binary buildings, KDE gradients, accessibility surfaces) have low information content per pixel — 512×512 is overkill and causes OOM. Can scale up later if justified.
> **Encoding**: B=building footprints (white=bldg, black=empty), G=POI entropy density (whiter=more diverse), R=accessibility surface (exponential decay).
> **Limitation**: Cannot decode RGB back to original geospatial data. Analysis limited to channel-level patterns.
> **Full research context**: See `STAGE2_RESEARCH_SYNTHESIS.md`

### Step 1.0: Pre-Flight Checks (BEFORE any implementation)
- [x] **Run memory estimation**: `python estimate_memory_usage.py --config configs/geospatial.yaml` at 256×256 — confirm fits on RTX 3090
  - Result: 4.311 GB peak allocated at batch_size=4 — 18% of 24GB
  - Current U-Net architecture (`channel_mults=[1,2,2,2]`) works fine, no changes needed
- [x] **U-Net architecture adaptation** for 256×256:
  - [x] Verified: Current 4-level architecture (`channel_mults=[1,2,2,2]`) already reduces 256→16 at bottleneck
  - [x] Verified: Attention is ONLY in `MiddleBlock` (line 237 of unet.py), not at every level — already bottleneck-only
  - [x] Memory test passed: 4.311 GB peak at batch_size=4 (18% of 24GB)
  - [x] Decision: No architecture changes needed. The Oracle's concern about OOM was based on incorrect assumption that attention was at every level.
- [ ] **Jacobian rank strategy**: Full SVD is infeasible beyond ~64×64 (at 256×256×3 the Jacobian would be ~200K×200K). Plan to use **Hutchinson trace estimator** (randomized rank estimation) instead of full SVD for Jacobian analysis at 256×256. Alternatively, compute Jacobian at the U-Net bottleneck layer (16×16) rather than the full input.

### Step 1.1: Data Pipeline Rework
- [x] Install dependencies (osmnx, pandana, geopandas, shapely, rasterio, pandas, etc.)
- [x] Download OSM POI data for HCMC (Overpass API)
- [x] Download OSM street network for HCMC (OSMnx, pedestrian mode)
- [x] Download OSM building footprints (Overpass API, ~20K polygons from 2/4 tiles)
- [x] Build 1km grid aligned to HCMC boundary (UTM Zone 48N, EPSG:32648)
- [x] Compute B channel: building footprint binary raster (current: intersects→scalar, NEEDS: rasterio rasterize → 256×256)
- [x] Compute G channel: POI Shannon entropy (current: per-cell scalar, NEEDS: coarse sub-cell entropy → bilinear to 256×256)
- [x] Compute R channel: gravity-model accessibility (current: flat bins, NEEDS: pandana exponential decay → interpolate to pixel grid)
- [x] Data quality report: Moran's I, cross-channel correlations, POI stats
- [x] **Snap grid boundary** to 1km grid lines in UTM so every cell is exactly 1000m × 1000m
- [x] **Rework B channel**: rasterio.features.rasterize building polygons → full-city binary canvas → slice 256×256 per cell
- [x] **Rework G channel**: coarse sub-cell entropy (e.g., 20×20 per cell) → bilinear interpolate to 256×256 → full-city canvas → slice per cell
- [x] **Rework R channel**: pandana `type="sum" decay="exponential"` at all network nodes → scipy.interpolate to regular grid → full-city canvas → slice per cell
- [x] **Fuse RGB**: Stack B+G+R → save per-cell PNG (256×256×3)
- [ ] **Fill building gaps**: Re-run Overpass for 2 timed-out tiles (~40K more buildings expected)
- [ ] **Data quality re-report**: Per-channel pixel statistics, visual spot-checks of sample cells

### Step 1.2: Dataset & Config
- [x] **Build geospatial data loader** — extend `datasets/dataset.py` for 256×256×3 RGB PNGs (current loader only handles MNIST). `get_dataset("geospatial")` must work.
- [x] Train/val/test split (80/10/10)
- [x] Create `configs/hcmc_vitality.yaml` for 256×256 3-channel training
- [x] Validate: loader returns correct shapes, values in [0,1] or [-1,1]
- [ ] Dataset characterization: marginal histograms per channel, joint scatter plots, PCA variance explained
- [ ] **Data augmentation** (conditional): Do NOT augment blindly. First find intrinsic dimension (Step 1.3). If dataset is insufficient, then add augmentation as a controlled experiment:
  - [ ] 90°/180°/270° rotations (urban structure is rotation-invariant at this scale)
  - [ ] Horizontal/vertical flips
  - [ ] Channel-independent color jitter
  - [ ] Compare: unaugmented vs augmented generalization window behavior
  - [ ] **Note**: Augmentation via rotation/flip does not add variance — it's a regularization technique. Since we want to *observe* generalization behavior, treat augmentation as an experimental variable, not a default.

### Step 1.3: DDPM Training
- [ ] **GPU**: Start DDPM training (20K-50K steps, logarithmic checkpointing)
- [ ] Monitor: train loss, checkpoint at logarithmic intervals (100, 250, 500, 1K, 2.5K, 5K, 10K, 20K, 40K)
- [ ] **CPU (parallel)**: Pre-classify training images by channel statistics (cluster into vitality tiers)
- [ ] **CPU (parallel)**: Compute intrinsic dimension via PCA/MLE on training set
- [ ] **CPU (parallel)**: Prepare evaluation scripts (see Step 1.4)
- [ ] **EMA consideration**: Evaluate whether EMA (Exponential Moving Average) is needed based on training dynamics. If overfitting oscillations appear on 609 images, implement EMA in `train.py`. (EDM codebase uses EMA; ddpm-win-master currently doesn't.)

### Step 1.4: Analysis & Evaluation
- [ ] **Geospatial data loader**: Build and validate (see Step 1.2) — this is a prerequisite, not afterthought
- [x] **DDIM sampling**: Implement DDIM for fast deterministic sampling (port from `Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Utils/Diffusion.py`)
- [x] **Memorization tracking** (PRIMARY METRIC): Port `compute_fmem.py` (k-NN gap ratio, threshold=0.333) — identifies generalization window. This is the domain-agnostic core metric.
- [ ] **SSCD reproducibility** (PRIMARY METRIC): Port SSCD descriptors from `Diffusion-Model-Generalizability-main/edm/sscd.py` — reproducibility + memorization scores. More robust than FID for non-natural images.
- [ ] ~~FID computation~~: **REMOVED.** FID uses InceptionV3 (ImageNet features) which has zero semantic overlap with urban rasters. The number it produces is uninterpretable for our data. Replaced with k-NN gap + SSCD as primary metrics.
- [ ] **Jacobian rank analysis**: Port from `Diffusion-Model-Generalizability-main/edm/jacobian.py`, adapted for 256×256 using **Hutchinson trace estimator** (randomized rank estimation) or computed at U-Net bottleneck (16×16) instead of full input. Full SVD at 256×256×3 would require ~200K×200K matrix — infeasible.
- [ ] **Intrinsic dimension estimation**: PCA + MLE on training set → identify saturation point
- [ ] Extract U-Net bottleneck features at key checkpoints
- [ ] Distribution comparison: Wasserstein per channel, Moran's I on generated samples, MMD
- [ ] **Baseline comparison**: Compare memorization dynamics with MNIST results (already available from ddpm-win-master testing) and Bonnaire et al. CelebA results for reference

### Step 1.5: Evaluation Scripts to Build
These are ports from research repos into ddpm-win-master:

| Script | Source | Purpose | Priority | Notes |
|--------|--------|---------|----------|-------|
| `datasets/dataset.py` (geo) | New | Geospatial PNG loader | CRITICAL | Current loader only handles MNIST | DONE |
| `sampling/ddim.py` | Why-Diffusion/Utils/Diffusion.py | Fast deterministic sampling | HIGH | Must adapt to ddpm-win-master config system | DONE |
| `evaluation/memorization.py` | Why-Diffusion/Evaluation/compute_fmem.py | k-NN gap ratio (F_mem) | CRITICAL | Primary metric for generalization window | DONE |
| `evaluation/reproducibility.py` | EDM/edm/sscd.py | SSCD reproducibility score | HIGH | Primary metric (replaces FID) |
| `evaluation/jacobian.py` | EDM/edm/jacobian.py | Jacobian rank via Hutchinson | HIGH | Must use Hutchinson estimator, not full SVD |
| `analysis/intrinsic_dim.py` | New | PCA/MLE intrinsic dimension | MEDIUM | |
| `analysis/latent_viz.py` | New | UMAP/t-SNE, clustering | MEDIUM | |

### Step 1.6: Visualization
> High-dimensional embedding space analysis requires visualization at every stage. This is a cross-cutting concern, not a one-time task.

- [ ] **Training dynamics**: Loss curves per checkpoint, sample grid at each log-checkpoint
- [ ] **Channel analysis**: Per-channel histogram comparison (real vs generated), spatial autocorrelation (Moran's I) on generated samples
- [ ] **Latent space**: UMAP/t-SNE of U-Net bottleneck features, colored by channel statistics and cell origin
- [ ] **Generalization window**: Plot memorization fraction vs training steps (the core thesis figure)
- [ ] **Jacobian rank evolution**: Rank estimate vs training steps (validates low-rank manifold learning)
- [ ] **Sample quality**: Side-by-side real vs generated cells at key checkpoints (early, mid, late training)
- [ ] **Spatial structure**: Moran's I comparison between real and generated rasters at each checkpoint

---

## Step 2: Geospatial Diffusion Model (Vector Embedding + Decode)

> **Format**: Buildings + roads + POIs → specialized encoders → unified 512-dim embedding → diffusion → structured decoders → synthetic geospatial data.
> **Key advantage**: Can decode back to original data types. Can manipulate physical inputs and observe vitality changes.
> **Scope**: Full pipeline (Path C) — custom encoders, fusion, diffusion, decoders, mapping model. PhD-level work.
> **Full research context**: See `STAGE2_RESEARCH_SYNTHESIS.md`

### Step 2.1: Data Quality & Collection

> **Goal**: Ensure sufficient high-quality geospatial data for Stage 2, across multiple cities with varying POI quality. Synthesize human-centric data if unavailable.

- [ ] **HCMC data gap fill**: Re-run Overpass for 2 timed-out tiles (building footprints)
- [ ] **HCMC human-centric data**: Find or synthesize proxy vitality indicators
  - [ ] VIIRS nighttime lights (free, ~500m resolution, standard vitality proxy)
  - [ ] Mobile phone data (Viettel, if available)
  - [ ] Social media check-in density
  - [ ] Google Places ratings/reviews density
- [ ] **Singapore pipeline**: Same code, different city (medium POI quality)
  - [ ] Building footprints from OSM
  - [ ] POI data (higher quality than HCMC)
  - [ ] Street network (pedestrian)
  - [ ] Human-centric data (higher availability)
- [ ] **London pipeline**: Same code, different city (good POI quality)
  - [ ] Building footprints from OS OpenData
  - [ ] POI data (excellent quality)
  - [ ] Street network (comprehensive)
  - [ ] Human-centric data (ONS, TfL, extensive open data)
- [ ] **Data quality comparison report**: Side-by-side POI count, category coverage, building count per km²
- [ ] **Pair data**: Create paired datasets (physical config → vitality proxy) for mapping model training

### Step 2.2: Geospatial Diffusion Model

> **Goal**: Build encoder-fusion-diffusion-decoder pipeline that operates on vector embeddings.

#### 2.2.1: Encoders
- [ ] **Building encoder**: Poly2Vec-style — polygon geometries → 256-dim vectors
  - [ ] Input: building footprint polygons + attributes (type, area, height if available)
  - [ ] Architecture: Fourier-based encoding preserving topology and directionality
  - [ ] Fallback: rasterize → ResNet18 (RegionDCL approach, already proven)
- [ ] **Road encoder**: HiFiNet-style GNN — graph structure → 256-dim vectors
  - [ ] Input: street network graph (nodes, edges, lengths, connectivity)
  - [ ] Architecture: hierarchical frequency-decomposition GNN
  - [ ] Fallback: simpler GCN (GraphSAGE) on dual graph representation
- [ ] **POI encoder**: DCA-style — categorized points → 256-dim vectors
  - [ ] Input: POI locations + categories + counts per cell
  - [ ] Architecture: distance-decay GAT + type aggregator
  - [ ] Fallback: category embedding + spatial KDE features
- [ ] **Fusion module**: Transformer cross-attention → 512-dim unified embedding
  - [ ] Input: three 256-dim vectors per cell
  - [ ] Architecture: multi-head self-attention with positional encoding
  - [ ] Output: single 512-dim embedding per 1km cell

#### 2.2.2: Diffusion Model
- [ ] **Latent diffusion on embeddings**: TabDDPM-style diffusion on 512-dim vectors
  - [ ] Gaussian diffusion for continuous features
  - [ ] Much faster than image diffusion (512-dim vs 256×256×3)
  - [ ] Logarithmic checkpointing (reuse ddpm-win-master infrastructure)
- [ ] **Alternative**: Direct discrete diffusion (DiGress-style) if continuous relaxation fails
- [ ] **Training**: Same protocol as Step 1 (20K-50K steps, log checkpoints)
- [ ] **Jacobian rank**: Use Hutchinson trace estimator (same approach as Step 1.4)

#### 2.2.3: Decoders
- [ ] **Building decoder**: Graph decoder → polygon reconstruction
  - [ ] Input: 256-dim building embedding
  - [ ] Output: set of building footprint polygons
  - [ ] Architecture: graph attention decoder → vertex sequence → polygon
  - [ ] Validation: round-trip encode → decode preserves building structure
- [ ] **Road decoder**: GNN decoder → adjacency matrix → network
  - [ ] Input: 256-dim road embedding
  - [ ] Output: street network graph (nodes, edges)
  - [ ] Architecture: graph generator (multi-block construction, GRAPHARM-style)
- [ ] **POI decoder**: Spatial clustering → point locations + categories
  - [ ] Input: 256-dim POI embedding
  - [ ] Output: set of (location, category) pairs
  - [ ] Architecture: conditional point generation + category classifier

#### 2.2.4: Sampling & Analysis
- [ ] **Sample from diffusion model**: Generate synthetic embeddings → decode to geospatial data
- [ ] **Sample quality metrics**:
  - [ ] Building: polygon count, total area, footprint distribution per cell
  - [ ] Road: edge count, connectivity, average degree, total length
  - [ ] POI: point count, category distribution, spatial clustering
- [ ] **Compare synthetic vs real**: distribution matching, spatial statistics
- [ ] **Generalization window**: track structural feature stability across checkpoints

### Step 2.3: Vitality Mapping & Evaluation

> **Goal**: Train a model that maps physical environment configurations to human-centric vitality indicators. Use it to evaluate synthetic configurations from Step 2.2.

#### 2.3.1: Mapping Model
- [ ] **Define vitality indicators** (demand-side):
  - [ ] Proxy 1: VIIRS nighttime lights intensity (free, standard activity proxy)
  - [ ] Proxy 2: POI visit density (from check-in/social media data)
  - [ ] Proxy 3: Pedestrian flow estimates (from space syntax + land use)
  - [ ] Proxy 4: Composite vitality index (5D framework: density, diversity, design, transit, accessibility)
- [ ] **Train mapping model**: physical embedding → vitality score
  - [ ] Architecture: MLP or gradient boosting (XGBoost)
  - [ ] Input: 512-dim unified embedding (from fusion module)
  - [ ] Output: vitality score(s)
  - [ ] Training data: real paired observations (physical config → measured vitality)
- [ ] **Validate mapping model**: R², SHAP values, cross-city generalization

#### 2.3.2: Intervention Experiments
- [ ] **Density manipulation**: Systematically vary building density in embedding space → observe vitality prediction
- [ ] **Diversity manipulation**: Vary POI category diversity → observe vitality
- [ ] **Accessibility manipulation**: Vary road connectivity → observe vitality
- [ ] **Threshold detection**: Compare model-predicted thresholds with empirical literature
  - [ ] Does the model reproduce POI density threshold at ~4,000/km²?
  - [ ] Does the model reproduce diversity threshold at ~1.5?
  - [ ] Does the model reproduce accessibility diminishing returns?
- [ ] **If thresholds match**: strong evidence diffusion model learned real urban structure
- [ ] **If thresholds differ**: investigate what the model learned instead — potentially new findings

#### 2.3.3: Comparison with Step 1
- [ ] Do Step 1 (image) and Step 2 (vector) identify the same generalization window?
- [ ] Does Step 2 reveal structure that Step 1 misses?
- [ ] Which approach produces more interpretable results?

### Step 2.4: Visualization (Vector Space)
- [ ] **Embedding space**: UMAP/t-SNE of 512-dim unified embeddings, colored by city, building density, POI diversity
- [ ] **Encoder representations**: t-SNE of each encoder's output (256-dim) to verify meaningful clustering
- [ ] **Decoded samples**: Visual comparison of real vs decoded building footprints, road networks, POI distributions
- [ ] **Mapping model**: SHAP value visualization showing which physical features drive vitality predictions
- [ ] **Intervention effects**: Heatmaps showing how vitality changes as physical features are swept through their range
- [ ] **Cross-city**: Side-by-side embedding visualizations for HCMC, Singapore, London

---

## Step 3: Three-City Comparison (Data Quality Gradient)

- [ ] Singapore data pipeline + training
- [ ] London data pipeline + training
- [ ] Side-by-side data quality reports (POI count, category coverage, building density)
- [ ] Compare latent space structures across cities
- [ ] Analyze: does POI quality affect what the DDPM learns?

## Step 4: Challenge WorldMove

- [ ] Locate WorldMove's raw input features (38-dim location profiles)
- [ ] Train our DDPM on their input features for a city they validated
- [ ] Compare our learned manifold structure with their 8-dim autoencoder embedding
- [ ] If they differ → WorldMove's compression loses structural information
- [ ] If they match → WorldMove's embedding is independently validated

## Step 5: Global South/North Structural Comparison

- [ ] Systematically add cities from both Global South and North
- [ ] Train separate DDPMs per group
- [ ] Compare generalization windows and learned manifold structures
- [ ] Test: is the "essence" of vitality universal or context-dependent?

---

## Parallel Execution Strategy

```
Step 1 (Current):
  [1.0 Pre-flight]──→[1.1 Pipeline]──→[1.2 Dataset]──→[1.3 Training + 1.4 Analysis (parallel)]
  GPU: DDPM training (20K-50K steps)
  CPU: eval scripts, intrinsic dim, pre-classification

Step 2 (Future):
  [2.1 Data Collection]──→[2.2 Geospatial DDPM]──→[2.3 Mapping + Evaluation]
  2.1 runs in parallel with late Step 1
  2.2 encoders can be prototyped during Step 1 training
  2.3 starts once 2.2 produces first samples

Step 2 internal parallelism:
  2.2.1 Encoders (3 encoders in parallel)
  2.2.3 Decoders (3 decoders in parallel, after encoders work)
  2.3.2 Intervention experiments (after mapping model trained)
```

---

## Key Technical Notes

### U-Net Architecture for 256×256
- Current: `channel_mults = [1, 2, 2, 2]`, `is_attn = [False, False, False, False]` (designed for 28×28 MNIST)
- Required: `channel_mults = [1, 2, 2, 2, 4]`, `is_attn = [False, False, False, False, True]` (5 levels, attention at bottleneck only)
- Memory estimate at 256×256, batch=4, channels=64: ~6-8 GB (comfortable on RTX 3090)
- Fallback: reduce `model_channels` to 32-48 if OOM

### Jacobian Rank at 256×256
- Full SVD at 256×256×3: ~200K×200K matrix — infeasible
- **Hutchinson trace estimator**: Randomized method for estimating matrix trace/rank without materializing the full matrix. Uses random vectors v and computes v^T · J · v for multiple samples. Much cheaper, gives good rank estimate.
- Alternative: Compute Jacobian at U-Net bottleneck (16×16) instead of full input
- EDM reference code only runs full SVD at 4×4 (MoLRG) or 32×32 (CIFAR-10)

### Metrics Strategy
- **Primary**: k-NN gap ratio (memorization fraction) — domain-agnostic, directly measures generalization window
- **Primary**: SSCD reproducibility score — self-supervised descriptors, robust to non-natural images
- **Removed**: FID — InceptionV3 features meaningless for urban rasters
- **Supplementary**: Per-channel Wasserstein distance, Moran's I, MMD
- **Structural**: Jacobian rank (Hutchinson), intrinsic dimension (PCA/MLE)

---

## Key Documents

| Document | Purpose |
|----------|---------|
| `STAGE2_RESEARCH_SYNTHESIS.md` | Research synthesis: non-linearity evidence, diffusion on non-image data, encoding methods, supply-demand frameworks |
| `INTEGRATION_PLAN.md` | Original 10-week integration plan (DDPM features, evaluation metrics) |
| `RESEARCH_METHODOLOGY_DESIGN.md` | Methodological rationale: discrete vs continuous, schedule choices |
| `discussion/ch2-ism-proposal.md` | ISM proposal: DDPMs as epistemic probes for manifold learning |
| `discussion/DDPM-experiment-design-framework.md` | Original experiment design (District 1 vs District 7) |
| `progress-260409.md` | Brainstorming session: RGB encoding decisions, sprint plan |
| `progress-260410.md` | Channel redesign: B=buildings, R=exponential decay, accessibility survey |
| `references/accessibility_measures_survey.md` | 630-line academic survey on accessibility measures |
