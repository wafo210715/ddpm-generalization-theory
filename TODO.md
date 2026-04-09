# TODO — DDPM Urban Vitality Sprint

## CRITICAL PRINCIPLE: Parallelize Wherever Possible

> **Do NOT run tasks sequentially when they can overlap.** Example: once intrinsic dimension is identified and training starts, begin dataset pre-classification and analysis in parallel. GPU trains while CPU analyzes.

---

## Phase 1: HCMC Data Pipeline & DDPM Demo (4-Day Sprint)

### Day 1: Data Pipeline
- [ ] Install dependencies in ddpm-win-master venv (osmnx, pandana, geopandas, shapely, rasterio, pandas, etc.)
- [ ] Download WorldPop 100m unconstrained population for HCMC region
- [ ] Extract OSM POI data for HCMC administrative boundary (Overpass API)
- [ ] Download OSM street network for HCMC (OSMnx, pedestrian mode)
- [ ] Build 1km grid aligned to HCMC boundary
- [ ] Compute B channel: WorldPop → aggregate to 1km → normalize [0,1]
- [ ] Compute G channel: POI Shannon entropy per 1km cell → normalize [0,1]
- [ ] Compute R channel: Gravity-model accessibility via pandana → normalize [0,1]
- [ ] Data quality report: POI count per cell, zero-POI %, coverage maps, Moran's I
- [ ] Visualize each channel as heatmap

### Day 2: Patch Extraction & Dataset Prep
- [ ] Extract 32x32 overlapping patches (stride=4, reflect-pad edges) from 3-channel grid
- [ ] Split: 80% train / 10% val / 10% test
- [ ] Dataset characterization: marginal histograms, joint scatter plots, PCA variance explained
- [ ] Create PyTorch dataset loader (extend `datasets/dataset.py` for urban patches)
- [ ] Create `configs/hcmc_vitality.yaml` for 32x32 3-channel training
- [ ] Validate: loader returns correct shapes, values in [0,1] or [-1,1]

### Day 3: Training + Parallel Analysis
- [ ] **GPU**: Start DDPM training (20K-50K steps, logarithmic checkpointing)
- [ ] **CPU (parallel)**: Pre-classify training patches by channel statistics (cluster into vitality tiers)
- [ ] **CPU (parallel)**: Compute intrinsic dimension via PCA/MLE on training set
- [ ] **CPU (parallel)**: Prepare evaluation scripts (FID, F_mem, latent rank extraction)
- [ ] Monitor: train loss, checkpoint at logarithmic intervals (100, 250, 500, 1K, 2.5K, 5K, 10K, 20K, 40K)

### Day 4: Analysis & Presentation
- [ ] Extract U-Net bottleneck features at key checkpoints
- [ ] UMAP/t-SNE visualization colored by channel stats and patch origin
- [ ] K-means clustering on latent features, silhouette score
- [ ] Generate samples via DDIM from final checkpoint
- [ ] Distribution comparison: Wasserstein per channel, Moran's I, MMD
- [ ] Prepare honest progress report (slides or document)

---

## Phase 2: Three-City Comparison (after sprint)
- [ ] Singapore data pipeline (same code, different city)
- [ ] London data pipeline (same code, different city)
- [ ] Train separate DDPMs per city
- [ ] Side-by-side data quality reports
- [ ] Compare latent space structures across cities
- [ ] Analyze: does POI quality affect what the DDPM learns?

## Phase 3: Challenge WorldMove (future)
- [ ] Locate WorldMove's raw input features (38-dim location profiles)
- [ ] Train our DDPM on their input features for a city they validated (e.g., Shanghai)
- [ ] Compare our learned manifold structure with their 8-dim autoencoder embedding
- [ ] If they differ → WorldMove's compression loses structural information
- [ ] If they match → WorldMove's embedding is independently validated

## Phase 4: Global South/North Structural Comparison (future)
- [ ] Systematically add cities from both Global South and North
- [ ] Train separate DDPMs per group
- [ ] Compare generalization windows and learned manifold structures
- [ ] Test: is the "essence" of vitality universal or context-dependent?

---

## Reference: Parallel Execution Strategy

```
Timeline (conceptual):

Day 1-2:  [=== Data Pipeline ===]
Day 2-3:      [=== Patch Extraction ===]
Day 3:              [==== DDPM Training (GPU) ====]
Day 3:              [Pre-classify (CPU)] [Eval scripts (CPU)]
Day 3-4:                  [== Latent Analysis ==]
Day 4:                          [== Presentation ==]
```

Key overlaps:
- **Patch extraction** can start as soon as first channel grid is ready (don't wait for all 3)
- **Training** starts once dataset loader is validated
- **Pre-classification & eval scripts** run on CPU while GPU trains
- **Latent analysis** begins as soon as first checkpoint is saved (don't wait for training to finish)
