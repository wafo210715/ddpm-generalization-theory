# Progress Report — 2026-04-09

**Session type**: Major brainstorming + planning session (spanned two context windows)
**Outcome**: Complete 4-day sprint plan, all key research decisions finalized, execution not yet started

---

## Discussion Flow

### 1. Problem Statement: Dataset Curation for Urban Vitality DDPM

User opened the session after a month-long gap. The DDPM architecture (`ddpm-win-master`) was ready (tested on MNIST, config-driven, dual RTX 3090 support). The missing piece: **what dataset to train on**.

The core challenge: urban vitality is an abstract concept. An unconditional DDPM takes images as input and learns to generate similar images — there is no text-to-image style input/output pair. We need to encode urban vitality into images the model can learn from.

User referenced two prior documents:
- `discussion/ch2-ism-proposal.md` — ISM proposal positioning DDPMs as "epistemic probes"
- `discussion/DDPM-experiment-design-framework.md` — earlier experiment design (District 1 vs District 7)

User described their thesis evolution: **urban growth → urban development → urban vitality**. Their "change of mind" was moving away from WorldMove dependency toward an independent methodology.

### 2. WorldMove Investigation

User shared the WorldMove GitHub repo. We explored their codebase and three sequential papers:

| Paper | Venue | Method | Key Innovation |
|-------|-------|--------|----------------|
| DeepMobility | PNAS Nexus 2025 | GAN | Bidirectional collaboration (individual ↔ population) |
| CoDiffMob | WWW 2025 | Diffusion | Collaborative noise priors (R²=0.95 correlation) |
| WorldMove | Nature Sci. Data 2026 | Diffusion | Global scaling to 1600+ cities |

**WorldMove pipeline**: 38-dim location features (1 pop + 34 POI + 1 popularity + 2 coords) → autoencoder (38→8 dim) → diffusion model → minimum-distance matching to real mobility data.

**Critical finding**: WorldMove does NOT specifically combat poor POI quality. They acknowledge it and rely on multi-source fusion + autoencoder abstraction. This became a key insight for our positioning.

### 3. Research Positioning (Key Decision)

User chose **independent methodology demo** over WorldMove-dependent study.

> "WorldMove's methodology is by nature the same I used for text-to-city and text-to-park that was in my ch2-ism-proposal.md."

The DDPM is not a generator — it's an **epistemic probe**: a manifold learning instrument that internalizes the low-dimensional structure of urban life. We don't predict mobility; we discover what structure exists.

### 4. Three-City Strategy

User chose a **data quality gradient** as a research feature:
- **HCMC** (poor OSM POI) → start here
- **Singapore** (medium POI quality)
- **London** (good POI quality)

> "If in our research, there is North/South discrepancy, then it may be due to POI quality. If there isn't, then the POI quality has been smoothed out and they are capturing generic pattern of human behaviour."

### 5. RGB Channel Encoding Design

Three brainstorming questions resolved the encoding:

**G channel (Diversity)**: POI Shannon entropy from OSM — standard, computable from open data.

**B channel (Density)**: WorldPop population — open data, global coverage, 100m resolution.

**R channel (Accessibility)**: This was the hard one. User cannot use OD/mobility data (no ground truth available). User expressed frustration: "this is the kind of questions that make me want to switch back to OD mobility data." After deep research, we settled on **gravity-model accessibility**:
```
A_i = Σ_j O_j × exp(-β × d_ij)
where d_ij = shortest network distance (pedestrian), β = 0.001/m
```
Computed via pandana on OSM street networks. Uses only OSM + WorldPop — no proprietary data needed.

### 6. Image Size Problem & Solution

1km per pixel means HCMC (~20km × 20km) gives only a 20×20 image — too small for meaningful DDPM training.

User chose **patch-based approach**: extract 32×32 overlapping patches with stride=4 from the city grid. With reflect-padding at edges, this generates hundreds of overlapping patches with urban spatial context.

### 7. Model & Presentation Setup

- **Model**: Use `ddpm-win-master` as-is (already tested, config-driven)
- **Presentation**: Honest progress report — interim results, not final claims
- **Scope**: User explicitly wanted "more questions first" before committing to execution

### 8. Methodology Deep-Dive

User's most detailed request:
> "How do we calculate the training dataset's distribution, how do we monitor the generalization window, how do we calculate the intrinsic dimension of training dataset and incrementally add the dataset to be trained on different direction, and how do we observe the change of model being trained on different dataset. How do we pre-classify the training dataset so that we know what direction we can manipulate and anticipate. How do we sample from the trained model, and compare the distribution with distribution of training dataset."

This was addressed in the final plan with specific metrics:
- **Dataset distribution**: Marginal histograms, joint scatter plots, PCA variance explained, Moran's I
- **Generalization window**: F_mem (k-NN gap, threshold=0.333), FID, latent rank
- **Intrinsic dimension**: Incremental data addition, FID vs dataset size saturation
- **Pre-classification**: Cluster patches by channel statistics (vitality tiers)
- **Sampling/comparison**: DDIM generation, Wasserstein distance per channel, Moran's I, MMD

### 9. Future Work: Challenging WorldMove

User explicitly requested a future plan:
> "This is our 4 days plan, we need a future plan that challenges WorldMove type of methods."

Four phases defined:
1. **HCMC sprint** (now) — prove methodology works
2. **Three-city comparison** — data quality gradient analysis
3. **Challenge WorldMove** — train our DDPM on their raw 38-dim features, compare manifold with their 8-dim embedding
4. **Global South/North** — systematic structural comparison

### 10. Parallel Execution Principle

User's final instruction before pausing:
> "Do parallel work, for example, we should get the model training after we identified its intrinsic dimension, then while the model is training, we analyze/pre-classify the model in the meantime. We do not proceed until I see a clear plan documented."

This principle is now saved to project memory and embedded in the TODO.

---

## All Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Research positioning | Independent methodology demo | WorldMove uses same black-box approach we already critique |
| First city | HCMC | Poor POI quality is a feature, not a bug |
| R channel | Gravity-model accessibility (OSM) | No OD data available; uses only OSM + WorldPop |
| G channel | POI Shannon entropy (OSM) | Standard diversity measure |
| B channel | WorldPop population density | Open data, global coverage |
| Image size | 32×32 patches, 1km/pixel, stride=4 | Overlapping patches from ~20×20 city grid |
| Model | ddpm-win-master as-is | Already tested on MNIST, config-driven |
| Presentation | Honest progress report | Interim results, not final claims |
| Execution | Parallel GPU/CPU overlap | Maximize throughput on dual RTX 3090 |

## Deliverables Created

| File | Purpose |
|------|---------|
| `.claude/plans/smooth-imagining-eclipse.md` | Complete 4-day sprint plan with technical specs |
| `TODO.md` | Full task breakdown with parallel execution strategy |
| `progress-260409.md` | This file — session summary |
| `.claude/skills/pdf/SKILL.md` | PDF reading skill (installed) |
| `.claude/skills/brainstorming/SKILL.md` | Brainstorming skill (installed) |

## What's NOT Done Yet (Execution)

- [ ] Install data pipeline dependencies (osmnx, pandana, geopandas, etc.)
- [ ] Download WorldPop, OSM POI, OSM network for HCMC
- [ ] Build 1km grids and compute R/G/B channels
- [ ] Extract patches and create PyTorch dataset
- [ ] Configure and run DDPM training
- [ ] Latent space analysis and presentation

---

## Session Metadata
- Spanned two context windows (first ran out of tokens)
- Duration: several hours of discussion
- Outcome: All design decisions finalized, plan written, execution pending user approval
