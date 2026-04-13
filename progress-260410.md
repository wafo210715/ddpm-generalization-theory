# Progress Report — 2026-04-10

**Session type**: RegionDCL analysis + accessibility methodology research + channel redesign planning
**Outcome**: Channel redesign finalized with academic rationale; accessibility survey completed; no code execution

---

## Discussion Flow

### 1. RegionDCL Repository Review

User asked to evaluate [RegionDCL](https://github.com/LightChaser666/RegionDCL) for potential reuse.

**RegionDCL** — "Urban Region Representation Learning with OpenStreetMap Building Footprints"

**Key finding**: RegionDCL is a **contrastive learning** framework (CLIP family), NOT a diffusion model. It uses InfoNCE loss, SimCLR pretraining, and adaptive triplet loss — fundamentally different from our DDPM.

**Components analyzed**:
- `data_util/preprocess.py`: OSM building polygons → 224×224 binary rasterized images using `rasterio`
- `model/resnet.py`: SimCLR-pretrained ResNet18 → 512-dim building features
- `model/biased_attention.py`: Distance-biased self-attention with `log((max_dist^1.5 + 1) / (dist^1.5 + 1))`
- `model/adaptive_triplet.py`: Triplet loss with adaptive margin = 100 × Wasserstein distance
- `model/trainer.py`: InfoNCE loss with KL-divergence similarity, temperature=0.05
- `experiment/evaluator.py`: Land use (MLP) + population density (Random Forest) downstream tasks

**Relevance to our project**:
- Data pipeline (OSM → raster) directly reusable as skeleton
- Evaluation framework (downstream tasks) could evaluate DDPM-generated patches
- Distance-biased attention formula worth noting for future spatial conditioning
- `wasserstein_distance()` function could be reused for distribution comparison
- **Core methods (contrastive learning) are architecturally incompatible with DDPM**

### 2. CLIP vs DDPM — Architectural Clarification

User asked about the relationship between CLIP and DDPM (noting Stable Diffusion uses CLIP).

| Dimension | CLIP / RegionDCL | DDPM |
|---|---|---|
| Learning paradigm | Contrastive (alignment) | Iterative denoising (generation) |
| Loss | InfoNCE / Triplet | MSE noise prediction |
| Output | Embeddings | New samples |
| Generative? | No | Yes |

**Stable Diffusion** = CLIP text encoder (frozen, conditioning) + UNet (diffusion denoiser) + VAE (image codec). Only the UNet is diffusion. CLIP is just the language interface.

### 3. Reviewing Existing Computed Results

User pointed out that the data pipeline (`data-processing/`) already had computed results from a previous session. Reviewed:

**Existing outputs** (`data-processing/data/hcmc/reports/`):
- `hcmc_channel_heatmaps.png` — B/G/R composite
- `hcmc_accessibility.png` — gravity model heatmap
- `hcmc_poi_entropy.png` — Shannon entropy heatmap
- `hcmc_population.png` — WorldPop heatmap
- `data_quality_report.txt` — full statistics

**Key data quality metrics**:
- Grid: 22×30, 1km resolution, 713 cells (36.4% land coverage)
- 16,515 POIs across 131 categories
- Moran's I: B=0.72, G=0.68, R=0.76 (all highly significant)
- Cross-channel correlations: B-G=0.64, B-R=0.52, G-R=0.49 (moderate, not collinear)

**Discrepancy found**: The brainstorming session (progress-260409.md) designed continuous exponential decay (`exp(-β × d_ij)`), but the implemented pipeline used stepped decay (flat counts at 500/1000/1500/2000m with halving weights). No documentation existed explaining this deviation.

### 4. Deep Research: Accessibility Measures in Urban Science

User requested a comprehensive academic survey on accessibility measures, with two framing questions:
1. Should accessibility be road-network-based?
2. Is replacing OD data with accessibility metrics solid?

**Research conducted** and saved to `references/accessibility_measures_survey.md` — a 630-line survey covering:

**Part 1 — Taxonomy**: 7 accessibility measure families with formulas, parameters, strengths, weaknesses:
- Gravity-based (Hansen 1959), Cumulative opportunity, 2SFCA (Luo & Wang 2003), Radiation model (Simini et al. 2012), Betweenness centrality, Utility-based

**Part 2 — OD vs Accessibility**: OD = revealed behavior (actual trips), Accessibility = potential for interaction. Key insight: **they measure different things, not better/worse versions of the same thing.** Levinson & Wu (2020) argue accessibility is the more fundamental concept.

**Part 3 — Network vs Euclidean**: Network distance strongly recommended for pedestrian accessibility. Apparicio et al. (2008, 784 citations) and Sevtsuk & Mekonnen (2012, 548 citations) show Euclidean systematically misestimates accessibility, especially near barriers.

**Part 4 — Gravity Model Parameters**: β ranges by context. Exponential decay: 0.1–0.5/km for urban, 0.1–0.3 for pedestrian. Calibration methods: PPML preferred over log-log OLS (Santos Silva & Tenreyro 2006).

**Part 5 — Accessibility & Vitality**: Mixed-use accessibility correlates most strongly with vitality. Walking scale (400–800m) most appropriate. POI-based destination weights capture diversity Jacobs emphasized.

**Part 6 — Recent Advances (2020–2026)**: Levinson & Wu (2020) general access theory, GNN-based accessibility, Global South studies, temporal accessibility.

### 5. Critical Conceptual Distinction: Supply Side vs Demand Side

User articulated a key insight that reframes the entire research:

> "OD is human-centric, while accessibility is focusing on the physical setting — built environment. There could be two urban vitality: one derived from built environment density, accessibility and diversity. Another based on people, population density, OD."

This led to a clear positioning:

| | **Built Environment** (supply) | **Human Behavior** (demand) |
|---|---|---|
| Density | Building footprint density | Population density, pedestrian flow |
| Diversity | POI functional diversity (Shannon) | Activity diversity (social media, street view) |
| Accessibility | Network reachability to POIs | OD — where people actually go |

**Conclusion**: We are NOT substituting accessibility for OD. We are studying a fundamentally different dimension. OD data is not a compromise — it's the wrong measure for our research question. Our research studies the **built environment conditions that enable vitality**, not the human behavior that constitutes vitality.

**Implication for research question**: Reframed from "DDPM discovers the structure of urban vitality" to "DDPM discovers the generative manifold of urban built environment configurations — the physical preconditions for vitality."

### 6. Channel Redesign Decisions

Three channel changes decided, each with full academic rationale:

**B Channel: Building Footprint Coverage (replace WorldPop)**
- Why: WorldPop is human-centric (where people sleep), not built environment. Building footprints directly measure physical fabric.
- How: OSM building polygons → area fraction per 1km cell → grayscale
- Future: At finer resolution (500m), becomes naturally binary

**R Channel: Continuous Exponential Decay (fix implementation)**
- Why: Stepped decay has artificial discontinuities. Continuous is the theoretical standard (Hansen 1959, Wilson 1971).
- β = 0.3/km (confirmed by user). At this value: 500m→0.86, 1km→0.74, 2km→0.55, 3km→0.41.
- Soft cutoff at 3km (3/β), beyond which contribution is <5%.
- POI count as mass (W_j=1) — no calibration data available, avoids conflating with G channel.

**G Channel: POI Shannon Entropy (keep, document limitations)**
- No algorithmic change.
- Limitation: loses semantic content, sensitive to category granularity.
- Future work: direct POI encoding for semantic awareness.

**Grid Resolution: Keep 1km for demo**
- User decided: 1km works for demo, 500m is future work.
- Plan documented for generalizing to other resolutions later.

### 7. Research Positioning: Jacobs' Four Conditions

The three channels map to three of Jacobs' (1961) four conditions:

| Channel | Jacobs' Condition | What it measures |
|---------|-------------------|-----------------|
| B (Building) | Concentration | Physical fabric density |
| G (POI entropy) | Mixed primary uses | Functional diversity |
| R (Accessibility) | Small blocks | Spatial structure / connectivity |
| — | Aged buildings | Not available from OSM (acknowledged limitation) |

---

## Deliverables Created

| File | Purpose |
|------|---------|
| `references/RegionDCL/` | Cloned repo (codebase analysis) |
| `references/accessibility_measures_survey.md` | 630-line academic survey with citations |
| `.claude/plans/cozy-honking-marble.md` | Channel redesign plan with rationale |

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RegionDCL reuse | Data pipeline skeleton, evaluation framework only | Contrastive learning, architecturally incompatible with DDPM |
| OD vs accessibility | Not a substitution — different research questions | Supply-side (built environment) vs demand-side (human behavior) |
| B channel | Building footprint area fraction (replace WorldPop) | Built environment measure, not human-centric |
| R channel | Continuous exponential decay, β=0.3/km | Theoretical standard, matches original brainstorming |
| G channel | Shannon entropy (unchanged) | Best available proxy for functional diversity |
| Grid resolution | Keep 1km for demo, 500m future work | Validate pipeline before adding complexity |
| Accessibility mass | POI count (W_j=1) | No calibration data; avoids conflating with G channel |

## What's NOT Done Yet

- [ ] Implement channel redesign in `build_hcmc_pipeline.py`
- [ ] Extract OSM building footprints for HCMC
- [ ] Recompute R channel with continuous exponential decay
- [ ] Regenerate quality report and visualizations
- [ ] Test different grid resolutions (500m)
- [ ] Write methodology chapter section with academic citations

---

## Session Metadata
- Duration: Several hours of research, analysis, and brainstorming
- Outcome: Methodology positioning clarified, channel redesign planned, accessibility survey completed
- Academic references compiled for thesis methodology chapter
