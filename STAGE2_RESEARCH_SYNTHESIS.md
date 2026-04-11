# Stage 2 Research Synthesis — Geospatial Diffusion for Urban Vitality

**Date**: 2026-04-11
**Status**: Research synthesis complete. **Plan finalized** — see `TODO.md` (authoritative task list) and `.sisyphus/plans/revised-research-plan.md` (plan source of truth).
**Trigger**: Realization that RGB image encoding (Stage 1) has an information-theoretic bottleneck — cannot decode back to original geospatial data. Stage 2 proposes direct vector embedding of heterogeneous geospatial data.
**Scope Decision**: Path C (full PhD pipeline) — custom encoders, fusion, diffusion, decoders, mapping model.
**Key Plan Changes (post-synthesis)**:
- Resolution: 512×512 → **256×256** (OOM risk at 512, urban rasters are low-information-density per pixel)
- FID: **REMOVED** (InceptionV3 features meaningless for urban rasters). Replaced by **k-NN gap + SSCD** as primary metrics.
- Jacobian: Full SVD infeasible at 256×256. Using **Hutchinson trace estimator** or bottleneck-level (16×16) computation.
- Data augmentation: Conditional experiment, not default. Treat as variable to study generalization behavior.
- EMA: Evaluate need based on training dynamics (609 images may show overfitting oscillations).

---

## 1. Core Hypothesis

### Stage 1 (RGB Image Encoding)
- Encode 3 built-environment dimensions as RGB channels → **256×256** images
- Train DDPM on images → sample → analyze pixel patterns
- **Limitation**: Cannot reverse B/G/R channels back to building footprints, road networks, POIs
- Analysis stops at "what combination of three channels works during generalization window"

### Stage 2 (Vector Embedding + Decode)
- Encode buildings, roads, POIs directly as vectors → unified embedding
- Train diffusion on embeddings → sample → **decode back to structured geospatial data**
- Train mapping model: physical environment → human-centric vitality indicators
- **Innovation**: Can manipulate inputs, observe non-linear vitality thresholds, and evaluate synthetic urban configurations
- **Scope**: Path C (full pipeline) — custom encoders, fusion, diffusion, decoders, mapping model. PhD-level work.

### The Supply-Demand Distinction

| | Built Environment (Supply) | Human Behavior (Demand) |
|---|---|---|
| Density | Building footprint density | Population density, pedestrian flow |
| Diversity | POI functional diversity | Activity diversity (social media, street view) |
| Accessibility | Network reachability to POIs | OD flows — where people actually go |

**Our research studies the built environment conditions that enable vitality, not the human behavior that constitutes vitality.** OD data is not a compromise — it's the wrong measure for our research question.

---

## 2. Non-Linearity Hypothesis — Empirical Backing

The relationship between physical built environment and vitality is NON-LINEAR with measurable thresholds:

### Density Thresholds
- POI density: diminishing returns beyond **4,000 facilities/km²** (Jinan, Nature Scientific Reports 2026)
- POI density: activation at 14/km², overload threshold at 100/km² (Beijing 2025)
- Building density: first increasing, then inhibiting (Chongqing 2025)
- Urban density: inverted U, peak at **0.497** (global study)

### Diversity Thresholds
- POI blending degree: suppression-promotion threshold ~**1.5** (Jinan 2026)
- Land use mix: functional diversity >1.5 shows strong positive effects (Guangzhou 2024)
- Optimal POI blending: **1.8–2.2** maximizes vitality (Jinan 2026)

### Accessibility Thresholds
- Openness ratio: beneficial <0.2, inhibitory >0.3 (Jinan 2026)
- Street width: increasing → plateau at 15m → negative >40m (Beijing 2025)
- Gravity model: β=0.3/km, soft cutoff at 3km (3/β)

### Built Environment Explains Vitality
- **R² = 0.514–0.64**: 5D framework explains 51-64% of vitality variance (multiple cities)
- **R² = 0.571**: 3D spatial network explains 57% of pedestrian flow (Hong Kong)
- **R = 0.756**: Satellite morphology predicts OD flows (Imagery2Flow, Nature Comms 2025)

### Jacobs' Four Conditions — Empirically Validated
Across 6+ cities: Italian cities (2016), Seoul (2015), Hong Kong (2022), Qingdao (2022), Quito (2023). All four conditions (mixed use, small blocks, aged buildings, concentration) show significant positive effects on vitality indicators.

**Implication**: A generative model that captures non-linear threshold effects is more scientifically valid than linear correlationist approaches. The diffusion model should implicitly learn these thresholds if trained on sufficient physical environment data.

---

## 3. Can Diffusion Work on Non-Image Data?

### Yes — Multiple Proven Architectures

| Method | Data Type | How It Works | Key Paper |
|--------|-----------|-------------|-----------|
| **TabDDPM** | Tabular | Gaussian + multinomial diffusion on feature vectors | Kotelnikov 2023, ICML |
| **DiGress** | Graphs | Discrete denoising on adjacency + node attributes | Vignac 2023, NeurIPS |
| **LGD** | Graphs | GNN encoder → latent diffusion → graph decoder | 2024 |
| **GlobalMapper** | Urban layouts | CNN encoder → latent → GAT decoder → building graphs | He & Aliaga 2023, ICCV |
| **COHO** | Urban layouts | Graph-based Masked AutoEncoder, 330 cities, 17M buildings | 2024, ECCV |
| **PolyDiff** | 3D meshes | Direct categorical diffusion on triangle vertices | Alliegro 2023 |
| **ControlCity** | Urban morphology | Stable Diffusion + ControlNet + building footprint decoder | 2024 |

### Decoding Strategies

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| **Direct discrete diffusion** | Categorical noise on graph nodes/edges | Strict structural validity (DiGress) |
| **Continuous relaxation + thresholding** | Gaussian diffusion on continuous proxy → round | Simple prototypes (Bit Diffusion) |
| **VAE + structured decoder** | Encode to continuous latent → decode to structure | Urban layouts (GlobalMapper) |
| **Latent diffusion + decoder** | Diffusion in latent space → multi-modal decoder | Flexible, best quality (LGD) |

### Recommended for Our Case: Latent Diffusion + Structured Decoders

```
Multi-modal encoders → unified 512-dim embedding → DDPM → decoders → structured output
```

---

## 4. Geospatial Encoding Methods

### Building Footprints
- **Poly2Vec** (ICLR 2024): Polymorphic Fourier-based encoding, 128-256 dim, preserves topology/directionality
- **RegionDCL** (KDD 2023): Rasterize → ResNet18 → 512-dim features (our current approach in Stage 1)
- **Pix2Poly** (WACV 2024): Transformer for polygon generation from images

### Road Networks
- **HiFiNet** (2025): Hierarchical Frequency-Decomposition GNNs, 256-512 dim
- **HyperRoad** (2023): Dual graph with hypergraph construction, 128-256 dim
- **RFN** (2020): Relational Fusion Networks — GCN for road networks (primal + line graph)

### POIs
- **DCA** (2024): Dual Context Aware Neural Network — distance-decay GAT + type aggregator, 128-256 dim
- **CaLLiPer** (2024): Contrastive Language-Location Pre-training
- **Category embeddings**: POI category embeddings with type hierarchies

### Multi-Modal Fusion
- **UrbanFusion** (2025): Stochastic Multimodal Fusion (SMF), transformer-based, 512-dim
- **Urban2Vec** (2020): Multi-modal with street view + POIs, bag-of-words approach
- **Early/Mid/Late fusion**: Data-level, feature-level, decision-level

### Recommended Encoding Stack
```
Buildings: Poly2Vec-style → 256-dim
Roads: HiFiNet-style GNN → 256-dim
POIs: DCA-style → 256-dim
Fusion: Transformer cross-attention → 512-dim unified embedding
```

---

## 5. Supply-Demand Two-Stage Approach — Academic Precedent

### Is Two-Stage (Physical → Human) Sound?

**Yes — strong academic precedent:**

| Framework | Pipeline | Validation |
|-----------|----------|------------|
| **LUTI models** | Land use allocation → Transport network loading | 700k-3.8M person cities |
| **AI4US** (2025) | Generate → Evaluate → Calibrate | Standard workflow for GenAI in urban science |
| **Imagery2Flow** (2025) | Satellite imagery → Deep learning → OD flow | R=0.756, Nature Comms |
| **MoPeD/MITO** (2023) | PAZ-level demand → Network loading | 15-20% better than single-step |
| **Space Syntax** (2017) | 3D network + land use → Agent model → Pedestrian flows | R²=0.571 |
| **5D Framework** (2025) | Built env features → Vitality index | R²=0.64 |

### The Mapping Model

The mapping model bridges supply → demand. It would be trained on real paired data (physical configuration → observed vitality). Even proxy vitality measures work — the key is having paired supply-demand observations.

**Critical insight**: If the mapping model is trained on real data, and you feed it synthetic physical configurations from the diffusion model, the predicted vitality scores reveal what the diffusion model "thinks" about urban vitality — including the non-linear thresholds it implicitly learned.

---

## 6. Complete Stage 2 Architecture

```
[Physical Data]                    [Human Data]
Building footprints ──┐             Population density
Road network ─────────┤             OD flows
POIs ─────────────────┤             Pedestrian counts
                      │                    │
                      ▼                    │
              [Specialized Encoders]        │
              (Poly2Vec, GNN, DCA)          │
                      │                    │
                      ▼                    │
              [Fusion Module]               │
              (Transformer, 512-dim)         │
                      │                    │
                      ▼                    │
              [DDPM / Latent Diffusion]      │
              learns P(physical config)     │
                      │                    │
                      ▼                    │
              [Sample Embedding]            │
                      │                    │
                      ▼                    │
              [Structured Decoders]         │
              (graph decoder, etc.)         │
                      │                    │
                      ▼                    │
         Sampled buildings, roads, POIs ──→ [Vitality Mapping Model]
                                                │
                                                ▼
                                         Synthetic Vitality
                                         (human-centric indicators)
```

---

## 7. Stage 1 vs Stage 2 Comparison

| Aspect | Stage 1 (RGB) | Stage 2 (Vector) | Why Stage 2 is Better |
|--------|--------------|-----------------|----------------------|
| Data representation | Pre-computed channels | Raw geospatial vectors | No information loss in encoding |
| Decode to original | Impossible | Possible with structured decoders | Can analyze what was generated |
| Non-linearity | Implicit in image patterns | Explicit in mapping model | Can study thresholds directly |
| Data manipulation | Modify pixel values (meaningless) | Modify building density, road connectivity | Meaningful intervention experiments |
| Generalization window | Observe image quality changes | Observe structural feature stability | More interpretable |
| Academic positioning | "Image generation for urban data" | "Generative urban science" | Paradigm shift |

---

## 8. Key Risks

1. **Decoder fidelity**: Decoding embeddings back to realistic building footprints/road networks is hard. Expect imperfect reconstructions.
2. **Mapping model data**: Need paired physical→vitality data. If unavailable for HCMC, use proxy measures or find cities with both supply and demand data.
3. **Embedding dimensionality**: 512-dim may be too high for 609 samples. Need dimensionality reduction (PCA → intrinsic dim).
4. **Stage 1 already answers the core question**: Generalization window and manifold structure are observable in image space. Stage 2 adds interpretability, not a fundamentally different finding.

---

## 9. Key References

### Diffusion on Non-Image Data
- TabDDPM: Kotelnikov et al., "TabDDPM: Modelling Tabular Data with Diffusion Models," ICML 2023
- DiGress: Vignac et al., "DiGress: Discrete Denoising Diffusion for Graph Generation," NeurIPS 2023
- LGD: "Latent Graph Diffusion," 2024, arXiv:2404.11199

### Urban Generation
- GlobalMapper: He & Aliaga, "GlobalMapper: Learning to Map Urban Buildings with Aerial Imagery and Map Data," ICCV 2023
- COHO: "COHO: Cross-Modal Hierarchical Ontology for Urban Layout Generation," ECCV 2024
- ControlCity: Zhou et al., "ControlCity: Multimodal Diffusion for Urban Morphology Synthesis," 2024

### Geospatial Encoding
- Poly2Vec: "Poly2Vec: A Polymorphic Fourier-based Encoding for Points, Polylines, and Polygons," ICLR 2024
- HiFiNet: "Hierarchical Frequency-Decomposition GNNs for Road Networks," 2025
- DCA: "Dual Context Aware Neural Network for POI Representation," 2024
- UrbanFusion: "UrbanFusion: Geo-Foundation Model with Stochastic Multimodal Fusion," 2025

### Non-Linearity & Vitality
- Jinan study: "Nonlinear effects of built environment on urban vitality," Nature Scientific Reports 2026
- Imagery2Flow: "Predicting human mobility from satellite imagery," Nature Communications 2025
- AI4US: "AI4US: Generative AI for Urban Science," 2025
- Jacobs validation: Multiple cities, 2015-2023

### Supply-Demand Models
- MoPeD/MITO: "MoPeD meets MITO: Integrated pedestrian demand modelling," 2023
- LUTI: "Spatial Economic Modeling for Sustainable Planning," 2025
- 5D Framework: "Investigating built environment and urban vitality," Nature Scientific Reports 2025
