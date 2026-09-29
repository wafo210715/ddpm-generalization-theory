# ISM Final Report: Paper Framework

## Core RQ

**Can a Denoising Diffusion Probabilistic Model (DDPM) learn the low-dimensional manifold of urban built environment from RGB-encoded geospatial data?**

This is a **methods paper** with **preliminary validation** on HCMC.

---

## Section-by-Section Framework

### 1. Introduction (keep, adjust)

**Current state**: Mostly good. Needs 3 adjustments:
1. Replace 3 RQs with single RQ
2. Tone down "paradigm shift" language → "methodological exploration"
3. Clarify scope: this is Stage 1 (BE supply-side only), not the full generative epistemology

**Logic flow**:
- Big data → quantification of UV → but stuck in correlationist paradigm
- DDPM's potential as epistemic probe (not design tool)
- Our approach: RGB channel synthesis to preserve heterogeneity without labeling
- **Single RQ**: Can DDPM learn meaningful manifold from this data?
- Paper contribution: a methodological framework + preliminary evidence

**Callbacks setup** (these concepts reappear later):
- [C1] "correlationist paradigm" → Section 2.2, Methods, Discussion
- [C2] "structure-preserving compressor" → Section 2.3, Methods 3.2, Experiments
- [C3] "generalization window" → Section 2.3, Experiments 4.1, Discussion
- [C4] "supply-demand" → Section 2.1, Methods 3.1, Discussion/Future Work
- [C5] "RGB channel synthesis" → Section 2.4, Methods 3.2, Experiments
- [C6] "epistemic probe" → woven throughout as framing device

**Visual needed**: None for intro.

---

### 2. Literature Review (mostly keep, targeted adjustments)

#### 2.1 From Jacobs to Quantification (keep)
- **Callback [C4]**: Supply-demand framework introduced here
- **Adjustment**: Add explicit statement that "this paper addresses Stage 1 (supply-side/BE) only; demand-side integration is future work"
- **Note**: The crossed-out text (~~This research repositions...~~) should be removed or rewritten

#### 2.2 The Limits of Correlationist Paradigm (keep)
- **Callback [C1]**: Detailed critique of correlationist methods
- **No major changes needed**

#### 2.3 Diffusion Models: From Image Synthesis to Manifold Learning (minor adjustments)
- **Callback [C3]**: Generalization window theory
- **Adjustment needed**: The inline todo note "Curate your usage of the 'latent space' terminology" → Replace "latent space" with "learned manifold" where referring to DDPM's representation, add footnote distinguishing pixel-space DDPM from latent-space LDM
- **Adjustment**: Strengthen the transition from theory to practice — make explicit that validation on real-world urban data (vs CIFAR-10/CelebA) is non-trivial

#### 2.4 Bridging the Gap (keep, small adjustments)
- **Callback [C2][C5]**: Structure-preserving compressor + RGB synthesis
- **Adjustment**: Clarify that this is the theoretical basis for the proposed methodology, not yet validated findings

**Visuals needed**:
- [V1] **Conceptual diagram**: RGB channel synthesis — how building/road/POI → rasterization → RGB fusion → DDPM input (conceptual, can produce)
- [V2] **UV conceptual evolution timeline**: Jacobs → Lynch → Montgomery → Landry → Big Data quantification (conceptual, nice-to-have)

---

### 3. Methods (rewrite based on 0412 notes)

**Structure** (reorganized for clarity):

#### 3.1 Data Conceptualization
- Supply-demand framework [C4] — BE as supply-side proxy
- Three dimensions: density (building), accessibility (road), diversity (POI)
- Why these three: commonly accepted in UV literature, geospatial availability
- Explicitly state: no pre-computed vitality score, no labeling → preserves heterogeneity [C1 callback]
- **Visual [V3]**: Data conceptualization table (3 dimensions × 3 attributes: data source, encoding, urban meaning)

#### 3.2 Data Preprocessing Pipeline
- 3.2.1 Geospatial rasterization
  - Vector → raster conversion: polygon/polyline/point → grid
  - Cell size vs calculation unit distinction (important methodological note from 0412)
  - Buffer analysis for POI diversity (Shannon index) and road accessibility (gravity decay)
- 3.2.2 RGB channel synthesis [C5]
  - Three B/W images → normalized [0,1] → mapped to [0,255]
  - Why separate channels: preserves raw relationships, forces model to learn mixing logic
  - Each channel is lossy compression of complex geospatial data — but synergies matter, not per-channel fidelity
- 3.2.3 Pre-encoding analysis
  - PCA / intrinsic dimension analysis
  - Typology classification of the dataset
  - **Visual [V4]**: Data pipeline diagram (end-to-end flowchart, conceptual)
  - **Visual [V5]**: Original dataset distribution (PCA scatter, RGB channel histograms) — **PRODUCIBLE if data exists**
  - **Visual [V6]**: Sample RGB fused images from dataset (grid of 8-12 examples) — **PRODUCIBLE**

#### 3.3 Model Architecture and Training
- DDPM architecture choices (U-Net, base channels, image resolution)
- Training configuration (steps, batch size, learning rate)
- The mathematical-theory-to-practice gap: CIFAR-10 is 32×32, our data is higher resolution
- **Visual [V7]**: Architecture diagram (standard DDPM U-Net, conceptual)

#### 3.4 Analysis Framework
- 3.4.1 Generalization window identification [C3]
  - Loss curve analysis (train/val)
  - Plan to use Bonnaire's f-mem metrics (future refinement)
- 3.4.2 Sampling strategy
  - Random sampling for distribution analysis
  - Nearest-neighbor analysis (Zhang et al. 2024 approach)
- 3.4.3 RGB channel decomposition
  - Decompose generated images back to R/G/B channels
  - Compare distributions with original data channels
  - **Visual [V8]**: Analysis pipeline diagram (conceptual)

---

### 4. Experiments (restructure)

**Key reframing**: This is not "results" — it's "preliminary validation" of the methodology.

#### 4.1 Study Area and Data
- HCMC (Ho Chi Minh City)
- Data sources and spatial extent
- Dataset size and characteristics
- **Visual [V9]**: Study area map with data coverage — **PRODUCIBLE**

#### 4.2 Training Dynamics and Generalization Window
- Training curve: 50K (no convergence) → 500K (convergence at ~86K)
- Train vs validation loss
- Where is the generalization window? Preliminary identification
- **Visual [V10]**: Training/validation loss curves — **PRODUCIBLE** (user has this data)
- **Visual [V11]**: Sample progression at different training stages — **PRODUCIBLE**

#### 4.3 Preliminary Sampling Analysis
- Random sampling results — visual comparison with training data
- Nearest-neighbor analysis (if executable)
- RGB channel decomposition of sampled images (if executable)
- **Visual [V12]**: Generated vs training sample comparison grid — **PARTIALLY PRODUCIBLE**
- **Visual [V13]**: RGB channel distribution comparison (original vs generated) — **NEEDS ANALYSIS PIPELINE**
- **Visual [V14]**: Nearest-neighbor pairs — **NEEDS ANALYSIS PIPELINE**

#### 4.4 The Theory-to-Practice Gap
- Mathematical theory (Bonnaire, Wang et al.) validated on simple datasets
- Real-world urban data challenges: resolution, base channels, batch size
- What works, what doesn't, what needs adaptation
- This is the honest contribution of a methods paper

---

### 5. Discussion (new section, replaces current Conclusions)

#### 5.1 Methodological Contribution
- What does the framework offer that correlationist methods don't? [C1 callback]
- RGB channel synthesis as a non-reductionist encoding strategy [C5 callback]
- The model as epistemic probe — what we can and cannot claim [C6 callback]

#### 5.2 Limitations
- Stage 1 only (supply-side) — demand-side is future work [C4 callback]
- Single city validation (HCMC) — multi-city comparison needed
- Analysis pipeline incomplete — generalization window boundary not precisely identified
- Terminology clarification: pixel-space DDPM vs latent-space LDM

#### 5.3 Future Work
- Multi-city comparison (HCMC, Singapore, London)
- Supply-demand joint learning (encoding BE + Human data together)
- Alternative architectures: latent diffusion (LDM), ViT-based (JiT), tabular DDPM
- Refined generalization window analysis with f-mem metrics
- Data composition manipulation experiments (original RQ3)
- Comparison with correlation-based methods (original RQ2)

---

## Visuals Summary

| ID | Description | Type | Status |
|----|-------------|------|--------|
| V1 | RGB channel synthesis diagram | Conceptual | Can produce now |
| V2 | UV conceptual evolution timeline | Conceptual | Nice-to-have |
| V3 | Data conceptualization table | Table | Can produce now |
| V4 | Data pipeline diagram | Conceptual | Can produce now |
| V5 | Original dataset distribution (PCA, histograms) | Data viz | **PRODUCIBLE if data exists** |
| V6 | Sample RGB fused images | Data viz | **PRODUCIBLE** |
| V7 | Architecture diagram | Conceptual | Can produce now |
| V8 | Analysis pipeline diagram | Conceptual | Can produce now |
| V9 | Study area map | Data viz | **PRODUCIBLE** |
| V10 | Training/validation loss curves | Data viz | **PRODUCIBLE** (user has) |
| V11 | Sample progression at training stages | Data viz | **PRODUCIBLE** |
| V12 | Generated vs training comparison | Data viz | **PARTIALLY PRODUCIBLE** |
| V13 | RGB channel distribution comparison | Data viz | **NEEDS ANALYSIS PIPELINE** |
| V14 | Nearest-neighbor pairs | Data viz | **NEEDS ANALYSIS PIPELINE** |

---

## Callback Map (cross-section references)

```
[C1] Correlationist paradigm
     Intro → Sec 2.2 (critique) → Methods 3.1 (why no composite score) → Discussion 5.1

[C2] Structure-preserving compressor
     Intro → Sec 2.3 (theory) → Methods 3.2 (RGB preserves relationships) → Experiments 4.3

[C3] Generalization window
     Sec 2.3 (Bonnaire theory) → Methods 3.4.1 (identification) → Experiments 4.2 (training curves) → Discussion 5.2 (boundary not precise)

[C4] Supply-demand framework
     Sec 2.1 (UV evolution) → Methods 3.1 (BE-only = supply side) → Discussion 5.2 (limitation) → Future Work (joint learning)

[C5] RGB channel synthesis
     Sec 2.4 (theoretical basis) → Methods 3.2.2 (implementation) → Experiments 4.3 (decomposition analysis)

[C6] Epistemic probe
     Intro (framing) → Sec 2.4 (theoretical positioning) → Discussion 5.1 (what we can claim) → Discussion 5.2 (what we cannot)
```
