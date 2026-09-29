# Handoff · diffusion reading (the argument's threats)

**Start a session with:**

> Read `discussion/HANDOFF_diffusion_reading.md` and do the next paper: `references/papers/ddpm-generalization-memorization/<file>`.

**Next up:** W1. Bonnaire et al. 2025, then Yoon 2023, then Wang et al. 2025
(subspace). The user is reviewing their own notes on these three in parallel;
each entry should end with a short "compare with the user's notes" line when the
user brings them. *(Update this line at the end of every session.)*

**Deadline:** a supervisor meeting in mid-October, after W3. The questions for
it are below.

This is the second of two reading tasks. The other, more urgent, is
[`HANDOFF_factor_loop.md`](HANDOFF_factor_loop.md). Keep them in separate
sessions.

---

## What this task is

The study trains a DDPM on real city tiles and reads what it learns through the
generalization window: the stretch of training between the model starting to
generate plausible samples and it starting to copy training cells (Bonnaire et
al. 2025). The April proposal ([`ch2-ism-proposal.md`](ch2-ism-proposal.md))
sets this against the "correlationist" paradigm of urban vitality research. The
user has two interests of equal weight: what the model reveals about the city,
and what learning behaviour a diffusion model shows on real urban geospatial
data.

**The argument is parked until this reading is done.** Each session writes one
bibliography entry and updates the threat ledger. It does not change the
pipeline, the analysis plan, or [`analysis_map.md`](analysis_map.md); where a
paper implies a change, write it in the ledger as a proposal for the user.

## The threats

**T1 · The instruments measure only correlation** (raised 23 Sep). Almost every
planned analysis is a first- or second-order statistic:
- each layer's histogram (B10);
- correlations between layers, per pixel and per tile (C1);
- the cell descriptor: mean, spread, zero share, skew, Moran's I, and pairwise
  correlation (C2);
- the same compared on generated cells (E3);
- correlation at a distance, bivariate Moran's I (E3c).

Build a random field with the same histograms and the same correlations,
within and between layers and at every distance: the **correlation twin**, a
Gaussian field with matched marginals. It passes every one of those tests by
construction, so matching them shows nothing a correlationist method couldn't.
On 23 Sep a probe on 40 random cells at 256 px found two things
(`analysis_map.md` §4, `report_descriptor_probe.py`):
- Moran's I at one-pixel range can't tell real tiles from interpolated noise on
  13 of 16 layers;
- the real data already breaks rules that should hold by construction: 21.3 %
  of pixels with a height have no building, and 11.5 % of pixels with entropy
  lack two places in their ring.

**T2 · The model may learn mostly the correlation twin** (raised 23 Sep).
- Li, Dai & Qu (NeurIPS 2024): as a diffusion model moves from memorization to
  generalization, its denoiser becomes more linear, close to the optimal
  denoiser for a Gaussian with the training data's mean and covariance. This
  shows most "when the model's capacity is relatively small compared to the
  training dataset size". When the model is heavily overparameterized, it
  appears early in training.
- Wang & Vastola (TMLR 2024): the Gaussian part dominates at moderate and high
  noise, and is learned early.
- Wang & Pehlevan (in the folder, W2) derive a Gaussian-equivalence theory of
  training. From their abstract: the time for a mode to match its target
  variance scales as τ ∝ λ⁻¹, so "high-variance (coarse) structure is mastered
  orders of magnitude sooner than low-variance (fine) detail". Convolutional
  U-Nets, however, show "rapid near-simultaneous emergence of many modes".

If the window holds mostly the covariance, then "what stays stable in the
window" is correlationist content. The claim then needs what the model adds
beyond it.

**T3 · The memorization measure may not be valid** (from the user's reading
order; the priority for the supervisor meeting). The window's end, τ_mem, is
measured by F_mem: the k-NN gap ratio below 1/3, a sample-level nearest-neighbour
distance (Bonnaire; our E2).
- Kaiser & Kollmann 2026, from their abstract: models "progressively overfit
  the denoising training objective" while "generalizing at the sample level".
  The gap is largest at intermediate noise and "does not translate to
  inference".
- If memorization in the loss and memorization in samples come apart, the
  window may be partly an artefact of which one we measure.
- Gu 2025 proposes EMM, the largest training set at which a model still
  reaches the theoretical (memorizing) optimum.
- Zhang et al. 2026 detect memorization from representations: localized,
  "spiky" for memorization, "balanced" for generalization.

**Related: rules and their clock.** The two-clocks paper (arXiv 2605.10019) gives
a second onset that doesn't depend on F_mem:
- τ_rule: the first step where sample-level rule accuracy reliably exceeds 0.90;
- τ_mem: the first step where the memorized fraction reliably exceeds 0.1.

Its "innovation window" lies between them. It vanishes when rules are complex,
meaning their interaction degree is high, as in parity over G pixels, which
correlation cannot see. Our data has rules of its own: height only on a
building; floor area as the 100 m average of footprint × height ÷ 3 m; entropy
only where the 300 m ring holds two places. So a τ_rule could define "structure
onset" (E4). Caveat: the real data at 256 px already breaks some of them.

## Questions to answer

**For the supervisor meeting**, from `reading-order.docx`, translated:
1. Does F_mem (a sample-level nearest-neighbour distance) survive Kaiser &
   Kollmann's decoupling critique? On curated data (584 images in the pilot,
   3 010 cells in v4), which measure is more valid?
2. Does Merger & Goldt's local-coverage theory predict anything for our
   multi-channel tensor, whose layers each have their own measurement scale, or
   only for natural images?
3. Are Kadkhodaie's adaptive-basis account and Wang & Pehlevan's spectral bias
   complementary or competing? Which should describe the window's content?
4. Is the generalization window a well-defined object, or an artefact of how
   it is measured?
5. Is there a gap in this theory for pixel-space, non-distilled DDPMs? Where is
   this study's theoretical contribution?

**The user's two standing questions** (23 Sep):
- **Q1.** How to analyse relationships between layers before and after
  training, so that the analysis sees more than correlation.
- **Q2.** What stays consistent and what varies through the generalization
  window, and across dataset composition (the C7 variants, or "topological
  data", below).

## Terms

**Topological data** *(proposed by the user, 29 Sep; definition to confirm).*
The user's word for the **minimal dataset** (C6: the smallest set of cells,
chosen by an explicit rule) together with the datasets built from it by adding
or removing typologies of fabric (C7: the "mutants", i.e. baseline, +A, +B, −A,
class-balanced, full). The user learned it from Prof. Hao Zheng. Until the user
says which meaning is intended, keep "minimal set" and "variants" in working
documents. The word already has two established meanings a reader will bring:

- **Mathematics and ML:** topological data analysis, the shape of a
  distribution (its clusters, loops and holes), measured with persistent
  homology. Read this way, adding or removing a typology changes the topology of
  the training distribution. That fits Wang et al.'s union of subspaces,
  Merger & Goldt's local coverage, and C3–C7.
- **Architecture:** topology as connectivity, as opposed to geometry (Zheng's
  own doctoral work is titled "Geometry and Topology"). Read this way,
  topological data means data about how elements connect: adjacency, networks.

Whichever is meant, the paper must define the term on first use and cite where
it comes from.

## Our setting, to compare every paper's conditions against

- **Cells:** N = 3 010 training cells, zoom-15 tiles of Ho Chi Minh City,
  1.2 km across. A curated set: tiles with ≥ 100 buildings.
- **Channels:** 13 grids at 256 px first (512 later), in data space, with no
  latent; plus 3 conditions per cell.
- **Data:** zero-heavy, skewed layers. Three are sharp (footprint, height,
  vegetation); ten are smooth fields interpolated from coarser grids.
- **Model:** a DDPM with a U-Net trained from scratch (`ddpm-win-master`), not
  distilled. Memorization is scored by the k-NN gap ratio below 1/3 (E2), above
  an untrained-model floor (D2).
- **Variants:** C7 cuts the dataset into variants such as the ≥ 1 000-building
  core (938 cells), so N and local coverage vary across runs.

## Reading order

This follows the user's `reading-order.docx` in the papers folder. Rows marked
**+** were added from the 23 Sep threats. Rows marked **missing** aren't in the
folder yet; ask the user for them, and don't download without asking. The year
in a file name is sometimes the version's year, not the year in the plan.

**W1 · Review, the phenomenon, and the foundations (fast)**

| paper | file | depth | why |
|---|---|---|---|
| Bonnaire et al. 2025 | `Bonnaire et al. - 2025 - …` | full, with the user's notes | the window theory and the source of F_mem; the repo's `Why-Diffusion-Models-Don-t-Memorize-main/` is its code |
| Yoon et al. 2023 | `Yoon 等 - …` | full, with the user's notes | the memorization–generalization dichotomy, which can hold per class |
| Wang et al. 2025 (subspace clustering) | `Wang et al. - 2025 - …` | full, with the user's notes | union of low-rank subspaces; minimal N against intrinsic dimension; code in `Diffusion-Model-Generalizability-main/` |
| Carlini et al. 2023 | `Carlini et al. - 2023 - …` | skim, figures | memorization is real: extraction attacks |
| Fefferman et al. 2013 | `Fefferman et al. - 2013 - …` | skim, vocabulary | the manifold hypothesis stated formally |
| Whiteley et al. | `Whiteley et al. - 2025 - …` | §2.3 only | the statistical reference on the manifold hypothesis |

**W2 · Mechanism: why diffusion models generalize (three deep reads, plus T2)**

| paper | file | why |
|---|---|---|
| Kadkhodaie et al. 2024 (ICLR), geometry-adaptive harmonic representations | **missing** | adaptive bases on low-dimensional data explain generalization; the counterweight to T2 |
| Zhang et al. 2024 (Qing Qu's group), reproducibility | **missing** | low-dimensional structure plus denoising dynamics lead to reproducibility; read against Kadkhodaie |
| Wang & Pehlevan, spectral bias | `Wang and Pehlevan - 2026 - …` | high-variance modes are learned first, under a Gaussian-equivalence principle; central to T2 and to "what the window contains" |
| **+** Li, Dai & Qu 2024, arXiv 2410.24060 | **missing** | T2: denoisers grow linear as models generalize |
| **+** Wang & Vastola 2024, arXiv 2412.09726 | **missing** | T2: the Gaussian score dominates at moderate and high noise |
| Biroli et al. 2024 | `Biroli et al. - 2024 - …` | don't reread in full; it is cited as the "structure-preserving compressor" in the user's `ch-intro.tex`, which isn't on this machine |

**W3 · Measurement and validity (three deep reads; the core of the supervisor meeting)**

| paper | file | why |
|---|---|---|
| **Kaiser & Kollmann 2026** | `Kaiser and Kollmann - 2026 - …` | **top priority.** Memorization in the loss versus in samples; whether the F_mem story holds (T3) |
| Gu et al. 2025 | `Gu et al. - 2025 - …` | EMM, the most mature candidate to replace F_mem; first settle whether EMM and F_mem measure the same thing |
| Merger & Goldt 2026 | `Merger and Goldt - 2026 - …` | memorization is local: sparse regions are memorized, dense regions generalize. Our curated 3 010 cells are their regime; it bears on Q2 and topological data |
| **+** two-clocks paper, 2026, arXiv 2605.10019 | **missing** | τ_rule as an onset independent of F_mem; bears on "is the window well-defined" |

**W4 · After the meeting, chosen by what it decides**

| paper | file | why |
|---|---|---|
| Zhang et al. 2026, balanced representation space | `Zhang et al. - 2026 - Generalization …` | representation-based memorization detection; a check to set against F_mem |
| Li et al., representation dynamics via low-dimensional modelling | `Li et al. - 2026 - …` | the theory for the user's "Track 3" latent walk |
| MatterGen chain: Zeni → Juelsholt → Kim | `Zeni et al. - 2024 - …`, `Juelsholt - 2026 - …`, `Kim et al. - …` | abstracts and figures only. A "flagship discovery" that turned out to reproduce training data; a mirror case for the presentation's RQ1 |

**In the folder but not in the plan:** Ventura et al. 2025, *Manifolds, Random
Matrices and Spectral Gaps*. Gaps in the spectrum of the score's Jacobian reveal
sub-manifolds and their dimension, and the generative process passes through
three phases. It bears on C5 and E5 (Jacobian rank). Optional; ask the user
where it goes.

**Dropped or deferred** (per the plan): Zhang et al. 2026 on probability-flow
distance (distillation only); Robinson et al. 2025, *Token embeddings violate
the manifold hypothesis* (the plan calls it "Dey 2025"; drop if time is short);
Tinaz et al. 2026 (sparse autoencoders; only if RQ2 needs interpretability);
Somepalli et al. 2022 (don't reread).

## A paper session, step by step

1. Find the paper in `references/papers/ddpm-generalization-memorization/`.
   Every PDF has a text layer beside it (`<same name>.txt`, 29 Sep). Cite lines
   as `L#`. For a new PDF, make one with `pdftotext -layout`.
2. Update its row in
   [`../references/annotated_bibliography.md`](../references/annotated_bibliography.md)
   (status *read*) and write its entry, tagged *generalization*. Use the
   template there; the "What it means for us" block is the important one.
3. Update the threat ledger below: what the paper adds to T1, T2, T3, the
   supervisor questions, Q1 or Q2, with a pointer to its entry. Say whether it
   **supports**, **weakens** or **answers** each, and under what conditions.
4. Add any concrete test it suggests for our data to the candidate-test list,
   with what it would cost on 3 010 cells × 13 × 256 × 256.
5. Update this file: the **Next up** line, and one line in the progress log.

## Threat ledger

*Filled in paper by paper. One line per finding, with a pointer to the entry.*

### T1 · instruments measure only correlation
- (none yet)

### T2 · the model may learn mostly the correlation twin
- (none yet)

### T3 · the memorization measure may not be valid
- (none yet)

### Rules and τ_rule
- (none yet)

### Supervisor questions 1–5
- (none yet)

### Q1 · analysing relationships between layers before and after training
- (none yet)

### Q2 · what stays consistent or varies through the window and across variants
- (none yet)

### Candidate tests for our data

The first four were proposed on 23 Sep, the last two from the reading order on
29 Sep. The reading should confirm, sharpen or drop each.

| test | answers | what it needs | status |
|---|---|---|---|
| Build the correlation twin (Gaussian field, matched marginals, all auto- and cross-covariances), then compare real, twin and model on every statistic | T1, T2 | real cells; FFT-based spectral fit per layer pair | proposed |
| Tests the twin should fail: by-construction rule validity; mutual information beyond what the correlation implies; footprint morphology (blob sizes, connectivity); distinct types at the same density | T1, Q1 | the B8 store; the real data must pass the rules first | proposed |
| Distance of the trained denoiser from the Gaussian (linear) denoiser, per noise level and checkpoint | T2, Q2 | a way to compute the Gaussian denoiser at 13 × 256² (see Li, Dai & Qu; Wang & Pehlevan) | proposed |
| τ_rule from our rules, set against τ_mem | T2, T3, Q2 | a tolerance taken from real cells; tile-edge margins (21 px for 100 m, 64 px for the 300 m ring) | proposed |
| Train loss against validation loss per noise level, beside F_mem, at every checkpoint | T3 | a held-out validation split (the store already writes splits) | proposed, from Kaiser & Kollmann |
| Memorization per cell against that cell's local coverage (density of training cells around it), and per C7 variant | T3, Q2 | nearest-neighbour density in standardized space; per-cell gap ratios | proposed, from Merger & Goldt |

## Progress log

| date | paper | what changed | next |
|---|---|---|---|
| 2026-09-23 | — | T1 and T2 raised; three papers identified (Li/Dai/Qu, Wang & Vastola, two clocks) | reading list |
| 2026-09-29 | — | handoff written | Bonnaire 2025 |
| 2026-09-29 | — | the user's `reading-order.docx` adopted (W1–W4); T3 and the supervisor questions added; text layers made for all 21 PDFs; five papers missing | W1: Bonnaire 2025 |
