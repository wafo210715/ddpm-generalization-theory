# Annotated bibliography

Shared by the two reading tasks:

- **vitality**: how the literature measures each factor. Read in the factor
  loops of [`discussion/layer_cards.md`](../discussion/layer_cards.md); handoff
  [`discussion/HANDOFF_factor_loop.md`](../discussion/HANDOFF_factor_loop.md).
- **generalization**: what diffusion-model theory says about what a model learns,
  and when. Read against the two threats in
  [`discussion/HANDOFF_diffusion_reading.md`](../discussion/HANDOFF_diffusion_reading.md).

The papers themselves are local in `references/papers/` (PDF plus a
`pdftotext -layout` text layer), which git ignores. Line references `L#` point
into that text layer. The older extraction in
[`discussion/vitality_provenance.md`](../discussion/vitality_provenance.md) cites
line numbers in text files on the Mac. Where an entry here re-cites a paper, the
new `L#` is the one to trust, and a disagreement with the old row is noted in
the entry.

One entry per paper. A paper used by several loops gets a section per factor,
added when its loop reaches it.

## Index

| paper | tag | loops / threats | file in `references/papers/` | status |
|---|---|---|---|---|
| Jacobs 1961, *The Death and Life of Great American Cities* | vitality | A (concentration), all loops | not in the folder | to find |
| Gómez-Varo et al. 2022 (JANE, Barcelona) | vitality | A, B, C, F, G | `urban-vitality/Gómez-Varo-2022-…` | to read |
| Ye et al. 2018 (Shenzhen, block density and typology, Spacematrix) | vitality | A | `urban-vitality/Ye-2018-…` | to read |
| Xia et al. 2020 (five cities, street blocks) | vitality | A, C, G | `urban-vitality/Xia-2020-Analyzing-…` | to read |
| Wu et al. 2023 (Shenzhen, Spacematrix) | vitality | A, B, C, E | `urban-vitality/Wu-2023-…` | to read |
| Jiang et al. 2024 (Beijing, TAZ) | vitality | A, C | `urban-vitality/Jiang-2024-…` | to read |
| Dogan & Lee 2024 (Seoul, physical complexity) | vitality | A, B, C | `urban-vitality/Dogan-2024-…` | to read |
| Tan et al. 2026 (Shenzhen, RF + SHAP) | vitality | A, B, C, F | `urban-vitality/Tan-2026-…` | to read |
| Doan et al. 2025, non-linear and threshold effects | vitality | A, D | `urban-vitality/Doan-2025-Nonlinear-…` | to read |
| Bonnaire et al. 2025, "Why Diffusion Models Don't Memorize" (arXiv 2505.17638) (W1) | generalization | T1, T2, T3, window | `ddpm-generalization-memorization/Bonnaire et al. - 2025 - …` | to read |
| Yoon et al. 2023, "Diffusion Probabilistic Models Generalize when They Fail to Memorize" (W1) | generalization | T3, Q2 | `ddpm-generalization-memorization/Yoon 等 - …` | to read |
| Wang et al. 2025, "Diffusion Models Learn Low-Dimensional Distributions via Subspace Clustering" (W1) | generalization | T2, Q2, C3–C7 | `ddpm-generalization-memorization/Wang et al. - 2025 - …` | to read |
| Carlini et al. 2023, "Extracting Training Data from Diffusion Models" (W1) | generalization | T3 (skim) | `ddpm-generalization-memorization/Carlini et al. - 2023 - …` | to read |
| Fefferman et al. 2013, "Testing the Manifold Hypothesis" (W1) | generalization | vocabulary (skim) | `ddpm-generalization-memorization/Fefferman et al. - 2013 - …` | to read |
| Whiteley et al., "Statistical exploration of the Manifold Hypothesis" (W1) | generalization | §2.3 only | `ddpm-generalization-memorization/Whiteley et al. - 2025 - …` | to read |
| Kadkhodaie et al. 2024 (ICLR), geometry-adaptive harmonic representations (W2) | generalization | T2 (counterweight) | missing | to find |
| Zhang et al. 2024 (ICML), "The Emergence of Reproducibility and Consistency in Diffusion Models" (W2) | generalization | T2, Q2 | missing | to find |
| Wang & Pehlevan, "An Analytical Theory of Spectral Bias in the Learning Dynamics of Diffusion Models" (W2) | generalization | T2, Q2 | `ddpm-generalization-memorization/Wang and Pehlevan - 2026 - …` | to read |
| Li, Dai & Qu 2024, "Understanding Generalizability of Diffusion Models Requires Rethinking the Hidden Gaussian Structure" (arXiv 2410.24060) (W2 +) | generalization | T2 | missing | to find |
| Wang & Vastola 2024, "The Unreasonable Effectiveness of Gaussian Score Approximation for Diffusion Models and its Applications" (arXiv 2412.09726) (W2 +) | generalization | T2 | missing | to find |
| Kaiser & Kollmann 2026, "Diffusion Models Memorize in Training -- and Generalize in Inference" (W3) | generalization | T3 (top priority) | `ddpm-generalization-memorization/Kaiser and Kollmann - 2026 - …` | to read |
| Gu et al. 2025, "On Memorization in Diffusion Models" (EMM) (W3) | generalization | T3 | `ddpm-generalization-memorization/Gu et al. - 2025 - …` | to read |
| Merger & Goldt 2026, "Local Coverage Governs Memorization in Diffusion Models" (W3) | generalization | T3, Q2 | `ddpm-generalization-memorization/Merger and Goldt - 2026 - …` | to read |
| "The two clocks and the innovation window: When and how generative models learn rules", 2026 (arXiv 2605.10019) (W3 +) | generalization | T1, T3, rules | missing | to find |

Rows cover the papers queued for reading; W1–W3 is the week in the user's diffusion reading order, and **+** marks a paper added from the 23 Sep threats. The diffusion folder's full plan, including W4 and what was dropped, is in `discussion/HANDOFF_diffusion_reading.md`. All 49 vitality PDFs, and which loop each serves, are listed in the factor-loop handoff under "The papers on disk". Add a row here when a paper is queued. Status: *to find* → *to read* → *read* → *annotated*.

## Entry template

```markdown
### Author Year — short title
`tag` · loops / threats · file: `references/papers/<name>.pdf` · read <date>

**Claim.** One or two sentences: what the paper says it shows.

**Setting.** City or data, unit of analysis, sample size, method.

**How it measures <factor>.** (vitality; one block per factor)
| measure | formula | denominator / areal unit | scale | data source | effect found | method | L# |

**What it means for us.** (generalization) Which threat it bears on, and
whether it strengthens it, weakens it, or answers it. The conditions under which
its result holds (model capacity, N, noise level, data type), and how our
setting compares: 3 010 cells, 13 channels, 256 px, a U-Net trained from scratch.

**Critique.** What the measure or result captures, what it hides, and what its
data source biases. Where it disagrees with other papers, or with the older
row in vitality_provenance.md.

**Take.** What we use, test or argue against, and where that lands (card, B8,
argument).
```

---

## Entries

*(none yet)*
