# Literature Review Complete: Status Summary

## Section 2.1 — From Jacobs to Quantification: The Conceptual Evolution of Urban Vitality
**File**: `section-2-1-text.md` | **References**: `reference-list-intro-s21.md`
- 4 paragraphs + Table 1
- P1: Doan & Zhang evolution narrative with Jacobs' four generators
- P2: Supply-demand framing, big data proxy traditions
- P3: Three conflation cases (Li 2021 POI↔POI, Li 2022 SVI circularity, Pan 2021 PLS-SEM) + NTL endogeneity + Williams ghost cities
- P4: Summation — the conflation is sustained by statistical architectures

## Section 2.2 — The Limits of the "Correlationist" Paradigm in Urban Vitality
**File**: `section-2-2-text.md` | **References**: `reference-list-section-2-2.md`
- 4 paragraphs
- P1: Data proxies + conceptual reductionism + 2.1 callback
- P2: Three limitations strengthened with new evidence:
  - Additivity: Doan & Zhang + Wu et al. + **Tan et al. 2026** (RF+SHAP+PDP+ISM+BN)
  - Context stripping: Jiang et al. + Li et al. + **Collins et al. 2024** (gender-spatial spillover)
  - Threshold effects: Doan & Zhang + **Tan et al. 2026** (FAR >0.25, food services oversaturation)
- P3: Epistemological dissonance with 2.1
- P4: Connectionist vs symbolic-correlationist (Huang 2025)

## Section 2.3 — Diffusion Models: From Image Synthesis to Manifold Learning
**File**: `section-2-3-text.md` | **References**: `reference-list-section-2-3.md`
- 5 paragraphs
- P1: DDPM mechanism + Manifold Hypothesis + generalization evidence (Wang, Kadkhodaie, Zhang)
- P2: Property 1 — curse of dimensionality + subspace clustering
- P3: Property 2 — generalization window + spectral bias + dataset caveats
- P4: Methodological opening for urban analysis — probing through inputs/outputs
- P5: Current urban applications miss the point (Jiang 2024 systematic review: 120 papers, all generator-only)

## Section 2.4 — Bridging the Gap: A Generative Epistemology for Urban Vitality
**File**: `section-2-4-text.md` | **References**: `reference-list-section-2-4.md`
- 3 paragraphs (simplified from current draft's 3 rambling paragraphs)
- P1: "Structure-preserving compressor" — connects 2.3 theory to urban analysis
- P2: RGB channel synthesis — the representational solution (user's own contribution)
- P3: Single postulate — generalization window as epistemic probe + scope (supply-side only)

---

## Key Changes from Current Draft (0414)

| Section | Major changes |
|---------|--------------|
| 2.1 | Restructured around Doan & Zhang narrative; three specific conflation cases; Table 1 added |
| 2.2 | Added Tan 2026 (additivity + thresholds), Collins 2024 (spatial spillover); restored connectionist reasoning text |
| 2.3 | Added Jiang 2024 systematic review; GlobalMapper/COHO; fixed "latent space" → "learned representation"; added methodology preview in P4 |
| 2.4 | **Removed** conditional diffusion paragraph; **removed** second postulate; **removed** wrong RGB citation; fixed terminology; added scope clarification |

## Total References in Literature Review
- Section 2.1: 15 references (14 in folder + 1 new: Smolensky 1988 for 2.2 P4 footnote)
- Section 2.2: 17 references (all in folder) + Smolensky 1988
- Section 2.3: 18 references (all in folder)
- Section 2.4: 7 references (all in folder, all reused from earlier sections)
- **New PDFs needed**: None — all references already in `relevant/references/`
- **Non-PDF reference**: Smolensky (1988) — user to add to Zotero

---

## Output Files

| File | Content |
|------|---------|
| `section-2-1-text.md` | Full prose |
| `section-2-2-text.md` | Full prose (revised with Tan 2026 + Collins 2024) |
| `section-2-3-text.md` | Full prose |
| `section-2-4-text.md` | Full prose |
| `reference-list-intro-s21.md` | References for Intro + 2.1 |
| `reference-list-section-2-2.md` | References for 2.2 |
| `reference-list-section-2-3.md` | References for 2.3 |
| `reference-list-section-2-4.md` | References for 2.4 |
| `section-2-1-revised.md` | 2.1 logic flow |
| `section-2-2-logic-flow.md` | 2.2 logic flow |
| `section-2-3-logic-flow.md` | 2.3 logic flow |
| `section-2-4-logic-flow.md` | 2.4 logic flow |
