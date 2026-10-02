# Diffusion reading · what to read it against

Papers: `~/Documents/wafo-base/literature/papers/` (flat folder, pdf + txt).

*Corrected 30 Sep 2026.* Old text: "Order: `references/papers/ddpm-generalization-memorization/reading-order.docx` (W1–W4)." Neither that folder nor any `reading-order.docx` exists on this machine (`find` over wafo-base and the repo). The W1–W4 order is lost; the reading order is now set in conversation.

**Three problems for the argument**
- **T1.** Nearly every planned analysis (histograms, correlations, Moran's I) is also reproduced by a Gaussian field with the same correlations, so it can't show learning beyond correlation.
- **T2.** Generalizing diffusion models may learn mostly that Gaussian structure (Li, Dai & Qu 2024; Wang & Vastola 2024; Wang & Pehlevan).
- **T3.** F_mem may not be a valid measure of memorization: loss-level and sample-level memorization come apart (Kaiser & Kollmann 2026).

**Supervisor questions (mid-October)**
1. Does F_mem survive Kaiser & Kollmann on curated data (3 010 cells)?
2. Does Merger & Goldt's local coverage predict anything for a multi-channel tensor?
3. Kadkhodaie's adaptive bases or Wang & Pehlevan's spectral bias: which describes what the window holds?
4. Is the generalization window well-defined, or an artefact of the measure?
5. Where is the gap for pixel-space, non-distilled DDPMs?

**Missing from the folder:** Li/Dai/Qu 2024 (arXiv 2410.24060), Wang & Vastola 2024 (2412.09726), Two Clocks 2026 (2605.10019), Ventura 2025.

*Corrected 30 Sep 2026.* Old text: "Missing from the folder: Kadkhodaie 2024, Zhang 2024 (reproducibility), Li/Dai/Qu 2024 …, Two Clocks 2026 (2605.10019). Ventura 2025 is in the folder but not in the order." Kadkhodaie 2024 and Zhang 2024 are now in the folder (pdf + txt); no Ventura file exists anywhere under wafo-base.
