# ISM Report — Skeleton Drift Diff (original logic flows vs current text)

**Inputs**: original skeleton = `discussion/logic-flow-260417/` (written
2026-04-17, the writing day, per the 2026-04-17 diary); current text =
`discussion/ISM_report.txt` (committed 2026-04-22, one 1-line change since, so
drift = the 04-17→04-22 revision window). Cross-checked against the
reverse-engineered skeleton (`ism_section_flow.md`, 2026-09-11).

**Headline**: drift is small and almost all of it is *strengthening*. The
current text is the skeleton executed faithfully, plus one major post-skeleton
addition (the three conflation examples) and two skeleton elements that were
cut. Tier implications are listed per item.

---

## Drift items

### D1 · §2.1 three conflation examples — post-skeleton ADDITION (major)

The original `lit-review-logic-flow.md` §2.1 ends at P3 with a 🔍 gap:
"Need 1-2 more examples of studies that explicitly conflate supply and demand
within a single index." The current text's P4 — the three graded examples
(Li (Q.) 2021 POI-vs-POI; X. Li 2022 SVI vehicles; Pan 2021 PLS-SEM latent) —
answers that gap and was written into `section-2-1-revised.md` during the
revision window (diary: "用户反馈修订: 根据用户补充证据重写").

- **Tier impact**: Li (Q.) 2021, X. Li 2022, Pan 2021 are Tier 1 by position —
  confirmed identically by original-skeleton gap, current text, and plan list.
- **Verification status**: all three txt-verified 2026-09-11 (sub-session
  pass); Li 2021 found to hit four anomaly types (T1+T5+T7), stronger than the
  text states.
- The skeleton's accompanying 🔍 ("1-2 more examples") is now over-fulfilled:
  V5–V7 (Qin 2025, Chen/Biljecki 2021, Dogan & Lee 2024) are additional
  verified cases available if §2.1 is ever extended.

### D2 · §2.2 third limitation ("emergence") — CUT

The first-pass lit-review flow had three correlationist limitations
(additivity / context stripping / emergence, each with only 1 citation —
flagged as the weakest part of the review, gaps E1–E3). The revised
`section-2-2-logic-flow.md` and the current text keep **two** limitations;
emergence was dropped rather than evidenced.

- **Tier impact**: the suggested emergence citations (Batty 2013, Portugali
  2012) were never added — nothing to tier. If emergence is ever restored as a
  third limitation, it needs new Tier 1 support.
- Tan 2026's FAR > 0.25 threshold entered via the revised 2-2 flow and is in
  the current text — Tier 1, verified 2026-09-11 (verbatim in txt).

### D3 · §2.4 conditional-diffusion / text-prompt paragraph — CUT

The original 2-4 flow had P2 on text-prompt curation (J. Huang et al. 2021)
with a 🔍 DISCONNECT note: the experiment uses unconditional DDPM, so the
paragraph should be cut or reframed. The current text cut it.

- **Tier impact**: J. Huang 2021 is not in the current references — correctly
  dropped, nothing to tier.

### D4 · §2.2 Collins 2024 — CUT

The revised 2-2 flow cited Collins et al. 2024 for cross-boundary spillover
(gender-differentiated externalities). The current text's spatial-axis
sentence cites only Y. Jiang 2023 and X. Li 2022; Collins does not appear in
the references.

- **Tier impact**: none (not cited). Collins stays in the vitality corpus as
  a clean-design example.

### D5 · §2.3 Y.-H. Wu 2025 — loose attribution visible already at skeleton time

The original 2-3 flow P4 lists "Y.-H. Wu et al. 2025: large learning rates
prevent memorization — **supports the idea of** distinguishing signal from
noise." The current text's stronger phrasing ("features that remain constant
in the window are the signal" attributed to Wu) grew from this seed. The plan
flags it as "possibly a loose attribution" — the skeleton confirms the
linkage was loose from the start.

- **Tier impact**: Y.-H. Wu 2025 stays Tier 1 and becomes the **highest-
  priority verification target** in 2b (check what the paper actually
  supports: large learning rates ↔ memorization prevention, nothing about
  window-stability-as-signal).

### D6 · §2.4 / §1 X. Liu 2024 RGB citation — skeleton already doubted it

The original 2-4 flow P1: "🔍 MISSING: RGB channel synthesis is cited only to
X. Liu et al. 2024… The current citation is 'Reinventing Planting Design in
Landscape Architecture' — **is this the right paper?**" The doubt was never
resolved; the citation is in the current text (Intro P5 + §2.4).

- **Tier impact**: X. Liu 2024 stays Tier 1 (RGB precedent) and is a 2b
  verification target — confirm the paper actually demonstrates multi-channel
  raster fusion, or replace the reference.

### D7 · Channel-letter assignment — internal skeleton inconsistency, resolved in text

Intro flow P5 wrote "building density (R), road accessibility (G), POI
diversity (B)"; the 2-4 and methods flows wrote R=accessibility, G=diversity,
B=density. The current text is consistent with the latter everywhere
(§1 P5, §2.4, §3.2). Resolved; no action.

### D8 · §3.7.3 Zhou year; Heusel year

Methods flow cites "[Zhou et al. - 2026]" for FID on urban layouts; current
text and references say Zhou et al. 2024 (arXiv 2409.17049). Heusel cited as
2017 in the flow, 2018 in references (arXiv 2017 / NeurIPS 2018).

- **Action**: cosmetic; verify Zhou year at 2b entry-writing time.

### D9 · §4/§5 — WIP in both skeleton and text

No drift; both are placeholder notes. The skeleton's callback maps promise
§4.1–4.3 and §5.1–5.3 resolutions — when those sections get written, they
should be built from the callback maps in `logic-flow-260417/`, not from
scratch.

---

## April evidence gaps — status now

| # | Gap (April 17) | Status (Sept 11) |
|---|---|---|
| E1–E3 | §2.2 three limitations under-cited | Moot — third limitation cut (D2); remaining two now carry Doan & Zhang + C. Wu + Tan (verified) |
| E4 | Bonnaire/Zhang/Kadkhodaie validation datasets | Became the June audit's seven claims — **2b work item, open** |
| E5 | X. Liu 2024 right paper? | Unresolved — 2b verification target (D6) |
| E6 | Manifold-hypothesis canonical citation | Current text uses Chapelle 2006 — acceptable, Tier 2 |
| E7 | State Jacobs's four generators | Resolved in current text (§2.1 P1 lists them) |
| E8 | Supply/demand framing attribution | Still our own framing; Williams 2019 decoupling is the nearest anchor (cited §2.1 P5) |
| E9 | More conflation examples | Over-fulfilled: V5–V7 in reserve |
| E10 | CS symbolic/connectionist supplement for Huang | Not added; Huang 2025 carries the frame alone (Tier 1) |
| E11 | GlobalMapper/COHO in §2.3 P5 | Added (the 75% claim) — **audited, 2b work item** |
| E12 | Conditional-diffusion paragraph | Cut (D3) |

## Final tier list (skeleton-confirmed)

**Tier 1 (支柱) — 25 papers**, each now traceable to a specific skeleton
paragraph AND a current-text sentence:

- Seven audited claims: Kadkhodaie 2024, P. Wang 2025, Zhang 2024,
  Cole & Lu 2024, Williams 2019, L. He & Aliaga 2023 + 2024, F. Jiang 2024
- DDPM-theory spine: Bonnaire 2025, Biroli 2024, S. Li 2024, Yoon 2023,
  Y.-H. Wu 2025 (⚠️ loose attribution — verify first), Rahaman 2019,
  B. Wang & Pehlevan 2025
- Urban-critique spine: Doan & Zhang 2025, Li (Q.) 2021, X. Li 2022,
  Pan 2021, Y. Jiang 2023, Tan 2026, C. Wu 2023, S.-Y. Huang 2025,
  X. Liu 2024 (⚠️ right paper? — verify early)
- Black-box claim group: Q. Wang 2025, M. He 2026, Zhuang 2024, Zhou 2024
- Pending user decision (skeleton-borderline): H. Liu 2022 (named-framework
  attribution — recommend promote), Zeng 2018, Gómez-Varo 2022, M. Wang 2025
  (promote only if the T4 taxonomy enters §2.1 text)

**Tier 2 (散户)**: everything else in the references — cited for what they are
(Jacobs, Whyte, Lynch, Maas, Montgomery, Landry, Chapelle 2006, Vapnik 2000,
Ho 2020, Song 2021, Karras 2022, Rombach 2022, Dhariwal & Nichol 2021,
Heusel 2018, Carlini 2023, Somepalli 2022, George 2025) or as examples
(Kim 2020, Liu 2019, Scepanovic 2021, Chen 2021, Y. Jiang 2024, Qin 2025,
Xia 2020, Mouratidis 2020, Dogan & Lee 2024, Gómez-Varo 2022, H. Liu 2022*,
M. Wang 2025*, Zeng 2018* [* = pending promotion decision]).

---
*Built 2026-09-11 by reading the full original flows against the full current
text; verification statuses from the 2026-09-11 sub-session pass recorded in
`vitality_indicators_review.md`.*
