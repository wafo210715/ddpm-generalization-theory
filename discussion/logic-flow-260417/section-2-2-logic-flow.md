# Section 2.2: The Limits of the "Correlationist" Paradigm — Logic Flow

## Overall Logic
```
P1: Huang's framework — symbolic vs connectionist → our adaptation: correlationist vs connectionist [C6]
P2: Current UV research as correlationist — data proxies + toolkit + reductionism [C1]
P3: Two limitations — evidence of correlationist failures [C1]
P4: Brief wrap-up — epistemological dissonance + bridge to 2.3 [C1→C6→Sec 2.3]
```

**Key theoretical move**: "Correlationist" is our domain-specific adaptation of Huang's "symbolic." The parallel:

```
Architecture (Huang):    symbolic rationality  → connectionist rationality
UV Research (This paper): correlationist paradigm → connectionist modeling
```

Both share: predefined structure → learn from data; explicit rules → probabilistic inference; expert-defined → data-inferred. But the methodological apparatus differs: geometric deduction (architecture) vs. statistical correlation (UV research).

---

### P1: Huang's Framework and Our Adaptation [C6]
**Core argument**: Establish the theoretical precedent (Huang's symbolic vs connectionist) then introduce "correlationist" as the UV-domain equivalent of "symbolic."

**Logic**:
- S1: Huang argues architecture is transitioning from symbolic rationality (predefined rules, fixed typologies, expert-defined schemata) to connectionist rationality (learned from data, probabilistic inference, latent navigation)
- S2: Urban vitality research has arrived at a structurally similar juncture — but expressed through different apparatus
- S3: What we term "correlationist" = adaptation of Huang's "symbolic" to UV studies. Produces knowledge as feature weights, regression coefficients, ranked factor importance
- S4: Connectionist alternative = learn geometry of urban data manifold, capture joint distribution
- S5: This distinction provides the analytical lens for the critique that follows

**Evidence**:
- ✅ [Huang - 2025]: symbolic vs connectionist in architectural intelligence
- ✅ Our adaptation: "correlationist" as UV-domain equivalent (stated as "what we term...")

**Purpose**: Establishes the theoretical precedent and our domain-specific terminology BEFORE diagnosing the field.

---

### P2: Current UV as Correlationist [C1]
**Core argument**: Contemporary UV research operates squarely within what we have termed the correlationist mode.

**Logic**:
- S1: Big data enabled unprecedented quantification but reinforced paradigm prioritizing associations over mechanisms
- S2: Vitality operationalized through digital traces:
  - LBS / mobile signaling → human mobility (Kim 2020; X. Li et al. 2022; Liu et al. 2019)
  - Social media → economic/cultural engagement (H. Liu et al. 2022; Qin et al. 2025; Scepanovic et al. 2021)
  - POI density → functional intensity (Jiang et al. 2024; C. Wu et al. 2023)
- S3: Standard toolkit — OLS, spatial regression, GWR, RF — to catalogue drivers
- S4: Results in "conceptual reductionism" (Doan & Zhang 2025)
- S5: **Callback to 2.1**: these proxies conflate supply and demand dimensions

**Evidence**: ✅ Well-supported. No gaps.

**Key framing**: "operates squarely within what we have termed the correlationist mode" — connects our adapted framework to concrete practice.

---

### P3: Two Fundamental Limitations [C1]
**Core argument**: The correlationist approach encounters two epistemic walls.

**Limitation 1: Additivity and Non-linearity** (thresholds as a type of non-linearity)
- S1: Treats urban factors as independent, linear contributors — fails to account for interaction effects, synergies, threshold behaviors
- S2: Doan & Zhang: "linearity, additivity, and homoscedasticity"; Wu et al.: "erroneous conclusions"
- S3: Tan et al. 2026 — most comprehensive multi-method response (RF+SHAP+PDP+ISM+BN): FAR must exceed ~0.25, food services show negative association beyond saturation
- S4: Thresholds are a type of non-linearity — linear coefficients qualitatively misleading

**Limitation 2: Context Stripping — Two Axes**
- S5: **Statistical axis**: variables analyzed in isolation, losing relational structure of how attributes co-occur
- S6: **Spatial axis**: geographic units treated as isolated, severing cross-boundary interdependencies
- S7: Both empirically consequential — spatial autocorrelation (Jiang 2024, Li 2022); cross-boundary spillover (Collins 2024: gender-differentiated externalities)

**Closing**: Together, these limitations mean the paradigm cannot access emergent properties.

**Evidence**:
| Claim | Citations | Assessment |
|-------|-----------|------------|
| Additivity assumption | Doan & Zhang 2025, Wu et al. 2023 | ✅ Synthesized |
| Non-linearity + thresholds | Tan et al. 2026 | ✅ Comprehensive multi-method evidence |
| Context stripping (statistical) | Argument from structure | ✅ Self-contained |
| Context stripping (spatial) | Jiang 2024, Li 2022, Collins 2024 | ✅ Synthesized |

**Writing principle**: Synthesize, don't quote-dump. Only two short quotes (Doan & Zhang, Wu et al.).

---

### P4: Brief Wrap-Up [C1→C6→Sec 2.3]
**Core argument**: Epistemological dissonance with 2.1 → necessity of connectionist alternative → bridge to 2.3.

**Logic**:
- S1: Correlationist paradigm generates feature lists but cannot access joint distributions / generative processes
- S2: Dissonance with 2.1 — Jacobs/Landry/Montgomery define vitality as emergent, paradigm predicts it with linear/additive drivers
- S3: Failure to transcend predict-and-optimize paradigm necessitates connectionist modeling
- S4: Bridge: "Section 2.3 examines the theoretical properties of diffusion models that enable such a reframing"

**Evidence**: ✅ Self-contained synthesis of 2.1 + 2.2 + Huang's framework.

**Key difference from previous version**: Condensed from two paragraphs (old P3+P4) into one brief wrap-up. The Huang content is now in P1, so P4 only needs to connect back to 2.1 and forward to 2.3.

---

## DDPM's Coverage of 2.2 Limitations (cross-reference for 2.4)

```
2.2 Limitation                    2.4 Claim           Section 5 Treatment
─────────────────────────────────────────────────────────────────────────
L1: Additivity / Non-linearity   → Directly addressed   Confirmed in Experiments
L2a: Context stripping            → Partially addressed  Partial validation via
    (statistical)                    (manifold position)   RGB decomposition
L2b: Context stripping            → NOT addressed        Acknowledged as
    (spatial/spillover)              (independent inputs)  genuine limitation
```

---

## Callback Map for Section 2.2

```
[C1] Correlationist paradigm
     ← Intro P1 (setup)
     → 2.2 P2 (diagnosed as symbolic-correlationist)
     → 2.2 P3 (two limitations)
     → 2.4 P1 (which dimensions DDPM addresses)
     → Methods 3.1 (no composite score)

[C4] Supply-demand
     ← 2.1 P2-P3 (conflation critique)
     → 2.2 P2 (callback: proxies conflate supply/demand)
     → 2.4 P3 (scope: supply-side only)

[C6] Epistemic probe
     ← Intro P2/P6
     → 2.2 P1 (Huang's distinction sets up the lens)
     → 2.3 P4 (methodological opening)
     → 2.4 P3 (operationalized)
```
