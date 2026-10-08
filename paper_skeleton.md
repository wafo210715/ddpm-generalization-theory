# ISM 论文骨架（基准）

ISM 这篇 methods paper 的骨架，用 2026-04-17 写作当天的原始 section flow。每读完一篇文献、写完 annotated bibliography 后，和 agent 一起讨论：这一篇对骨架有什么影响。结论记在 § 4，标上日期。

- **来源**：`archive/ism-report-2026-04/logic-flow-260417/`。§ 1–3 的内容原样摘自 `paper-framework.md` 和各节的 `*-logic-flow.md`，没有改写；每个段落要点、证据和缺口都在原文件里。
- **正文**：`archive/ism-report-2026-04/ISM_report.txt`（2026-04-22 提交）。原骨架和这份正文的差别（§2.1 加了三个混用的例子，砍了 emergence 这条局限，等等）见 `archive/ism-report-2026-04/ism_skeleton_drift.md`。
- **改动规则**：§ 1–3 是 4 月原文，不改。骨架要变时，在 § 4 记下来；用户确认要改骨架时，在原文旁写新版和日期，旧文保留（删掉的划线）。
- **注意（agent 2026-10-06）**：4 月骨架的前提是 RGB 三通道和 4 182 个格子；现在的设计是 3 010 个格子（`discussion/indicator_specification.md`），活力的看法见 `vitality_stance.md`。讨论一篇文献时，如果碰到这类过时的地方，记进 § 4，不要直接改骨架。

---

## 1. 核心问题与六条贯穿概念（`paper-framework.md`）

> **Can a Denoising Diffusion Probabilistic Model (DDPM) learn the low-dimensional manifold of urban built environment from RGB-encoded geospatial data?**
>
> This is a **methods paper** with **preliminary validation** on HCMC.

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

## 2. 各节结构

### §1 Introduction（`introduction-logic-flow.md`）
```
P1: Problem diagnosis → correlationist paradigm + conceptual reductionism
P2: Theoretical opening → DDPM as epistemic probe [C2][C6]
P3: Mechanism → generalization window + manifold learning [C3]
P4: THE GAP → DDPM theory validated on simple data, not urban data ← NEW
P5: Our approach → RGB channel synthesis [C5] + single RQ
P6: Dual significance → methodological [C1-C2-C3] + epistemological [C4-C6]
P7: Paper positioning → methods paper, scope, roadmap
```

### §2.1 From Jacobs to Quantification（`lit-review-logic-flow.md`）
```
Jacobs (qualitative, normative) → Lynch (organism, performance) → Maas/Montgomery (emergent property) → Landry (holistic interplay)
    → big data makes UV "observable" → two proxy traditions emerge
    → indices conflate supply/demand → this conflation is the root problem [C4]
```
P1 Theoretical Roots · P2 The Invisibility Problem + Two Proxy Traditions · P3 Multidimensional Indices + Conflation Critique

### §2.2 The Limits of the "Correlationist" Paradigm（`section-2-2-logic-flow.md`）
```
P1: Huang's framework — symbolic vs connectionist → our adaptation: correlationist vs connectionist [C6]
P2: Current UV research as correlationist — data proxies + toolkit + reductionism [C1]
P3: Two limitations — evidence of correlationist failures [C1]
P4: Brief wrap-up — epistemological dissonance + bridge to 2.3 [C1→C6→Sec 2.3]
```

### §2.3 Diffusion Models: From Image Synthesis to Manifold Learning（`section-2-3-logic-flow.md`）
```
P1: DDPM mechanism + Manifold Hypothesis + empirical evidence of generalization [C2][C3]
P2: Property 1 — curse of dimensionality + subspace clustering [C2]
P3: Property 2 — generalization window + spectral bias [C3]
P4: Methodological opening for urban analysis — probing through inputs/outputs [C6]
P5: Current urban applications miss the point — black box generators [C1 callback]
```

### §2.4 Bridging the Gap: A Generative Epistemology for Urban Vitality（`section-2-4-logic-flow.md`）
```
P1: Structure-preserving compressor [C2] → connects 2.3 theory to 2.2 limitations, with honest boundary
P2: RGB channel synthesis [C5][C1] → the representational solution that avoids reductionism
P3: Single postulate — generalization window as epistemic probe [C3][C6] → scope + preview of Methods
```

### §3 Methods（`section-3-methods-logic-flow.md`）
```
3.1 Research Framework [C4][C1][C6] — Pipeline overview with epistemic probe annotations
3.2 Data Preparation [C5] — From geospatial vectors to RGB tensors
    3.2.1 Rasterization
    3.2.2 RGB Fusion
    3.2.3 Pre-encoding Analysis
3.3 Model Architecture [C2] — DDPM formulation and U-Net specification
    3.3.1 Denoising Diffusion Probabilistic Model
    3.3.2 U-Net Noise Prediction Network
3.4 Noise Schedules — Implemented schedules and rationale for baseline choice
3.5 Training Procedure and Checkpointing Strategy [C3] — Protocol for manifold learning
3.6 Sample Generation — DDPM and DDIM sampling
3.7 Evaluation Metrics [C6] — How we probe what the model learns
    3.7.1 Memorization Fraction
    3.7.2 Distributional Fidelity
    3.7.3 Fréchet Inception Distance (FID)
    3.7.4 Inter-Channel Synergy Analysis
3.8 Irreducible Characteristics Detection [C6] — Identifying learned structural signals
```

### §4 Experiments（`paper-framework.md`；4 月时只有提纲）
> **Key reframing**: This is not "results" — it's "preliminary validation" of the methodology.

4.1 Study Area and Data · 4.2 Training Dynamics and Generalization Window · 4.3 Preliminary Sampling Analysis · 4.4 The Theory-to-Practice Gap

### §5 Discussion（`paper-framework.md`；4 月时只有提纲）
5.1 Methodological Contribution [C1][C5][C6] · 5.2 Limitations [C4] · 5.3 Future Work

## 3. 怎么讨论一篇文献的影响

用户写完这篇的 annotated bibliography 后，agent 依次问：

1. **落在哪里**：这篇碰到哪一节、哪一段（如 §2.2 P3），或哪条贯穿概念（C1–C6）？
2. **什么作用**：是给某段补证据（填 🔍/❓ 缺口）、威胁某段的说法、需要新加一段，还是让某段该删？
3. **和 4 月骨架的距离**：这篇是否说明骨架的某个前提已经过时（RGB 三通道、只做供给侧、单城市……）？
4. **要不要改**：只记下讨论的结论；改不改骨架由用户决定。

说法要带原文句子和行号，分开"作者写的"和"我们推的"。没有影响也记一行，写"无"。

## 4. 每篇文献对骨架的影响（按日期往下加）

| 日期 | 文献 | 落在哪里（节/段/C） | 作用（补证据 / 威胁 / 新段 / 该删 / 无） | 讨论结论 | 骨架改了吗 |
|---|---|---|---|---|---|
| 2026-10-08 | Dogan & Lee 2024 | 2.1 P1、P2、P3；2.2 P2、P3、P4；Methods 3.2 | 证据为主；反论点 ×2 | 证据：谱系出处；单一数值认识论里最用力的一篇（三个数各自回归）；活力分装了供给；饱和；维度间的边检验不了；空间自相关可能被重采样抬高。反论点：多源合成（2.1 P3 已回应）；一致性可评估韧性（记入 stance，暂不预先回应）。转化层问题留给 Methods 3.2 | 逐段骨架已改（`skeleton/section-2-1.md`、`section-2-2.md`）；§1–3 未动 |
| 2026-10-08 | Fan et al. 2023 | 2.1 P2、P3；2.2 P2、P3；open_questions A1、C1 | 证据 / 伙伴 | 两期设计是方法谱系往前一步；功能替代只停在叙事；结论落在改善建成环境；「近水」两种操作化结果不同。不作混淆例子（字面检查后非 T1、不标 T4）。时间维度在研究问题之外，记 future research 不记局限 | 逐段骨架已改；§1–3 未动 |
| 2026-10-08 | Garau & Annunziata 2022 | — | 无 | 用户判断质量不足，跳过深入讨论；台账保留 6 行数据源 | 否 |
| 2026-10-08 | Yue et al. 2021 | 2.1 P2、P3 | 论点出处 + 证据 | 「capacity of a built environment」作供给读法的出处；POI、居住用地放进「人的活动」是数据换角色的例子。2.2 不适用（无回归）。§5.2 小街区解释活力是循环论证，不引作证据。唯一 HCMC 本地研究，数据源可获取性由子会话核实中 | 逐段骨架已改；§1–3 未动 |
