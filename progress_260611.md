# Progress 2026-06-11: Deep Verification + v3 主线重构

> **当前分支**: `wafo/5090-setup`
> **会话类型**: 对抗性核查 + 主线重构（3 轮）
> **状态**: 
> - 对原 progress 的 5 个承重主张做了 deep verification（6 个 explore agent + 1 个 librarian agent）
> - 发现 6 个事实错误并修正
> - 发现 G 通道 97.5% 零值、MLE 实现 bug、generalization window 被错误否定
> - 用户纠正了我对 data curation、SDXL、generalization window 的误解
> - 重构出 v3 主线（6 phase），替代原 TODO 列表

---

## 一、本会话三轮结构

**第 1 轮**: 评估原 progress_260611.md（已删除），按 deep-verification skill 的 6 动作核查
**第 2 轮**: 用户提出自己的主线，我补充缺口、修正误解、重构为 v3 计划
**第 3 轮**: 用户纠正我对 data curation / SDXL / window 的错误理解，最终对齐后定稿 v3

---

## 二、Deep Verification 结果摘要

### 承重主张清单与核查结论

| # | 承重主张 | 结论 | 关键发现 |
|---|---------|------|---------|
| C1 | F_mem ≈ 0 at all checkpoints (max 0.0209 at 360K) | ⚠️ 50-sample 正确但误导 | 1000-sample: max=0.012 at 500K, 5/19 checkpoints 非零 (240K-500K) |
| C2 | Coupled regime at 160K-500K (GR r=0.87, BR r=0.75, BG r=0.65) | ✅ 数据精确匹配 | 但遗漏了 300K 也是 coupled；`regime_evolution.png` 不存在 |
| C3 | SUBDIV=20 是真正瓶颈 | ✅ 代码精确匹配 | 但 "20m" 不存在——SUBDIV=20 是计数，实际 sub-cell=50m |
| C4 | 732 cells 数据集 | ❌ 错误 | 实际 4,182 cells（`data_quality_report.txt`） |
| C5 | 7 篇外部文献 | ✅ 全部真实 | Poly2Vec venue 内部文档冲突（ICLR 2024 vs ICML 2025），外部确认 ICML 2025 正确 |

### 已执行的修正（6 处）

| # | 文件 | 修正内容 |
|---|------|---------|
| 1 | `README.md` | 732 → 4,182 cells; 9.88M → 157.8M U-Net; 更新 task status table |
| 2 | `STAGE2_RESEARCH_SYNTHESIS.md` | Poly2Vec ICLR 2024 → ICML 2025 (×2 处) |
| 3 | `post-training-evaluation.md` | Tasks 6/7/9 `[x]` → `[ ]`（实际未完成） |
| 4 | `experiments_summary.md` | ddpm_sample.py 139 → 117 lines |
| 5 | `progress_260611.md` (本文件) | 添加 1000-sample F_mem 补注 |
| 6 | `progress_260611.md` (本文件) | regime_evolution.png ✅ → ⚠️ 缺失 |

### 新发现（非承重但影响计划）

| # | 发现 | 严重度 | 证据 |
|---|------|--------|------|
| F1 | **G 通道 97.5% 零值**——P95 归一化后退化为几乎全零图像 | 高 | `data_quality_report.txt`: G non-zero=2.49% |
| F2 | **MLE=3.93 实现有 bug**——用了简化 Levina-Bickel 公式 | 中 | `pre_encoding_analysis.py:298-302` vs `multi_resolution_id.py:127-136` |
| F3 | **pre_encoding 输出目录不存在**——gitignored 且磁盘上也丢失 | 高 | `data-processing/data/hcmc/reports/pre_encoding/` → Test-Path = False |
| F4 | **R 通道主导训练和生成**——唯一有空间自相关的通道 | 高 | Moran's I at 500K: R=0.251, G=-0.005, B=-0.003 |
| F5 | **Generalization window 被错误否定**——1000-sample 数据显示 window ≈ 0-200K | 高 | `fmem_curve_1000samples.csv` + `synergy_metrics.csv` |
| F6 | **ISM report R/G/B 通道分配矛盾**——line 23 vs line 121 不一致 | 中 | `ISM_report.txt:23` vs `ISM_report.txt:121` |
| F7 | **训练 loss 曲线不可恢复**——500K 步训练没有日志文件 | 中 | `train.py` 已修复但旧 run 丢失 |

---

## 三、关键认识对齐（用户纠正我的地方）

### 3.1 SUBDIV ≠ 20m

| 参数 | 值 | 物理含义 | ISM report 用语 |
|------|---|---------|----------------|
| SUBDIV | 20 | 每边切 20 份（无量纲计数） | "a 20×20 grid of calculation units" (line 109) |
| CALC_UNIT_M | **50m** | 每个 sub-cell 的物理尺寸 = 1000m/20 | "calculation unit (50 m)" (line 109) |
| BUFFER_M | 150m | 每个计算点的分析半径 | "150 m buffer" (line 111) |
| TARGET_PX | 128 | 输出图像像素数 | "128×128 pixels" (line 117) |
| 像素分辨率 | **7.8125 m/px** | 1000m/128（派生值） | 未显式命名 |

**两层分辨率**：计算层 50m（真正信息分辨率）vs 输出层 7.8m/px（双线性插值产物）。
**Source**: `build_hcmc_pipeline_v2.py:57-60`, `ISM_report.txt:109-117`, `data_quality_report.txt:5`

### 3.2 Generalization Window 存在（0-200K）

| 信号 | 来源 | 含义 |
|------|------|------|
| Coupled regime at 160K-200K | `synergy_metrics.csv` | 结构学习完成 |
| F_mem > 0 at 240K (0.004) | `fmem_curve_1000samples.csv` | 记忆开始 |
| F_mem rises monotonically to 0.012 at 500K | 同上 | 记忆持续增加 |

`experiments_summary.md` (4/20) 基于 50-sample 说 "No generalization window detected"——被 5/23 的 1000-sample 数据推翻，但 summary 从未更新。
**Source**: `progress_260421.md:14-22`, `fmem_curve_1000samples.csv`, `synergy_metrics.csv`

### 3.3 Data Curation 是当前论文核心，不是下一篇论文

ISM report 的核心研究问题（`ch2-ism-proposal.md:21`）：
> "How does the systematic manipulation of training data composition alter the model's learned manifold?"

计划的对比实验（`DDPM-experiment-design-framework.md:156-201`）：
| 实验组 | 训练数据 | 目的 |
|--------|---------|------|
| Control | 全部 HCMC | 基线 |
| Exp A | 特定 typology 子集 | 看模型学到什么 |
| Exp B | 另一 typology 子集 | 对比 |
| Exp C | Minimal dataset (~ID 个样本) | 最小充分集 |

**Source**: `ISM_report.txt:5,19,69,83`, `ch2-ism-proposal.md:21,51`, `TODO.md:89-91`, `progress_260421.md:219-253`

### 3.4 SDXL / Latent Diffusion 是 Future Work

当前理论基础是 pixel-space diffusion。JiT 和 latent diffusion 不在当前 scope。
Pro 6000 Blackwell 96GB 的 headroom 用于更大模型或更高分辨率，不用于 latent diffusion。

### 3.5 RGB Fusion 参考文献

**Liu, X., Yang, N., & Tian, R. (2024). *Reinventing Planting Design in Landscape Architecture: A Generative AI Approach*. Wichmann Verlag. DOI: 10.14627/537752020**

ISM report line 23 引用此文献作为 RGB channel synthesis 方法论先例。
**Source**: `ISM_report.txt:23,455`, `ch2-ism-proposal.md:47,95`

### 3.6 采样方法澄清

| 方法 | 步数 | 用在哪 | 状态 |
|------|------|--------|------|
| DDIM | 50 step | **所有** F_mem 评估、channel stats、synergy | 已用 |
| DDPM | 1000 step | `sample_v3.py` 可选 | **从未在 v3 checkpoints 上跑过** |

用户之前说的"DDIM 1000 更有意义"可能是混淆了 **1000 samples**（统计功效）和 **1000 steps**（采样质量）。两者都更有意义，但原因不同。
**Source**: `ddim.py` (default ddim_steps=50), `ddpm_sample.py`, `experiments_summary.md:176`

---

## 四、G 通道问题详解

### 现状

| Channel | 非零像素 | Raw Mean | Raw Max | P95 | 生成 Moran's I (500K) |
|---------|---------|----------|---------|-----|----------------------|
| B (density) | 7.31% | 0.006 | 1.34 | 0.013 | -0.003 |
| G (diversity) | **2.49%** | 0.005 | 0.59 | 0.419 | **-0.005** |
| R (accessibility) | **54.30%** | 27.36 | 3766.9 | 98.55 | **0.251** |

**Source**: `data_quality_report.txt`, `minimum_viable_report.md`

### 发现

1. G 通道 97.5% 像素为零 → P95 归一化后变成巨大零峰 + 细尾的退化分布
2. G 的 Wasserstein 距离最低（EMD=0.047 at 400K）——但这是**假阳性**，匹配全零分布本来就不难
3. Synergy 分析标 "diversity-driven" 为 dominant regime（8/19 checkpoints）——但 G 本身几乎不携带空间信息
4. G-R 原始相关性 0.8844——共享底层城市结构（POI 聚集在街道旁），但 G 的信号太稀疏
5. R 是唯一有空间自相关的通道（Moran's I ~0.25 from 160K onward）

**Source**: `synergy_summary.csv`, `synergy_metrics.csv`, `minimum_viable_report.md`, `data_quality_report.txt`

### 为什么这很重要

G 通道问题影响所有下游分析：
- F_mem 评估中 G 通道贡献几乎为零
- Synergy 分析的 "diversity-driven" 标签有误导性
- Data curation 的聚类（基于 B/G/R mean+std）被 G 的零值主导
- 如果 G 通道需要重新编码，后面所有步骤都要重来

---

## 五、MLE Bug 详解

### 现状

`pre_encoding_analysis.py:298-302` 的 MLE 实现：
```python
log_ratios = np.log(distances[:, k] / (distances[:, 1] + eps) + eps)  # 只用 d_k / d_1
d_mle = len(X) / (np.sum(log_ratios) + eps)
```

`multi_resolution_id.py:127-136` 的正确实现：
```python
for j in range(k - 1):
    log_ratios += np.log(d_k / distances[:, j])  # sum of log(d_k / d_j) for j=1..k-1
m_k = (k - 2) / log_ratios  # per-point estimate, then average
```

### 发现

- 简化公式只用第 k 近邻和第 1 近邻的距离比，忽略了中间所有近邻
- 标准 Levina-Bickel (2004) 公式需要对 j=1..k-1 求和
- 简化公式可能**低估**真实 MLE 维度
- Smoke test (`multi_resolution_id.py`) 跳过了 MLE（CSV 中显示 "nan"），所以没有交叉验证
- PCA k95=33 已被独立验证，没问题

### 为什么这很重要

MLE=3.93 是论文方法学部分的关键数字（"数据内在维度约 4"）。如果真实值是 8-15，论文的 manifold hypothesis 论证需要调整。

**Source**: `pre_encoding_analysis.py:282-306`, `multi_resolution_id.py:127-136`, `id_vs_resolution_smoke/id_vs_resolution.csv`

---

## 六、v3 主线计划

### Phase 0: 数据完整性审计（0.5 天）

**目的**：确保论文中每个数字都能溯源到生产现场。

#### Task 0.1: 重跑 pre_encoding_analysis.py

- **现状**: `data-processing/data/hcmc/reports/pre_encoding/` 目录不存在。`.gitignore` 包含 `data-processing/data/`，所有报告输出被 gitignore。`progress_260421.md` 声称 4/27 跑过，但文件已丢失。
  - Source: `Test-Path data-processing/data/hcmc/reports/pre_encoding/` → False
  - Source: `.gitignore` 包含 `data-processing/data/`
  - Source: `progress_260421.md:39,43` 声称 k95=33, MLE=3.93
- **发现**: PCA k95=33 已被 `multi_resolution_id.py` 独立验证。MLE=3.93 使用了错误的简化公式（见第五节）。
  - Source: `id_vs_resolution_smoke/id_vs_resolution.csv` 确认 k95=33
  - Source: `pre_encoding_analysis.py:298-302` vs `multi_resolution_id.py:127-136`
- **为什么这么做**: `intrinsic_dimension.txt` 和 `typology_labels.csv` 是 Phase 2 data curation 的前置依赖。没有 typology labels 就无法设计实验组。同时需要修复 MLE bug 得到正确的内在维度估计。
- **产出**: `intrinsic_dimension.txt`（含修正后的 MLE）, `typology_labels.csv`, `pca_scree.png`, 其他 pre-encoding 输出
- **命令**: `python data-processing/scripts/pre_encoding_analysis.py --input data-processing/data/hcmc/processed/cells/ --output data-processing/data/hcmc/reports/pre_encoding/`

#### Task 0.2: 修复 MLE bug

- **现状**: `pre_encoding_analysis.py:298-302` 使用简化 Levina-Bickel 公式（只用 `log(d_k/d_1)`）。`multi_resolution_id.py:127-136` 有正确实现（sum over `log(d_k/d_j)` for j=1..k-1）。
  - Source: `pre_encoding_analysis.py:282-306`
  - Source: `multi_resolution_id.py:127-136`
- **发现**: 简化公式可能低估真实 MLE 维度。真实值可能在 5-15 范围（基于类似数据集的文献）。
- **为什么这么做**: MLE 是论文方法学的关键数字。如果从 3.93 变成 8-15，manifold hypothesis 的论证力度不同。
- **产出**: 修正后的 `estimate_mle_intrinsic_dim()` 函数，与 `multi_resolution_id.py` 的实现一致

#### Task 0.3: 整合 1000-sample F_mem 到 experiments_summary.md

- **现状**: `experiments_summary.md` 写于 4/20，基于 50-sample 数据。1000-sample CSV 于 5/23 生成但从未整合。
  - Source: `experiments_summary.md:3` (date: 2026-04-20)
  - Source: `fmem_curve_1000samples.csv` (timestamp: 2026/5/23 10:30:11)
- **发现**: 50-sample max F_mem=0.0209 at 360K 是小样本波动。1000-sample max=0.012 at 500K，且 5/19 checkpoints 非零（240K-500K），呈单调上升趋势。
  - Source: `fmem_curve.csv` (50 samples) vs `fmem_curve_1000samples.csv` (1000 samples)
- **为什么这么做**: 论文应使用统计功效最高的数据。1000-sample 数据还揭示了 generalization window 的存在（见 Task 0.4）。

#### Task 0.4: 确认 generalization window = 0-200K

- **现状**: `experiments_summary.md:71` 说 "No generalization window detected"（基于 50-sample）。1000-sample 数据显示不同图景。
  - Source: `experiments_summary.md:71`
  - Source: `fmem_curve_1000samples.csv`
  - Source: `synergy_metrics.csv`
- **发现**:
  - 结构学习完成: coupled regime at 160K-200K（`synergy_metrics.csv`，所有 pairs |r|>0.5）
  - 记忆开始: F_mem > 0 at 240K（`fmem_curve_1000samples.csv`，4/1000 memorized）
  - 记忆增长: F_mem 单调上升到 0.012 at 500K
  - **Generalization window ≈ 0-200K**
- **为什么这么做**: 这是论文的核心叙事基础。"Window 存在"比"no window"更符合数据，也更容易与 Bonnaire et al. 的理论框架对接。

#### Task 0.5: 修正 ISM report R/G/B 通道分配矛盾

- **现状**: `ISM_report.txt:23` 说 "R=building density, G=road accessibility, B=POI diversity"。`ISM_report.txt:121` 说 "B=building density, G=POI diversity, R=accessibility"。代码（v2 pipeline）用的是 line 121 的版本。
  - Source: `ISM_report.txt:23` vs `ISM_report.txt:121`
  - Source: `build_hcmc_pipeline_v2.py` 的 `np.stack([R_uint8, G_uint8, B_uint8], axis=-1)`
- **发现**: Line 23 和 line 121 的 R/G/B 分配完全相反。
- **为什么这么做**: 论文方法学部分不能有这种基本矛盾。

#### Task 0.6: 确认 train/val/test split 实际数字

- **现状**: README 说 "584/73/74 split"（基于旧的 732 cells）。实际 4,182 cells 的 split 数字未知。
  - Source: `README.md:40` (旧数字)
  - Source: `data/hcmc/processed/splits/train_cells.txt` 等文件存在
- **发现**: splits 目录有 3 个文件（train/val/test），总 cells = 3263+407+409 = 4079（另有 ~103 cells 未分配）。
  - Source: `splits/` 目录文件计数
- **为什么这么做**: 论文方法学需要准确的 split 数字。

**🚪 Phase 0 验证门**: 列出论文将引用的 5 个关键数字，每个能指向一个磁盘上的源文件。

| 数字 | 源文件 | 当前状态 |
|------|--------|---------|
| 4,182 cells | `data_quality_report.txt` | ✅ 已验证 |
| k95=?, MLE=? | `intrinsic_dimension.txt` | ❓ 需重跑（Task 0.1 + 0.2） |
| F_mem max=? at step ? | `fmem_curve_1000samples.csv` | ✅ 已验证（0.012 at 500K） |
| 19 checkpoints | `model_ckpts/hcmc_vitality_v3_pro6k/` | ✅ 已验证 |
| Final synergy r values | `synergy_metrics.csv` | ✅ 已验证 |

---

### Phase 1: 数据编码反思（1-2 天）

**目的**：诊断 G 通道问题，决定是否需要重新编码。

#### Task 1.1: G 通道稀疏性诊断

- **现状**: G 通道 97.5% 像素为零。P95 归一化后变成退化分布。R 通道 54.3% 非零，是唯一有空间自相关的通道。
  - Source: `data_quality_report.txt` (G non-zero=2.49%, R non-zero=54.30%)
  - Source: `minimum_viable_report.md` (Moran's I: R=0.251, G=-0.005, B=-0.003 at 500K)
- **发现**: G 的 Shannon entropy 计算要求 buffer 内 ≥2 个 POI 且 ≥2 个类别。大部分 50m sub-cell 不满足条件 → 零值。这不是 bug 而是数据本身的稀疏性。
  - Source: `build_hcmc_pipeline_v2.py` 的 entropy 计算逻辑
- **为什么这么做**: 如果 G 通道需要重新编码（比如用 POI density 代替 entropy，或对非零值做 log transform），整个 pipeline 需要重跑，所有下游分析都要重来。这个决策必须在其他 phase 之前做出。
- **可能的修复方案**:
  1. 用 POI count（而非 entropy）作为 G 通道
  2. 对非零 entropy 值做 log transform 后再 P95 归一化
  3. 用 P50 of non-zero values（而非全局 P95）做归一化
  4. 保持现状，在论文中讨论 G 通道稀疏性作为 limitation

#### Task 1.2: Normalization 策略对比

- **现状**: 当前使用 global P95 normalization（所有 cells 的所有值 flatten → 取 P95 → clip to [0,1] × 255）。
  - Source: `build_hcmc_pipeline_v2.py` 的 normalization 代码
- **发现**: P95 normalization 对稀疏通道（B, G）效果差——大部分值被映射到 0，少数极端值被 clip 到 255。R 通道因为非零覆盖率高（54.3%），P95 效果较好。
  - Source: `data_quality_report.txt` (P95 values: B=0.013, G=0.419, R=98.55)
- **为什么这么做**: Normalization 策略直接影响模型能学到什么。如果 G 通道的归一化方式改变，模型可能学到不同的 joint structure。
- **实验**: 对 3 种归一化策略生成直方图对比，不需要重跑 pipeline（只需重新归一化已有的 20×20 原始值）

#### Task 1.3: 读 Liu et al. (2024) 原文

- **现状**: RGB fusion 方法论引用了 Liu, X., Yang, N., & Tian, R. (2024)。但只读了 ISM report 中的引用描述，没有读原文。
  - Source: `ISM_report.txt:23,455`
- **发现**: 原文是关于 landscape architecture 和 planting design 的 generative AI，不是 urban vitality。项目借用了其 RGB channel synthesis 技术。
- **为什么这么做**: 需要确认原文中 RGB fusion 的具体方法论细节——用什么类型的 black and white PNG data，如何处理多通道，有没有讨论通道稀疏性问题。这可能为 Task 1.1 的 G 通道修复提供方法论指导。
- **TODO**: 获取原文（DOI: 10.14627/537752020），重点关注：
  1. RGB channel synthesis 的具体实现
  2. 输入数据的类型和预处理
  3. 是否讨论了通道间信息量不平衡的问题

#### Task 1.4: 决策——当前 RGB 编码是否足够

- **现状**: 基于 Task 1.1-1.3 的结果做决策。
- **为什么这么做**: 这个决策决定了后续所有 work 的基础。如果改编码 → 重跑 pipeline → 重跑 pre-encoding → 重训模型。如果不改 → 在论文中讨论 limitation。
- **决策标准**:
  - 如果 G 通道修复后 ID 显著变化（比如从 33 降到 20 以下）→ 值得改
  - 如果修复只影响 G 通道的分布但不改变整体 ID → 可以不改，作为 limitation 讨论

**🚪 Phase 1 验证门**: 能回答"G 通道 97.5% 零值对模型学习的影响是什么"，以及"我们的 RGB 编码方法论是否有文献支撑"。

---

### Phase 1.5: 理论核实 + 训练规模决策（1 天）

**目的**：理解为什么当前训练 F_mem 只在 240K 才出现 0.4% 的记忆化，决定是否需要更长训练或不同模型。

#### Task 1.5.1: 读 Bonnaire et al. (2025) 的 scaling law

- **现状**: 157.8M 参数模型在 4,182 样本上训练 500K 步，F_mem 最高只有 0.012。不清楚这是训练不够长、模型太大、还是数据太简单。
  - Source: `fmem_curve_1000samples.csv` (max F_mem=0.012 at 500K)
  - Source: `hcmc_vitality_v3_pro6k.yaml` (ch=128, ~157.8M params)
- **发现**: Bonnaire et al. 的理论说记忆化时间 τ_mem ∝ n（数据集大小），但也受 model capacity 影响。具体公式需要从论文中提取。
  - Source: `Why-Diffusion-Models-Don-t-Memorize-main/` (reference implementation)
- **为什么这么做**: 如果理论预测 τ_mem >> 500K，那延长训练就能观察到更强的记忆化。如果理论预测 τ_mem << 500K 但 F_mem 仍然 ≈ 0，那问题在模型/数据匹配。
- **产出**: 理论 τ_mem 估计值 + 对应的行动建议

#### Task 1.5.2: 核实 generalization window 的理论基础

- **现状**: 数据显示 window ≈ 0-200K（coupled regime at 160K-200K, F_mem onset at 240K）。但需要确认这是否符合 Bonnaire et al. 的理论预期。
  - Source: `synergy_metrics.csv` (coupled at 160K-200K)
  - Source: `fmem_curve_1000samples.csv` (F_mem > 0 at 240K)
- **发现**: Bonnaire et al. 的 "generalization window" 定义为结构学习完成到记忆化开始之间的时间段。我们的数据显示 ~40K 步的窗口（160K→240K 或 200K→240K）。
- **为什么这么做**: 如果 40K 步的窗口在理论上合理，论文的叙事就有理论基础。如果不合理，可能需要重新解释数据。
- **同时考虑**: minimal dataset 的理论基础——Bonnaire et al. 是否讨论了数据集大小对 window 宽度的影响？如果数据集太小导致 window 太宽（模型永远不记忆），那 data curation 的目的不仅是控制变量，还是确保 window 可观测。

#### Task 1.5.3: 训练规模决策

- **现状**: 三种可能解释当前 F_mem ≈ 0 的原因。
- **为什么这么做**: 这个决策直接影响 Phase 3 的实验设计。
- **决策矩阵**:

| 解释 | 含义 | 行动 |
|------|------|------|
| A. 训练不够长 | τ_mem > 500K | 延长训练到 1M+ 步 |
| B. 模型太大 / 数据太小 | 157.8M 对 4,182 样本，容量远超数据复杂度 | 用小模型（~10M, ch=32）重训 |
| C. 数据 ID 太低 | MLE≈4 的数据太简单 | 不是问题——用 "no memorization" 作为结果 |

- **注意**: 解释 B 和 data curation 有交互——如果用 curated subset（比如 241 个 dense-core cells），模型/数据比更极端，记忆化更难出现。可能需要同时缩小模型。

#### Task 1.5.4: 分辨率决策

- **现状**: 计算层 50m（SUBDIV=20），输出层 7.8m/px（TARGET_PX=128）。升级分辨率需要先升级 SUBDIV。
  - Source: `build_hcmc_pipeline_v2.py:57-60`
- **发现**: 提高 TARGET_PX 而不提高 SUBDIV = 相同信息的双线性插值。要让 256×256 有意义，需要 SUBDIV=40（25m sub-cells）。
  - Source: `progress_260611.md` 的 Q1 分析
- **为什么这么做**: 如果 Phase 1 决定改编码（重跑 pipeline），可以顺便升级 SUBDIV。如果不改编码，维持 128×128。
- **依赖**: Phase 1.4 的决策

**🚪 Phase 1.5 验证门**: 明确回答"为什么当前训练没有清晰的 memorization phase"，并给出具体的训练参数决策（模型大小、训练步数、分辨率）。

---

### Phase 2: Data Curation 设计（1-2 天）

**目的**：设计对比实验的训练数据子集。这是论文的核心实验设计。

#### Task 2.1: 改进聚类

- **现状**: 当前 k=3 typology 聚类（基于 B_mean, G_mean, R_mean, B_std, G_std, R_std）不够 distinctive。用户原话："three clusters just don't seem so distinctive"。
  - Source: `requests_260421.txt:34-35`
  - Source: `progress_260421.md:219-228`
- **发现**: G 通道 97.5% 零值导致 G_mean 和 G_std 对聚类贡献极小。聚类实际上只被 B 和 R 驱动。
  - Source: `data_quality_report.txt` (G non-zero=2.49%)
- **为什么这么做**: 聚类质量直接决定实验组的区分度。如果实验组之间没有本质差异，对比实验就没有意义。
- **改进方案**:
  1. 扩展特征：加 Moran's I（空间自相关）、channel skewness、non-zero fraction
  2. 尝试 DBSCAN（不需要预设 k，能发现任意形状簇）
  3. 对 G 通道做 log(1+x) transform 后再聚类
  4. 用户确认 typology 分类在地理上合理（可视化在 HCMC 地图上）
- **前置依赖**: Task 0.1（需要重跑 pre-encoding 得到 typology_labels.csv）

#### Task 2.2: 设计实验组

- **现状**: 项目文档中已有多个实验组方案。
  - Source: `DDPM-experiment-design-framework.md:156-201` (4-group district comparison)
  - Source: `TODO.md:89-91` (minimal, cluster-balanced, dense-core-only)
  - Source: `INTEGRATION_PLAN.md:240-268` (Phase 7 subset strategies)
- **发现**: 方案之间有不一致——district-based vs typology-based vs minimal dataset。需要统一。
- **为什么这么做**: 论文需要一个清晰的实验设计矩阵，每个实验组有明确的假设。
- **建议的实验组**:

| 实验组 | 训练数据 | 样本数 | 假设 |
|--------|---------|--------|------|
| Control | 全部 cells | ~3,263 (train) | 基线：学到完整 joint distribution |
| Minimal | ~k95 个样本（覆盖 ID） | ~33 | 最小充分集：F_mem 更高？window 更窄？ |
| Dense-core | Cluster 2 only | ~241 | 高密度子集：学到不同的 synergy regime？ |
| Cluster-balanced | 每个 typology 等量 | ~3×min_cluster_size | 去偏：消除 class imbalance 的影响 |
| [可选] Geographic | 特定行政区 | varies | 地理控制：不同城市形态的泛化差异 |

#### Task 2.3: 确定每个实验组的训练参数

- **现状**: 当前训练参数（157.8M, 500K steps, 128×128）是基于全数据集选择的。子集训练可能需要不同参数。
  - Source: `hcmc_vitality_v3_pro6k.yaml`
- **发现**: Bonnaire et al. 的 scaling law 说 τ_mem ∝ n。如果 Minimal 组只有 33 个样本，τ_mem 应该远小于全数据集。但 157.8M 模型对 33 个样本可能永远不记忆（容量太大）。
  - Source: `Why-Diffusion-Models-Don-t-Memorize-main/` (Bonnaire theory)
- **为什么这么做**: 每个实验组需要合适的模型大小和训练步数，否则对比没有意义。
- **决策**:
  - Minimal 组（~33 samples）：可能需要 ch=32（~10M params）的小模型
  - Dense-core 组（~241 samples）：ch=64（~40M params）？
  - Control 组：维持 157.8M
  - 所有组的训练步数基于 Phase 1.5 的 τ_mem 估计

#### Task 2.4: 写出实验假设

- **现状**: 实验设计需要明确的假设，否则变成"跑一堆实验看结果"。
- **为什么这么做**: 论文是 hypothesis-driven 的。每个对比实验应该有一个可证伪的预测。
- **示例假设**:
  - H1: Minimal 组的 F_mem 上升更快（因为数据少，模型更容易记忆）
  - H2: Dense-core 组的 coupled regime 出现更早（因为数据更同质）
  - H3: Cluster-balanced 组的 irreducible characteristics 与 Control 组不同
  - H4: 跨组泛化测试（train on A, test on B）的 FID 显著高于组内泛化

**🚪 Phase 2 验证门**: 实验设计表 + 每个实验组的假设 + 评估指标 + 训练参数。用户确认 typology 分类合理。

---

### Phase 3: 对比实验执行（3-7 天 GPU）

**目的**：在 curated datasets 上训练模型，收集对比数据。

#### Task 3.1-3.5: 训练各实验组

- **现状**: 当前只有 Control 组的训练结果（500K 步，157.8M，全部 cells）。
  - Source: `model_ckpts/hcmc_vitality_v3_pro6k/` (19 step checkpoints + 1 final)
- **为什么这么做**: 对比实验是论文的核心贡献。没有对比数据，论文只有描述性结果。
- **每个 run 需要**:
  - 训练（含 logarithmic checkpointing）
  - F_mem 评估（1000 samples, DDIM 50-step）
  - Synergy analysis（19 checkpoints）
  - Channel statistics（19 checkpoints）
- **GPU 时间估计**: 取决于 Phase 1.5 的模型大小决策
- **前置依赖**: Phase 2（实验组设计）+ Phase 1.5（训练参数决策）

#### Task 3.6: 每个 run 的评估

- **现状**: 评估脚本已存在（`compute_fmem_1000samples.py`, `channel_statistics.py`, `synergy_analysis.py`）。
  - Source: `scripts/compute_fmem_1000samples.py`, `analysis/channel_statistics.py`, `analysis/synergy_analysis.py`
- **为什么这么做**: 每个 run 需要相同的一套评估指标，才能做有意义的对比。

**🚪 Phase 3 验证门**: 所有 runs 完成，评估数据存在且格式一致。

---

### Phase 4: 采样与证据生成（1-2 天，与 Phase 3 部分并行）

**目的**：生成论文需要的所有视觉证据。

#### Task 4.1: DDIM 50-step F_mem 评估（所有 runs, 1000 samples）

- **现状**: 评估脚本已存在。Control 组的 1000-sample 数据已存在。
  - Source: `scripts/compute_fmem_1000samples.py`
  - Source: `fmem_curve_1000samples.csv` (Control 组)
- **为什么这么做**: F_mem curve 是论文的核心 figure 之一。需要所有实验组的 F_mem curve 做对比。

#### Task 4.2: DDPM 1000-step 采样（关键 checkpoints）

- **现状**: `ddpm_sample.py` 已实现但从未在 v3 checkpoints 上跑过。
  - Source: `sampling/ddpm_sample.py` (117 lines)
  - Source: `experiments_summary.md:176` ("not run on v3 checkpoints")
- **发现**: DDPM 1000-step 理论上生成更高质量样本。`sample_v3.py` 和 `generate_grids.py` 都支持 `--sampling_method ddpm`。
  - Source: `sample_v3.py`, `generate_grids.py`
- **为什么这么做**: 论文发表级 figure 需要最高质量的样本。DDIM 50-step 用于 F_mem（一致性），DDPM 1000-step 用于视觉展示（质量）。

#### Task 4.3: 跑 multi_signal_window.py（每个 run）

- **现状**: 脚本存在（470 lines），但 PNG 从未生成。
  - Source: `analysis/multi_signal_window.py` (470 lines)
  - Source: `analysis/section_2_4/multi_signal_window.png` → 不存在
- **为什么这么做**: 这是论文主图——合并 F_mem, Wasserstein, synergy regime 的 2×2 figure。

#### Task 4.4: 跑 synergy_analysis.py + regime plots（每个 run）

- **现状**: CSV 数据存在（`synergy_metrics.csv`），但所有 PNG plots 缺失。
  - Source: `analysis/section_2_4/synergy/synergy_metrics.csv` ✅
  - Source: `analysis/section_2_4/synergy/regime_evolution.png` → 不存在
  - Source: `analysis/section_2_4/synergy/` 目录只有 3 文件（2 CSV + 1 MD）
- **为什么这么做**: Regime evolution plot 是 synergy 分析的核心视觉证据。

#### Task 4.5: 跑 generate_grids.py（每个 run 的 key checkpoints）

- **现状**: 脚本存在（382 lines），但从未跑过。
  - Source: `scripts/generate_grids.py` (382 lines)
- **为什么这么做**: 论文需要 real vs. generated 的 side-by-side 对比图。3 种模式：grid, comparison, channels。

#### Task 4.6: Paper figure checklist

- **为什么这么做**: 确保论文的每个 figure 都有对应的磁盘文件，没有遗漏。

| Figure | 内容 | 源文件 | 状态 |
|--------|------|--------|------|
| Fig 1 | Pipeline diagram | 手动制作 | 🔲 |
| Fig 2 | F_mem curves（所有 runs 对比） | `fmem_curve_1000samples.csv` × N runs | 🔲 |
| Fig 3 | Multi-signal window（Control） | `multi_signal_window.png` | 🔲 Task 4.3 |
| Fig 4 | Regime evolution（所有 runs 对比） | `regime_evolution.png` × N runs | 🔲 Task 4.4 |
| Fig 5 | Sample grids（real vs generated） | `publication_grids/` | 🔲 Task 4.5 |
| Fig 6 | Channel histograms | `channel_histograms_*.png` | ✅ 存在 |
| Fig 7 | Wasserstein evolution | `wasserstein_evolution.png` | ✅ 存在 |
| Table 1 | Pre-encoding (ID, typologies) | `intrinsic_dimension.txt` | 🔲 Task 0.1 |
| Table 2 | Synergy at key steps | `synergy_metrics.csv` | ✅ 存在 |
| Table 3 | 实验组对比结果 | Phase 3 产出 | 🔲 |

**🚪 Phase 4 验证门**: 100% figure 覆盖。每个 figure 有对应的磁盘文件。

---

### Phase 5: 论文写作（2-3 天）

**目的**：基于 verified evidence 写 §4 和 §5。

#### Task 5.1-5.6: 各节写作

- **§4.1 Dataset & Pre-encoding**: 用 Phase 0 verified 数字
- **§4.2 Memorization Dynamics**: F_mem curves 对比（所有 runs）
- **§4.3 Structure Learning**: Synergy regime 对比（所有 runs）
- **§4.4 Data Curation Experiments**: 核心对比（论文主要贡献）
- **§4.5 Irreducible Characteristics**: 跨 runs 稳定的特征
- **§5 Conclusions**: 综合发现

**🚪 Phase 5 验证门**: 对 §4 每个定量声明跑一遍 deep verification 动作 A（溯源）——每个数字能指向 CSV 或 figure 的具体行/像素。

---

## 七、依赖图

```
Phase 0 (数据审计)
    │
    ├──→ Phase 1 (数据编码反思)
    │         │
    │         ▼
    ├──→ Phase 1.5 (理论核实 + 训练规模)
    │         │
    │         ▼
    └──→ Phase 2 (Data Curation 设计)
              │
              ▼
         Phase 3 (对比实验执行)
              │
              ├──→ Phase 4 (采样与证据生成, 部分并行)
              │         │
              └─────────┘
                        │
                        ▼
                   Phase 5 (论文写作)
```

Phase 1 和 Phase 1.5 可以部分并行（理论核实不依赖 G 通道诊断结果）。
Phase 4 的 Task 4.3-4.5 可以在 Phase 3 的每个 run 完成后立即执行（不需要等所有 runs 完成）。

---

## 八、开放决策点

| # | 决策 | 依赖 | 决策时机 |
|---|------|------|---------|
| D1 | G 通道是否重新编码 | Phase 1.1-1.3 | Phase 1 结束时 |
| D2 | 分辨率维持 128 还是升级 256 | Phase 1.4 + 1.5.4 | Phase 1.5 结束时 |
| D3 | 模型大小（157.8M vs 更小） | Phase 1.5.1-1.5.3 | Phase 1.5 结束时 |
| D4 | 训练步数（500K vs 1M+） | Phase 1.5.1 | Phase 1.5 结束时 |
| D5 | 聚类方法（k-means vs DBSCAN） | Phase 2.1 | Phase 2 开始时 |
| D6 | 实验组数量和组成 | Phase 2.2 | Phase 2 结束时 |

---

## 九、关键引用索引

### 项目内部文档

| 文件 | 关键内容 | 行号 |
|------|---------|------|
| `ISM_report.txt` | RGB channel synthesis 方法论 | 23, 109-123 |
| `ISM_report.txt` | WIP Experiments/Conclusions | 365, 375 |
| `ISM_report.txt` | Future work directions | 387-391 |
| `ISM_report.txt` | Liu et al. (2024) 引用 | 455 |
| `ch2-ism-proposal.md` | RQ3: data manipulation | 21, 51 |
| `experiments_summary.md` | F_mem + synergy 结论 | 71, 88, 111-128 |
| `DDPM-experiment-design-framework.md` | 4-group 对比实验设计 | 156-201 |
| `TODO.md` | k95=33, MLE=3.93 | 88 |
| `TODO.md` | 3 curated datasets | 89-91 |
| `progress_260421.md` | 1000-sample F_mem 发现 | 6, 14-22 |
| `requests_260421.txt` | 用户对聚类的不满 | 34-35 |
| `INTEGRATION_PLAN.md` | Phase 7 subset training | 240-268 |

### 数据文件

| 文件 | 关键数字 |
|------|---------|
| `data_quality_report.txt` | 4,182 cells, G non-zero=2.49% |
| `fmem_curve.csv` | 50-sample F_mem (max 0.0209 at 360K) |
| `fmem_curve_1000samples.csv` | 1000-sample F_mem (max 0.012 at 500K) |
| `synergy_metrics.csv` | Per-checkpoint Pearson r, regime labels |
| `synergy_summary.csv` | Per-checkpoint regime summary |
| `minimum_viable_report.md` | Per-channel mean/std/EMD/Moran's I |

### 代码文件

| 文件 | 关键参数 |
|------|---------|
| `build_hcmc_pipeline_v2.py:57-60` | SUBDIV=20, CALC_UNIT_M=50, TARGET_PX=128 |
| `pre_encoding_analysis.py:298-302` | MLE 简化公式（有 bug） |
| `multi_resolution_id.py:127-136` | MLE 正确公式 |
| `hcmc_vitality_v3_pro6k.yaml` | ch=128, ~157.8M, bs=8, 500K steps |
| `ddim.py` | Default ddim_steps=50, eta=0 |
| `ddpm_sample.py` | DDPM 1000-step (117 lines) |

### 外部文献

| 引用 | 用途 | 验证状态 |
|------|------|---------|
| Bonnaire et al. (2025) | Generalization window 理论 + scaling law | ✅ 存在 |
| Liu, X. et al. (2024) | RGB channel synthesis 方法论 | ✅ DOI: 10.14627/537752020 |
| Haldar et al. (AISTATS 2024) | ID 与分辨率无关 | ✅ PMLR 238:1090-1098 |
| Poly2Vec (ICML 2025) | 统一 Fourier 编码器 | ✅ PMLR 267:55511-55532 |
| Simple Diffusion (ICML 2023) | "Downsampling helps" | ✅ PMLR 202:13213-13232 |

---

## 十、下一轮 chat 候选

按优先顺序:
1. **Phase 0 执行** — 重跑 pre-encoding + 修复 MLE bug + 整合 1000-sample 数据
2. **Phase 1.1** — G 通道稀疏性诊断 + 修复方案实验
3. **Phase 1.3** — 读 Liu et al. (2024) 原文，提取 RGB fusion 方法论细节
4. **Phase 1.5.1** — 读 Bonnaire et al. 的 scaling law，估计 τ_mem
5. **Phase 2.1** — 改进聚类（扩展特征 + DBSCAN 实验）
