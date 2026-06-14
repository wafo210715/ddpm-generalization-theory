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

核心 research question：

  Can a Denoising Diffusion Probabilistic Model learn the low-dimensional manifold of urban built environment from RGB-encoded geospatial data?

  通过三个 analytical dimensions 来检验：

  1. Whether a generalization window emerges during training on urban data
  2. What structural patterns persist during this window
  3. How the distributions of generated samples compare to those of the training data

### "学到 manifold" 的可操作定义（6/14 复查补充）

RQ 中的 "learn the low-dimensional manifold" 必须有可操作定义，否则三个 analytical dimensions 的证据链是松散的。判定标准（按强度递增）：

| 强度 | 判据 | 当前状态 |
|------|------|---------|
| 弱 | 生成样本的 Wasserstein 距离 < 阈值（分布匹配） | ✅ 已满足（500K 处 EMD ~0.05-0.28） |
| 中 | 生成样本的 synergy regime 与训练数据一致（结构匹配） | ❓ 训练数据 synergy 未算（**Phase 0 Task 0.7** 衍生） |
| 强 | 生成样本的内在维度 ≈ 训练数据的内在维度（manifold 匹配） | ❓ 生成样本 ID 未算（可作为 Phase 4 衍生 task） |

论文应明确说"我们用强度 X 的判据"，避免审稿人质疑"凭什么说学到了 manifold"。最严谨的版本是强度 3（ID 匹配），但工作量大；强度 2 是合理折中。

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
| 4 | `experiments_summary.md` | ~~ddpm_sample.py 139 → 117 lines~~ ⚠️ 修正有误：实际文件 139 行，117 是错误数字（6/12 deep verification 发现） |
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
| F7 | **训练日志存在但 summary 是中途快照**——`training_log.csv` 完整跑到 step=500000（2653 行），但 `training_summary.txt` 显示 "Total steps: 260000"（中途快照，未更新） | 低 | `logs/hcmc_vitality_v3_pro6k/.../training_log.csv` line 2653 + `training_summary.txt` line 4 |

---

## 三、关键认识对齐（用户纠正我的地方）

### 3.1 通道表示问题诊断（B/G 通道 Moran's I ≈ 0 的根因）

**对齐结论**：B 和 G 通道在生成图片中 Moran's I ≈ 0，根因是**计算方式抹平了空间模式**，不是"建筑/POI 在城市里没有空间结构"。训练数据的 Moran's I 尚未验证（代码只对生成样本计算；`data_quality_report.txt` 的 Moran's I 因缺 `libpysal` 失败）。

**B 通道**：150m 圆形 buffer 将建筑多边形退化为单一标量（覆盖比），丢失了建筑的形状、朝向、排列模式。92.7% 零值 + P95=0.013 → 生成的图片几乎是全黑底上的稀疏亮点 → Moran's I 自然趋近零。
**Source**: `build_hcmc_pipeline_v2.py:297-344`（`total_area / buffer_area` 计算）

**G 通道**：Shannon entropy 要求 ≥2 POI 且 ≥2 类别，97.5% sub-cell 不满足条件 → 数据本身在 50m 尺度极度稀疏。
**Source**: `build_hcmc_pipeline_v2.py`（entropy 计算逻辑）

**R 通道**（对照）：gravity model 的指数衰减天然产生平滑连续场 → 唯一保留空间结构的通道（Moran's I = 0.251）。

**Liu Xun (2024) 方法对比**：每个通道独立使用各自最高分辨率生成黑白 gradient 图片，然后 RGB fusion。我们的 pipeline 强制三通道对齐到统一 20×20 网格再插值到 128×128。关键区别：她做 image-to-image（融合后的彩色图 → 预测），不需要从 RGB 分解回各通道，因此不需要通道间像素对齐。
**Source**: DOI: 10.14627/537752020

#### 怎么做（3 项行动）

1. **Binary mask / 高分辨率表示**：用 binary mask（有建筑/无建筑）代替建筑覆盖比，保留建筑形态。需要明确"像素是否有建筑"的裁决方式。SUBDIV=20（50m）太粗——单体建筑通常 10-20m，需要 SUBDIV≥40，或直接从 GIS 矢量导出高分辨率 PNG（分辨率由导出设置/DPI 决定）。也可以参考 Liu 方法让各通道独立用最优表示：R=gravity model 栅格、B=矢量栅格化 binary mask、G=POI kernel density，各用各自最适合的分辨率。

2. **150m buffer 检查**：单独拿出数据制作流程，验证 150m buffer 是否符合 HCMC 城市实际情况。当前覆盖比计算对 buffer 大小敏感，需要从实际数据中抽取典型区域（密集城区 vs 郊区）可视化不同 buffer 下的覆盖比分布，判断 150m 是否造成过度平滑或欠平滑。
   **Source**: `build_hcmc_pipeline_v2.py:57-60`（`BUFFER_M = 150`）

3. **计算训练数据的 Moran's I**：当前 `channel_statistics.py` 只对生成样本计算 Moran's I。需要补算训练数据 B/G/R 各通道的 Moran's I，才能判断"训练数据本身就没有空间结构"（→ 表示方式问题，需要改编码）还是"训练数据有结构但生成数据没学到"（→ 模型学习问题，需要调整训练参数或模型容量）。此结果直接指导 Phase 1.4 决策（当前 RGB 编码是否足够）。

**与 Phase 1 的衔接**：行动 1 和 3 的结果共同决定 Task 1.4（编码决策）。如果训练数据 B 通道 Moran's I 也 ≈ 0 → 表示方式确实是瓶颈 → 改编码有价值。如果训练数据 B 通道 Moran's I > 0 但生成数据 ≈ 0 → 模型没学到 → 需要先看训练参数/模型容量问题。

### 3.2 SUBDIV ≠ 20m

| 参数 | 值 | 物理含义 | ISM report 用语 |
|------|---|---------|----------------|
| SUBDIV | 20 | 每边切 20 份（无量纲计数） | "a 20×20 grid of calculation units" (line 109) |
| CALC_UNIT_M | **50m** | 每个 sub-cell 的物理尺寸 = 1000m/20 | "calculation unit (50 m)" (line 109) |
| BUFFER_M | 150m | 每个计算点的分析半径 | "150 m buffer" (line 111) |
| TARGET_PX | 128 | 输出图像像素数 | "128×128 pixels" (line 117) |
| 像素分辨率 | **7.8125 m/px** | 1000m/128（派生值） | 未显式命名 |

**两层分辨率**：计算层 50m（真正信息分辨率）vs 输出层 7.8m/px（双线性插值产物）。
**Source**: `build_hcmc_pipeline_v2.py:57-60`, `ISM_report.txt:109-117`, `data_quality_report.txt:5`

### 3.3 Generalization Window 存在（160K-240K, 约 80K 宽）

Window 由两个独立信号界定：**结构学习 onset**（synergy 首次达到 coupled）→ **记忆化 onset**（F_mem 首次 > 0）。

| 信号 | 来源 | 数值 | 含义 |
|------|------|------|------|
| Synergy 首次 coupled | `synergy_metrics.csv` | **160K**（200K, 300K, 500K 再现） | 结构学习 onset |
| F_mem 首次 > 0 | `fmem_curve_1000samples.csv` | **240K** (f_mem=0.00394, n_mem=4/1000, CI=[0.001,0.008]) | 记忆化 onset |
| F_mem 持续上升至 500K | 同上 | 0.012 at 500K（360K 处小幅回落 0.006→0.005） | 记忆化进行中 |

**Window 宽度 = 240K - 160K = 80K**（不是 40K，也不是 0-200K）。

**⚠️ 关键细节（之前 plan 漏写）**：240K 处 synergy 状态是 **density-driven**，不是 coupled。意味着记忆化开始的瞬间，模型刚从 coupled 退回到单通道主导状态。这本身就是值得论文讨论的发现——记忆化 onset 与 synergy 结构回落同时发生，而非发生在 coupled 稳态中。

**阈值披露**：coupled 判定来自 `synergy_analysis.py:351-356`——要求 BG/BR/GR 所有 pair `|r| > 0.5`。改阈值会移动 window 边界。论文方法学必须披露这个 0.5 阈值。

**统计显著性**：F_mem CI 在 240K 不包含 0（CI=[0.001, 0.008]），onset 是 robust 的。但绝对水平极低（< 1.2%）——需要未训练 baseline 对照才能判断"1.2% 是否高于噪声"（见 Phase 0 Task 0.7）。

`experiments_summary.md` (4/20) 基于 50-sample 说 "No generalization window detected"——被 5/23 的 1000-sample 数据推翻。**6/14 复查发现 summary 仍未更新**（见 Task 0.3 修复）。
**Source**: `progress_260421.md:14-22`, `fmem_curve_1000samples.csv`, `synergy_metrics.csv`, `synergy_analysis.py:351-356`

### 3.4 Data Curation 是当前论文核心，不是下一篇论文

ISM report 的核心研究问题（`ch2-ism-proposal.md:21`）：
> "How does the systematic manipulation of training data composition alter the model's learned manifold?"

计划的对比实验（`DDPM-experiment-design-framework.md:156-201`）：
| 实验组 | 训练数据 | 目的 |
|--------|---------|------|
| Control | 全部 HCMC | 基线 |
| Exp A | 特定 typology 子集 | 看模型学到什么 |
| Exp B | 另一 typology 子集 | 对比 |
| Exp C | Minimal dataset (~ID 个样本) | 最小充分集 |

#### Typology 现状（未解决）

**当前定义方式**：`pre_encoding_analysis.py` 的 `classify_typology()` 从每张 128×128 cell 图片提取 6 个特征（`B_mean, G_mean, R_mean, B_std, G_std, R_std`），StandardScaler 标准化后跑 KMeans，测试 k=3~7 选 silhouette 最高的 → 当前 k=3 胜出，产出 `typology_labels.csv`。
**Source**: `pre_encoding_analysis.py:137-201`

**问题**：
1. G 通道 97.5% 零值 → `G_mean` 和 `G_std` 几乎不提供区分度 → 6 个特征中 2 个是噪声 → 聚类实际只被 B 和 R 驱动
2. 用户评估："three clusters just don't seem so distinctive"（`requests_260421.txt:34`）
3. typology 结果奇怪，根因与 dataset 本身有关（G 通道稀疏性、B 通道覆盖比计算方式——见 3.1）
4. 未在 HCMC 地图上验证聚类的地理合理性

**未解决的依赖**：3.4 实验组中的"特定 typology 子集"和"另一 typology 子集"具体是什么，取决于 typology 改进结果。

#### 已提出的改进思路

- **方法一**（per-image 特征聚类）：保持 per-image 均值/标准差作为特征，但先可视化 3D scatter 分布（B_mean, G_mean, R_mean 三维空间），根据分布形态选择 k-means 或 DBSCAN。可尝试扩展特征（加 Moran's I、skewness、non-zero fraction）和对 G 通道做 log(1+x) transform。
  **Source**: `requests_260421.txt:34`, `TODO.md:54-55`

- **方法二**（per-pixel 聚类 → 图片 typology）：直接对每个 50m pixel 的三通道值（B, G, R）做 DBSCAN 聚类，然后每张图片的 typology 由其内部 pixel cluster 的组成比例决定。更精细但计算量更大。
  **Source**: `requests_260421.txt:34`, `TODO.md:56`

**与计划的关系**：Phase 2 Task 2.1 已计划改进聚类，Phase 0 Task 0.1 需先重跑 `pre_encoding_analysis.py` 产出当前的 `typology_labels.csv` 作为基线。

**Source**: `ISM_report.txt:5,19,69,83`, `ch2-ism-proposal.md:21,51`, `TODO.md:54-56,89-91`, `progress_260421.md:219-253`

### 3.5 SDXL / Latent Diffusion 是 Future Work

当前理论基础是 pixel-space diffusion。JiT 和 latent diffusion 不在当前 scope。
Pro 6000 Blackwell 96GB 的 headroom 用于更大模型或更高分辨率，不用于 latent diffusion。

### 3.6 RGB Fusion 参考文献

**Liu, X., Yang, N., & Tian, R. (2024). *Reinventing Planting Design in Landscape Architecture: A Generative AI Approach*. Wichmann Verlag. DOI: 10.14627/537752020**

ISM report line 23 引用此文献作为 RGB channel synthesis 方法论先例。
**Source**: `ISM_report.txt:23,455`, `ch2-ism-proposal.md:47,95`

### 3.7 采样方法澄清

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

#### Task 0.2: 修复 MLE bug ✅ 已完成（6/12 复查）

- **现状**: ~~`pre_encoding_analysis.py:298-302` 使用简化 Levina-Bickel 公式~~
- **6/12 复查结果**: 代码已修正为完整 Levina-Bickel 公式（`for j in range(k-1): log_ratios += np.log(d_k / (distances[:, j] + eps) + eps)`）
  - Source: `pre_encoding_analysis.py:307-311`（当前已修复）
- **未完成的部分**: 修复后的代码**尚未在真实数据上重跑**，新 MLE 值未知（旧值 3.93，预期 5-15）。重跑在 Task 0.1 中执行。
- **为什么重要**: MLE 是论文方法学的关键数字。如果从 3.93 变成 8-15，manifold hypothesis 的论证力度不同。

#### Task 0.3: 整合 1000-sample F_mem 到 experiments_summary.md

- **现状**: `experiments_summary.md` 写于 4/20，基于 50-sample 数据。1000-sample CSV 于 5/23 生成但从未整合。
  - Source: `experiments_summary.md:3` (date: 2026-04-20)
  - Source: `fmem_curve_1000samples.csv` (timestamp: 2026/5/23 10:30:11)
- **发现**: 50-sample max F_mem=0.0209 at 360K 是小样本波动。1000-sample max=0.012 at 500K，且 5/19 checkpoints 非零（240K-500K），整体呈上升趋势（360K 处有小幅回落）。
  - Source: `fmem_curve.csv` (50 samples) vs `fmem_curve_1000samples.csv` (1000 samples)
- **为什么这么做**: 论文应使用统计功效最高的数据。1000-sample 数据还揭示了 generalization window 的存在（见 Task 0.4）。

#### Task 0.4: 确认 generalization window = 160K-240K（80K 宽）

- **现状**: `experiments_summary.md:71` 说 "No generalization window detected"（基于 50-sample）。1000-sample 数据显示不同图景。
  - Source: `experiments_summary.md:71`
  - Source: `fmem_curve_1000samples.csv`
  - Source: `synergy_metrics.csv`
  - Source: `synergy_analysis.py:351-356`（coupled 阈值 |r|>0.5）
- **发现**:
  - 结构学习完成: coupled regime at 160K, 200K, 300K, 500K（`synergy_metrics.csv`，所有 pairs |r|>0.5）
  - 记忆开始: F_mem > 0 at 240K（`fmem_curve_1000samples.csv`，4/1000 memorized, CI=[0.001,0.008]）
  - 记忆增长: F_mem 整体上升到 0.012 at 500K（360K 处小幅回落 0.006→0.005）
  - **Generalization window = 160K→240K = 80K**（不是 0-200K，也不是 40K）
  - **关键细节**: 240K 处 synergy 是 density-driven，不是 coupled——记忆化 onset 与 synergy 结构回落同时发生
- **为什么这么做**: 这是论文的核心叙事基础。"Window 存在"比"no window"更符合数据，也更容易与 Bonnaire et al. 的理论框架对接。但 window 的精确边界（160K→240K = 80K）必须严谨，否则方法学审查会挑出问题。

#### Task 0.5: 修正 ISM report R/G/B 通道分配矛盾 ✅ 已确认无矛盾（6/12 复查）

- **6/12 复查结果**: line 23 与 line 121 **两处一致**（B=density, G=diversity, R=accessibility），只是列举顺序不同。原本怀疑的"矛盾"是误读。
  - Source: `ISM_report.txt:23` vs `ISM_report.txt:121`
- **不需要任何修复动作**——这个 task 保留作为记录，避免后续重复怀疑。

#### Task 0.6: 确认 train/val/test split 实际数字

- **现状**: README 说 "584/73/74 split"（基于旧的 732 cells）。实际 4,182 cells 的 split 数字未知。
  - Source: `README.md:40` (旧数字)
  - Source: `data/hcmc/processed/splits/train_cells.txt` 等文件存在
- **发现**: splits 目录有 3 个文件（train/val/test），总 cells = 3263+407+409 = 4079。
- **⚠️ 待解释的差异**: 4,182 - 4,079 = **103 cells 未在 splits 中**。可能原因（需在 Task 0.6 执行时确认）：
  - (a) 4,182 = pipeline 保存的总数，4,079 = 进入训练流程的实际数（被数据质量过滤）
  - (b) splits 脚本只覆盖了部分 cells（地理边界、行政区过滤）
  - (c) 其他原因
- **产出**: 4,182 → 4,079 的具体过滤路径 + splits 文件的精确计数
- **为什么这么做**: 论文方法学需要准确的 split 数字。103 cells 的差异若不解释，审稿人会问。

#### Task 0.7: 计算训练数据 Moran's I（Phase 1 前置裁决，迁移自原 Task 1.5）

**🚨 这个 task 必须在 Phase 1 开始之前完成——它是整个 Phase 1 推理链的裁决实验。**

- **现状**: `channel_statistics.py:1029-1036` 只对**生成样本**计算 Moran's I（且只取第一张：`imgs_01[0, ch_idx, :, :]`）。`data_quality_report.txt` 的 Moran's I 因缺 `libpysal` 失败（line 27: "Moran I failed: No module named 'libpysal'"）。训练数据各通道的 Moran's I **从未成功计算过**。
  - Source: `channel_statistics.py:1029-1036`（只取第一张生成样本）
  - Source: `data_quality_report.txt:27`（libpysal 缺失）
- **为什么必须前置**: 没有训练数据的 Moran's I 就无法区分两种情况：
  - **(A)** 训练数据本身就没空间结构 → 表示方式是瓶颈 → Phase 1 改编码有价值
  - **(B)** 训练数据有结构但模型没学到 → 训练问题是瓶颈 → Phase 1 不需要改编码，应该看训练参数
  
  如果不先做 Task 0.7，Phase 1 Task 1.4（编码决策）就是悬空的——可能在错误的方向上做大量工作。
- **产出**: B/G/R 各通道在训练数据（3,263 张）上的 Moran's I（均值 + 分布），与生成数据（500K checkpoint）对比
- **实现**: 写独立脚本（不修改 channel_statistics.py），需安装 `libpysal` + `esda`，或自实现 Moran's I
- **预计耗时**: 30 分钟（含依赖安装）

#### Task 0.8: 未训练 DDPM 的 F_mem baseline 对照

**🚨 这个 task 决定 F_mem=0.012 是否在统计上算"开始记忆化"。**

- **现状**: 当前 F_mem onset 在 240K (0.004, CI=[0.001,0.008])，500K 达到 0.012。**但没有未经训练的 baseline 对照**——不知道 1.2% memorization fraction 是高还是低。
  - Source: `fmem_curve_1000samples.csv`（仅训练后的 F_mem）
- **为什么必须前置**: 论文 claim "F_mem onset at 240K" 隐含 baseline = 0。但如果未训练 DDPM 的 F_mem 也 ≈ 0.005（k-NN gap 方法的噪声水平），那 240K 处的 0.004 可能不是真信号。要稳健地说"onset at 240K"，必须证明 240K 的 F_mem 显著高于未训练 baseline。
- **产出**: 未训练 DDPM（随机初始化权重）的 1000-sample F_mem 数值
- **实现**: 加载随机初始化的 model（`DDPMUnet` 直接实例化，不加载 checkpoint），跑 1000 DDIM 50-step sample，计算 F_mem
- **预计耗时**: 1-2 小时 GPU



| 数字 | 源文件 | 当前状态 |
|------|--------|---------|
| 4,182 cells | `data_quality_report.txt` | ✅ 已验证 |
| k95=?, MLE=? | `intrinsic_dimension.txt` | ❓ 需重跑（Task 0.1 + 0.2） |
| F_mem max=? at step ? | `fmem_curve_1000samples.csv` | ✅ 已验证（0.012 at 500K） |
| 19 checkpoints | `model_ckpts/hcmc_vitality_v3_pro6k/` | ✅ 已验证 |
| Final synergy r values | `synergy_metrics.csv` | ✅ 已验证 |
| 训练数据 Moran's I (B/G/R) | 待产出 | ❓ Task 0.7（Phase 1 前置裁决） |
| 未训练 baseline F_mem | 待产出 | ❓ Task 0.8（onset 显著性裁决） |

---

### Phase 1: 数据编码反思（1-2 天）

**目的**：诊断 B/G 通道问题（见 3.1），决定是否需要重新编码。

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

- **现状**: 基于 Task 0.7（前置）+ Task 1.1-1.3 + 1.6-1.7 的结果做决策。
- **为什么这么做**: 这个决策决定了后续所有 work 的基础。如果改编码 → 重跑 pipeline → 重跑 pre-encoding → 重训模型。如果不改 → 在论文中讨论 limitation。
- **决策标准**（综合 3.1 对齐结论 + Phase 0 Task 0.7 结果）:
  - 训练数据 B 通道 Moran's I ≈ 0（**Phase 0 Task 0.7**）→ 表示方式本身没有空间结构 → 改编码（binary mask 等）有价值
  - 训练数据 B 通道 Moran's I > 0 但生成数据 ≈ 0 → 模型没学到 → 先看训练参数/模型容量，再考虑编码
  - 150m buffer 检查（Task 1.6）显示过度平滑 → 调整 buffer 或换表示方式
  - G 通道修复后 ID 显著变化（比如从 33 降到 20 以下）→ 值得改
  - 如果修复只影响 G 通道的分布但不改变整体 ID → 可以不改，作为 limitation 讨论

#### ~~Task 1.5: 计算训练数据的 Moran's I~~ → 已迁移到 Phase 0 Task 0.7

**6/14 复查修正**：原 Task 1.5 是 Phase 1 推理链的**前置裁决实验**，不应该埋在 Phase 1 内部。已迁移到 Phase 0 Task 0.7，必须在 Phase 1 启动前完成。

理由：Task 1.4（编码决策）依赖 Task 1.5 的结果。如果 Task 1.5 还在 Phase 1 内部，那 Phase 1 启动时 Task 1.4 的输入还没就绪——典型的反向依赖。把 Task 1.5 提到 Phase 0 后，Phase 1 可以基于已知的 Moran's I 结果直接做 Task 1.4 决策。

Phase 1 后续 task 对 Task 0.7 的依赖：
- **Task 1.7 (binary mask)**: 依赖 Task 0.7 结果——只在 (A) 训练数据 B 通道 Moran's I ≈ 0 时才值得做 binary mask
- **Task 1.4 (决策)**: 依赖 Task 0.7 + Task 1.1-1.3 + 1.6-1.7

#### Task 1.6: 150m buffer 检查（3.1 行动②）

- **现状**: `build_hcmc_pipeline_v2.py:57-60` 硬编码 `BUFFER_M = 150`。B 通道的覆盖比 = 建筑 footprint 面积 / 150m 圆形 buffer 面积。覆盖比对这个参数高度敏感——buffer 越大越平滑，越小越保留细节。用户认为 150m 不符合对 HCMC 城市形态的认知。
  - Source: `build_hcmc_pipeline_v2.py:57-60`（`BUFFER_M = 150`）
  - Source: `build_hcmc_pipeline_v2.py:297-344`（覆盖比计算逻辑）
- **为什么这么做**: 如果 150m 造成过度平滑（密集城区所有 sub-cell 都接近满覆盖），那 B 通道的低 Moran's I 可能部分源于 buffer 太大。如果 150m 在郊区造成欠平滑（buffer 内只有零星建筑），那覆盖比的方差可能过大。需要从实际数据中验证。
- **产出**: 从 HCMC 数据中抽取 3-5 个典型区域（密集城区如 District 1、混合区如 Bình Thạnh、郊区如 Củ Chi），可视化不同 buffer 大小（50m / 100m / 150m / 200m）下的覆盖比分布
- **不需要重跑 pipeline**: 直接从 `hcmc_buildings.gpkg` 和 grid 数据做独立分析

#### Task 1.7: Binary mask / 高分辨率表示实验（3.1 行动①）

- **现状**: 当前 B 通道用 150m buffer 覆盖比（连续标量），丢失了建筑的形状、朝向、排列模式（见 3.1 对齐结论）。Binary mask（有建筑/无建筑）可以保留建筑形态，但需要明确每个 pixel 的裁决方式。
  - Source: 3.1 对齐结论
  - Source: `build_hcmc_pipeline_v2.py:297-344`（当前覆盖比计算）
- **为什么这么做**: 如果 **Phase 0 Task 0.7** 确认训练数据 B 通道 Moran's I ≈ 0（表示方式问题），binary mask 是最直接的修复方案。同时参考 Liu Xun (2024) 方法——各通道独立用最优分辨率，不要求通道间像素对齐。
- **实验方案**:
  1. **SUBDIV 升级**: SUBDIV=40（25m）或更高，用建筑多边形直接栅格化为 binary mask（pixel 中心在建筑内=1，否则=0）
  2. **GIS 矢量导出**: 从 `hcmc_buildings.gpkg` 直接渲染高分辨率 PNG（分辨率由 DPI 决定）
  3. **各通道独立分辨率**（Liu 方法）: R=gravity model 栅格（天然平滑，当前分辨率可能够用）、B=binary mask（需要高分辨率）、G=POI kernel density（待实验）
- **产出**: Binary mask 版本的 B 通道样本图片 + Moran's I 对比（vs 当前覆盖比版本）
- **前置依赖**: **Phase 0 Task 0.7** 的结果（确认是否是表示方式问题）

**🚪 Phase 1 验证门**: 能回答以下三个问题：
1. "训练数据的 B 通道有没有空间结构？"（**Phase 0 Task 0.7** → Moran's I 数据，Phase 1 启动前必须就绪）
2. "150m buffer 是否合适？"（Task 1.6 → 可视化证据）
3. "我们的 RGB 编码方法论是否有文献支撑？"（Task 1.3 → Liu 方法对比）

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

- **现状**: 数据显示 window = 160K→240K = **80K**（coupled 首次在 160K, F_mem onset at 240K）。注意 coupled 也在 200K, 300K, 500K 再现——240K 处反而是 density-driven（不是 coupled）。
  - Source: `synergy_metrics.csv` (coupled at 160K, 200K, 300K, 500K)
  - Source: `fmem_curve_1000samples.csv` (F_mem > 0 at 240K)
  - Source: `synergy_analysis.py:351-356` (coupled 阈值 |r|>0.5)
- **发现**: Bonnaire et al. 的 "generalization window" 定义为结构学习完成到记忆化开始之间的时间段。我们的数据显示 **80K 步的窗口**。
- **关键张力**: 240K 处 synergy 已经从 coupled 退回到 density-driven——记忆化 onset 与 synergy 结构回落**同时**发生。这跟 Bonnaire 框架"coupled 稳态中开始记忆化"的预期不完全吻合。要么 (a) 重新解释为"density-driven 是记忆化 onset 的预兆信号"，要么 (b) 承认 synergy 与 F_mem 的耦合关系比理论预期复杂。
- **为什么这么做**: 如果 80K 步的窗口在理论上合理，论文的叙事就有理论基础。如果不合理，可能需要重新解释数据。
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
  5. **方法一**（per-image 特征）：先可视化 3D scatter 再选 k-means vs DBSCAN（见 3.4）
  6. **方法二**（per-pixel 聚类）：对每个 50m pixel 的三通道值做 DBSCAN，图片 typology = pixel cluster 组成比例（见 3.4）
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
  - Source: `sampling/ddpm_sample.py` (139 lines)
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
Phase 0 (数据审计 + 前置裁决)
    │  含: Task 0.7 (训练 Moran's I) ── 裁决 → Phase 1 推理链
    │       Task 0.8 (未训练 baseline) ─ 裁决 → F_mem onset 显著性
    │
    ├──→ Phase 1 (数据编码反思: B/G 通道诊断)
    │         │  内部依赖:
    │         │  ├── Task 0.7 (Phase 0) ──┐
    │         │  ├── Task 1.6 (buffer)    ├→ Task 1.4 (决策)
    │         │  └── Task 0.7 → 1.7 ─────┘
    │         │        (binary mask)
    │         ▼
    ├──→ Phase 1.5 (理论核实 + 训练规模)
    │         │
    │         ▼
    └──→ Phase 2 (Data Curation 设计, 依赖 Phase 1 Task 1.4 决策)
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

- **Phase 0 Task 0.7 和 Task 0.8 是 Phase 1 启动前的前置裁决实验**——不做这两个，Phase 1 推理链悬空。
- Phase 1 内部：Task 1.6（buffer 检查）独立；Task 1.7（binary mask）依赖 Phase 0 Task 0.7 的结果；Task 1.4（决策）依赖 Task 0.7 + 1.1-1.3 + 1.6-1.7 全部完成。
- Phase 1 和 Phase 1.5 可以部分并行（理论核实不依赖 B/G 通道诊断结果）。
- Phase 2 依赖 Phase 1 的 Task 1.4 决策结果（是否改编码决定了 Phase 2 的聚类基础）。
- Phase 4 的 Task 4.3-4.5 可以在 Phase 3 的每个 run 完成后立即执行（不需要等所有 runs 完成）。

---

## 八、开放决策点

| # | 决策 | 依赖 | 决策时机 |
|---|------|------|---------|
| D1 | B/G 通道是否重新编码（含编码方式：覆盖比 vs binary mask vs 各通道独立分辨率） | Phase 1.1-1.7 | Phase 1 结束时 |
| D2 | 分辨率维持 128 还是升级 256 | Phase 1.4 + 1.5.4 + 1.7 | Phase 1.5 结束时 |
| D3 | 模型大小（157.8M vs 更小） | Phase 1.5.1-1.5.3 | Phase 1.5 结束时 |
| D4 | 训练步数（500K vs 1M+） | Phase 1.5.1 | Phase 1.5 结束时 |
| D5 | 聚类方法（k-means vs DBSCAN；per-image vs per-pixel，见 3.4 两种方法） | Phase 2.1 | Phase 2 开始时 |
| D6 | 实验组数量和组成 | Phase 2.2 | Phase 2 结束时 |
| D7 | 150m buffer 维持还是调整 | Phase 1.6 | Phase 1 结束时 |

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
| `TODO.md` | 两种聚类改进方法（per-image vs per-pixel） | 54-56 |
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
| `build_hcmc_pipeline_v2.py:57-60` | SUBDIV=20, CALC_UNIT_M=50, BUFFER_M=150, TARGET_PX=128 |
| `build_hcmc_pipeline_v2.py:297-344` | B 通道覆盖比计算（150m buffer） |
| `pre_encoding_analysis.py:137-201` | Typology 聚类（6 特征 + KMeans） |
| `pre_encoding_analysis.py:298-302` | MLE 简化公式（有 bug） |
| `multi_resolution_id.py:127-136` | MLE 正确公式 |
| `channel_statistics.py:1029-1036` | Moran's I 只对生成样本第一张计算 |
| `hcmc_vitality_v3_pro6k.yaml` | ch=128, ~157.8M, bs=8, 500K steps |
| `ddim.py` | Default ddim_steps=50, eta=0 |
| `ddpm_sample.py` | DDPM 1000-step (139 lines) |

### 外部文献

| 引用 | 用途 | 验证状态 |
|------|------|---------|
| Bonnaire et al. (2025) | Generalization window 理论 + scaling law | ✅ 存在 |
| Liu, X. et al. (2024) | RGB channel synthesis 方法论 | ✅ DOI: 10.14627/537752020 |
| Haldar et al. (AISTATS 2024) | ID 与分辨率无关（实验验证 64/128/320px ID 近似不变） | ✅ PMLR 238:1090-1098, "Effect of Ambient-Intrinsic Dimension Gap on Adversarial Vulnerability" |
| Poly2Vec (ICML 2025) | 统一 Fourier 编码器 | ✅ PMLR 267:55511-55532 |
| Simple Diffusion (ICML 2023) | "Downsampling helps" | ✅ PMLR 202:13213-13232 |

---

## 十、下一轮 chat 候选

按优先顺序（6/14 复查后调整）:
1. **Phase 0 Task 0.7** — 计算训练数据 B/G/R 各通道 Moran's I（**Phase 1 启动前的前置裁决实验**，30 分钟）
2. **Phase 0 Task 0.8** — 未训练 DDPM 的 F_mem baseline（决定 F_mem=0.012 是否算"开始记忆化"，1-2 小时 GPU）
3. **Phase 0 Task 0.1** — 重跑 pre-encoding（MLE 公式已修正，需重跑获取新数值）+ 整合 1000-sample 数据
4. **Task 1.6** — 150m buffer 检查（从 `hcmc_buildings.gpkg` 抽典型区域可视化）
5. **Phase 1.1** — G 通道稀疏性诊断 + 修复方案实验
6. **Phase 1.3** — 读 Liu et al. (2024) 原文，提取 RGB fusion 方法论细节
7. **Task 1.7** — Binary mask 实验（依赖 Phase 0 Task 0.7 结果）
8. **Phase 1.5.1** — 读 Bonnaire et al. 的 scaling law，估计 τ_mem
9. **Phase 2.1** — 改进聚类（per-image vs per-pixel，见 3.4 两种方法）

---

## 十一、6/12 Deep Verification 复查

对本文档的 7 条承重主张做了第二轮 deep verification（6 个 explore agent），发现并修正以下问题：

### 已修正的错误（5 处）

| # | 文件 | 修正内容 |
|---|------|---------|
| 7 | `experiments_summary.md:156` | ddpm_sample.py 行数 117 → 139（上一轮修正有误） |
| 8 | `STAGE2_RESEARCH_SYNTHESIS.md:233` | Poly2Vec 标题更正为官方标题 "Polymorphic Fourier-Based Encoding of Geospatial Objects for GeoAI Applications" |
| 9 | `hcmc_vitality_v2.yaml:2` | 732 → 4,182 cells（残留旧数字） |
| 10 | `progress_260611.md` (本文件) | F_mem 描述从"单调上升"改为"整体上升（360K 处小幅回落 0.006→0.005）" |
| 11 | `progress_260611.md` (本文件) | Coupled regime 从"160K-200K"扩展为"160K, 200K, 300K, 500K"（完整列表） |

### 经验证确认正确的主张

| # | 原怀疑 | 核查结果 |
|---|--------|---------|
| Haldar et al. (AISTATS 2024) | 论文主题是"对抗鲁棒性"而非"ID 与分辨率无关" | ✅ 论文实验确实验证了 ID 在 64/128/320px 下近似不变（arXiv:2403.03967），引用有效 |
| ISM R/G/B 通道分配 | line 23 vs line 121 是否矛盾 | ✅ 两处一致（B=density, G=diversity, R=accessibility），只是列举顺序不同 |
| MLE 公式 | pre_encoding_analysis.py 是否仍用简化公式 | ✅ 代码已修正为完整 Levina-Bickel 公式（for loop over j=1..k-1），上一轮模型已修复 |

---

## 十二、6/14 Deep Verification 复查（基于 plan 自身，非底层 claim）

对 progress_260611.md 这个 plan **本身**做了第三轮 deep verification（不是核查事实 claim，而是核查 plan 的内部一致性、结构性盲区、反向依赖）。方法：按 deep-verification skill 的 6 动作逐条核查 5 条承重主张 + 8 条结构性盲区。

### 修正的问题（共 9 处）

| # | 位置 | 问题 | 修正 |
|---|------|------|------|
| 12 | Section 3.3 + Task 0.4 + Task 1.5.2 | Window 长度三处不一致（0-200K / 40K / 80K） | 统一为 **160K→240K = 80K**；明确 240K 处 synergy 是 density-driven |
| 13 | Section 3.3 | synergy regime 阈值 \|r\|>0.5 未披露 | 加阈值披露（来自 `synergy_analysis.py:351-356`） |
| 14 | Phase 0 Task 0.2 | 已完成但仍列在 Phase 0 待办 | 标记 ✅ 已完成（6/12 已修代码，新值待 Task 0.1 重跑） |
| 15 | Phase 0 Task 0.5 | 已确认无矛盾但仍列在 Phase 0 待办 | 标记 ✅ 已确认（line 23 vs 121 一致） |
| 16 | Phase 0 / Phase 1 | Task 1.5（训练 Moran's I）反向依赖——是 Phase 1 推理链的裁决实验却被埋在 Phase 1 内部 | 迁移到 Phase 0 Task 0.7，必须在 Phase 1 启动前完成 |
| 17 | Phase 0（新增） | 缺 F_mem baseline 对照——F_mem=0.012 无法判断是否高于噪声 | 新增 Task 0.8（未训练 DDPM baseline F_mem） |
| 18 | Phase 0 Task 0.6 | 4,182 vs 4,079 splits 差 103 cells 未解释 | 加待解释差异说明（3 种可能原因） |
| 19 | RQ 部分 | "学到 manifold" 无可操作定义 | 加 3 强度判据表（弱=分布匹配 / 中=结构匹配 / 强=ID 匹配） |
| 20 | 二节 F7 | "训练 loss 曲线不可恢复" — 错误，日志实际存在 | 修正为 "summary 是 260K 中途快照，log 完整跑到 500K" |

### 同步修改的文件

| 文件 | 修改 |
|------|------|
| `discussion/experiments_summary.md` | Section 4 全部重写：从"No window"改为"window = 160K-240K"；加 6/14 changelog；Section 1 F_mem 行更新；Outstanding Item #5 修正（日志存在） |

### 经核查确认正确的主张

| # | 主张 | 核查结果 |
|---|------|---------|
| C3 数据集规模 | 4,182 cells | ✅ `data_quality_report.txt:9` 确认（splits 差异已在 Task 0.6 标注） |
| C5 训练规模 | 500K steps, 157.8M params, 19+1 checkpoints | ✅ `training_log.csv` line 2653 + checkpoint 文件大小（1.77GB ≈ 157.8M × 12 bytes model+Adam） |
| C4 MLE 修复 | 已用完整 Levina-Bickel | ✅ `pre_encoding_analysis.py:307-311` |

### 关键净结论

**plan 整体靠谱度：高**（约 80%）。事实基础扎实，6 phase 研究逻辑合理，决策点 D1-D7 抓得准。

**主要风险**已通过本论修正前置到 Phase 0：
- Task 0.7（训练 Moran's I）：**Phase 1 启动前必须做**——决定 Phase 1 整个推理链是否成立
- Task 0.8（未训练 baseline）：**论文 claim F_mem onset 前必须做**——决定 onset 是否统计显著

**未在本轮修正但已在 plan 内标注的开放项**：
- Task 1.5.2 中 240K 处 density-driven（不是 coupled）的理论解释——这是 plan 自己已经识别但没消化的张力
- Bonnaire et al. 的 window 预测是否与 80K 实测一致——需要 Phase 1.5.1 读论文后回答
