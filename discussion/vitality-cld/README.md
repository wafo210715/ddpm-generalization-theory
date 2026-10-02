# Per-paper CLD 提取（issue #3）— 模板 v2

**目的**：`vitality_indicators_review.md` §4/§5 的去重表是节点目录，但按它画出来的图没有回路。
本目录一篇论文一个文件，把每篇的 findings 转成可回溯的 CLD 片段，最后再整合。
方法讨论见 issue #3 及其评论。

**版本**：v2.1（2026-09-15 晚）。v1 pilot 三篇后按 `pilot-notes-260915.md` T1–T6 与三个方法论问题（符号冲突 / 非线性 / 指标等价）修订；v2.1 加 `perspective` 字段、一段话与英文条目、图依赖边规则。三篇 pilot 已按 v2 补齐，其中 Tan-2026 的 ISM 边已按 Fig. 8 核对。

**铺开**：2026-09-15 晚，剩余 26 篇按 method-class 分 4 批由子会话提取完成（提示词 `EXTRACTION-PROMPT.md`），29 篇齐。交付表、模板问题汇总、整合前待裁定见 `rollout-notes-260915.md`。**模板 v3 尚未修订**：rollout-notes §2 的枚举缺值、字段缺位、词表变动待合并。

## 0. 三条原则（先于模板）

1. **边极性是 ceteris paribus 的偏效应。** 回归总效应、SHAP 平均值都不是极性，是多条路径的叠加。一条边只能有一个极性；出现"+ 和 −"先拆路径，不并存。
2. **节点是 construct，不是 indicator。** 同一 construct 的不同操作化（POI 熵 vs land-use 熵 vs 到各类 POI 距离）挂在同一节点下的 indicator ledger 里；两个 indicator 稳定给出相反符号 = construct 该拆，不是 indicator 该删。
3. **CLD 不表达形状。** U 型 / N 型属于方程层。CLD 层的做法是拆成多个单调机制的叠加，或标回路主导权切换（dominance regime）；只在兜底时用边注 `| 条件`。
5. **论文把某个 construct 分了类型的，图上保留类型，不压成一个节点。**（2026-09-17 用户要求）不同类型的 capacity 和 functionality 不同，对活力的边也不同：路网类型（gridiron / organic / radial / no pattern，Chen 2021）各自有均值、机制叙事和回路；绿量类型（垂直绿化 / 行道树 / 草地 / 小公园 / 大公园 / 森林）吸水、降温、可达、遮蔽的功能都不同。单篇能拆的当场拆；跨篇的类型表（typology ledger）在合并时建。
4. **同一 construct 从不同视角采集，可能在因果链上扮演不同角色。** 卫星 NDVI（top-down）和街景绿视率（eye-level）都叫 greenness，但前者测的是地表覆盖、后者测的是行人看到什么；一个可能是开发强度的副产物（上游被决定），另一个可能是感知的直接输入（下游中介）。同理 building density（底图，top-down）vs sky view（街景，eye-level）。节点表必须填 `perspective`，整合时 perspective 不同的 indicator 不当作同一条边的重复证据。

## 1. 每篇文件结构

```text
一段话          中文，平实句子：作者怎么测活力、发现了什么、用 CLD 看对我们有什么启发；术语留给表格
EN entry        英文 ≤ 300 词（2026-10-02 起，原为 ≤ 200），供 annotated bibliography；同上三层内容
0. 元数据        论文、txt 路径、method-class、单元/尺度、地域、样本期
1. DV            这篇的 vitality 是哪个 stock；原句 + 行号；指向圈内哪个子节点
2. 节点表        论文变量 → construct 节点 + indicator / data / unit-scale / sample-range
3. 边表          见 §3 字段
   3a coef       系数 / SHAP / SEM 直接效应
   3b corr       相关矩阵、共线性剔除、双变量 vs 偏回归对照
   3c path       SEM 间接效应、BN 路径
   3d interaction PDP / 交互 / 阈值
   3e hierarchy  ISM 层级（expert）
   3f narrative  Discussion 机制叙事；每个负号/反直觉符号作者怎么解释（必填）
   3g limitation 作者承认的反向因果、双向、自选择
   3h DV→DV      多维 DV 之间、DV → 下游变量
4. Mermaid       实边 -->  虚边 -.->  专家层级 ==>  调节变量虚线指边
5. 回路          论文内闭合的；论文只给了一半的；dominance regime
6. 提取备注      与去重表的出入、词表缺行、模板不够的地方、**图依赖边**
```

**图依赖边**：SEM 路径图、ISM 层级图、BN 网络图、PLS-SEM 结构图里的边，如果正文文字没有逐条写出来，txt 里是恢复不了的。这种情况在 §6 列 `图依赖边：Fig. N，需人工看图`，只写文字能支持的边，**不猜**。Tan-2026 的 ISM 边就是按截图补的先例。

## 2. 节点字段

| 字段 | 取值 | 说明 |
|---|---|---|
| `node` | 词表 slug（§5） | construct 层；不在词表的标 `new-node` |
| `construct` | 语义 | 例：功能混合度 |
| `indicator` | 操作化 | 例：14 类 POI Shannon 熵 |
| `data` | 数据源 | 例：高德 POI 2019 |
| `unit-scale` | 分析单元 + 缓冲半径 | 例：街道段，300 m |
| `sample-range` | min / max / mean | 决定 operating range；表里没有写 `未报告`；txt 表格错位写 `未提取（txt 错位）`，**不猜** |
| `perspective` | top-down / eye-level / 3d-computed / survey / transactional / registry | 数据采集视角，类别依数据定，可细分（如 top-down.satellite / top-down.footprint）：卫星、底图、路网、POI 坐标 = top-down；街景、实地计数 = eye-level；**由 3D 模型 / 点云 / 建筑高度计算后压成 2D 数值的 = 3d-computed**（如 DEM 算的天空可视域、绿量体积、FAR / 建筑体量、句法 integration）；问卷 / Place Pulse = survey；点评、房交、手机信令 = transactional；人口登记、房产登记 = registry。**图上每个节点都要显示这个标签**（同名不同视角的 greenery 靠它区分） |
| `role` | IV / DV / mediator / moderator / control / component | 论文自己给的角色 |
| `region` | 城市 | 从元数据继承，多城论文逐节点标 |

## 3. 边字段

| 字段 | 取值 | 说明 |
|---|---|---|
| `source` / `target` | 节点 slug | target 指向 vitality 圈时**必须写子节点**：`vitality_social` / `vitality_economic` / `vitality_cultural` / `vitality_perceived` / `vitality_presence`（在场人口）/ `vitality_composite`（合成分） |
| `sign_direct` | + / − / ns / ? | 偏效应；只有 coef / path / interaction 能填 |
| `sign_total` | + / − / ns | 总效应、平均边际效应、双变量相关 |
| `sign_conditional` | + / − | 条件成立时的符号；配 `condition` |
| `condition` | 文字 | 例：`FAR > 0.25`、`伴 bus stop 多` |
| `moderator` | 节点 slug | 调节这条边的变量；Mermaid 里虚线指边 |
| `effect-type` | direct / total / average-marginal / conditional / bivariate | 与 sign 列对应 |
| `evidence` | coef / corr / path / interaction / hierarchy / narrative / limitation / inferred | 见 §4 |
| `operating-range` | source 变量在样本中的区间 | 从节点 `sample-range` 继承；有阈值时写阈值两侧 |
| `line` | txt 行号 | 主键 |
| `quote` | 原句 | 必须 grep 核过 |

**符号冲突处理顺序**（整合时，对每一对跨论文冲突）：
DV 子节点是否相同 → effect-type 是否相同（直接 vs 总）→ 是否有 moderator（地域 = 隐藏调节变量）→ 任一篇 Discussion 是否把方向反过来（3f/3g）→ 最后才判为非单调，按原则 3 拆机制。

## 4. evidence 取值与画法

| 值 | 含义 | Mermaid |
|---|---|---|
| `coef` | 回归系数、SEM 直接效应、SHAP 符号 | `-->` 实 |
| `path` | SEM 间接效应、BN 路径 | `-->` 实 |
| `interaction` | PDP / 交互 / 阈值 | `-->` 实，标签 `\| 条件` |
| `hierarchy` | ISM 层级；邻接矩阵是专家给定 | `==>` 粗，标 expert |
| `corr` | 相关矩阵、共线性剔除、双变量 vs 偏回归翻转 | `-.->` 虚 |
| `narrative` | Discussion 机制叙事 | `-.->` 虚 |
| `limitation` | 作者承认的反向因果 / 双向 / 自选择 | `-.->` 虚 |
| `inferred` | 我们的推理 | `-.->` 虚，标 inferred |

调节变量：`M -. "moderates" .-> edge_id`（Mermaid 不支持指边，用一个小节点 `((e12))` 代表边，M 指向它）。

## 5. 节点词表

节点 id = construct。同一 construct 的不同 indicator 记在各篇的节点表里，整合时汇成 ledger。

| slug | construct | 已知 indicator 变体 |
|---|---|---|
| `POI_density` | 设施供给量 | POI 计数 / 核密度 / 分类子集（`POI_food`、`POI_business`、`POI_life_service`、`POI_shopping`、`POI_cultural`） |
| `functional_mix` | 功能混合度 | `POI_entropy`（POI 熵）/ `landuse_mix`（land-use 熵）/ Kim balance index / Tan mixing degree。**v2 起 `POI_entropy` 与 `landuse_mix` 并入此 construct，作为 indicator 变体** |
| `dist_facility` | 到特定设施距离 | `.medical` / `.school` / `.mall` / `.plaza` …（Tan、Zeng、Gómez-Varo）。**v2 新增** |
| `road_density` | 路网密度 | 全路网 / 主干路（Li22）/ 300 m 内道路数（Li21） |
| `intersection_density` | 局部连通性 | 交叉口密度；受 `morphology_type` 调节（Chen21） |
| `space_syntax` | 构型可达性 | global / local integration、choice |
| `FAR` | 开发强度 | FAR / FSI / 类 FAR |
| `building_density` | 建筑覆盖 | GSI / BCR / 单位长度建筑数（Li21） |
| `building_height` | 体量 | MBH / 层数 |
| `sky_view` | 街道开敞度 | 街景天空占比（Li21、Tan）/ openness（Wu）。**v2 新增** |
| `greenness` | 绿量 | NDVI / 绿视率 / 绿地占比 / street tree（Chen24）。**合并时按类型拆**（typology ledger）：垂直绿化 / 行道树 / 草地 / 口袋公园 / 大公园 / 森林；现有证据：大面积公园 −（Mouratidis、Jiang MGD、Gómez-Varo/Cai border vacuum）、小公园 +（Scepanovic <1 km²、口袋公园）、行道树 +（Chen24）、街景绿视率 −（Li21 经/文、Wu） |
| `morphology_type` | 路网类型 | gridiron / organic / radial / no pattern（Chen21）；图上保留四个子节点；调节 `building_density → vitality`（radial 建筑密度最高活力最低） |
| `dist_center` | 区位 | 到市中心距离 |
| `dist_park` | 到绿地距离 | |
| `transit_access` | 公交可达 | `.bus` / `.subway`（计数或距离，距离方向相反） |
| `house_price` | 场所社会经济属性 | 房价 / 地价 / 租金 |
| `building_age` | 更新动态 | |
| `terrain` `dist_water` `rail_barrier` `morphology_type` `sidewalk` `lighting` | 同 §4 | |
| `pop_density` | 静态人口 | census / WorldPop / 居民数 |
| `employment_density` | 就业密度 | 招聘核密度（Li21）/ 工作人口（Lee-Kang）。**v2 新增** |
| `heatmap` `mobile_signal` `OD_flow` `social_media` `NTL` | 需求侧代理 | 多数作 DV；作 IV 时是 vitality 的另一 proxy，不是上游 |
| `perception_*` | 感知 | `_safe` `_lively` `_beautiful` `_wealthy` |
| `human_capital` `GDP` `demography` | 同 §5 | |
| `social_cohesion` | vitality 下游 | Mouratidis。**v2 新增** |

## 6. vitality 圈（sector boundary）

vitality 按 stock 类型拆子节点，画图时用一个圈把它们围起来，其他因子的箭头可以指进圈；**边表里 target 必须写子节点**。子节点：

| 子节点 | stock | 例 |
|---|---|---|
| `vitality_presence` | 在场人数 / 设备数 | 手机信令、百度热力、CDR |
| `vitality_social` | 到达/出发流量 | Li21 OD 点 |
| `vitality_economic` | 需求加权商业存量 | Li21 点评加权商户 |
| `vitality_cultural` | 文化供给存量 | Li21 文化 POI |
| `vitality_perceived` | 居民感知 | Mouratidis 量表、Paköz |
| `vitality_composite` | 合成指数 | Zeng、Yue、Cai … |

圈内子节点之间的边（Li21 三维互促）记在 3h。

## 7. 回路字段

| 字段 | 说明 |
|---|---|
| `loop-id` | R-集聚、B-饱和 … |
| `segments` | 每段边的 evidence 与来源论文 |
| `closed-in-paper` | 是 / 否（只给了哪几段） |
| `dominance-regime` | 什么条件下该回路主导；阈值写在这里而不是边上 |

## 8. 整合（未开始；2026-09-15 用户定义）

**目的**：展示现在的 vitality 研究在概念层面能揭示什么。图不追求准确，而是要在视觉上反映现有研究的不足：哪些边是真正分析过的，哪些回路只是推理出来的。

**契约（2026-09-17）**：CLD 的目标是画出所有 indicator 之间的因果回路。先一篇一篇画，再合并。**合并只收 DV 是单一 indicator 的论文**（presence / perceived / 单一流量）；DV 是 composite 的论文单独放，先看到纯 indicator 版本再处理。多尺度暂不考虑。

三件事：
1. 叫法不同、意思相同的节点**并起来**（按 construct）
2. 叫法相同、数据表征不同的节点**分开**（按 perspective / indicator；典型：卫星 NDVI vs 街景绿视率）
3. 不同论文各给了一段的回路**接起来**

**视觉约定**：
- 实线 = 论文实际分析过的边（coef / path / interaction / hierarchy），不区分直接还是间接
- 虚线 = 推理出来的边（narrative / limitation / corr / inferred），回路闭合靠的多半是这种
- 颜色 = 供给侧 vs 需求侧（节点和边都标）；T4 类合成指数含供给成分，在圈内用供给色标出
- vitality 用一个圈围住子节点，其他因子的箭头按需指进圈
- **每个节点带 perspective 小标签**（top-down / eye-level / 3d-computed / survey / transactional / registry）
- **每张图带方法徽章**（图顶或图角）：① 模型类型：linear / nonlinear / path (SEM) / descriptive；② 变量筛选：有无共线性剔除 / 逐步剔除 / 指标剪枝，剔了什么（没报告写 n.r.）。用户看到徽章再决定要不要细聊

流程：同一条边多篇证据时按 §3 冲突处理顺序判断是否真冲突，真冲突才进原则 3；回路只在整合层闭合。逐批讨论顺序：A（SEM 3 + ML 4）→ B（回归 6）→ C（6）→ D（描述 10）。
