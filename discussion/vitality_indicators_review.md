# Vitality indicators from the literature — correlationist 认知清单

**Status: v2 (2026-09-12). 按用户新框架重做：提取单位是「因子 + 效应形态」，
不是指标系统本身。分类是我们做的（作者自己不做 BE/Human 划分），
效应形态从各文的回归/关联结果里读出来。**

**这份文件的最终产出是两张去重因子表（§4 BE、§5 Human），它们是 causal loop
diagram 的节点清单，也是 HCMC 数据采集的对照表。**

配套文件：
- 逐篇明细（含 txt 行号与短引文证据）：`vitality-extraction/batch-A.md` 至 `batch-D.md`
- 决策表（选哪些因子做成通道、半径、数据源）：`indicator_specification.md`
- 旧版 v1（指标家族视角，含 T1–T7 与 V1–V8 的事实层，已被抽查 17/18）：本节 §3 沿用其异常清单，附注使用须知

## 分类约定（2026-09-10 用户裁定，比论文自己的划分优先）

| 指标类型 | 归 | 备注 |
|---|---|---|
| 任何 POI 计数/密度/熵/核密度 | BE | 无论原文放在哪一侧 |
| 静态人口存量（census、居住人口密度） | Human | |
| 夜间灯光 | Human | 经济活动代理，provisional |
| 主观感知分数（Place Pulse 类、问卷量表） | Human | 无论原文作 IV 还是 DV |
| 房价/地价、社区年龄、地形、环境质量 | 待裁定 | 单列，不硬塞 |

「原角色」列记论文自己给这个角色是什么（IV / DV 成分 / 分组变量 / 控制变量），
与我们的 BE/Human 分类互相独立。

---

# 1. Per-paper 总览：vitality 如何被表征

**表征列的读法**：`flat` = 活力是一个可直接观测/询问的量（人数、签到数、感知评分）；
`composite` = 活力是一个指数系统，由若干成分加权/合成。

| # | 论文 | 城市 / 尺度 | 表征 | vitality 具体测的是什么 |
|---|---|---|---|---|
| 1 | Zeng 2018 | Chicago + Wuhan / block group、community | composite | DALD 四维 12 指标 + spatial TOPSIS 排名 |
| 2 | Liu 2019 | Nanjing / 14,236 基站单元，半小时粒度 | flat | 手机信令在线人数时间序列 |
| 3 | Xia 2020 | 五市 / street block | flat ×2（昼/夜） | 昼：点评小餐饮核密度×评论数；夜：VIIRS 辐射 |
| 4 | Kim 2020 | Seoul / block group | flat | 工作日 6am–9pm 手机信令服务人口 |
| 5 | Yang & Pan 2020 | 中国 326 市 / city 面板 | flat | 平均人力资本 HC（EIC 法）；NTL 作验证 DV |
| 6 | Mouratidis & Poortinga 2020 | Oslo / neighborhood | flat | 3 题 Likert 感知量表（α=0.904） |
| 7 | Pan 2021 | 成渝 140 区县 / district-county | composite | PLS-SEM 潜变量 UV（NTL + POI 计数） |
| 8 | Li 2021 | 成都三环内 / 4434 街道段 | flat ×3（社/经/文） | 社=taxi+共享单车 OD；经=点评评分加权商户数；文=300 m 文化 POI 计数 |
| 9 | Chen/Biljecki 2021 | 9 城 / ~1 km grid | composite | vitality score 0–100（POI 核密度 + NTL + WorldPop 人口 + Airbnb；tweet 剔除） |
| 10 | Yue 2021 | HCMC vs Shanghai / sub-district | composite | projection pursuit，三维 9 指标合成 |
| 11 | Li 2022 | 武汉 / 1025 社区 | composite | 街景 6 类要素计数 + 微博签到，两者平均 |
| 12 | Liu 2022 | 长三角 27 市 / city | composite | vital triangle，6 维 28 指标，AHP+熵权 |
| 13 | Scepanovic 2021 | 意大利 6 城 / census district | flat | TIM 移动互联网活动密度 |
| 14 | Gómez-Varo 2022 | Barcelona Nou Barris / 50 m grid | composite | JANE Index v2，22 变量 6 条件分（潜在活力条件） |
| 15 | Wu 2023 | 深圳 / 500 m grid | flat | 腾讯 EasyGo 实时人口密度（昼/夜分开） |
| 16 | Jiang 2024 | 北京 / TAZ | composite | PCA 三源（NTL + 手机信令 + OD 流）合成 |
| 17 | Chen 2024 | SF / census tract + block group | flat | IUNV = Flickr 地理照片计数（12 主题聚类辅助解读） |
| 18 | Collins 2024 | 意大利 7 城 / TIM cell | flat | CDR 活动量，15 分钟粒度；**DV 是男女活力差** |
| 19 | Dogan & Lee 2024 | 首尔 / 500 m grid | composite | 10 变量 FA 合成 3 指标（intensity/diversity/consistency） |
| 20 | Wang 2025 | 深圳 / 300 m grid | composite | 改进高斯 2SFCA：9 类点评评论数供给 ÷ 居住区人口需求 |
| 21 | Cai 2025 | 北京 / block | composite | 6 维 14 指标 z-score 加权 + SOFM 聚 4 类 |
| 22 | Tan 2026 | 深圳 / 500 m grid | flat | 百度慧眼 14 天平均人口热力 |
| 23 | Shi 2021 | 上海 / 16 区年度面板 | composite | C1–C18 三系统，离差最大化权重 |
| 24 | Lee & Kang 2022 | 首尔 / 424 行政洞，日频面板 | flat | SKT 浮动人口访问密度（6:00–21:00） |
| 25 | Paköz & Işık 2022 | Istanbul / 39 district，问卷 n=337 | composite | PCA 两因子；vitality 因子主载荷 vitality 0.711 + mobility 0.799 |
| 26 | Shi 2023 | 绵阳 / 栅格-斑块 | composite | NTL + 百度热力 + 夜间 POI 叠加，自然断点 4 级 |
| 27 | Liu 2025 | 南京 / 712 block | composite | V = P + S·H（物理 + 社会功能 × Place Pulse Lively 修正） |
| 28 | Qin 2025 | 南京 / 509 TAZ | composite | 社会/文化/经济/创新四维熵权合成 |
| 29 | Williams 2019 | 成都+沈阳 / 300 m 居住网格 | composite（反向） | Hansen 重力 amenities score，低分聚类=鬼城 |
| 30 | Doan & Zhang 2025 | —（综述） | — | PRISMA 941→191 篇；确认定义无共识 |
| 31 | Nathansohn & Lahat 2022 | 海法 / 质性 | — | 概念重构 vitalisation，无量化指标 |
| 32 | Ruszczyk 2023 | 孟加拉两小城 / 质性混合 | — | liveability 参与式评估，无量化指标 |

**分布**：29 篇量化论文中 flat 12、composite 17。composite 占多数，且其中
10 篇（旧版 T4 清单）的指数里含供给指标——这正是 §2.1 批判的对象。

---

# 2. Per-paper 因子与效应形态

## 表 A · flat 论文（12 篇）

效应列读法：`↑` 正相关 / `↓` 负相关 / `阈值[x]` / `倒U` / `非线性` /
`不显著` / `未报告` / `n/a`。逐篇全量因子见 `vitality-extraction/batch-*.md`。

| 论文 | vitality DV | BE 因子（→ 效应） | Human 因子（→ 效应） | 方法 |
|---|---|---|---|---|
| Liu 2019 | 手机信令人数 | POI 密度↑、POI 熵↓（有序混合好）、richness/Simpson 不显著 | — | SOM 聚类后特征对比 |
| Xia 2020 | 昼餐饮核密度 / 夜 NTL | FAR↑、BCR↑、土地利用混合↑ | — | Moran's I + LISA |
| Kim 2020 | 手机信令服务人口 | FSI↑、住宅 GFA↑、balance index 1/4↑、gamma index↑、tare space↓、mesh↓、plot 占比↓ | — | 空间自回归（17 IV） |
| Yang & Pan 2020 | 平均人力资本 | — | HC↑、人口↑ | OLS-FE（NTL 验证） |
| Mouratidis 2020 | 3 题感知量表 | local amenities↑、距市中心↓、绿地↓、公交不显著 | 邻里人口密度↑ | SEM |
| Li 2021 | 社/经/文三维各自 flat | POI density↑（三维）、POI 熵↑、bus↑、sky 社↑经↓文↓、building density 社↑经↓文↓、road sparsity 社↓文↑、subway 社↓文↑、work convenience 社↑文↑ | Crowd counting↑（三维） | 多元回归 ×3 |
| Scepanovic 2021 | 移动互联网活动 | intersection density↑（最强 0.497）、小公园↑、anisotropicity↑、block size↓、building height↓、land use mix↑（最弱） | — | 两阶段回归 |
| Wu 2023 | 腾讯 EasyGo 昼/夜 | integration↑、road density↑、road area ratio↑、POI diversity↑、BPOI/IPO↑、GPOI/RPOI 夜↑昼不显著、FSI↓、GSI↑、building continuity↑、greenness↓、openness↑、walkability↑、choice 不显著 | safe↑（夜更大）、wealthy↑、beautiful↓、boring↓、lively 不显著 | RFR 重要性 + SLR |
| Chen 2024 | Flickr 照片计数 | UGS 可达性↑、land use entropy↓、tree↑、sculpture↑、person↓、wall/fence↓、chair↑ | 17 岁以下人口占比↑ | Pearson + OLS + SEM |
| Collins 2024 | CDR 活动量（**DV=男女差**） | third place density↑（最强）、POI density↓、building density↑、cycling/walking intersection↑、road intersection↓（不显著） | — | Kendall τ + SAR |
| Tan 2026 | 百度热力 14 天均值 | FAR↑ **阈值[0.25]**（SHAP 第一）、global integration↑、life service↓、food↓、road density↓、距地铁↓（越近越高）、business residence↑、mixing degree↓ | safe↑、wealthy↓ | RF + SHAP + PDP |
| Lee & Kang 2022 | SKT 访问人口密度 | 总建筑面积↑、商业用地↑、居住用地↓、工业用地↓、土地混合↑、地铁/公交/步行道↑、walk score↑、邻里公园↓、主题/自然公园↑、教育机构↓、公共设施/医院/市场↑ | 常住人口密度↑、工作人口密度↑（系数最大 0.41） | SAC 空间面板（工作日/周末分开） |

## 表 B · composite 论文（17 篇）

成分按 BE/Human 拆分（按我们的约定，不是原文的划分）。
「关联分析」列记论文是否拿指数去和别的变量做过分析。

| 论文 | 指数成分里的 BE | 指数成分里的 Human | 待裁定 | 关联分析 |
|---|---|---|---|---|
| Zeng 2018 | 道路/建筑密度、4 类距离、POI 计数、土地利用 Shannon | 人口密度 | — | n/a（仅制图排名） |
| Pan 2021 | POI 计数 | NTL | — | PLS-SEM：SV↑0.402、CV↑0.602、EcV 间接 0.909、EnV 间接 0.444（EnV→CV ↓−0.286） |
| Chen 2021 | POI 核密度、Airbnb 核密度 | NTL、WorldPop 人口 | — | LightGBM 重要性（无方向）；no pattern↓；gridiron 高活力端↑ |
| Yue 2021 | 混合土地利用、交叉口密度、POI 密度、居住密度、道路连通性、物理隔离、amenities | 人口密度 | 老建筑数量 | n/a（仅制图） |
| Li 2022 | （无 BE 成分） | 街景 6 类计数、微博签到 | — | SLM/GWR 21 因子：人口密度↑、FAR↑（仅基础模型）、open space↑、人行道/路灯↑、购物/公交/休闲 POI↑、road density↓、NDVI↓、距公园↓、integration↑（尺度依赖） |
| Liu 2022 | 休闲 POI 熵/密度、23 类 POI 熵、道路密度、绿化覆盖、人均公园绿地 | 人口 5 项、经济 6 项、政府 4 项、外部链接 4 项、NTL | PM2.5、空气优良天数 | 效标相关（人均 GDP 等 R²>0.5）；环境维与其他维负/不相关 |
| Jiang 2024 | （无 BE 成分） | NTL、手机信令、OD 流 | — | OLS+MGWR 12 形态 IV：FSI↑、MBH↑、MBA↑、RD↑、TFD↑、LUM↑（100% TAZ）、GSI↓、MBD↓、LGPI↓ |
| Dogan & Lee 2024 | POI 密度、POI 多样性 | 流入人口 4 项、经济活动、博客点评、Airbnb | — | 22 个物理复杂度 IV：Tot_NB↑+非线性（先增后衰减）、Ent_BCS↑、Var_ED↓、Tot_CS↓、Tot_FW↑ |
| Wang 2025 | 9 类点评评论数（供给端） | 居住区人口（需求端） | — | n/a（仅 Moran's I 聚类） |
| Cai 2025 | 商业/设施密度、建筑功能混合、RNR/BNB、交叉口密度、距公交/铁路距离 | 人口密度 | 建筑平均年龄↓、房价↓、开敞度↓ | n/a（SOFM 聚类）；15 km 半径阈值（空间格局） |
| Shi 2021 | 文化场馆数、人均绿地、建成区绿化覆盖、空气质量优良率 | 人均 GDP 等 13 项社经 | — | n/a（仅指数 Moran's I） |
| Paköz 2022 | （无 BE 成分） | vitality/mobility/safety 感知 | — | Kruskal-Wallis：区密度类别↑感知活力 |
| Shi 2023 | 夜间经济 POI 8 类 | NTL、百度热力 | — | 多项 logistic：三级路网↑（最重要）、水体↑、铁路↑、火车站↓ |
| Liu 2025 | POI 熵、建筑密度、类 FAR、POI 类型×密度、交叉口数 | Place Pulse Lively（乘性修正） | — | n/a（仅与信令 GT 的 Pearson 验证 r=0.84） |
| Qin 2025 | 文化 POI 密度、创新 POI 密度 | 信令人口密度、NTL | — | OLS 保留 7 因子全↑（商圈数、道路密度、楼层数、建筑密度、TAZ 面积、公交站密度），距市中心↓；MGWR 空间异质 |
| Williams 2019 | 8 类设施可达性 | Landscan 人口（需求权重） | — | n/a（Moran's I 聚类 + 实地验证） |
| Mouratidis 2020 | — | — | — | （见表 A，非 composite） |
| Gómez-Varo 2022 | 商业/设施密度、BuMI、CFMIX、RNR/BNB、交叉口密度、介数中心性、6 类距离/有无 | 人口密度 | 建筑年份×2、租金、坡度 | n/a（仅制图） |

---

# 3. Anomaly 素材（论文用，非关键论证）

沿用 v1 的 T1–T7 与 V1–V8 清单，事实层已抽查 17/18 成立。

## T1–T7 分类详情

**T1 · 同数据集子集 / 角色互换。** 不同的活力维度、或 DV 与 IV，是同一数据集的子集。
Canonical：**Li 2021**（文化活力 DV = 300 m 缓冲区内文化 POI 计数；IV = 全类 POI 密度 + 14 类 POI 熵——POI 对 POI）。
另：Yue 2021（POI 密度放在 human activity 轴——角色互换）、Dogan & Lee 2024（POI 在 Intensity DV 内 vs BE 复杂度 IV）、Qin 2025（文化/创新活力 = POI 密度 vs 商圈/公交站 IV）、Chen 2021（POI 密度在活力分数内，成分级）、Xia 2020（成分级）。

**T2 · 供需混合因子。** 一个指标内部混合了供给与需求信号。
Canonical：**Li 2022**——街景机动车计数：私家车混合（人在用车 + 基础设施可见），公交是纯供给（反映公交基础设施密度，本身是 IV）；两者加总进 DV。
另：Chen 2024（IUNV = 街景物理绿量供给 + Flickr 人的活动）、Shi 2023（VAN = NTL+热力需求融合 POI 供给）、Pan 2021（EnV 潜变量内部混合人口密度与道路密度）。
疑似（NTL 的电力基础设施污染）：Xia 2020、Qin 2025、Jiang 2024、Yang & Pan 2020。

**T3 · causal pathway 不明朗。** 统计上不可区分的路径。
Canonical：**Pan 2021**——PLS-SEM 潜变量 UV 由 NTL（载荷 0.842）+ POI 计数（0.941）测量；「道路密度 → UV」可能走 道路→POI→UV（纯供给传导）或 道路→人的活动→UV（需求效应），不可区分。
处理：只标记；依赖此类路径的主张失去因果解释力。（1 例明确命中；该模式可推广到任何混合供需指标的潜变量 DV。）

**T4 · 供给入尺（定义循环——活力测量内含供给指标）。** 活力分数按构造就含供给侧指标；「前置条件」被当成「结果」测量。
发现型，**最频繁：10 例完整命中**——Zeng 2018（DALD 全供给）、Yue 2021、Gómez-Varo 2022（弱：说的是"条件"却命名为"Index of urban vitality"）、Liu 2022（道路密度测"mobility"）、Dogan & Lee 2024、Wang 2025（"vitality … a function of high-density built environments, accessibility, population"）、Cai 2025（14 个全供给指标）、Liu 2025（物理活力 = POI 熵 + 建筑密度/体量 + 设施 + 交叉口）、Shi 2021（C1–C18）、Williams 2019（amenity 供给分数 → 空置推断——decoupling 论文自己就跑在供给代理上）。成分级：Chen 2021、Qin 2025。
与 T1 的区别：T4 不需要回归——任何后续把这种分数当需求侧 DV 使用都自动触发 T1。

**T5 · 同侧自证（需求对需求）。** DV 与 IV 都是人的活动代理；回归变成对同一现象的两种测量互相验证。
命中：**Li 2021**（人口热力均值作 IV vs taxi/单车 OD 社会活力 DV）。T5 是 T1 的需求侧镜像。

**T6 · 概念漂移（测量-概念错配）。** 指标测的是另一个概念（可持续性、人力资本、基础设施）却叫"活力"；或维度名 ≠ 实测内容。
命中：Shi 2021（空气质量、财政支出进入"活力" ≈ 可持续发展指数）、Yang & Pan 2020（疑似：活力 ≈ 人力资本）、Liu 2022（成分级）、Cai 2025（成分级：设施密度代理"人口集中"）、Scepanovic 2021（命名级：Jacobs 的前置条件被称为"vitality proxies"）。

**T7 · 需求加权供给存量。** 供给存量（商户/POI）用需求痕迹（评论/评分）加权，产物被当作需求侧活动呈现。权重是需求，底是供给。
命中：**Li 2021**（经济活力 = 商户数 × 点评评分权重）、Xia 2020（日间活力 = 餐饮核密度 × 评论数权重）、Wang 2025（活力 = POI 可达性 × 评论权重）。
T2 的结构子型——更隐蔽，因为"我们用了评论数据"看起来像需求侧。

## 命中统计

| 类型 | 命中数 | 论文 |
|---|---|---|
| T1 | 5（+1 成分级） | Li 2021, Yue 2021, Dogan & Lee 2024, Qin 2025, Chen 2021（成分级）；Xia 2020（成分级） |
| T2 | 3 明确 + 4 疑似 | 明确：Li 2022, Chen 2024, Shi 2023。疑似（NTL）：Xia 2020, Qin 2025, Jiang 2024, Yang & Pan 2020 |
| T3 | 1 | Pan 2021 |
| T4 | 10（+2 成分级） | Zeng 2018, Yue 2021, Gómez-Varo 2022, Liu 2022, Dogan & Lee 2024, Wang 2025, Cai 2025, Liu 2025, Shi 2021, Williams 2019 |
| T5 | 1 | Li 2021 |
| T6 | 2 + 3 部分/疑似 | Shi 2021, Liu 2022；Yang & Pan 2020（疑似）, Cai 2025, Scepanovic 2021（命名级） |
| T7 | 3 | Li 2021, Xia 2020, Wang 2025 |
| Clean | 8 | Liu 2019, Kim 2020, Mouratidis & Poortinga 2020, Wu 2023, Collins 2024, Tan 2026, Lee & Kang 2022, Paköz & Işık 2022 |
| n/a | 3 | Nathansohn & Lahat 2022, Ruszczyk 2023, Doan & Zhang 2025（综述） |

**Li 2021 同时命中四类（T1+T5+T7，外加其熵 IV）**——报告的第一个 conflation 例子比 §2.1 现稿更强。
**Mouratidis & Poortinga 2020** 是可引用的正面范例（"precondition… not part of the vitality measure itself"）。

## V1–V8 违规清单（逐条已核对原文）

- V1 · Yue 2021：POI 密度放在 human-activity 轴（T1 角色互换）
- V2 · Li 2021：文化 POI 作 DV vs POI 密度/熵作 IV（T1）；外加 T5、T7
- V3 · Li 2022：街景机动车计数（供给可见性）加总进 DV（T2）
- V4 · Pan 2021：POI 与 NTL 同入 UV 潜变量（T3）
- V5 · Qin 2025：文化/创新活力 = POI 密度（T1）
- V6 · Chen 2021：POI 密度在 vitality score 内部（T1 成分级）
- V7 · Dogan & Lee 2024：POI 密度在 Intensity 指标内部（T1）
- V8 · Xia 2020：餐饮商家存量作日间活力（T7）
- Field-level：Y. Jiang 2023 综述引文逐字核实（"some studies used POI data as the representation of urban vitality while some others use POI data as an explanatory variable…"，txt L580–582）

## 使用须知

> [!warning] 四条结构层注意
> 1. **T4 判据太宽**：「指数含供给指标」对绝大多数 AHP/熵权法活力指数都成立，区分力低。引用时补限定：「且该分数随后被当作需求侧结果解释/回归」。
> 2. **T6 基准循环**：判「概念漂移」需要 vitality 的标准定义，而本文立场是无共识。引用时重述为「指标与该文自己给出的定义不匹配」。
> 3. **「Clean 8」混两种含义**：设计上真干净（Mouratidis 问卷、DV 与 BE 完全分离）vs 本轮只审了一半（Collins 的 IV 侧未审）。引用前者可直接用，引用后者需注明。
> 4. **计数表不互斥**：T7 是 T2 子型、T5 是 T1 镜像、T4 用了就触发 T1。33 个 flag 落在 21 篇上，Li 2021 一篇占三格。引用时给例子不给总数。

---

# 4. BE 因子去重表

**这是进论文的表。** 每行 = 一个去重因子。「被几篇用过」含其在指数内部（成分）
与外部关联分析（IV）两种出现方式。「各文报告的效应」按方向汇总。

| 因子 | 被几篇用过 | 原文角色 | 各文报告的效应 | 数据源可行性（HCMC） |
|---|---|---|---|---|
| POI 计数/密度 | 13（Zeng, Liu19, Xia, Li21, Chen21, Yue, Li22, Liu22, Gómez-Varo, Wu, Dogan, Wang, Qin, Collins, Cai, Tan） | 多数作 IV 或指数成分；Li21/Qin 作 DV 维度（异常） | 绝大多数 ↑；Li21 社↑经↓文↓（分维度变号）；Collins ↓（对男女差） | **OSM POI 在 HCMC 覆盖差**（B6：50 m 环熵定义率 1.1%）；SVI-to-POI 是替代路线 |
| 道路密度 | 9（Zeng, Kim, Li21, Li22, Liu22, Wu, Chen21, Qin, Collins, Jiang） | IV | 多数 ↑；Li22 ↓；Tan ↓；Chen21 方向未报告 | OSM roads 已 fetch，可行 |
| 交叉口密度 | 7（Kim, Li22, Yue, Gómez-Varo, Scepanovic, Collins, Dogan, Cai） | IV | Scepanovic ↑（最强 0.497）；Li22 不显著；Collins ↓（不显著）；Dogan ↑（十字口） | OSM 可算 |
| 土地利用混合（Shannon 熵） | 8（Zeng, Xia, Kim, Yue, Liu22, Li22, Scepanovic, Qin, Chen21, Chen24） | IV 或成分 | 多数 ↑；Liu19 ↓（有序混合好）；Chen24 ↓；Li22 用居住占比替代（↓） | 需要 land-use map 或用 POI 熵近似 |
| FAR / FSI | 6（Xia, Kim, Li22, Jiang, Wu, Tan, Qin） | IV | 多数 ↑；Wu ↓；Li22 仅基础模型显著；**Tan 阈值[0.25]** | **GBA 高度图可行**；OSM `building:levels` 覆盖率未知 |
| 建筑密度/覆盖率 | 7（Zeng, Xia, Kim, Li21, Wu, Jiang, Collins, Qin） | IV 或成分 | 多数 ↑；Jiang ↓（GSI）；Li21 社↑经↓文↓ | OSM footprints 已有，可行 |
| 建筑高度/体量 | 4（Kim, Li22, Jiang, Scepanovic, Liu25） | IV | Jiang MBH↑；Scepanovic ↓；Li22 新建高层肌理↓ | GBA 高度图可行 |
| 距市中心距离 | 3（Mouratidis, Li22, Qin） | IV | 全部 ↓（越远活力越低） | 简单，可算 |
| 距公园/绿地距离 | 4（Mouratidis, Li22, Scepanovic, Qin） | IV | Mouratidis ↓、Li22 ↓（越近越低，反直觉）、Scepanovic ↑（小公园倒数） | OSM 可算 |
| 公交/地铁可达性 | 7（Zeng, Kim, Li21, Li22, Wu, Qin, Lee-Kang, Cai） | IV | 多数 ↑；Li21 subway 社↓文↑；Lee-Kang ↑ | OSM transit 可算 |
| 空间句法 integration/choice | 3（Li22, Wu, Tan） | IV | Li22 integration↑（尺度依赖）；Wu integration↑、choice 不显著；Tan 全部 ↑ | OSM road graph 可算 |
| 绿地/NDVI/绿视率 | 5（Li22, Wu, Chen24, Mouratidis, Cai, Shi21） | IV 或成分 | Li22 NDVI↓、Wu greenness↓、Mouratidis 绿地↓、Chen24 tree↑ | Sentinel-2 NDVI 可行 |
| 土地利用混合（POI 熵版） | 5（Liu19, Li21, Wu, Tan, Liu25, Dogan） | IV | 多数 ↑；Liu19 ↓（有序）；Tan ↓ | 同 POI 覆盖问题 |
| 房价/地价 | 3（Li21, Li22, Cai, Tan） | IV | Li21 社不显著/经↑文↑；Li22 不显著；Cai 预设↓；Tan ↓ | 待裁定；HCMC 无链家类数据 |
| 建筑/社区年龄 | 3（Yue, Gómez-Varo, Cai, Li22, Xia） | IV 或成分 | Yue 老城区↑；Li22 社区年龄↑；Cai 预设↓；Gómez-Varo 待裁定 | OSM `start_date` 极稀疏，不可行 |
| 地形（坡度/海拔/起伏） | 3（Kim, Lee-Kang, Shi23, Gómez-Varo） | 控制/IV | Kim 不显著；Lee-Kang 海拔↓；其余未报告 | SRTM DEM 可行 |
| 水体/河流距离 | 3（Li22, Shi23, Cai, Yue） | IV | Shi23 ↑（越近越高）；Li22 不显著 | OSM water 已有 |
| 铁路/高速隔离 | 2（Yue, Cai, Shi23） | IV 或成分 | Yue 预设↓；Shi23 铁路↑火车站↓；Cai 预设↑（远离好） | OSM 可算 |
| 建筑形态类型 | 2（Li22, Chen21） | IV | Li22 新建高层↓；Chen21 no pattern↓、gridiron 高活力端↑ | 可从 footprints 派生 |
| 人行道/步行设施 | 3（Li22, Wu, Lee-Kang, Gómez-Varo） | IV | 全部 ↑ | OSM sidewalk 标签稀疏 |
| 照明 | 2（Li22, Gómez-Varo） | IV | Li22 ↑ | VIIRS 或 OSM lit 标签 |

---

# 5. Human 因子去重表

| 因子 | 被几篇用过 | 原文角色 | 各文报告的效应 | 数据源可行性（HCMC） |
|---|---|---|---|---|
| 人口密度（静态） | 10（Zeng, Kim, Yue, Li22, Chen21, Dogan, Cai, Pan, Gómez-Varo, Lee-Kang） | 多数作 IV 或成分 | 全部 ↑（Lee-Kang 系数最大 0.41） | WorldPop / census 可行 |
| 手机信令/服务人口 | 5（Liu19, Kim, Jiang, Qin, Scepanovic） | 多数作 DV | —（是活力本身） | **HCMC 无运营商数据** |
| 签到/点评/社交媒体 | 5（Li21, Li22, Chen21, Chen24, Dogan） | 多数作 DV 或成分 | Li21 crowd counting↑（作 IV 时） | 微博/点评不可用；Flickr 稀疏 |
| NTL 夜间灯光 | 7（Xia, Pan, Chen21, Jiang, Qin, Shi23, Yang-Pan） | 多数作成分或验证 | —（是活力代理） | VIIRS DNB 可行（但 ~500 m 粒度在 1.2 km tile 内几乎恒定） |
| 百度/腾讯热力 | 3（Tan, Li21, Shi23） | 多数作 DV | —（是活力本身） | **HCMC 无百度/腾讯** |
| 浮动人口/OD 流 | 3（Lee-Kang, Dogan, Jiang） | 多数作 DV 或成分 | Lee-Kang 访问人口密度是 DV | **HCMC 无运营商数据** |
| 感知评分（Place Pulse/问卷） | 4（Mouratidis, Paköz, Wu, Tan, Liu25） | 多数作 IV 或修正项 | Wu safe↑、beautiful↓；Tan safe↑wealthy↓；Liu25 作乘性修正 | **需要街景 + 感知模型** |
| 人力资本/教育 | 2（Yang-Pan, Liu22） | IV 或成分 | Yang-Pan HC↑ | census 可行 |
| 经济总量（GDP/三产/零售） | 4（Yang-Pan, Liu22, Shi21, Pan） | 成分或控制 | 全部 ↑ | 统计年鉴可行 |
| 性别/年龄结构 | 2（Collins, Chen24, Paköz） | 分组变量 | Collins 男女差是 DV；Chen24 17 岁以下↑ | census 可行 |

---

# 6. Handoff

## 6.1 这份文件的推理逻辑

文献不管有没有活力共识定义，都已经建立起一套认知：城市活力与哪些东西相关。
这套认知是 correlationist 的——绝大多数用线性回归、空间自回归、Moran's I，
报告的效应以单调为主（↑ 或 ↓），少数 ML 研究（Tan 2026 的 SHAP、Dogan & Lee 的
MINE、Xie 2025 的 XGBoost+SHAP）报告了非线性与阈值。

我们要做的是：把这套认知里的**所有因子**（§4 + §5 两张表）作为输入通道，
让 DDPM 在 HCMC 的栅格上学习它们的联合分布。如果 DDPM 能学到 correlationist
方法看不到的非线性交互（比如 FAR 阈值、设施密度的过饱和拐点），
那就是 connectionist 方法论对活力认知的增量。

## 6.2 数据采集看哪两张表

- **§4 BE 因子去重表** → 每行的「数据源可行性」列就是采集清单。
  已可行：OSM buildings/roads/transit/water、Sentinel-2 NDVI、SRTM DEM。
  需要补：GBA 高度图（FAR/建筑高度）、SVI-to-POI（POI 覆盖）、land-use map（混合度）。
  不可行：建筑年龄（OSM `start_date` 稀疏）、房价（HCMC 无链家类数据）。
- **§5 Human 因子去重表** → 本文只做供给侧（§2.4），Human 表是下一阶段的对照。
  但 NTL 和人口密度已在 `indicator_specification.md` 候选行里（作 condition 不作 grid）。

## 6.3 HCMC 现实问题结合点

不只是方法论上要区分 BE 和 Human——HCMC 的现实问题要求这个区分：

- **非正规性**：HCMC 大量建成环境是非正规的（巷弄、自建、混合用途），
  OSM 的 POI 覆盖在这里编码的是制图投入而非城市功能（B6 实测：
  第二密瓦片 6866 栋建筑只有 4 个 POI）。用 OSM POI 做 BE 因子会系统性低估
  非正规区——这不是数据质量问题，是 HCMC 的城市形态问题。
- **供给与需求的脱节**：HCMC 的快速城市化意味着大量新建区的物质形态已经到位
  但人的活动还没跟上（新区空置），同时老城区的物质形态密集但承载的活动
  远超设计容量。只有分开测 BE 和 Human，才能看到这种脱节——
  合在一起的「活力指数」会把两种完全不同的状态混成同一个分数。
- **我们能贡献的**：correlationist 方法在 HCMC 这样的数据稀缺环境里
  只能依赖少数可得变量做线性回归。DDPM 可以从多通道栅格里学习
  联合分布，不依赖任何单一变量的完备性——这对 HCMC 特别重要，
  因为没有任何单一数据源在这里是完备的。

## 6.4 A1 综合判断（沿用 v1，与 §4/§5 的效应证据互证）

1. **BE 共识核心是五个指标**：建筑密度/覆盖率、道路密度、交叉口密度、POI 计数、熵混合。`indicator_specification.md` 的候选行已全覆盖。
2. **FAR 是最高价值的补充**：五研究使用（Li 2022、Jiang 2024、Qin 2025、Xia 2020、Tan 2026），且携带旗舰非线性证据（Tan 的 0.25 阈值）。HCMC 高度源：GlobalBuildingAtlas 已实测可获取（`asiawest/e105_n15_e110_n10.json`，RMSE 1.5–8.9 m，亚洲高层低估需注意），OSM `building:levels` 覆盖率仍未测——两个源二选一或互校。
3. **「混合」指标分三个来源**：POI 熵（OSM 可行但小半径稀疏）、土地利用熵（需 land-use map，Yue 2021 为 HCMC 用过 12 类图）、Kim 2020 的 balance index（第三变体）。
4. **空间句法有三研究先例**（Li 2022、Wu 2023、Tan 2026），可从 OSM 路网图计算；在 1 201 m tile 内的方差是否存活是 B10 的问题。
5. **老建筑**（Jacobs generator，HCMC 相关的 Yue 2021 与 Cai 2025 链家数据都用了）几乎不可能从 OSM 获得（`start_date` 稀疏）——记为已知缺口，不设候选行。
6. **Human 侧指标是本文之后的阶段**（§2.4：本文只做供给侧）。§5 那张表是联合供需研究的档案；H-1 在场强度是最自然的第一需求通道；NTL 已在候选行里（作为 condition，非 grid）。
7. **文献的测量尺度远比我们的栅格粗**（TAZ / community / 500 m / 300 m buffer；最细 Cai 2025 的 100 m grid）。没有任何文献验证过 50 m 环；环半径必须从 HCMC 实地条件论证（B6/B10），不能从 TAZ 尺度的研究继承。
8. **异常分类直接喂 §2.1**：T4 把批判从回归循环推进到指数构造本身——即使不做驱动因子回归，一个含供给的活力分数也不能当需求侧 DV 用。

---

# 附录 A · 边缘待裁定指标（不归任一侧）

| 指标 | 使用者 | 为什么暧昧 |
|---|---|---|
| 房价 / 地价 | Tan 2026, Li 2022（kriging）, Cai 2025（ARR，链家）, Li 2021 | 场所的社会经济属性；既非建成形态也非人的活动 |
| 社区/建筑年龄 | Tan 2026, Yue 2021, Gómez-Varo 2022（YEARM/YEARSD）, Cai 2025（BAA/DBA）, Li 2022, Xia 2020（城市化时期） | 更新动态代理 |
| 教育-收入复合（EIC） | Yang & Pan 2020 | 城市尺度人力资本，非空间指标 |
| 政府/财政指标（C12/C13/C16, d1–d4） | Shi 2021, Liu 2022 | 城市尺度治理能力——概念漂移领域（T6） |
| 地形（坡度/海拔/起伏） | Kim 2020, Lee-Kang 2022, Shi 2023, Gómez-Varo 2022 | 自然条件，二分法未覆盖 |
| 环境质量（PM2.5/PM10/空气优良率） | Liu 2022, Lee-Kang 2022, Shi 2021 | 同上 |
| 水体距离 | Shi 2023, Li 2022 | 自然要素 |

# 附录 B · 文献覆盖清单（32 项 + 7 本理论书）

状态说明：2026-09-11 全部 31 篇完成指标级核对（两子会话）；2026-09-12 由 4 个子会话按新框架重做逐篇提取（表征 + 因子 + 效应形态），明细见 `vitality-extraction/batch-A` 至 `batch-D.md`。

| # | 论文 | 轴1 class | 表征 | 异常旗标 |
|---|---|---|---|---|
| 1 | Zeng et al. 2018（Chicago/Wuhan） | BE | composite（DALD） | T4 |
| 2 | Liu et al. 2019（Nanjing） | Human | flat（信令 SOM） | clean |
| 3 | Xia et al. 2020（五市） | Mixed | flat ×2（昼/夜） | T7, T1 成分级, T2 疑似 |
| 4 | Kim 2020（Seoul） | Human | flat（服务人口） | clean |
| 5 | Yang & Pan 2020（中国 326 市） | Human 社经 | flat（人力资本） | T6 疑似, T2 疑似 |
| 6 | Mouratidis & Poortinga 2020（Oslo） | Human 感知 | flat（问卷量表） | clean —— 正面范例 |
| 7 | Pan et al. 2021（成渝） | Mixed | composite（PLS-SEM） | T3（+T1/T2 成分级） |
| 8 | Li et al. 2021（= Q. Li 2021, 成都） | Mixed | flat ×3（社/经/文） | T1 + T5 + T7 |
| 9 | Chen/Biljecki et al. 2021（9 城） | Mixed | composite（vitality score） | T4 成分级, T2 疑似 |
| 10 | Yue et al. 2021（HCMC vs Shanghai） | Mixed | composite（projection pursuit） | T1 角色互换 + T4 |
| 11 | Li et al. 2022（Wuhan） | Human | composite（街景+签到） | T2 |
| 12 | Liu et al. 2022（Vital Triangle） | Mixed | composite（28 指标） | T4 + T6 成分级 |
| 13 | Scepanovic et al. 2021（意大利 6 城） | Human | flat（移动互联网活动） | clean（T6 命名级） |
| 14 | Gómez-Varo et al. 2022（JANE） | BE 条件 | composite（22 变量） | T4 弱 |
| 15 | Wu et al. 2023（Shenzhen, SVI） | Human | flat（EasyGo 昼/夜） | clean |
| 16 | Jiang et al. 2024（Beijing） | Human | composite（PCA 三源） | T2 疑似（NTL） |
| 17 | Chen et al. 2024（SF, IUNV） | Mixed | flat（Flickr 照片） | T2 |
| 18 | Collins et al. 2024（意大利 7 城） | Human | flat（CDR，DV=男女差） | clean |
| 19 | Dogan & Lee 2024（首尔） | Human | composite（FA 3 指标） | T1 + T4 |
| 20 | Wang et al. 2025（深圳） | Human 代理 | composite（2SFCA） | T4 + T7 |
| 21 | Cai et al. 2025（北京, SOFM） | Mixed | composite（6 维 14 指标） | T4 + T6 成分级 |
| 22 | Tan et al. 2026（深圳） | Human | flat（百度热力） | clean |
| 23 | Shi et al. 2021（上海, Land） | Mixed | composite（C1–C18） | T4 + T6 |
| 24 | Lee & Kang 2022（首尔） | Human | flat（SKT 访问人口） | clean |
| 25 | Paköz & Işık 2022（Istanbul） | Human 感知 | composite（PCA 感知） | clean |
| 26 | Nathansohn & Lahat 2022 | 质性 | — | n/a |
| 27 | Shi et al. 2023（绵阳, 夜间 VAN） | Human 代理 | composite（NTL+热力+POI） | T2 |
| 28 | Ruszczyk et al. 2023（孟加拉） | 质性混合 | — | n/a |
| 29 | Liu et al. 2025（南京, EPB） | Mixed | composite（P+S·H） | T4 |
| 30 | Qin et al. 2025（南京） | Mixed | composite（四维熵权） | T1 + T2 疑似 + T4 成分级 |
| 31 | Williams et al. 2019（鬼城） | 对照方法 | composite 反向（amenities） | T4 |
| R | Doan & Zhang 2025（系统综述） | 对照材料 | — | n/a |

**身份注记**：ISM 报告的 "Q. Li et al. 2021" = Li et al., "Multidimensional Urban Vitality on Streets"（`Li-et-al-2021-…txt`，一作 Qian Li）；期刊元数据为 ISPRS IJGI 2022, 11(1):2，引用年份需在报告参考文献冻结前统一。

**理论书（7 本，无量化指标）**：Jacobs 1961, Whyte 1980, Lynch 1981, Maas 1984, Montgomery 1998, Landry 2007/2009, Gehl 2010。

**未读（2026-09-11 索引，[0]）**：Amer et al. 2025, Jiang & Ma 2025——下一轮候选。

---

*v2 重做说明：提取由 4 个并行子会话完成（2026-09-12，kimi-k3 medium），
每篇因子带 txt 行号与短引文，逐篇明细见 `vitality-extraction/batch-*.md`。
v1 的 BE/Human 双母节点结构与分类约定沿用；anomaly 清单沿用 v1 事实层，
附四条结构层使用须知。PDF 页码未核。*
