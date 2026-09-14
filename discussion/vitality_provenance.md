# Vitality indicators — provenance 表（数据源 × 计算口径 × 尺度）

**Status: v1 (2026-09-14). 4 个子会话按 batch 分片提取，父会话合并。**
配套 [`vitality_indicators_review.md`](vitality_indicators_review.md)：该文件的去重因子表（§4/§5）回答"哪些因子、效应如何"；本文件回答"**每个因子在每篇论文里由什么数据源、什么计算口径表示**"——同名因子在不同论文的 bias 不同，效应不可直接比较，读去重表的效应冲突时先回本表核对口径。

**行号约定**：所有 `L#` 指 `/Users/guanhongli/Documents/wafo-base/literature/papers/` 下对应 txt 提取层的行号；PDF 页码未核。「原文未说明」= txt 全文检索后确认无此信息，非遗漏。

---

# 1. 跨篇口径分歧矩阵（同名因子的不可比性证据）

## 1.1 POI 熵 / 混合度 —— 至少六种口径

| 口径变体 | 论文 | 数据源 |
|---|---|---|
| Shannon 按 POI **计数**占比 | Li 2021（14 类，300 m buffer）、Tan 2026（口径未细述）、Liu 2022（c1 6 类 / c3 23 类，AMap） | AMap |
| **exp(Shannon)**（值大=无序，方向反转！） | Liu 2019（24→12 类）、Wu 2023（6 类） | AMap |
| Shannon 按 **gross floor space** 加权 | Jiang 2024（15 类） | AMap POI + 建筑面积 |
| 归一化 Shannon（−Σp·lnp / ln n） | Yue 2021（**12 类土地利用图**，VAST/上海地调）、Cai 2025（BFM/CFM，AMap）、Chen 2024（Song et al. 法） | 土地利用图 / AMap |
| 三大类熵（÷log 3） | Scepanovic 2021（Urban Atlas 20→3 类） | Urban Atlas (EEA) |
| balance index（非熵） | Kim 2020（GFA 平衡，住宅基准）、Gómez-Varo（RNR/CFMIX/BNB，地籍+零售库）、Cai 2025（RNR/BNB，AMap） | 官方地籍 / AMap |
| **Simpson**（非 Shannon） | Dogan & Lee 2024（13 类）、Liu 2019（evenness 行） | Kakao Map / AMap |

**土地混合还有源分裂**：土地利用图（Yue、Scepanovic、Qin resdc、Li 2022 武汉规划信息中心、Shi 2023 绵阳自然资源局）vs POI 平台（AMap/Baidu/OSM/Kakao）——两者测的不是同一个 construct。

## 1.2 交叉口密度 —— 分母与定义全部分裂

| 论文 | 数据源 | 口径 |
|---|---|---|
| Scepanovic 2021 | 原文未说明 | 数 ÷ census district 面积 |
| Yue 2021 | OSM（拓扑分析） | 每 sub-district 计数 |
| Chen 2021 | OSM（+部分官方） | 数 ÷ 1 km 格网面积 |
| Gómez-Varo 2022 | 市议会 Road Graph 2019 | 50 m 格网内计数 |
| Li 2022 | OSM | 数 ÷ 社区净面积 |
| Collins 2024 | OSM | 分模式（行人/骑行/驾驶）可达交叉口均值 |
| Qin 2025 | OSM | 数 ÷ TAZ 面积 |
| Liu 2025 | AMap 系公共地图 | 数 ÷ block 面积 |
| Cai 2025 | **正文 GCM 2021 vs Table 1 Amap——矛盾待核** | 数 ÷ block 面积 |
| Dogan & Lee 2024 | 首尔市政府 2020 中心线 | 仅**十字口**计数（另有尽端路计数） |

效应冲突（Scepanovic ↑0.497 vs Collins ↓ n.s. vs Li22 n.s.）在此口径分布下**不可作实质解读**。

## 1.3 路网（道路密度/句法/距离）—— 四类数据源

- **OSM**：Chen 2021、Collins 2024、Jiang 2024、Li 2022、Qin 2025、Liu 2022、Yue 2021、Chen 2024
- **官方**：Shi 2023（绵阳自然资源和规划局）、Lee-Kang 2022（NGII）、Dogan & Lee 2024（首尔市政府）、Cai 2025（GCM 2021）、Gómez-Varo 2022（市议会）
- **商业平台**：Liu 2025（AMap 系）、Li 2021（OSM + 百度补充；且其"road sparsity"是**计数**非长度密度）
- **原文未说明**：Zeng 2018、Wu 2023、Tan 2026、Wang 2025（2SFCA 路网）、Mouratidis 2020

## 1.4 建筑密度 / FAR —— 命名与分母

- **Qin 2025 命名反转**：其 "building density" = FSI（惯例 FAR），"volume rate" = GSI（惯例覆盖率）——读其表必须按公式对齐。
- Liu 2025：F2 = 基底/面积（覆盖率）、F3 = 基底×层高/面积（类 FAR）。
- FAR 分母：Xia（街区面积，高德轮廓+层数）、Kim（BGA，NSDI 按用途 GFA）、Wu 2023（深圳规划局）、Qin（TAZ 面积）、Jiang（TAZ 面积）、Tan（来源未说明）。
- GSI/覆盖率：Xia BCR（高德）、Wu 2023（深圳规划局）、Qin（资源环境科学数据中心⚠ 正文另称 OSM/AMap，两说不一致）。

## 1.5 DV（vitality 表征）数据源 —— 32 篇无两篇同源

信令系：中国电信 UDR（Liu 2019）、Seoul Open Data Plaza（Kim 2020）、北京某运营商（Jiang 2024）、南京规划资源局（Qin 2025）、未具名运营商 30% 覆盖（Liu 2025）。平台系：腾讯 EasyGo（Wu 2023、Li 2021 热力）、百度慧眼（Tan 2026）、SKT（Lee-Kang 2022、Dogan & Lee 2024）、TIM 移动互联网（Scepanovic 2021）、TIM CDR（Collins 2024）。社交媒体系：大众点评（Xia 2020、Wang 2025 评论、Williams 2019）、微博签到（Li 2022）、Flickr（Chen 2024）、Kakao/BC Card/Airbnb（Dogan & Lee）。遥感系：VIIRS（Xia、Yang & Pan 验证）、珞珈一号（Pan、Jiang、Qin、Shi 2023、Cai 融合）。问卷系：Mouratidis、Paköz。合成指数系（无外部活力源）：Zeng、Shi 2021、Gómez-Varo、Liu 2022、Liu 2025（P+S·H，手机信令仅作 GT 验证）。

# 2. 「原文未说明」高频缺口（provenance 无法补全处）

| 篇目 | 缺口 |
|---|---|
| Tan 2026 | **全部客观 IV 的供应商与年份**（POI/路网/建筑/房价/社区年龄仅列类别名，L424-427）；唯一例外是百度街景 |
| Wu 2023 | 路网/POI/SVI 年份；空间句法的路网源 |
| Mouratidis 2020 | 5 个 BE 因子中 4 个底层数据源（仅绿地 = Hansen et al. 2013） |
| Zeng 2018 | density/diversity 两维数据源（仅称 "open data platform"）；密度公式与分母 |
| Scepanovic 2021 | block/街道几何与小公园数据来源 |
| Kim 2020 | 各空间图层自身的测绘年份 |
| Dogan & Lee 2024 | 活力数据年份（Table 5 Year 列被 txt 抽取打散） |
| Chen 2021 | 全部数据的采集年份 |
| Wang 2025 | 2SFCA 的路网数据源 |
| Shi 2023 | 距离五级分类的阈值；DEM 数据源 |
| Li 2022 | POI 采集年份；距离类因子的欧氏/网络距离 |
| Liu 2022 | AMap POI 具体年份（仅能以"多数指标 2019"挂靠） |

# 3. 勘误与不一致记录

1. **Pan 2021 EnV4/EcV4**：batch-B.md 附注称"同为高德 POI 计数"，但 txt 数据源章节明确 18 个统计数据集全部来自统计年鉴+电话访谈（L869-875，仅 NLD 与 POI 除外）——provenance 以 txt 为准。
2. **Cai 2025 交叉口密度**：正文（L748-754）指向 GCM 2021（自然资源部），Table 1 数据源列显示 amap.com/2021——txt 抽取错位或原文表文不符，**需回 PDF 核对**。
3. **Shi 2023 POI**：表 1 命名 "Baidu POI data"，获取方式栏写 "Downloaded from Google Map in June 2020"——如实标注，**需回 PDF 核对**。
4. **Chen 2024 census 口径漂移**：census 变量定义为 65+（ACS 2016-2020），Table 3 模型变量名 Age>60，门槛不同，原文未解释。
5. **Qin 2025 建筑轮廓源两说**：Table 1 为资源环境科学数据中心（218,958 polygon），正文 L168-169 称取自 OSM/AMap。
6. **Liu 2025 表 1 版式错乱**：建筑轮廓、路网等多因子数据源未独立标明，与同表 AMap POI 行关系需回 PDF 核对。

---

# 4. 逐篇明细（31 篇 + 1 综述）

## Batch A

### Zeng et al. 2018（Chicago + Wuhan）

**论文级**：Chicago 中心区 block group（正文 2159 个 L378；Table 2 记 2192 个 L425）+ Wuhan community neighborhood（1076 个）；POI 与行政边界 2016–2017（L489）。密度类指标分母公式原文未给出。

**DV**：composite——无外部活力数据源，12 指标经 spatial TOPSIS 合成（L498），权重信息熵法（L442-443）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 人口密度 | 原文未说明（仅称 "open data platform"，L930-931） | 原文未说明 | L493-494 |
| 道路密度 | 同上 | 原文未说明 | L493-494 |
| 建筑密度/覆盖率 | 同上 | 原文未说明 | L493-494 |
| 距学校/医院/商店/景点距离 | Chicago = Google Places API；Wuhan = 百度 LBS 开放平台，2016-2017（L478-489） | "distance to schools, hospitals, shops, and tourist attractions"（Eq 2.3）；距离类型未说明 | L930-931 |
| POI 计数（livability 4 类） | 同上双平台；8 部门两级分类重组（Table 3，L505-511） | 单元内 4 类设施计数（Eq 2.4）；是否除以面积未说明 | L482-485 |
| 土地利用混合（Shannon 熵） | 原文未说明 | SD(land use)（Eq 2.5）；地类数与粒度未说明 | L497 |

**因子关联表**：n/a（无回归/关联分析）。

### Liu et al. 2019（Nanjing）

**论文级**：基站改进 Voronoi 单元（DBSCAN 合并 <20 m 基站，14,236 个，L600-604）；手机数据 2018-06-11~17；POI 2018-08（L725）。

**DV**：中国电信 UDR 信令，每半小时记录，一周 17.17 亿条（L565-568）；剔除过境用户（停留 <5 h 或均速 >80 km/h，L594-596）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| POI 密度 | 高德 AutoNavi 开放接口，2018-08，371,273 个 POI（L723-725） | ΣNumᵢ/Area（Eq 9），分母=活力区面积；24→12 类 | L708-709 |
| POI 熵 | 同上 | **exp(−Σpᵢ·ln pᵢ)**（Eq 11），值大=无序 | L709-711 |
| POI richness | 同上 | 类别数（Eq 10） | L708-709 |
| Simpson（evenness） | 同上 | 1 − Σpᵢ²（Eq 12） | L711-713 |
| 主导 POI 类型 | 同上 | 占比最大类别（Eq 13） | L714-716 |

### Xia et al. 2020（五市）

**论文级**：street block（建成区勾边+道路 buffer 切分，L379-382）；建成区取自 NLUD-C 1995/2015；DV/NV 2015、POI 2016-01、建筑道路 2017 年后（L329-340）。

**DV**：昼 = 大众点评小餐饮核密度（500 m 半径）× 评论数加权（L432-440）；夜 = VIIRS DNB 2015 年均（15″≈500 m，重采样 30 m 加权到街区，L337-338）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| FAR / FSI | 高德建筑轮廓（含层数），2017 后（L334-336） | FARᵢ = Fᵢ/Aᵢ（街区总楼面/街区面积，Eq 1） | L389-390 |
| BCR | 同上 | BAᵢ/Aᵢ（Eq 2） | L391-393 |
| 土地利用混合（hybrid/single） | 百度 POI 2016-01，按 GB/T 21010-2017 重分 6 类，POI 按楼面面积设权重（L568-574） | 单类占比 >50% = single；≤30% = balanced hybrid（L604-607） | L604-607 |
| 城市化时期 | NLUD-C（中科院，Landsat 为主，精度 >90%） | 1995 前 = 老城区，1995-2015 = 新区 | L368-370 |

### Kim 2020（Seoul 四区）

**论文级**：自定义 block group（快速路与主干道围合，n=217，L674-738）；统一分母 = BGA；空间库 = Seoul Open Data Plaza [32] / juso.go.kr [33] / NSDI [34]，2020-11-20 访问（L2378-2380）；各图层测绘年份未说明。

**DV**：Seoul Open Data Plaza 服务人口（手机信号估算），2018-04 工作日 6:00-21:00（L778-846）；底层运营商未说明。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| tare space | 未逐变量说明（土地利用分区图 [34]） | tare area / BGA（Table A1） | L864-865 |
| slope | 同上 | "average elevation"（Table A1）vs 正文 "average incline"——表述不一致 | L866 |
| mesh | 道路名地址图 [33] | mesh = 2 / 路网密度；路网密度 =（内部路网长 + 外部/2）/ BGA | L872-874 |
| gamma index | 同上 | 边数 / 3×(节点数−2) | L874-876 |
| 公交/地铁可达性 | 站点位置 [32]/[33] | 地铁出入口数 / BGA；公交站数 / BGA | L877-878 |
| building entrance density | 建筑布局数据 [34] | 建筑入口数 / BGA | L881-882 |
| plot 系 4 指标 | [34] NSDI 系 | 组团地块面积比、平均地块面积、地块数/km² | L882-888 |
| FAR / FSI | 建筑布局数据 [34]（按用途 GFA） | GFA / BGA | L890-891 |
| balance index 1–4 | 同上 | 1 − \|GFA住宅 − GFA非住宅\| / (和)；住宅基准 | L898-900 |

### Yang & Pan 2020（中国 326 市）

**论文级**：地级市整市；326 市 × 2000/2005/2010 三期面板（L653）。

**DV**：概念活力 = 平均人力资本（EIC 法，自算）；验证 DV = GNLD 夜光中国城市调整版（CNRDS 下载，L386-387）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 人力资本/教育 | 6 个微观数据集（Mini Census/CGSS/CSS/CLDS/CFPS/CMDS，L366-369） | h = Σ p_edu·e^{β·d}；教育 6 类；β 逐市 Mincer 方程按义务教育 9 年分段 | L366-369 |
| 人口规模/经济总量 | CEIC（L404-406） | 控制变量（log） | L404-406 |
| NTL（验证 DV） | GNLD 中国调整版（CNRDS） | 市域灯光指数；像元级口径未说明 | L386 |

### Mouratidis & Poortinga 2020（Oslo）

**论文级**：45 社区（对应 postal zone）；问卷 2016-05/06，n=1344（L231-237）；BE 底层地理数据源未逐项说明。

**DV**：问卷 3 题 1–5 分量表，α=0.904（L269-275）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 人口密度 | 原文未说明 | 人/ha（社区居民数/面积） | L256-257 |
| local amenities（POI 计数族） | 原文未说明 | 社区质心 1000 m buffer 内咖啡馆+餐厅+社区中心+酒吧聚合计数 | L257-259 |
| 距市中心距离 | 原文未说明 | 质心沿步行路网到市中心 km | L253-255 |
| 绿地 | **Hansen et al. 2013 全球森林覆盖变化数据** | 250/500/1000 m buffer 绿地占比；终模型取 1000 m | L263-265 |
| 公交可达性 | 原文未说明 | 质心 500 m buffer 内所有站点高峰小时班次总数 | L259-261 |

### Shi et al. 2021（Shanghai）

**论文级**：16 行政区；2010–2019 年度面板。

**DV**：C1–C18 经离差最大化定权 + 加权和合成（L676-682）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 全部 18 项（C1–C18） | 《上海统计年鉴（2010-2019）》+ 各区统计；缺失值估算补全 | 年鉴口径；正文仅给单位不给公式（Table 1，L569-610） | L180-182 |

### Paköz & Işık 2022（Istanbul）

**论文级**：39 城市区（4 个 rural district 剔除）；毛人口密度分组（含非居住用地、剔水面，L862-864）；区人口 TURKSTAT 2020 年末；问卷 2020-06-01~05，n=337。

**DV**：六维感知评分（−1/0/+1 编码），PCA 降两因子（vitality 0.711 / mobility 0.799 主载荷）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 人口密度（分组变量） | TURKSTAT（2021） | 区总人口/区总面积（剔水面）；按变异断点分三类 | L862-864 |
| gender 等分组变量 | 在线问卷自报（n=337） | 二分类分组检验 | L870-871 |

## Batch B

### Pan et al. 2021（成渝）

**论文级**：140 区县；分母=区县面积；数据年份 2019。

**DV**：PLS-SEM 潜变量 = ① 珞珈一号 01 星 NTL（130 m，2019-08，ENVI ROI 算区县 DN 均值，L1048-1049）+ ② 高德 POI 计数（2019-08-15，5 大类 15 子类求和 219,839 条，L1268-1270）；z-score 后 SmartPLS 3.0。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| NTL | 珞珈一号 01 星，2019-08 | 区县 DN 均值 | L942 |
| POI 计数 | 高德 API，2019-08-15 | 区县求和；无分母无 buffer | L1168, L1268 |
| 人口密度（EnV1） | 统计年鉴 2019 + 电话访谈 | 年末常住人口/区县面积 | L572 |
| 道路密度（EnV2） | 同上 | km/km² | L573-575 |
| 经济总量系（EcV/SV/EnV3） | 同上 | 区县级统计量（三产占比、人均 GDP 等） | L490-560 |
| 交通/娱乐设施计数（EnV4/EcV4） | **统计年鉴+电话访谈（非高德 POI——勘误 1）** | 设施件数 | L575, L869-875 |
| 城市化率/教育计数（SV1/CV1-5） | 同上 | 区县级统计量 | L618-684 |

### Li et al. 2021（= Q. Li 2021，成都）

**论文级**：街道段（两连续交叉口间，4434 段）；多期混用（2016 出租车 / 2018 多数 / 2021 点评与街景，原文承认）。

**DV**：① 社会 = 滴滴出租车 OD（2016-11）+ 摩拜单车 OD（2018-08）地图匹配到街段；② 经济 = 大众点评评论 PCA 综合分加权商户数（2021）；③ 文化 = 高德 POI 五类文化设施 300 m buffer 计数（L1157-1158）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 绿视率/天空可视率 | 百度街景 2021（垂直 10°、120° 视场），DeepLabV3 | 采样点植被/天空像素占比均值 | L1375-1381 |
| 建筑密度/覆盖率 | 高德建筑矢量面 2018 | 300 m buffer 建筑栋数 ÷ 街道长度 | L1382-1384 |
| 土地利用混合（POI 熵版） | 高德 POI 2018-12-01 | 14 类 Shannon 熵（300 m buffer） | L1385-1387 |
| POI 计数/密度 | 同上 | 300 m buffer POI 总数 ÷ 街道长度 | L1388-1390 |
| 房价/地价 | 链家 2018-12-01（26,956 小区） | 最近小区房价 | L1391-1393 |
| 道路密度（road sparsity） | OSM 2018 + 百度补充 | **300 m buffer 道路条数（计数非长度！）** | L1394-1396 |
| 公交/地铁可达性 | 高德 POI 2018 | 公交站 500 m / 地铁站 800 m buffer 覆盖街道数 | L1397-1402 |
| 百度/腾讯热力（crowd counting） | 腾讯 EasyGo 热力 API 2018-12-07 | 300 m buffer 热力均值 | L1403-1405 |
| 就业密度 | 智联招聘 2018-12-01（100,653 条） | 核密度后重分类 1-9 级 | L1399-1401 |

### Chen et al. 2021（Biljecki，9 城）

**论文级**：~1 km 格网（对齐 WorldPop 1 km）；9 城；数据年份原文未说明。

**DV**：vitality score 五成分——POI 核密度（OSM）、推文密度（Twitter API，终剔除）、NTL（VIIRS DNB 月均）、人口（WorldPop 2018）、Airbnb 核密度（Inside Airbnb）（L683-710）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 建筑密度/覆盖率 | OSM + 部分城市官方开放数据（L469） | footprint 总面积 ÷ 格网面积 | Table 1, L530-545 |
| 交叉口密度 | 同上 | 交叉口总数 ÷ 格网面积 | 同上 |
| 街区密度/平均街区面积 | 同上 | block 总面积 ÷ 格网面积 | 同上 |
| 道路密度 | 同上 | 道路总长 ÷ 格网面积 | 同上 |
| 土地利用混合（LUM） | 土地利用 shapefile：OSM + 各市政府数据 | Shannon 熵（Eq 1） | L537 |
| 建筑形态类型（四概率） | OSM 路网经 OSMnx 渲染 CRHD，ResNet-34 | 半径 1 km CRHD 输入，softmax 概率 | L448, L377-380 |

### Yue et al. 2021（HCMC vs Shanghai）

**论文级**：sub-district；多数指标分母 = sub-district 面积；数据约 2016（人口 2015 普查）。

**DV**：PPM 合成 3 维 9 指标（无单一外源活力观测）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 土地利用混合（12 类熵） | 土地利用图 2016：上海地质调查研究院 / **越南科学技术院（VAST）** | MLU = −Σpᵢln(pᵢ)/ln(n)，12 类 | L344, L422-425 |
| 交叉口密度 | OSM 2016 + 行政图，拓扑分析 | 每 sub-district 路口数 | L331, L484-485 |
| 老建筑 | 上海：住建委历史建筑名录；**HCMC：无名录，从土地利用图提取保护用地图斑转点** | 每 sub-district 计数 | L483-484, L580-583 |
| 人口密度 | 两地 2015 普查 | 人口 ÷ 面积 | L494-495 |
| POI 计数/密度 | OSM POI 2016 | POI 数 ÷ 面积；剔除无关 POI | L496-497, L481-483 |
| 居住密度 | 土地利用图 2016 | 居住用地 ÷ 面积 | L500-501 |
| 道路密度 | OSM 2016 | 总长 ÷ 面积（km/km²） | L511 |
| 铁路/高速隔离 | 河流图 + OSM + 行政图 | 河流/国道/高速 30 m buffer + 铁路 50 m buffer 面积占比 | L516-518 |
| amenity availability | OSM POI 2016 | （公交站+车站+广场+公园）÷ 面积 | L524-525 |

### Scepanovic et al. 2021（意大利 6 城）

**论文级**：census district（平均 2.47 km²）；DV 2014-02~10；Sentinel-2 2018。

**DV**：TIM 移动互联网连接数（34% 份额），Voronoi 按面积比例分摊到 district（Eq 9）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 交叉口密度 | 原文未说明（总述 L209 仅列 "Urban Atlas, OSM, mobile phone data"） | 数 ÷ district 面积（Eq 6） | L591-596 |
| 距小公园距离 | 原文未说明 | block 质心到最近小公园（<1 km²）距离的平均倒数（Eq 3） | L555-556 |
| 土地利用混合 | Urban Atlas（EEA，20 类） | 三大类熵 ÷ log 3（Eq 2） | L533-541 |
| 平均楼层数 | ISTAT | 按高度类别加权（Eq 4） | L579 |
| 街区尺寸/各向异性 | 原文未说明 | block 面积均值；面积/外接圆面积比 | L582-585, L617-620 |
| OSM PoI 三类得分 | OSM amenity | sustenance/transportation/entertainment 三类；imagelet 内 score = log(1+n) | L1070-1072 |

### Nathansohn & Lahat 2022 — n/a（纯质性）

### Gómez-Varo et al. 2022（Barcelona, JANE）

**论文级**：50 × 50 m 格网；四规则归算（100 m 半径均值/有无/计数/距离，L508-512）；Nou Barris；2016–2019。

**DV**：JANE Index v2，22 变量 z-score 按权重加总 6 条件分（测"条件"非活动）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 人口密度（DENS） | 市议会 City blocks 2016 | hab./km²，100 m 半径均值 | L738 |
| 商业/公共设施密度 | 市议会 Retail inventory 2016 | /km²，100 m 半径 | L740 |
| 建筑功能混合熵（BuMI） | 财政部 Municipal cadaster 2017 | 六类用途地籍图斑熵 −Σpᵢln(pᵢ)/ln(n) | L742, L407-411 |
| RNR / CFMIX / BNB | 同上 + Retail inventory | 平衡指数（0–1） | L423-425, L643-650 |
| 交叉口密度（INTERSEC） | 市议会 Road Graph 2019 | 格网内计数 | L744, L660 |
| 句法（介数中心性） | City blocks 2016 + 普查人口权重 | 半径 600 m，最短路径 × 人口权重 | L375, L433-441 |
| 建筑年龄（YEARM/YEARSD） | Cadaster 2017 | 建造年份均值/标准差 | L677-690 |
| 租金（RENT） | 市议会 Rental Market 2018 | 邻里平均租金 | L752 |
| 公交可达性 | 市议会站点 2019 | 加权距离（铁路 40%/地铁 30%/公交 20%/单车站 10%） | L457-460 |
| 坡度（INCLIN） | 加泰罗尼亚制图所 15 m DEM 2019 | 街道坡度° | L756 |
| 照明/限速区/人行道/座椅 | 1:1000 地形图 2018 | 有无/距离 | L718-729 |
| 距 border vacuum | Land use cartography 2016 | 到大型单一功能建筑(≥5000 m²)/大公园/干道/停车场/空地距离 | L475-480 |

### Lee & Kang 2022（Seoul）

**论文级**：424 行政洞；日频面板 2018-11-01~2019-04-30，6:00-21:00；全变量 log。

**DV**：SKT 浮动人口（50×50 m 测量网格 181,929 个）；访问人口 = 总人口 − 常住（夜间 >50%）− 工作（工作日 >30%）（L460-466）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| PM10 | KT 监测网络 330 测点 | Kriging（Gaussian）插值 50 m 栅格聚合到洞 | L378-384 |
| 总楼面/地块面积 | 国土交通部 | m² | L540 |
| 土地利用占比（4 类） | 行政安全部 | 设施用地 ÷ 洞面积 | Table 1 |
| 土地利用混合（熵） | Seoul open data plaza | 熵；居住/商业商务/工业/公园绿地 | L394-397 |
| 地铁/公交可达性 | NGII | 站点数 ÷ 面积 | L548 |
| 步行道路长度 | NGII | m/m² | Table 1 |
| Walk score | Kim et al. 2019 韩国版指数 | 0–100，直接采用 | L843-846 |
| 公园占比 ×5 | 国土交通部 | 各类型公园 ÷ 洞面积 | L582 |
| 生活 SOC 设施 ×5 | NGII | 设施数 ÷ 面积 | L630 |
| 海拔 | NGII | 洞平均海拔（坡度因共线性剔除） | L630 |
| 常住/工作人口密度 | SKT | 同 DV 阈值定义 | L632, L460-466 |

## Batch C

### Wu et al. 2023（Shenzhen）

**论文级**：500 m grid（4526 个）；EasyGo 2019-05-04~17；POI/建筑/SVI 年份原文未说明。

**DV**：腾讯 EasyGo LBS 实时人口密度（原始 25 m，聚合 500 m × 2 h，20:00 分昼夜，L246-249）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 空间句法（Choice/Integration） | **路网源原文未说明** | sDNA | L277-281 |
| 道路密度 | 原文未说明 | 总长/面积 | L438 |
| POI 六类计数 | AMap，年份未说明 | 500 m grid 内各类个数 | L286-288, L439-444 |
| POI 熵 | 同上 | **exp(−Σpᵢln pᵢ)**（Eq 1） | L296 |
| FSI / GSI | 深圳市规划和自然资源局建筑数据 | 总楼面占比 / 覆盖面积占比 | L330-334, L446-449 |
| 绿视率/开敞度/步行占比 | 百度 SVI（PSPNet+ADE20K，4 方向） | 像素占比公式（Eq 2/3） | L345-360 |
| 感知评分 5 维 | MIT Place Pulse 2.0 | SVI 感知评分 | L377-379 |

### Jiang et al. 2024（Beijing）

**论文级**：六环内七区 TAZ；信令与 OD 2019-05；珞珈影像 2018-12；POI/建筑/绿地年份未说明。

**DV**：三源 ÷ TAZ 面积后 PCA 合成——信令（250 m 网格聚合）、珞珈一号 LJ1-01（130 m）、OD（>30 min stationary 判定，top 60,000）（L329-376）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| MBH/MBA/MBD | 北京建筑数据集（656,564 栋；提供方与年份未说明） | TAZ 内均值 | L384-386, L461-463 |
| GSI / FSI | 同上 | Gx/Ax；Fx/Ax（Spacematrix） | L480-490 |
| 道路密度（RD） | OSM | TAZ 级（公式未给） | L405-407 |
| TFD/IFD/SHD | 未显式绑定（全文设施仅 AMap POI 与 OSM 两源） | 公式未给 | L503-515 |
| LUM（MIX） | AMap 15 类 POI | −ΣPᵢln Pᵢ，**按 gross floor space 加权**（Eq 3） | L524-535 |
| MGD/LGPI/GED | Lang et al. 2023 全球冠层高度数据集（GEDI+Sentinel-2） | 绿地距离/最大斑块占比/边缘密度 | L409-413, L555-568 |

### Chen et al. 2024（SF, IUNV）

**论文级**：census tract（N=241）+ block group/block 双尺度；ACS 2016-2020；Flickr 2013-2022。

**DV**：Flickr 公园照片（自写爬虫，清洗后 3434 张）按 block 计数；Google Vision API top-10 标签 + 层次聚类（L547-582）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| UGS 可达性 | OSM 'leisure park'（442 个 UGS） | block group 中心 1000 m 服务分布 | L463-471 |
| 土地利用熵 | 原文未说明 | −Σ(pⱼ·ln pⱼ)/ln k（Song et al. 2013 法） | L478-491 |
| 道路密度 | OSM | L/S（Eq 6） | L315-316, L566-571 |
| census 9 变量 | CDC SVI 2020 / ACS 2016-2020 | tract 级（⚠ 65+ vs 模型 Age>60 口径漂移） | L299-303 |
| Tree 等 15 街景要素 | Google Street View（10 m 取点 26,810 个）+ ADE20K | 像素占比按街道平均 | L318-326, L388-395 |

### Collins et al. 2024（意大利 7 城）

**论文级**：TIM 不规则 CDR 网格；CDR 2015-02-28~04-30；census ISTAT 2011；OSM 2017。

**DV**：TIM CDR 15 分钟粒度，性别自 SIM 注册；核心 DV = 女性−男性活力差（L157-191）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| POI 密度（Foursquare 14 类） | OSM 2017 | ρ = N/A（Eq 2） | L265-270 |
| Third place 密度/多样性 | OSM 标签子集（Jeffres 4 类 + 自加 1 类） | 同上 + Shannon | L272-296 |
| 建筑/路网密度 | OSM 2017 | ρ = N/A；分模式网络长度 | L222-230 |
| 交叉口密度 | OSM 2017 | 分模式可达交叉口均值 | L226-230 |
| 多样性（POI/building/highway） | OSM 各特征集 | Shannon-Wiener log2（Eq 3） | L242-254 |
| census 对照 | ISTAT 2011 | areal interpolation 到 CDR 网格 | L298-301 |

### Dogan & Lee 2024（Seoul）

**论文级**：500 m grid（1881 个）；建筑与街道 = 首尔市政府 2020；活力数据年份无法确认（表格打散）。

**DV**：10 变量 min-max × FA 合成三 facet；源 = Seoul Open Data Plaza（living migration）、BC Card、Kakao Map POI（13 类 **Simpson**）、Airbnb（L892-958）。

**IV**：22 个物理复杂度指标全部由首尔市政府 2020 footprints + 中心线计算（ArcGIS/Rhino/Grasshopper/R）：十字口/尽端路计数、覆盖尺寸/高度/街长熵、街道朝向熵（Boeing 2019 法）、3D isovist、box-counting 均匀度、街段统计、建筑凸度/方正度（Basaraner & Cetinkaya 2017）、全城可达性；高差方差源未说明（L511-612, L729-744）。

### Wang et al. 2025（Shenzhen）

**论文级**：300 m grid（GB 50180-2018 五分钟生活圈；5,250 个居住格）；WorldPop 2020、Amap 居住 POI 2024-04、Dianping 2023。

**DV**：改进高斯 2SFCA——供给端 Sj = 大众点评评论数（243,923 条，对数变换防刷评，零评论剔除后 209,815、9 类）；需求端 Dk = WorldPop 人口；d0 = 1000 m 路网距离；**路网源原文未说明**（L327-443）。

**因子表**：n/a（仅指数 + 空间自相关验证）。

### Cai et al. 2025（Beijing, SOFM）

**论文级**：block（《北京城市更新专项规划》2022）；人口 2021、Amap POI 2022、GCM 2021、百度 SV 2022、链家 2022。

**DV**：14 指标 z-score 固定权重合成 6 维（Eq 6-11）→ SOFM 4 簇。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 人口密度 | 微信位置 × 珞珈一号融合（文献[33]），100 m 网格 | ArcGIS fusion 聚合到 block 取均值 | L880-882, L1036-1038 |
| CD/PFD（POI 密度） | Amap POI 2022 | block 内总数 ÷ 面积 | L550-554, L1040-1042 |
| BFM/CFM（POI 熵） | 同上 | −Σ(pᵢ·ln pᵢ)/ln n | L1050-1062 |
| RNR/BNB | 同上 | 平衡指数 0-1（Eq 3/4） | L1064-1076 |
| 交叉口密度 | **正文 GCM 2021 vs Table 1 Amap——矛盾（勘误 2）** | 数 ÷ block 面积 | L748-754, L1098-1101 |
| 开敞度 | 百度街景 2022（360°），SegNet | 天空像素占比（Eq 5） | L760-766 |
| 建筑年龄/房价 | 链家 2022 | 平均年龄/标准差/均价 | L884-887, L1112-1116 |
| DPT/DRLP | GCM 2021（自然资源部） | 居住设施到公交站/火车站/大型公园最短距离 | L1121-1130 |

### Tan et al. 2026（Shenzhen）

**论文级**：500 m grid（6714 个）；DV 2024-03-11~24（14 天）；**全部 IV 年份原文未说明**。

**DV**：百度慧眼——Location SDK 去重设备数，14 天平均，500 m 聚合（L464-469）。

**IV**：原文仅列类别名（"real estate transaction data, property registration data, road and building data, road network data, POIs, and street view images"，L424-427），**供应商与年份均未说明**。逐因子：房价（grid 均价）、社区年龄、13 类 POI 计数（均匀网格下计数≈密度，L817-820）、POI 熵（Shannon）、FAR、道路密度、9 个距离因子、句法（半径 800 m）、视觉感知 6 项（百度街景 205,000 张 + Mask2Former）、情感感知 6 项（ResNet-152 + Place Pulse 2.0）。

## Batch D

### Qin et al. 2025（Nanjing）

**论文级**：TAZ（509 个，交管部门划分）；全部数据 2018（信令 2018-10、NTL 2018-10-29、POI/路网/建筑 2018 年末）。

**DV**：四维熵权——信令（南京市规划和自然资源局，510,836 条 OD）、珞珈一号单幅 20181029、文化 POI 密度（AMap 14,868 条）、创新 POI 密度（AMap 30,873 条）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 平均楼层数 | 建筑轮廓 218,958 polygon（资源环境科学数据中心；⚠ 正文另称 OSM/AMap，勘误 5） | TAZ 内平均 | L222-226 |
| **"Building density" = FSI（命名反转）** | 同上 | FSIj = Fj/Sj | L395-399 |
| **"Volume rate" = GSI（命名反转）** | 同上 | GSIj = Gj/Sj | L401-405 |
| 土地利用混合（stepwise 剔除） | resdc.cn，6 一级 + 25 二级类 | E = −Σpᵢ·ln(pᵢ)，面积占比 | L246-249, L389-394 |
| 道路密度 | OSM 2018（分段计数：县道/城市/国道/高速/省道） | 总长 ÷ TAZ 面积 | L229-240, L407-408 |
| 交叉口密度（剔除） | OSM 2018 | 数 ÷ TAZ 面积 | L413 |
| 公交站密度 | 未单列（总述 OSM/AMap 2018 年末） | TAZ 内数量 | L411 |
| 距地铁站/市中心/公园、商圈数量 | 原文未说明（市中心点定义未给出） | 距离 km / 计数 | L409-419 |

### Liu et al. 2025（Nanjing, EPB）

**论文级**：urban block（712 个，道路网围合）；POI 2019-04-01、街景 2019-03、信令 2019-04-12/13。

**DV**：V = (P+S)×H；GT = 未具名运营商信令（约 30% 人口，31,820,372 条日志）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| F1 POI 熵（17 类） | AMap 公共地图 2019-04-01（397,399 条） | −Σpᵢ·ln(pᵢ)，按 POI 数量占比 | L207-211, L276 |
| F2 建筑密度（覆盖率） | 建筑轮廓及高度（⚠ 表 1 版式错乱，源未独立标明，勘误 6） | Σaᵢ/Area | L277 |
| F3 Built area（类 FAR） | 同上（含层高 hᵢ） | Σ(aᵢ×hᵢ)/Area | L278 |
| F4 Service perfection | AMap POI | 类型数 × 密度 | L253-256 |
| F5 交叉口 | 四级路网（源未独立标明） | Junc 数 ÷ 面积 | L283 |
| H 主观活力 | Place Pulse 2.0（366,802 对）+ 百度街景 45,751 张（100 m 采样，2019-03） | TrueSkill → DenseNet121 → block 内 lively 比例 | L191-194, L331-371 |

### Liu et al. 2022（Vital Triangle，长三角 27 市）

**论文级**：城市整体；多数指标 2019，人口七普 2020，建成区 2018；AHP（10 学者）与熵权取平均。

**DV**：28 指标逐行源（Table 3）——中国城市统计年鉴（多数）、七普（2020）、百度迁徙、企查查、AMap POI、OSM（c4 道路密度 m/km²）、NCEI 夜光、阿里研究院（d3 数字政府）、中国环境监测总站（e1/e2）、中国城市建设统计年鉴（e4）、携程（f3）、Web of Science（f4）。c1 = 6 类休闲 POI 熵、c2 = 6 类休闲 POI/建成区面积、c3 = 23 类 POI 熵（均 AMap）。

### Shi et al. 2023（Mianyang, VAN）

**论文级**：NTL 栅格-斑块单元（VAN-1/2/3 按 Grade 4 出现频次）；NTL 2018-10-15 与 2019-02-17 两期、百度热力 2021-05-12、POI 2020-06。

**DV**：珞珈一号两期融合 → 低通滤波 → 自然断点 4 级；叠加百度热力（21:00-21:30 截图 12 幅拼接）与 8 类夜经济 POI。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| POI（DV 成分，8 类） | 表 1 命名 "Baidu POI"，**获取栏写 "Downloaded from Google Map in June 2020"（勘误 3）** | Rowe 夜经济 8 类（餐饮 15,423 等） | L427-434, L489-493 |
| X1-X8 距离（路网/水体/机场/火车站/铁路/中央公园） | 绵阳市自然资源和规划局基础地理信息数据 | **5 级分类变量，各级阈值原文未说明** | L406-414, L604-672 |
| 土地利用（10 类）/混合度/LUCC | 同上 + 0.5 m 遥感影像（Google Map 2020-06 校正） | 分类/连续/变化三变量；混合度公式未说明 | L396-404, L676, L770-772 |
| 坡度/起伏度 | 原文未说明 | 连续变量 | L680-681 |

### Ruszczyk et al. 2023 — n/a（质性混合方法）

### Williams et al. 2019（成都+沈阳）

**论文级**：300 m fishnet 居住网格（两级 Queen 邻接）；Landscan 2014、Dianping 2015-12/2016-01、Amap 2016-04。

**DV**：amenities score——Hansen 重力模型（β=2 中国城市尺度修正，di,j = 质心到每类最近 10 个设施欧氏距离，需求侧 Vj = Dianping 评论数，公共/私人权重 wk 不同）；8 类设施 = Dianping 挖掘；人口 = Landscan 2014（1 km，24-h ambient）+ GMM 划分城/郊。注：摘要提 Fang/Baidu，方法部分未用 Fang；Baidu Total View 仅验证用（L372-404, L637-645）。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 8 类设施可达性 | Dianping 2015-12/2016-01 | Hansen 重力（β=2，Δr=1.5 km） | L372-373, L394-404 |
| Landscan 人口 | Landscan 2014 | 面积加权到 300 m；GMM 多中心拟合 | L258-269 |
| 居住网格界定 | Amap POI API 2016-04（选 Amap 因对比后更准确） | 去重 + 坐标纠偏 + Queen 邻接 | L314-340 |
| 验证 | 实地踏勘 + Baidu Total View | 逐网格人工标注 typology | L637-705 |

### Li et al. 2022（Wuhan）

**论文级**：community block（1025 个；分母 = 社区净面积）；街景 2019-05；签到一个月（月份未说明）；人口普查口径（年份未说明）。

**DV**：① 百度街景 67,761 张（~30 m 采样），FCN+ADE20K 六类要素计数标准化求和；② 微博签到 114,145 条标准化；③ 两者取平均。

| 因子 | 数据源 | 计算口径 | 证据 |
|---|---|---|---|
| 人口密度 | 社区常住人口普查（武汉市自然资源和规划局）+ GPW 补充 | 人/净面积（100/ha） | L894-895, L676-678 |
| 社区年龄/房价 | 搜房 4,988 + 58 同城 2,520 小区 | 普通克里格（指数 variogram），zonal 均值 | L897-901, L953-955 |
| FAR | 武汉国土资源和规划信息中心专题图（L679-682） | 总楼面 ÷ 地块面积 | L977-979 |
| Open space rate | 同上 | 非基底占用面积占比 | L979-981 |
| 交叉口/道路密度 | OSM | 数 ÷ 净面积（/ha）；主要街段总长 ÷ 净面积 | L679-681, L983-985 |
| POI 9 类密度 | 百度地图 POI（lbsyun.baidu.com，278,652 条归并 9 类；年份未说明） | 社区 500 m buffer 密度（/ha），取对数；地铁站稀疏致零膨胀仅保留公交 | L811-819 |
| 土地利用混合 | 武汉规划信息中心专题图 | **居住用地占比（刻意替代 Shannon 熵，方向解读需反转）** | L887-890 |
| 距离类 7 因子 | 原文未说明 | 质心到最近目的地 km；欧氏/网络未说明 | L959-970 |
| NDVI | Landsat（国家地球系统科学数据中心 geodata.cn） | (NIR−VIS)/(NIR+VIS)，负值置 0 | L682-684, L1000-1008 |
| 空间句法 | OSM 路网；计算软件未说明 | Integration = RA/RRA；Betweenness；半径 800-2500 m（1500 拟合最佳） | L1015-1040, L1421-1426 |

### Doan & Zhang 2025 — n/a（PRISMA 综述，191 篇）

论证价值句：L40 "previous studies … often rely on linear models, neglecting nonlinearity, interactions among factors, and causality"。

---

*提取：4 个子会话（batch A–D，kimi-k3 max，2026-09-14）；合并与 §1–§3 综合：父会话。所有行号为 txt 提取层行号，PDF 页码未核。*
