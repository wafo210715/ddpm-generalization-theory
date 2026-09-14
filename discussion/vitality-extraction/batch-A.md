# Batch A 逐篇提取记录（8 篇）

提取日期：2026-09-12。证据行号均指 `/Users/guanhongli/Documents/wafo-base/literature/papers/` 下对应 txt 提取层。分类按 2026-09-10 用户裁定执行（POI 计数/密度/熵→BE；静态人口存量→Human；夜间灯光→Human；主观感知分→Human；房价/社区年龄→待裁定）。

---

### Zeng 2018

**城市 / 尺度**：Chicago 与 Wuhan 核心区 / block group（Chicago，2159 个）、community neighborhood（Wuhan，1076 个）（L328「spatial scales of block group and community neighborhoods」；L378「2159 block groups in Chicago, and 1076 community」）

**vitality 表征**：composite index
- **定义**：将活力分解为 density / accessibility / livability / diversity 四维各 3 个共 12 个指标（L924「We choose 12 indicators in total in these four」），用 spatial TOPSIS 排序合成（L388「weight in formulating the spatial TOPSIS model」；L498「apply spatial TOPSIS to assess and rank urban vitality」）；POI 与社经数据 2016–2017（L489）。
- **构成**：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 人口密度 | Human | population density（L493-494「population density, road density, and building density」，density 维度） |
| 道路密度 | BE | road density（L493-494） |
| 建筑密度 | BE | building density（L493-494） |
| 距学校/医院/商店/景点距离 | BE | distance to schools, hospitals, shops, and tourist attractions（L495-496） |
| 银行/餐饮/休闲/美容汽车服务 POI 计数 | BE | number of banks, food service sites, sites for leisure and recreation…（L496-497） |
| 土地利用 Shannon 多样性 | BE | Shannon diversity of urban land use（L497-498） |

**因子表**：n/a（仅制图/描述）——全文只做维度对比、空间格局描述与 TOPSIS 排名，未做活力对因子的回归/关联分析（检索 regression、coefficient、correlation 均无实证关联结果）。

## 附注：活力指数内含供给指标（livability 的 POI 计数与 accessibility 的设施距离，属 T4）；两城口径不完全可比（block group vs community neighborhood）。

---

### Liu 2019

**城市 / 尺度**：Nanjing / 手机基站改进 Voronoi 单元（14,236 个空间统计单元，L604「Nanjing was divided into 14,236 spatial statistical units」），半小时时间粒度，2018-06-11 至 17 一周（L560-561；L565「recorded periodically (every half hour)」）

**vitality 表征**：flat
- **定义**：中国电信手机信令（UDR）推算的各基站单元每半小时在线用户数时间序列，作为时空活力代理（L236-237「dynamic human activity data…used as a proxy for urban spatiotemporal vitality」）；SOM 聚类把单元划分为 10 个（工作日）/8 个（周末）活力区。

**因子表**（§5.3 对各活力区做 POI 土地利用特征关联分析，Table 4）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| POI 密度 | BE | 驱动因素（活力区特征指标） | ↑ | L1017-1018「spatial concentration of land use has a significant impact on vitality」（中低活力区 POI 密度低于高活力区） |
| POI 熵（exp Shannon，值大=无序） | BE | 驱动因素 | ↓ | L1024-1025「orderly mixed land use is conducive to enhancing regional vitality」（高活力区熵更低） |
| POI richness（类别数） | BE | 驱动因素 | 不显著 | L1028-1029「correlation between land use richness, evenness, and urban vitality is not significant」 |
| Simpson 指数（evenness） | BE | 驱动因素 | 不显著 | L1028-1029（同上句，evenness 即 Simpson） |
| 主导 POI 类型（Max_Type） | BE | 驱动因素 | n/a（仅制图/描述） | L714-716 仅用于标注活力区主导功能，无方向性检验 |

## 附注：clean——POI 指标只用于解释活力区，不进活力测量本身；熵的方向注意是「有序混合→高活力」，非「熵越高越好」。

---

### Xia 2020

**城市 / 尺度**：Beijing、Changsha、Tianjin、Shenyang、Wuhan 五市（L300-301）/ street block（L403「definition of street blocks in this study」）

**vitality 表征**：flat（日间/夜间两个独立测度）
- **定义**：DV（daytime vitality）= 2015 年大众点评小餐饮商家核密度（500 m 半径）× 评论数加权（L432-433「Kernel density analysis was adopted to measure the distribution of small catering businesses」；L440「weight…according to the review counts」；数据 L338-339）；NV（night-time vitality）= 2015 年 VIIRS DNB 年均夜间灯光辐射，~500 m 分辨率重采样加权到街区（L337-338；L440-446）。

**因子表**（双变量 Moran's I + LISA 空间关联分析）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| FAR（容积率） | BE | IV（土地利用强度维度之一） | ↑ | L457-459「Significant positive spatial autocorrelations were found…all Moran's I values > 0 and p-values < 0.01」；数值例（沈阳）L465-466「FAR-DV: 0.44, FAR-NV: 0.52」 |
| BCR（建筑覆盖率） | BE | IV（土地利用强度维度之一） | ↑ | 同上；L465「BCR-DV: 0.42, BCR-NV: 0.58」 |
| 土地利用混合（hybrid vs single 街区） | BE | 分组对比变量 | ↑ | L872-874「high-density mixed-use street blocks, which are more likely to have relatively high urban vitality」；L706-708「hybrid street blocks have the largest proportion in HH type」 |
| 城市化时期（1995 前老城区 vs 1995–2015 新区） | 待裁定 | 分组对比变量 | ↑（老城区） | L872-874「…relatively high urban vitality, are largely located in old urban areas」 |
| 土地功能类型（六类 single/hybrid） | BE | 分组对比变量 | 未报告（categorical） | L690-691「green street blocks have the highest vitality, whereas industry street blocks generally have the lowest」（类别排序，无数值方向检验） |

## 附注：DV 本身是餐饮商家存量 × 评论数——供给指标直接当活力测量（T7）；NV 用 NTL（电力基础设施污染嫌疑，T2 suspected）；强度与活力存在局部失配（LH/HL 街区），全局正相关不代表处处成立。

---

### Kim 2020

**城市 / 尺度**：Seoul 四个区 / block group（由快速路与主干道围合的步行者地理边界，L674「a block group surrounded by expressways and arterials」）

**vitality 表征**：flat
- **定义**：手机信号估算的服务人口（service population），2018 年 4 月工作日 6am–9pm，Seoul Open Data Plaza（L778「6 a.m. to 9 p.m. on all weekdays (Monday to Friday) in April 2018」），作空间自回归 DV（L18-19）。

**因子表**（Table 3 空间自回归终模型；txt 中表格打散：数值块 L1230-1340，变量名块 L1359-1390）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| tare space（不可利用空间占比，含绿带/水面） | BE | IV（natural context） | ↓ | L1230「−129.930」（p=0.034）；L1419「has a negative coefficient in the model」 |
| slope（平均坡度） | BE | IV（natural context） | 不显著 | L1231-1232（p=0.695） |
| mesh（平均街区长度） | BE | IV（network form） | ↓（边缘显著 p=0.085） | L1237；L1544「coefficient of the mesh variable is −0.175」（标准化，txt 抽取形） |
| gamma index（网络连通度） | BE | IV（network form） | ↑ | L1246「159.510」（p=0.024） |
| subway entrance density | BE | IV（network form） | ↑（边缘显著 p=0.073） | L1247「1.428」 |
| bus stop density | BE | IV（network form） | 不显著 | L1248（p=0.969） |
| building entrance density | BE | IV（plot form） | 不显著 | L1255（p=0.119） |
| share of plot for building group | BE | IV（plot form） | ↓ | L1271「−200.020」（p<0.05） |
| plot for building group / built area | BE | IV（plot form） | 不显著 | L1272（p=0.475） |
| average plot area | BE | IV（plot form） | 不显著 | L1273（p=0.162） |
| number of plots per km² | BE | IV（plot form） | 不显著 | L1274（p=0.891） |
| floor area ratio（FSI） | BE | IV（building form & land use） | ↑ | L1300「19.739」（p=0.018） |
| residential buildings（住宅总建筑面积） | BE | IV（building form & land use） | ↑ | L1301「55.269」（p<0.01） |
| balance index 1（邻里商业 vs 住宅 GFA 平衡） | BE | IV（土地利用平衡） | ↑ | L1302「149.560」（p<0.01）；标准化系数最大 0.320（L1398-1400「the largest in the balance index…0.320」） |
| balance index 2（非日常商业 vs 住宅） | BE | IV | 不显著 | L1303（p=0.825） |
| balance index 3（办公 vs 住宅） | BE | IV | 不显著 | L1304（p=0.325） |
| balance index 4（教育福利 vs 住宅） | BE | IV | ↑ | L1305「74.420」（p=0.014） |
| area of unit（单元面积） | BE | 控制变量 | ↑ | L1335「41.126」（p=0.001） |
| type of district（区类型） | BE | 控制变量 | 不显著 | L1336（p=0.361） |

## 附注：clean；network density 与 open space ratio 因共线性被剔除（L1393-1395「network density has a correlation coefficient of −0.667 with the mesh」），故网络密度只有 mesh/gamma 两个代理进入终模型。

---

### Yang & Pan 2020

**城市 / 尺度**：中国 326 个地级市 / city，2000、2005、2010 三期面板（L408；L459-460「326 cities」）

**vitality 表征**：flat
- **定义**：用平均人力资本（average HC）替代人口作为城市活力指标，EIC（education-income combined）法结合教育与收入回报维度（L124「education-income combined (EIC) approach」；L27-28「proposes an alternative indicator, average human capital (HC), to investigate city vitality」）；用夜间灯光作经济活力代理验证 HC 的解释力（L263-265）。

**因子表**（OLS + 城市/年份固定效应，NTL 作 DV，Table 2；稳健性检验用 GDP 作 DV，Table 3）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| 平均人力资本 HC | Human | IV | ↑ | L445-446「one unit increase in average HC is correlated with a 1.759 increase in night-time light」；纳入人口后仍 ↑（L450-451「1.337 increase in light」） |
| 人口规模 | Human | IV | ↑ | L446「The coefficient of population is 0.001」 |
| GDP（总量/一二三产）、FDI、固定资产投资、面积（控制） | Human | 控制变量 | 未报告 | L404-406 列出控制变量来源，正文只报告 HC 系数在控制后稳健（L456-457），未给各控制变量方向 |
| （稳健性）HC → 三产 GDP | Human | IV | ↑ | L426-427「higher tertiary GDP by about 173 billion RMB」 |

## 附注：vitality ≈ 人力资本，概念漂移（T6 suspected）；验证侧用 NTL 作经济活力（T2 suspected）；城市尺度，无空间单元内的 BE 因子。

---

### Mouratidis & Poortinga 2020

**城市 / 尺度**：Oslo 大都市区 / neighborhood（受访者所居社区，问卷 + 地理空间数据）

**vitality 表征**：flat
- **定义**：居民对社区「liveliness / interesting things happening / opportunities for entertainment」三题 1–5 分评价合成活力量表，Cronbach's α=0.904（L269-275「The urban vitality scale has a Cronbach's alpha of 0.904」）。

**因子表**（SEM 路径分析，Table 3）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| neighborhood density（人/ha，居住人口密度） | Human | IV（建成环境特征） | ↑ | L570-571「neighborhood density and local amenities both have strong positive associations with urban vitality」 |
| local amenities（1000 m 内咖啡馆/餐厅/社区中心/酒吧计数） | BE | IV | ↑ | L570-571（同上句） |
| distance to city center（步行路网距离） | BE | IV | ↓ | L569-570「distance to city center has a strong negative association with urban vitality」 |
| green space（1000 m buffer 绿地占比） | BE | IV | ↓ | L572「Green space has a negative association with urban vitality」 |
| public transport accessibility（500 m 内高峰班次/小时） | BE | IV | 不显著 | L572-573「association between public transport and urban vitality is non-significant」 |

## 附注：clean——明确「density 是活力的前置条件，不属于活力测量本身」（L147-148），是分类纪律的正面范例；摘要中的 land use mix 在模型里操作化为 local amenities（L29 vs L257-259）。

---

### Shi 2021

**城市 / 尺度**：Shanghai / 16 个行政区，2010–2019 年度面板（L1135「16 districts in Shanghai from 2010 to 2019」）

**vitality 表征**：composite index
- **定义**：按可持续发展评估框架从经济、社会、环境三系统选 18 个指标（C1–C18），用离差最大化（DM）法定权合成活力综合指数（L32-34；摘要 L25-28），再对指数本身做 Moran's I / LISA 空间自相关。
- **构成**（Table 1，L566-583 起）：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 人均 GDP [C1] | Human | Per capita gross domestic product (L579) |
| GDP 增速 [C2] | Human | GDP growth rate (L580) |
| 城镇化水平 [C3] | Human | Urbanization level (L581) |
| 人均财政收入 [C4] | Human | Per capita fiscal revenue (L582) |
| 消费品零售额/GDP [C5] | Human | Proportion of retail sales…to GDP (L583) |
| 固定资产投资/GDP [C6] | Human | Proportion of investments in fixed assets to GDP |
| 第二产业占比 [C7] | Human | Proportion of secondary industry to GDP |
| 第三产业占比 [C8] | Human | Proportion of tertiary industry to GDP |
| 人口增长率 [C9] | Human | Population growth rate |
| 人口密度 [C10] | Human | Population density（静态人口存量） |
| 劳动力占比 [C11] | Human | Proportion of labor force to total population |
| 社保就业支出/财政支出 [C12] | Human | Social security and employment expenditures… |
| 科技支出/财政支出 [C13] | Human | Science and technology expenditures… |
| 文化场馆数量 [C14] | BE | Number of cultural venues（设施计数） |
| 人均绿地面积 [C15] | BE | Per capita green area |
| 环保投资/财政支出 [C16] | Human | Proportion of environmental investments to total fiscal expenditure（财政投向） |
| 建成区绿化覆盖率 [C17] | BE | Rate of green coverage rate of built-up areas |
| 空气质量优良率 [C18] | BE | Rate of air quality excellence rate |

**因子表**：n/a——全文只对活力综合指数本身做空间自相关分析（L1013 Moran's I；L1672-1680「Moran's I index was positive in all years…p-values less than 0.05」），未做活力对任何因子的关联/回归分析。

## 附注：指数 ≈ 可持续发展评估（含财政支出、空气质量、环保投资等治理/环境指标，T4 + T6 双陷阱）；「活力」口径与 Jacobs 意义上的人群活动差异最大的一篇。

---

### Paköz & Işık 2022

**城市 / 尺度**：Istanbul / district（39 个城市区按毛人口密度分低/中/高三类，L862「gross population density of districts」；L889-891）；在线问卷 n=337，2020-06-01 至 05（L871「337 participants…between 1 and 5 June 2020」）

**vitality 表征**：composite index（PCA 因子）
- **定义**：居民对社区 vitality / mobility / safety / healthiness / cleanliness / orderliness 六维感知评分，PCA 降为两因子，「urban vitality」因子主载荷为 vitality（0.711）与 mobility（0.799）（L1789-1796 Table 8「Component 2: Urban Vitality…Vitality 0.711…Mobility 0.799」）。
- **构成**（Component 2「Urban Vitality」，L1789-1796；仅列主要载荷）：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| vitality 感知（载荷 0.711） | Human | Vitality（主观感知分） |
| mobility 感知（载荷 0.799） | Human | Mobility（主观感知分） |
| safety 感知（载荷 0.167，弱） | Human | Safety |

**因子表**（Kruskal-Wallis / Mann-Whitney 分组检验，Tables 7 & 9）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| 行政区密度类别（低/中/高，毛人口密度） | Human | 分组变量（district characteristic） | ↑ | L1749-1751「as the density of the district increases, the positive perception in terms of 'urban vitality' increases significantly」；原始六维层面同样 ↑（L1224-1225） |
| gender | Human | 分组变量（personal characteristic） | 显著，方向未报告 | L1745-1748「the perception of 'urban vitality' differs by gender significantly」（未给出哪组更高） |

## 附注：vitality 因子含 mobility 强载荷（0.799），不是纯活力测量（与 Mouratidis 的三题量表是不同工具，不可合并）；safety 感知随密度呈倒 U（中密度最高，L1226-1229），但它属于 healthy environment 维度而非 vitality，仅供阈值效应参考。
