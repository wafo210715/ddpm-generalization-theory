# Annotated bibliography

Each entry is a citation and one paragraph of at most 300 words (was 200 until 2 Oct 2026). Line references
(L#) point into the paper's text layer in wafo-base `literature/papers/`；p.X 指 PDF 页码，用于没有 txt 的论文（如 Doan 2025，从附件 PDF 读）。

现行规则（2026-10-06）：读过的文献逼自己写一篇总结；入选规则以后再定——到时候觉得太水的文章可以删。每篇能发表的文章都在自己的认识论和方法论里自洽。

---

Gómez-Varo, I., Delclòs-Alió, X., & Miralles-Guasch, C. (2022). Jane Jacobs reloaded: A contemporary operationalization of urban vitality in a district in Barcelona. *Cities, 123*, 103565.

The authors update the JANE index: 22 variables grouped into Jacobs' conditions,
mapped on a 50 m grid over Nou Barris, a low-income district of Barcelona, from
City Council and cadastre data. Variables are z-scored and summed with weights
the authors chose; no activity data is used, so the index measures conditions,
not vitality, and is never validated (L305–313, L686–698). Concentration is
0.4 population density + 0.3 retail density + 0.3 facility density, each
averaged within 100 m (L310–317). There is no building term, although the paper
says Jacobs named building density (L116–118). The consequence shows on the map:
high-rise housing projects score high on concentration, while a compact low-rise
estate scores low (L468–470, L502–503). Ground coverage would rank them the
other way round. For us, the paper is a reason to keep coverage (GSI), floor area
(FAR) and population apart rather than merge them into one "intensity", and a
test to run: where do our three disagree? Its 100 m kernel matches ours. Its
population measure can't be reproduced independently in HCMC, because WorldPop
is itself spread over building footprints.

---

Chen, M., Cai, Y., Guo, S., Sun, R., Song, Y., & Shen, X. (2024). Evaluating implied urban nature vitality in San Francisco: An interdisciplinary approach combining census data, street view images, and social media analysis. *Urban Forestry & Urban Greening*, 95, 128289. https://doi.org/10.1016/j.ufug.2024.128289

Gu, Y., Liang, X., Fan, Z., Hou, Y., & Biljecki, F. (2027). Street view video for urban sensing: Potential for measuring urban vitality. *Computers, Environment and Urban Systems*, 131, 102509. https://doi.org/10.1016/j.compenvurbsys.2026.102509

两篇都从"人在哪里、做什么"来看活力。Chen 等人用旧金山公园和城市自然空间里的 Flickr 地理照片数量（3 434 张）衡量"隐含城市自然活力"（IUNV），用照片内容聚类看人们拍下了哪些风景和活动，再把照片数量与公园可达性、用地混合度、街景要素（树、雕塑等）做回归。Gu 等人用墨尔本 CBD 的 892 条 YouTube 步行视频（街景视频，SVV），按 50 米路段统计行人数量、走 / 站 / 坐等动作，以及声音（音量、说话、音乐、车声），并按 Gehl 的框架分成"移动"和"社交"两个维度。两篇对活力的定义本身没有错；但从这样的定义出发，提升活力只能靠改善建成环境质量，两篇给出的规划建议也确实如此：增加绿地与配置资源，改善照明，交通稳静化，拓宽人行道（Chen L940–947；Gu L974–975、L989–991）。这类改善往往资源密集，而且是单向投入。我主张的城市活力是系统层面的稳定，而不是单方面消耗资源去改善建成环境，以"提升"一个数值。从系统的视角看，时空分析必不可少：Chen 的照片是十年累计，看不出时间上的变化；Gu 虽然区分了昼夜，却发现同一条街白天和夜间的人数并不相关（L837–838），说明活力不是一个静态的水平。之后我将沿着"表征时空间、理解时空间、推理时空间"的技术路线，应对城市这个复杂系统和城市活力这个复杂课题，而不是把城市当作一个可以求解的动态多目标优化问题。

---

Doan, Q. C., Ma, J., Chen, S., & Zhang, X. (2025). Nonlinear and threshold effects of the built environment, road vehicles and air pollution on urban vitality. *Landscape and Urban Planning*, *253*, 105204. https://doi.org/10.1016/j.landurbplan.2024.105204

按照我们对活力的标准，这篇的因变量是干净的：只有街景图里的行人数，没掺供给侧因子。但作者把车辆计数（自行车、汽车、公交、摩托）放在自变量一侧——行人和车辆由同一批图像数出，是同一次曝光的两种读数互相解释（每格多张图平均能冲掉单次曝光的噪声，但冲不掉同地繁忙：热闹的格子行人多、车也多）——最后归因出"推自行车和公交提升活力"的政策建议（推广共享单车项目、优化公交线路，p.15），这个建议没有因果依据。如果以人的 presence 为活力，就应该区分不同类型的 presence。如果非要回答"车辆是否带来行人"，第一感是要做供给-需求分析，类似 walkability 和 mobility 的问题。在我们自己的 connectionist 框架里，这个问题变成：其他条件不变时，车辆通道与行人通道的变化是什么关系；担心的是几千个 tile 里找不到足够多其他条件类似的 tile 来隔离两者关系，样本太少。

文章把曼哈顿划成 5 679 个 100 m 格，用 XGBoost + SHAP 归因：建成环境 77.5%、车辆 15.2%、臭氧 1.5%；因子作用普遍是阈值式的（臭氧负向到 27.5 ppb 后变平、POI 混合熵过 0.4 陡增、路口数过 5 饱和、医疗服务倒 U 峰 0.9）。对我们最有用的反而是它自己没讨论的：购物服务 OLS 系数 −1.39、商业面积 −1.47，非线性模型下都为正——同一批数据，线性给出显著负号，非线性给出正号，这是“线性 baseline 有偏”的直接证据。另一条发现是，目的地可达性主导活力，这是常见的经验发现。但可达性分很多种，活力对不同人也不同；经验之外的反直觉更有意思：高可达却低在场（大枢纽站，人不停留），低可达却高在场（HCMC 的 hẻm，正式可达性低、巷内却热闹）。这些反例并不推翻"可达性主导"，而是说明操作化出来的可达性，和人实际使用的可达性本来就不重合。密度和多样性同样：即使看到联合分布，内部还有 typology 要识别——多样性是文化、商业还是餐饮？密度是建筑密度、人口密度还是夜光强度？通常只有问到足够具体的问题才细化 typology：Sarah Williams 回应 ghost city 时就把供给和需求拆开（设施可达为供给、点评数为需求权重）来识别空置。所以我对我们方法论的期望是把因子的 typology 拆得足够细；挑战是样本太少，或因子多到模型能学但 descriptor 难解析。可以用反事实验证，代价是生成的反事实可信度本身要检验。

---

Dogan, O., & Lee, S. (2024). Jane Jacobs's urban vitality focusing on three-facet criteria and its confluence with urban physical complexity. *Cities*, 155, 105446. https://doi.org/10.1016/j.cities.2024.105446

作者按 Jacobs「不同的人在不同的时间做不同的事」，把首尔 1 881 个 500 m 格子的活力拆成三个维度：强度、多样性、一致性。再用 22 个建筑与路网的形态变量，对每个维度各做一次线性回归。三个维度里取自同一份移动数据的部分，只是同一批出行记录的不同字段：条数、年龄与来源地的熵、时段的熵。这本身没有问题。问题在于合成。强度把人数、刷卡、POI 数、点评和 Airbnb 用因子载荷加权相加，其中载荷最大的是 POI（0.63）。多样性把来访者年龄的熵、来源地的熵和格子里 POI 种类的多样性加在一起，最后我都不知道这个多样性到底是什么。这篇对我最大的启发在尺度。它的街段和街区没有裁到格子里（平均街段长最大 3 986 m，而格子对角线约 707 m）；交通小区的数据怎么分到格子里也没交代，小区比格子大时，摊开的数据内部其实没有空间结构。强度的 Moran's I 本身可能被抬高，作者再由此推出绅士化（全文没有测过绅士化），就显得很弱了。但多种测量尺度不等于跨尺度分析。跨尺度不是把同一份数据缩放到几个尺度，而是每个尺度换用适合它的数据，比如你想验证雅各布斯的密度在不同尺度下是否可以提升活力，那密度应该在不同尺度有对应的表征：微观用 POI 或人流，中观用建筑密度，宏观用路网或不透水面。

---

Fan, Y., Kuang, D., Tu, W., & Ye, Y. (2023). Which spatial elements influence waterfront space vitality the most?—A comparative tracking study of the Maozhou River renewal project in Shenzhen, China. *Land*, 12(6), 1260. https://doi.org/10.3390/land12061260

作者看的是一条河改造前后的变化：茅洲河 61 个岸线单元（离水 10–50 m 的带），取 2018 年 10 月（改造前）和 2020 年 10 月（改造后）两期，把 11 个空间要素和 15 个活力指标都换成增长率，每个活力指标各跑一个线性回归。活力分两类：社会活力来自手机信令里的停留人数（居住、工作、访问三类人）和访客停留时长；经济活力来自周边 POI 个数和种类的熵。主要发现：步道密度和文教设施密度的增加，对停留人数和停留时长都是最大的正作用；地铁站覆盖是负的或无效（郊区站点离河太远）。最有意思的是一个反号：同样这些要素，对「经济活力强度」（周边企业、商业 POI 数）大多是**负**的。作者的解释是功能替代：工业码头、仓库被换成展馆和公园，企业和商业减少、文教设施增加。用 CLD 看，这是一条改造挤出原有经济功能的平衡关系，只在叙事里，没有估计。这篇是我们读过的唯一有前后两期的研究，增长率对增长率相当于拿每个单元自己的过去当对照，但没有对照河段，又叠加了疫情。

---

Yue, W., Chen, Y., Thy, P. T. M., Fan, P., Liu, Y., & Zhang, W. (2021). Identifying urban vitality in metropolitan areas of developing countries from a comparative perspective: Ho Chi Minh City versus Shanghai. *Sustainable Cities and Society*, 65, 102609. https://doi.org/10.1016/j.scs.2020.102609

Yue 等（2021）把活力定义为"建成环境促进社会活动的能力"，明说不直接测社会活动，而用三个维度九个供给条件代替：建成环境（12 类土地利用熵、路口数、老建筑数）、人的活动（人口密度、POI 密度、居住用地占比）、人–环境联系（道路长度密度、高速 / 铁路 / 河流缓冲占比、公交站与广场公园密度），在胡志明市 317 个、上海 229 个 sub-district（坊 / 社、街道 / 镇）上用 projection pursuit 合成一个分。全文没有回归，Table 1 里各指标对活力的方向（8 正 1 负）是先验，不是实测。结果：两城活力都从核心向外递减，HCMC 单中心、上海多中心；高分区与前租界的欧式小街区重合；被高速、国道切开的郊区得分低。真正有信息量的是两城的差别：同一套指标和方法，一城画出单中心、一城画出多中心，这不是方法预先规定的。但 §5.2 是循环论证：作者说"small blocks help increase urban vitality"（Fig. 6），而路口数本身就是活力分的一项，方向还预设为正。讨论里有三条因果叙事——多中心战略 → 副中心用地混合 → 活力；小街区、密路口 → 步行与接触 → 活力；大型交通基建 → 物理隔离 → 郊区活力下降——都把规划政策放在建成环境的上游，但没有量化。这是少有的 HCMC 本地研究：用地图来自越南科学技术院，POI 来自 OSM 2016，老建筑是保护用地图斑转成的点，数的是图斑不是楼。因为这篇只问格局、不分析影响因素，指标到活力的边全是作者定义的；这让我意识到，提取 CLD 时要区分作者定义的、作者分析得出的、作者分析后推理或推测的边（作者演绎做得认真的称推理，只是顺着结果猜的称推测）。
