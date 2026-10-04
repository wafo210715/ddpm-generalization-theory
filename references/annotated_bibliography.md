# Annotated bibliography

Each entry is a citation and one paragraph of at most 300 words (was 200 until 2 Oct 2026). Line references
(L#) point into the paper's text layer in `references/papers/`.

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
