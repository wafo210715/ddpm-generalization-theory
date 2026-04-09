Probing the Latent City: Unveiling the Low-Dimensional Manifolds of Urban Vitality via Generative AI

Abstract: Despite the influx of big data, urban vitality research remains constrained by a "correlationist paradigm," reducing complex urban synergies to linear, additive feature lists. This proposal advances a generative epistemology, repositioning Diffusion Models (DMs) from tools of automated design to epistemic probes for manifold learning. We argue that diffusion models function as structure-preserving compressors, capable of internalizing the low-dimensional manifold of urban life. Central to this inquiry is the generalization window - a training phase where structural concepts are learned before memorization occurs. By curating minimal, intrinsic-dimension-aligned datasets via RGB channel synthesis, this research aims to isolate the robust, irreducible "essence" of vitality from idiosyncratic noise. Moving beyond prediction, this study reframes the latent space as a computational laboratory, transforming the architect into a curator to uncover the hidden generative grammar of the city.

Keywords: Generative AI, Urban Vitality, Latent Space, Generalization Window, Intrinsic Dimension, Typology

1.  Introduction

The proliferation of multi-source urban big data has fundamentally altered the landscape of urban vitality research. What was once a qualitative inquiry into the social life of small urban spaces has evolved into a rigorous science of quantification, where location-based services, social media data, and built-environment metrics are routinely synthesized to map the pulse of the city (Kim, 2020; X. Li et al., 2022; Liu et al., 2019). Yet, despite this empirical expansion, the field remains theoretically constrained by a "correlationist" paradigm. Dominant methodologies ranging from spatial regression to random forest algorithms, prioritize the identification of statistical associations over the explication of generative mechanisms (Jiang et al., 2024; C. Wu et al., 2023; Zeng et al., 2018). Within this framework, urban vitality is treated as a dependent variable to be predicted by independent factors, effectively flattening the complex, emergent synergies of the city into linear, additive lists of drivers (X. Li et al., 2022; Pan et al., 2021). As recent systematic reviews indicate, this approach frequently suffers from "conceptual reductionism," where the lived experience of the city is severed from its context and reduced to isolated digital signals (Doan & Zhang, 2025).

This proposal argues that the true significance of contemporary generative AI, specifically Denoising Diffusion Probabilistic Models (DDPMs), lies not in its ability to automate urban design or visualize "future cities" (He et al., 2026; Q. Wang et al., 2025; Zhuang et al., 2024), but in its potential to function as an epistemic probe into this complexity. Unlike the correlationist approach, which strips context to isolate variables, diffusion models operate as "structure-preserving compressors" (P. Wang et al., 2025). By learning to reconstruct urban data from noise, these models do not merely memorize pixels; they internalize the low-dimensional manifold of the city---the hidden, intrinsic constraints and joint probability distributions that govern how physical form and social activity coalesce into vibrant environments (P. Wang et al., 2025).

From this perspective, the training of a diffusion model is not a technical task of image synthesis, but a methodological experiment in manifold learning. If, as recent computer science literature suggests, diffusion models exhibit a distinct "generalization window" where structural concepts are learned before noisy details (Bonnaire et al., 2025), then the training process itself offers a window into the "essence" of urban vitality. By observing what the model prioritizes and preserves, we can distinguish between the fundamental, irreducible drivers of vitality and the incidental noise of specific locales.

However, operationalizing this shift requires moving beyond the "black box" application of AI. It demands a rigorous framework that treats the generative model as a hypothesis-testing instrument. To address this, this research proposes a generative epistemology centered on the role of the architect-as-curator and the diffusion model as an analytical lens. The study is guided by three primary research questions:

(1) Can diffusion models learn the low-dimensional manifold of urban vitality from heterogeneous data sources?

(2) What essential features of vitality are captured in the latent space, and how do they differ from features identified through correlation-based methods?

(3) How does the systematic manipulation of training data composition alter the model's learned manifold, and what does this reveal about the heterogeneity of vitality across different urban typologies?

This research does not seek to propose a new tool for automated planning. Instead, it offers a theoretical reframing: interpreting the latent space of the diffusion model as a computational laboratory. By probing this space, we aim to move beyond predicting where vitality occurs, to uncovering the generative grammar of how it emerges.

2.  Literature Review

    1.  The Limits of the "Correlationist" Paradigm in Urban Vitality

The methodological revolution in urban vitality research, driven by the proliferation of multi-source big data, has enabled unprecedented quantification of urban phenomena but has simultaneously reinforced a "correlationist" paradigm. This epistemology prioritizes the identification of statistical associations over the understanding of generative mechanisms. Contemporary research operationalizes vitality as an observable outcome proxied through digital traces: Location-based services(LBS) and mobile signaling data capture human mobility (Kim, 2020; X. Li et al., 2022; Liu et al., 2019), social media check-ins and reviews serve as indicators of economic and cultural engagement (H. Liu et al., 2022; Qin et al., 2025; Scepanovic et al., 2021), and Point of Interests (POIs) density metrics quantify functional intensity (Jiang et al., 2024; C. Wu et al., 2023). While this quantification has allowed for analyses ranging from street-level grids to city-wide assessments (Pan et al., 2021; Xia et al., 2020), it often results in "conceptual reductionism", where complex social interactions and experiential dimensions are flattened into digita signals (Doan & Zhang, 2025).

Within this paradigm, researchers employ a standard toolkit, ranging from OLS and spatial regression models (Jiang et al., 2024; X. Li et al., 2022) to Geographically Weighted Regression (GWR) (Jiang et al., 2024; X. Li et al., 2022) and machine learning algorithms like Random Forest (C. Wu et al., 2023), to catalogue positive and negative drivers of vitality (C. Wu et al., 2023; Xia et al., 2020; Zeng et al., 2018). However, this approach encounters an epistemic wall defined by three fundamental limitations. First, it relies on an additivity assumption, treating urban factors as independent variables that contribute linearly to vitality, ignoring the synergistic nature of urban system (X. Li et al., 2022). Second, it suffers from context stripping, where spatial units are treated as isolated data points, severing the functional and cultural interdependencies that shape lived experience (Doan & Zhang, 2025). Finally, these methods fail to capture emergence, operating at single analytical scales without modelling how micro-scale interactions aggregate into macro-scale vitality states (Pan et al., 2021). Ultimately, the correlationist paradigm generate actionable feature lists but offers limited insight into the joint probability distributions and generative processes that produce vibrant ubran environments.

2.  Diffusion Models: From Image Synthesis to Manifold Learning

    The emergence of DDPMs offers a theoretical pathway beyond correlation, shifting focus from deterministic mapping to stochastic manifold learning. Unlike traditional generative models, DDPMs operate through probabilistic inference, reversing a diffusion process to learn the gradient of the data's log probability density (P. Wang et al., 2025). This mechanism is underpinned by the Manifold Hypothesis, which posits that high-dimensional urban data resides on low-dimensional manifolds governed by intrinsic, albeit often unobservable, constraints. In classical learning theory, accurately capturing this underlying distribution is the prerequisite for generalization (Vapnik, 2000). Diffusion models succeed not by memorizing pixel-level correlations, but by learning the geometry of this manifold. As recent empirical studies demonstrate, models trained on non-overlapping subsets can reproduce consistent outputs (Zhang et al., 2024), suggesting they are capturing robust generative laws rather than merely replicating training instances (Kadkhodaie et al., 2024).

    Two theoretical properties make diffusion models particularly potent for urban analysis. First, they fundamentally address the \"curse of dimensionality.\" Wang et al. demonstrate that optimizing diffusion training is mathematically equivalent to subspace clustering (P. Wang et al., 2025). This aligns with findings that score-based models can learn distributions with sample complexities scaling linearly with the data's intrinsic dimension (Cole & Lu, 2024). This suggests that even with minimal datasets that meet the intrinsic dimensional threshold, the model can organize data into meaningful regimes without human supervision.

    Second, and most critical for our hypothesis, is the phenomenon of implicit dynamical regularization. While large models are capable of a \"memorization regime\" - learning the empirical score to perfectly reproduce training data, they do not start there (Biroli et al., 2024; S. Li et al., 2024). Instead, Bonnaire et al. identify a distinct "generalization window" (Bonnaire et al., 2025). This behavior is driven by \"spectral bias,\" where neural networks learn low-frequency, structural patterns significantly faster than high-frequency noise (Rahaman et al., 2019; B. Wang & Pehlevan, 2025).

    This provides a crucial methodological opening for urban vitality research. Because the internal logic of the latent space remains a \"black box,\" we cannot directly observe how the model encodes urban rules. However, we can probe the model's behavior through its inputs and outputs. By analyzing the stability of generated forms throughout the generalization window, we can infer the \"essence\" of vitality: features that remain constant during this window represent the learned structural signal, while those that appear only later represent memorized noise (Y.-H. Wu et al., 2025).

    Despite this potential, current applications in urban planning largely treat diffusion models as "black box generators." Recent "Text-to-City" frameworks utilize these models primarily for automated design synthesis and visualization (He et al., 2026; Q. Wang et al., 2025; Zhuang et al., 2024). These studies often focus on instrumental metrics of visual fidelity, inadvertently pushing models into the memorization regime where they risk becoming engines of \"digital forgery\" (Carlini et al., 2023; Somepalli et al., 2022). By treating the model solely as a drafting tool, the field neglects its internal clustering and generalization behaviors (George et al., 2025). Consequently, they miss the opportunity to use the latent space as a computational laboratory for simulating urban dynamics and identifying the hidden structures of the city.

3.  Bridging the Gap: A Generative Episemology for Urban Vitality

    To transcend the limitations of the correlationist paradigm, this research reframes the diffusion model as a "structure-preserving compressor" of urban complexity. In this framework, the forward diffusion process systematically abstracts the high-dimensional "urban surface" - the noisy amalgam of built form and activity data - while the reverse process learns to reconstruct the underlying data-generating mechanism. By learning the joint probability distribution of urban elements, the model inherently addresses the failures of additivity and context stripping, preserving the relational structures that define how a city functions (Q. Li et al., 2021; P. Wang et al., 2025). However, operationalizing this structure-preserving compressor presents a critical representational challenge: converting multidimensional vitality data into the 2D tensor format required by vision-based diffusion architectures without reverting to reductionist metrics. If the methodology relies on pre-calculating a composite vitality score to generate a single-channel gradient map, it effectively bakes the "correlationist" assumptions into the input data, defeating the purpose of unsupervised manifold learning. To circumvent this, we propose utilizing RGB channel synthesis (X. Liu et al., 2024), a technique that maps distinct vitality proxies (e.g., physical density, social media activity, functional mix) into separate color channels. This approach preserves the raw, unweighted relationships between variables, forcing the model to learn the mixing logic in the latent space rather than imposing it through pre-processing.

    Furthermore, the deployment of conditional diffusion models (e.g., Stable Diffusion or ControlNet) introduces a second challenge regarding the curation of textual prompts. As Huang et al. argue, the coupling of Natural Language Processing (NLP) and computational generative design creates a vital new interface between human intent and machine logic (J. Huang et al., 2021). Mere keyword tagging is insufficient for capturing the complex "semantic design expression" of urban environments. Consequently, the curation of text prompts must be treated as a rigorous methodological step.

    This generative epistemology transforms our methodological approach into a hypothesis-driven probing experiment centered on two core postulates. First, we posit that the diffusion model's generalization window, the training phase where structural patterns are learned before memorization sets in, functions as an epistemic filter for urban vitality. We hypothesize that whatever morphological or functional characteristics remain consistent during this window constitute the essence of vitality, representing the robust generative laws of urban life as opposed to the incidental noise of specific locales. Second, we leverage the finding that diffusion models require a minimal, intrinsic-dimension-aligned dataset to generalize (P. Wang et al., 2025), hypothesizing that by systematically manipulating the composition of the training dataset, we can perform a heterogeneous analysis of vitality drivers. This reframes the analytical process around data selection rather than algorithmic tweaking, aligning with Huang's assertion that the architect's role has shifted from originator of form to curator of datasets, navigator of latent spaces, and interpreter of model outputs, where epistemic agency is exercised through the inclusion, exclusion, and weighting of urban data (S.-Y. Huang, 2025).

<!-- -->

3.  Methods

4.  Experiments

5.  Conclusions

**References**

Biroli, G., Bonnaire, T., Bortoli, V. de, & Mézard, M. (2024). Dynamical Regimes of Diffusion Models. *Nature Communications*, *15*(1), 9957. https://doi.org/10.1038/s41467-024-54281-3

Bonnaire, T., Urfin, R., Biroli, G., & Mézard, M. (2025). *Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training* (No. arXiv:2505.17638). arXiv. https://doi.org/10.48550/arXiv.2505.17638

Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramèr, F., Balle, B., Ippolito, D., & Wallace, E. (2023). *Extracting Training Data from Diffusion Models* (No. arXiv:2301.13188). arXiv. https://doi.org/10.48550/arXiv.2301.13188

Cole, F., & Lu, Y. (2024). *Score-based generative models break the curse of dimensionality in learning a family of sub-Gaussian probability distributions* (No. arXiv:2402.08082). arXiv. https://doi.org/10.48550/arXiv.2402.08082

Doan, Q. C., & Zhang, X. (2025). A systematic review of urban vitality studies: Trends and research opportunities. *Land Use Policy*, *158*, 107745. https://doi.org/10.1016/j.landusepol.2025.107745

George, A. J., Veiga, R., & Macris, N. (2025). *Denoising Score Matching with Random Features: Insights on Diffusion Models from Precise Learning Curves* (No. arXiv:2502.00336). arXiv. https://doi.org/10.48550/arXiv.2502.00336

He, M., Liang, Y., Wang, S., Zheng, Y., Wang, Q., Zhuang, D., Tian, L., & Zhao, J. (2026). Human-guided urban form generation using multimodal diffusion models. *Building and Environment*, *287*, 113892. https://doi.org/10.1016/j.buildenv.2025.113892

Huang, J., Johanes, M., Kim, F. C., Doumpioti, C., & Holz, G.-C. (2021). On GANs, NLP and Architecture: Combining Human and Machine Intelligences for the Generation and Evaluation of Meaningful Designs. *Technology\|Architecture + Design*, *5*(2), 207--224. https://doi.org/10.1080/24751448.2021.1967060

Huang, S.-Y. (2025). The Connectionist Turn: How Contemporary Generative AI Reshapes Architectural Rationality. *Architecture*, *5*(4), 132. https://doi.org/10.3390/architecture5040132

Jiang, Y., Huang, Z., Zhou, X., & Chen, X. (2024). Evaluating the impact of urban morphology on urban vitality: An exploratory study using big geo-data. *International Journal of Digital Earth*, *17*(1), 2327571. https://doi.org/10.1080/17538947.2024.2327571

Kadkhodaie, Z., Guth, F., Simoncelli, E. P., & Mallat, S. (2024). *Generalization in diffusion models arises from geometry-adaptive harmonic representations* (No. arXiv:2310.02557). arXiv. https://doi.org/10.48550/arXiv.2310.02557

Kim, S. (2020). Urban Vitality, Urban Form, and Land Use: Their Relations within a Geographical Boundary for Walkers. *Sustainability*, *12*(24), 10633. https://doi.org/10.3390/su122410633

Li, Q., Cui, C., Liu, F., Wu, Q., Run, Y., & Han, Z. (2021). Multidimensional Urban Vitality on Streets: Spatial Patterns and Influence Factor Identification Using Multisource Urban Data. *ISPRS International Journal of Geo-Information*, *11*(1), 2. https://doi.org/10.3390/ijgi11010002

Li, S., Chen, S., & Li, Q. (2024). *A Good Score Does not Lead to A Good Generative Model* (No. arXiv:2401.04856). arXiv. https://doi.org/10.48550/arXiv.2401.04856

Li, X., Li, Y., Jia, T., Zhou, L., & Hijazi, I. H. (2022). The six dimensions of built environment on urban vitality: Fusion evidence from multi-source data. *Cities*, *121*, 103482. https://doi.org/10.1016/j.cities.2021.103482

Liu, H., Gou, P., & Xiong, J. (2022). Vital triangle: A new concept to evaluate urban vitality. *Computers, Environment and Urban Systems*, *98*, 101886. https://doi.org/10.1016/j.compenvurbsys.2022.101886

Liu, X., Yang, N., & Tian, R. (2024). *Reinventing Planting Design in Landscape Architecture: A Generative AI Approach*. Wichmann Verlag. https://doi.org/10.14627/537752020

Liu, Zhang, & Long. (2019). Urban Vitality Area Identification and Pattern Analysis from the Perspective of Time and Space Fusion. *Sustainability*, *11*(15), 4032. https://doi.org/10.3390/su11154032

Pan, H., Yang, C., Quan, L., & Liao, L. (2021). A New Insight into Understanding Urban Vitality: A Case Study in the Chengdu-Chongqing Area Twin-City Economic Circle, China. *Sustainability*, *13*(18), 10068. https://doi.org/10.3390/su131810068

Qin, J., Wang, Z., Sheng, Y., Xue, L., Cai, X., & Zhang, K. (2025). Relationship between the built environment and urban vitality of Nanjing's central urban area based on multi-source data. *Environment and Planning B: Urban Analytics and City Science*, *52*(1), 168--185. https://doi.org/10.1177/23998083241256246

Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F. A., Bengio, Y., & Courville, A. (2019). *On the Spectral Bias of Neural Networks* (No. arXiv:1806.08734). arXiv. https://doi.org/10.48550/arXiv.1806.08734

Scepanovic, S., Joglekar, S., Law, S., & Quercia, D. (2021). Jane Jacobs in the Sky: Predicting Urban Vitality with Open Satellite Data. *Proceedings of the ACM on Human-Computer Interaction*, *5*(CSCW1), 1--25. https://doi.org/10.1145/3449257

Somepalli, G., Singla, V., Goldblum, M., Geiping, J., & Goldstein, T. (2022). *Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models* (No. arXiv:2212.03860). arXiv. https://doi.org/10.48550/arXiv.2212.03860

Vapnik, V. N. (2000). *The Nature of Statistical Learning Theory*. Springer New York. https://doi.org/10.1007/978-1-4757-3264-1

Wang, B., & Pehlevan, C. (2025). *An Analytical Theory of Spectral Bias in the Learning Dynamics of Diffusion Models* (No. arXiv:2503.03206). arXiv. https://doi.org/10.48550/arXiv.2503.03206

Wang, P., Zhang, H., Zhang, Z., Chen, S., Ma, Y., & Qu, Q. (2025). *Diffusion Models Learn Low-Dimensional Distributions via Subspace Clustering* (No. arXiv:2409.02426). arXiv. https://doi.org/10.48550/arXiv.2409.02426

Wang, Q., Liang, Y., Zheng, Y., Xu, K., Zhao, J., & Wang, S. (2025). Generative AI for urban planning: Synthesizing satellite imagery via diffusion models. *Computers, Environment and Urban Systems*, *122*, 102339. https://doi.org/10.1016/j.compenvurbsys.2025.102339

Wu, C., Ye, Y., Gao, F., & Ye, X. (2023). Using street view images to examine the association between human perceptions of locale and urban vitality in Shenzhen, China. *Sustainable Cities and Society*, *88*, 104291. https://doi.org/10.1016/j.scs.2022.104291

Wu, Y.-H., Marion, P., Biau, G., & Boyer, C. (2025). *Taking a Big Step: Large Learning Rates in Denoising Score Matching Prevent Memorization* (No. arXiv:2502.03435). arXiv. https://doi.org/10.48550/arXiv.2502.03435

Xia, C., Yeh, A. G.-O., & Zhang, A. (2020). Analyzing spatial relationships between urban land use intensity and urban vitality at street block level: A case study of five Chinese megacities. *Landscape and Urban Planning*, *193*, 103669. https://doi.org/10.1016/j.landurbplan.2019.103669

Zeng, C., Song, Y., He, Q., & Shen, F. (2018). Spatially explicit assessment on urban vitality: Case studies in Chicago and Wuhan. *Sustainable Cities and Society*, *40*, 296--306. https://doi.org/10.1016/j.scs.2018.04.021

Zhang, H., Zhou, J., Lu, Y., Guo, M., Wang, P., Shen, L., & Qu, Q. (2024). *The Emergence of Reproducibility and Generalizability in Diffusion Models* (No. arXiv:2310.05264). arXiv. https://doi.org/10.48550/arXiv.2310.05264

Zhuang, J., Li, G., Xu, H., Xu, J., & Tian, R. (2024). *Text-to-City: Controllable 3D Urban Block Generation With Latent Diffusion Model*. 169--178. https://doi.org/10.52842/conf.caadria.2024.2.169

  ----------------------------------- -----------------------------------

  ----------------------------------- -----------------------------------
