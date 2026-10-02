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
