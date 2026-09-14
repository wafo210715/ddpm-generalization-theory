# Choosing a second study area, and why the study needs one

Assembled 14 Sep 2026, in response to a problem raised about the indicator
review: the review records *which* factors each paper used and *what effect*
each reported, but never *what data* each paper built its factors from. This
page works out what follows from that, and which city fixes it.

## The problem is worse than the missing column

The missing-source column is real. Take the flagship case. Tan 2026's
floor-area-ratio threshold at 0.25 is the strongest non-linear finding in the
review. A ratio computed from Shenzhen floor-area records is a different
quantity from ours, which is a modelled height divided by an assumed storey
height over machine-derived footprints, and which we have already measured as
biased low in high-rises. The number 0.25 does not transfer. The same ambiguity
sits under building density, mix, and accessibility.

**But that is the second-order problem.** The first-order one is this:

The method this paper argues against regresses a vitality proxy against
built-environment predictors. To run that control ourselves we need a vitality
dependent variable. **Ho Chi Minh City has none** — mobile signalling, platform
heat maps and flow data are all absent, which is exactly the no-source block in
`hcmc_data_acquisition.md`. The plan's C1 step is defined as correlations among
indicators on the same data, which flags duplicate channels but is not the
driving-factor analysis being critiqued.

So the control condition cannot be run here at all. A second city is not a
robustness check; it is the only place the head-to-head can happen.

### What that changes about the choice

The city should be picked for **whether a public human-activity dataset exists
at usable resolution**, not for morphology and not for which papers studied it.
Everything on the supply side is already solved and global — Overture,
3D-GloBFP, the GHSL epochs, WorldCover, Sentinel-2, Copernicus DEM, WorldPop —
so the marginal cost is a boundary file, a dependent variable, and a training
run. 3D-GloBFP is distributed as world-grid tiles; its own documentation names a
France-and-Spain tile as the example.

### One framing decision to settle first

Every candidate dependent variable measures a *mode*, not a population: transit
riders, taxi passengers, cyclists, pedestrians past a sensor, workers at a desk.
Adopting one means picking an available proxy and calling it activity, which is
close to what §2.1 criticises the literature for doing.

The resolution is to be explicit about what the second city is for. **We are not
measuring vitality there.** We are running the control: what a correlationist
regression finds on a dataset, against what the diffusion model finds on that
same dataset. The dependent variable only has to be a defensible, documented,
reproducible measure of human activity, and the paper must say it is regressing
that measure and never call it vitality. That is a stricter standard than most
of the reviewed papers hold themselves to, and it turns the weakness into a
contribution.

A related correction: the regression does **not** have to run on our tile grid.
The reviewed papers run at whatever unit their dependent variable comes in, with
predictors computed in a buffer. What must be held constant between the two
methods is the data source and the factor definitions, not the spatial unit.

## The candidates

| City | Dependent variable | Public | Recent | Geometry | Measures | Precedent |
|---|---|---|---|---|---|---|
| **Barcelona** | Footfall counters; national trips by district | CC-BY 4.0 | **2017–2025** | ~115 sensors; districts complete | **Pedestrians** | **Yes, direct** |
| **Milan** | Telecom activity grid | ODbL | **2013–14 only** | **10 000 cells at 235 m, complete** | **Presence** | Related |
| New York | Subway hourly; taxi; bike; jobs | Open | To within weeks | 428 points; 37 588 blocks | Modes | Yes |
| London | Station counts; 15-min profiles; bike | Open | 2025–26 | 470 stations; 800 docks | Modes | None found |
| Madrid | Bike, bus, traffic sensors | Open | Mixed | Line or city level | Modes/vehicles | — |

### What each one really offers

**Barcelona** is the only candidate that is public, recent, pedestrian-based and
already has a published study doing exactly our control design: Rames et al. in
*Cities* (2024/25) regress footfall from the city's counter network against
traffic calming, transit access, shop density and green proximity. A second
documented precedent, Graells-Garrido et al. in *PLoS ONE* (2021), runs a gravity
model and geographically weighted regression over 73 neighbourhoods with the
method fully described and the full text open. The national transport ministry
also publishes district-level daily trips through 2025 with maintained R and
Python packages, giving a second, complete-coverage dependent variable as a
robustness check. Barcelona is also in our own review, via Gómez-Varo 2022.

**Milan** has the best-shaped dataset anyone found: an openly licensed telecom
activity grid, 10 000 cells at 235 m — finer than our tiles, so it aggregates up
cleanly — measuring connection activity rather than one travel mode, documented
in a *Scientific Data* descriptor. Its flaw is decisive on its own terms: it
covers November 2013 to January 2014. That mismatch would be fatal in HCMC. In a
mature European city it is arguable, and Urban Atlas publishes a 2012 vintage if
we want the eras to match.

**New York** has the most current and best-engineered data, and three published
regressions against built-environment predictors. It loses on the land-use
argument, because Urban Atlas is Europe-only.

**London** has the best data engineering of all — station counts, fifteen-minute
demand profiles and journey-level bike hire on scriptable endpoints — but no
study framing it as vitality, and the one true footfall grid, 350 m hexagons
updated weekly, is restricted to borough and business-district officers.

**Madrid** fails the bar. Its one fine-grained series was discontinued in 2023
and replaced by city-wide daily totals; bus data is line-level; the dense sensor
network counts vehicles.

## The land-use factor, which a European city recovers

Land-use mix is the one §4 factor with **no** HCMC equivalent, and ten of the
reviewed studies use it. Copernicus Urban Atlas covers all the European
candidates at its 2021 vintage, free under CC-BY 4.0, with 19 urban classes at a
0.25 ha minimum mapping unit.

State its limits precisely rather than overselling it. Urban Atlas separates
residential fabric by **density**, not by activity, and bundles industrial,
commercial, public and military into a single class. So it supports roughly five
meaningful mix categories — residential fabric, mixed economic and institutional,
transport, green and leisure, and vacant or under construction. That is a real
gain over land cover, which distinguishes none of them, and still coarser than
the four-way split the phrase "land-use mix" implies.

## Recommendation

**Barcelona**, with Milan held as a methodological alternative.

Barcelona is the only city that clears every bar at once: a public,
currently-maintained, pedestrian-based dependent variable; a second independent
one at district level for robustness; two published precedents using documented
and reproducible methods, one of them open access; Urban Atlas for the land-use
factor; and presence in our own review. Its morphology, a planned nineteenth
century grid, also contrasts about as sharply with HCMC's organic alley fabric
as any European city could, which is what makes a generalization claim
interesting rather than trivial.

Take Milan instead if we decide the dependent variable's geometry matters more
than its date — its complete 235 m grid is the only one that would let the
regression and the diffusion model share a spatial unit exactly.

### Known costs and frictions

- **Scope.** The report bounds itself to one city and defers multi-city work.
  This needs supervisor agreement before it goes in; the roadmap's Step 2 gate is
  the place for it.
- **A second training run**, plus the analysis code run twice.
- **Barcelona's portal challenges automated downloads** of raw files, though not
  its metadata API. Confirmed directly: the catalogue API answers normally, and
  a raw CSV request returns a bot-detection page instead of the file. A browser
  session is needed. Not a blocker, just manual.
- **The pedestrian claim is not yet verified, and there is contrary evidence.**
  The counter datasets are real, CC-BY 4.0, and current — `aforaments-descriptiu`
  and `aforaments-detall` both carry nine annual files through 2025. But the
  family is named *aforament*, gauging, and its sibling dataset
  `aforament-tipus-equipament` describes itself explicitly as the technology for
  reading **vehicle** mobility. Whether the network counts people or vehicles
  could not be settled from here, because the file that would say so is behind
  the bot check. **Settle this before committing**: if these are vehicle
  counters, Barcelona's headline advantage over Milan disappears, and the
  published pedestrian precedent must be drawing on a source other than this
  portal series.
- **115 sensors is a small regression sample.** The district-level trip data is
  the answer to that, not a reason to drop the footfall series.
- **The calibration idea does not survive contact.** I had hoped to reproduce a
  published study's coefficients as an external check on our factor
  construction. The cleanest candidate, Šćepanović et al. 2021, is built on 2014
  telecom data that was never released, and its project page offers only a
  visualization. Barcelona's precedents are reproducible in *method* but their
  sensor-level data still has to be pulled ourselves.

## Update, 14 Sep: the missing-source column has been filled

This section previously said the per-factor source gap was still open and needed
whoever has the PDFs. That work has since landed in `vitality_provenance.md`:
all 32 papers, 153 factor rows carrying source, computation and an evidence line
reference, plus a cross-paper divergence matrix.

**It does not merely close the gap — it strengthens the case for a second city.**
The problem turns out not to be that sources are undocumented, but that where
they *are* documented the constructs diverge past reconciliation:

- **POI entropy has at least six computations.** One of them, `exp(Shannon)`,
  runs in the opposite direction, so a positive coefficient there means the
  opposite of a positive coefficient elsewhere.
- **Intersection density splits on the denominator** in all ten papers that use
  it — census district, 1 km grid, 50 m grid, TAZ, block, net community area —
  and one counts only four-way junctions.
- **No two of the 32 papers share a dependent-variable source.**
- Individual traps: Li 2022 substitutes residential share for Shannon entropy,
  so its sign reads backwards; Qin 2025 names FSI "building density" and GSI
  "volume rate", inverting the convention.
- **27 factor rows record that the paper never states its source at all.**

So the effect-direction conflicts in the review's §4 — intersection density at
+0.497 in one study and a non-significant negative in another — are not
substantive disagreements to be adjudicated. They are unit mismatches. That
removes comparison-by-citation as an option permanently, rather than pending
better documentation, which is precisely why the control has to be one we run
ourselves.
