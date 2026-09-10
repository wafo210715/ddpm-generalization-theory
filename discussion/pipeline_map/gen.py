#!/usr/bin/env python3
"""Single source for the v4 pipeline map and its companion explainer.

usage: gen.py ARTIFACT_DIR [EXPLAINER_URL]
Writes content-only pages into ARTIFACT_DIR (for the Artifact host) and
standalone pages + the draw.io file into ../ (the repo's discussion folder).
"""
import html, math, os, re, sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_OUT = os.path.normpath(os.path.join(HERE, ".."))
ART_DIR = sys.argv[1] if len(sys.argv) > 1 else HERE
EXPLAINER_URL = sys.argv[2] if len(sys.argv) > 2 else "cells_channels_units.html"
MAP_URL = "https://claude.ai/code/artifact/46636ca5-e20b-432c-9bff-f174767a08e6"
CSS = open(os.path.join(HERE, "theme.css"), encoding="utf-8").read()
FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@500;600&display=swap">'

KIND_LABEL = {"action": "action", "gate": "checkpoint", "future": "later"}

STAGES = [
  {"code": "A", "title": "Before any code", "ordinal": "your first and second", "rows": [["A1"], ["A2", "A3"]]},
  {"code": "B", "title": "Build the dataset", "ordinal": "your third", "rows": [["B1"], ["B2"], ["B3", "B4", "B5"], ["B6"], ["B7"], ["B8", "B9"], ["B10"], ["B10b"], ["B11"]]},
  {"code": "C", "title": "Pre-training analysis", "ordinal": "your fourth", "rows": [["C1", "C2"], ["C3"], ["C4", "C4b"], ["C5"], ["C6"], ["C7"]]},
  {"code": "D", "title": "Train", "ordinal": "your fifth", "rows": [["D1", "D2"], ["D3"]]},
  {"code": "E", "title": "Sample and analyze", "ordinal": "your fifth and sixth", "rows": [["E1"], ["E2", "E3", "E3b", "E3c"], ["E4"], ["E5"]]},
  {"code": "F", "title": "Paper", "ordinal": "your seventh, as future work", "rows": [["F1", "F2"]]},
]

N = {
 "A1": {"kind":"action","title":"Write the indicator specification table","why":[
   "A footprint is a data source. Binary raster, density, and floor-area ratio are three different indicators of it.",
   "Each row fixes subject, source, formula, measurement scale, ring scale, and expected sparsity.",
   "The row count sets the channel count, so this comes before any loader change."]},
 "A2": {"kind":"action","title":"Search for human-centric layers for HCMC","why":[
   "Your seventh stage: joint form and activity, the ISM proposal's original wording.",
   "VIIRS night lights is the one layer known to be available, but at 500 m it is nearly flat inside a 1.2 km tile. Treat coarse layers as cell-level conditions, or find building-scale activity data.",
   "Decides whether activity enters this paper or the next. The model no longer forces the choice."]},
 "A3": {"kind":"action","title":"Move the raw HCMC vectors and the Python environment to the Linux box","why":[
   "The 96 GB Pro 6000 is here. The OSM geopackages, grid, and venv are not.",
   "The pilot's RGB cells need not come; the raw vectors must, since every channel is recomputed."]},
 "B1": {"kind":"action","title":"Cut the study area into zoom-15 map tiles (1,201 m)","why":[
   "In generalization theory (Bonnaire et al.) the memorization onset grows linearly with sample count. Four times the tiles of zoom 14 widens the observable generalization window.",
   "Typology is assigned per cell. 1.2 km cells are purer than 2.4 km cells, which keeps mutants clean.",
   "Reversible: four zoom-15 tiles merge into one zoom-14 tile. Tile addressing standardizes other cities later.",
   "Verified: 1,200 to 1,203 m across all of HCMC. No overlapping windows: near-duplicates corrupt the k-NN gap ratio."]},
 "B2": {"kind":"action","title":"Make the calculation unit equal to the pixel: 1024 px per tile, a 1.17 m unit","why":[
   "The unit is where an indicator's value is read, not where it is measured. Measurement scale belongs to each indicator: its decay radius, its ring, the polygon itself. So the unit is free above a floor set by the finest indicator, the footprints.",
   "Your call, given HCMC's ground conditions: at 1.17 m a 3 m alley is two to three pixels wide and a 5 m rowhouse keeps its outline. The pilot's 50 m units under 7.8 m pixels broke the definition.",
   "Cost grows with pixel count, not meaning: 64× the pilot's compute per step, so a 500K-step run is weeks by naive scaling. Debug at 128 or 256, calibrate at 512, run the experiments at 1024. Smaller mutants memorize sooner and need fewer steps.",
   "At this size the noise schedule must be shifted and the U-Net needs more levels; Simple Diffusion (Hoogeboom et al. 2023) trained 1024 px in pixel space exactly this way. Reasoning drawn in the companion page."]},
 "B3": {"kind":"action","title":"Buildings at 2 to 3 m: rasterize footprints, binary and density","why":[
   "Rowhouses are 4 to 6 m wide and alleys 2 to 4 m. Coarser units fuse them into blocks.",
   "The pilot's 150 m buffer coverage ratio flattened them to a scalar: Moran's I near 0."]},
 "B4": {"kind":"action","title":"Accessibility: gravity model on the road network, decay of a few hundred metres","why":[
   "Exponential decay yields a smooth field. Its meaning lives in the decay radius, so reading it at 2.35 m costs nothing and loses nothing.",
   "It was the only pilot channel with spatial structure (Moran's I 0.25)."]},
 "B5": {"kind":"action","title":"Diversity: Shannon entropy over a ring of a few hundred metres, plus POI count","why":[
   "Entropy needs at least two POIs and two categories. At 50 m the pilot channel was 97.5 % zeros.",
   "Count travels as its own channel, so diversity does not collapse into density.",
   "Semantic categories only if OSM coverage holds up. The checkpoint below decides."]},
 "B6": {"kind":"gate","title":"Checkpoint: POI-per-ring coverage report","why":[
   "The fraction of units whose ring is non-degenerate must be high before any training.",
   "You asked for this check explicitly."]},
 "B7": {"kind":"action","title":"Resample every channel to the common pixel grid (Liu et al. 2024)","why":[
   "Each channel is computed at its natural scale first. Alignment happens only at the end.",
   "Liu's image-to-image work fused independently rendered layers the same way."]},
 "B8": {"kind":"action","title":"Store cells as multi-channel tensors, not RGB PNGs","why":[
   "The U-Net reads its channel count from the config; one channel or six both work. Only the PNG loader forces three.",
   "No fusion step to defend, and activity layers become extra channels later.",
   "Cost: no single picture of all channels, and no pretrained perceptual metric. FID and SSCD were already unusable on urban rasters."]},
 "B9": {"kind":"action","title":"Decide how multi-channel cells will be shown","why":[
   "Six greyscale grids cannot be one picture. Fix the conventions before the first sample exists: small multiples, one panel per channel; false-colour composites, any three channels to RGB, the remote-sensing habit; one perceptually uniform colour map per indicator.",
   "Starting references, from memory, verify before citing: Tufte 1990, Envisioning Information, on small multiples; Crameri, Shephard and Heron 2020, Nature Communications, on colour maps; Munzner 2014, Visualization Analysis and Design, on encoding channels."]},
 "B10": {"kind":"gate","title":"Checkpoint: per-channel marginals, zero fraction, skew, tails","why":[
   "Chooses the transform for each channel.",
   "Gaussianity is not required. The mixtures in the reference papers are solvable toy models, not data assumptions."]},
 "B10b": {"kind":"gate","title":"Checkpoint: the joint pattern of the six channels, as well as each one alone","why":[
   "Not a replacement for B10. Each channel's distribution across the whole city is its own finding, and stays one: it is what says whether an indicator is sparse, skewed, or flat, and it is what chooses the transform.",
   "What it adds: six channels that each look right separately can still combine in ways the city never does, and nothing in a set of marginals can see that.",
   "A unit's six numbers are one point in six-dimensional space. The real city occupies only part of that space: dense building with no shops is ordinary, high entropy with no POIs is impossible by construction. The occupied region is itself a structure the model can get right or wrong.",
   "Same object as C3's clustering, which already treats a unit's column of scores as a point. This measures it as a distribution instead, so it can be compared against generated cells at E3.",
   "Read it twice: in data space to say what the combinations mean, and in standardized space for anything using distance, since otherwise the widest channel decides which units are alike."]},
 "B11": {"kind":"action","title":"Transform, then standardize each channel; store the parameters","why":[
   "A low-variance channel is pure noise over most of the diffusion trajectory. Standardization equalizes the signal-to-noise schedule across channels.",
   "Sparse, heavy-tailed channels get log1p or a quantile map first.",
   "The stored parameters are the decoder: unpacking a sample is just undoing this step."]},
 "C1": {"kind":"action","title":"Correlationist baseline: correlations among indicators across the study area","why":[
   "The control condition the paper argues against, on the same data.",
   "Not feature selection. The network handles correlated, non-linear inputs by itself.",
   "Flags near-duplicate channels. Pilot diversity and accessibility correlated at 0.88."]},
 "C2": {"kind":"action","title":"Write the cell descriptor: one function for real and generated cells","why":[
   "Per-channel mean, std, zero fraction, Moran's I, skewness; within-cell cross-channel correlation; unit-class composition.",
   "Must be computable from the raster alone. Generated cells have no vectors behind them.",
   "The pilot's statistics ran only on generated samples through separate code paths."]},
 "C3": {"kind":"action","title":"Typology: HDBSCAN on per-unit indicator vectors, then cell composition","why":[
   "DBSCAN needs only a vector and a distance. A unit's column of scores is one point, never split into one-hots.",
   "Never on raw full-resolution tensors, where distance is meaningless. HDBSCAN handles varying density.",
   "Typologies are the subspaces of Wang et al. That unifies intrinsic dimension, minimal set, and mutants."]},
 "C4": {"kind":"action","title":"Intrinsic dimension: global PCA k95, Levina-Bickel MLE, PCA inside each typology","why":[
   "PCA is linear and overestimates on curved manifolds. MLE follows curvature; the formula bug is fixed.",
   "Wang et al. size the minimal set by per-subspace dimension.",
   "The pilot's k95 of 33 was accessibility-dominated and must be re-estimated."]},
 "C4b": {"kind":"action","title":"Map the real dataset in two dimensions, coloured by typology","why":[
   "PCA to about 50 numbers per cell first, then UMAP or t-SNE. One map for the whole stack and one per grid, fitted on real cells only.",
   "Shows whether typologies separate and where the minimal set sits. It is the before picture, made before any model exists.",
   "Use UMAP if generated cells must later land on this same map: UMAP can place new points on a fitted map, t-SNE cannot."]},
 "C5": {"kind":"gate","title":"Checkpoint: does each typology sit in a low-dimensional subspace?","why":[
   "The Wang et al. assumption is a union of low-dimensional subspaces, a joint geometry, not marginal normality.",
   "If not, revisit the clustering before sizing any dataset. Mirrored later by the denoiser's Jacobian rank."]},
 "C6": {"kind":"action","title":"Select the minimal baseline set by an explicit rule","why":[
   "PCA gives directions, not a subset. Use k-center greedy in PCA space or one representative per typology.",
   "Sized per typology from C4.",
   "Below the intrinsic-dimension threshold the model memorizes, so this set is the control where the generalization window should be short or absent."]},
 "C7": {"kind":"action","title":"List the dataset mutants: baseline, +A, +B, −A, class-balanced, full","why":[
   "The one place where correlationist and connectionist views can disagree.",
   "Invariants across mutants are the irreducible structure.",
   "Each mutant gets its own model size and steps, because the memorization onset scales with n. Hypotheses H1 to H4 from June."]},
 "D1": {"kind":"action","title":"Train the DDPM directly on the cell tensors, with no latent space","why":[
   "Noise is added to the indicator grids themselves, so what the model memorizes or generalizes is measured in the data, not in a black box. A learned VAE latent would move it there and impose natural-image priors.",
   "\"Pixel\" here means unit, not colour. The tensor is C grids of unit scores; see the companion page.",
   "Model size and steps per mutant follow from the memorization onset scaling with n. At 1024 px the noise schedule must be shifted and the U-Net needs more levels.",
   "Runs on the Pro 6000 once A3 is done."]},
 "D2": {"kind":"action","title":"Score F_mem on an untrained model to get the metric's noise floor","why":[
   "Random weights produce nonsense cells. Whatever memorization fraction they score is the floor; only values above it mean the trained model copied.",
   "Independent of encoding, so it survives the redesign. Feeds the onset threshold in E4."]},
 "D3": {"kind":"gate","title":"Checkpoint: locate the memorization onset on one short run first","why":[
   "Every mutant is its own multi-day run, and the step budget only makes sense once tau_mem is known for this data. Guessing it wastes the budget in whichever direction the guess is wrong.",
   "Measured on the Pro 6000: a 500K-step run at 1024 px is 4 to 8 days. Sizing four mutants from a guess risks a month.",
   "The dataset is small. Of 1,881 zoom-15 tiles only 617 hold any building and 139 hold a hundred or more, and generalization theory has the onset growing with sample count, so the window may open and close early here.",
   "Run one mutant with logarithmic checkpoints until F_mem clears the D2 noise floor, then set every other run's budget from where that happened."]},
 "E1": {"kind":"action","title":"Sample every checkpoint; undo the standardization","why":[
   "A sample is a tensor in data space, not an embedding.",
   "Unpacking is the stored shift and scale in reverse, plus an optional three-grid rendering for humans."]},
 "E2": {"kind":"action","title":"Memorization: k-NN gap ratio on the standardized tensors","why":[
   "Raw distance is dominated by the highest-variance channel. The pilot measured accessibility memorization.",
   "The descriptor is too coarse to detect copying of a specific cell."]},
 "E3": {"kind":"action","title":"Distribution: descriptor on real vs generated, per mutant, per checkpoint","why":[
   "Same function on both sides.",
   "Irreducible characteristics: invariant across mutants and stable through the generalization window."]},
 "E3b": {"kind":"action","title":"Put generated cells on the real dataset's 2-D map, per grid and per checkpoint","why":[
   "Same PCA and UMAP fitted in C4b, so real and generated points share one map. Where generated points fill, avoid, or pile onto real ones shows what each grid learned.",
   "Per grid isolates which indicator's distribution drifted. Repeated over checkpoints it becomes a film of the generalization window.",
   "Qualitative: distances on such a map are not measurements. The descriptor histograms and Wasserstein distances carry the numbers."]},
 "E3c": {"kind":"action","title":"Spatial relations between grids: real versus generated, before and after training","why":[
   "Within a cell, how the grids sit relative to each other in space: bivariate Moran's I between two grids, co-location of unit classes, cross-correlograms by distance.",
   "The pilot's synergy analysis compared cell-level averages, which cannot tell whether buildings sit on accessible units or merely share a cell with them.",
   "Computed once on real cells and at every checkpoint on generated cells, so the change reads from what to what."]},
 "E4": {"kind":"action","title":"Locate the window per mutant: structure onset to memorization onset above the noise floor","why":[
   "Structure onset from E3, E3b and E3c; memorization onset from E2, above the D2 floor.",
   "Disclose thresholds. The pilot's coupled threshold |r| > 0.5 moved the boundary."]},
 "E5": {"kind":"future","title":"Jacobian rank of the trained denoiser (Hutchinson estimator)","why":[
   "Model-side test of the subspace structure from C5.",
   "Full SVD is infeasible at this size."]},
 "F1": {"kind":"action","title":"Contrast the correlationist baseline with the mutant invariants","why":[
   "The falsifiable claim: features invariant across mutants and stable through the generalization window are structure. Correlation importance is the comparison."]},
 "F2": {"kind":"future","title":"Joint form + activity run with extra channels","why":[
   "The ISM proposal's original wording. Depends on A2.",
   "No architecture change: activity layers are more channels. Conditional sampling then answers \"given this form, what activity?\" directly."]},
}

GLOSS = [
 ("cell", "Cell", "One map tile. One training example. Everything the model sees at once."),
 ("unit", "Unit", "One pixel inside a cell: the patch of urban space that each indicator gives a score to."),
 ("channel", "Channel", "One indicator's grid of unit scores for a cell. Six indicators make six greyscale grids. The pilot's RGB file was three of these in an image container; the colours were never colours."),
 ("tensor", "Tensor", "The whole stack of grids for one cell. What the model learns from and what it generates."),
 ("standardize", "Standardization", "Shifting and scaling each channel so its values centre on zero with a spread of one. The shift and scale are stored, so the step can be undone on a generated sample."),
 ("descriptor", "Descriptor", "A fixed recipe that turns one cell into a short list of numbers: per-channel average, spread, share of empty units, spatial clumping, class mix. Run on real and generated cells alike, then the two lists of numbers are compared."),
 ("fmem", "F_mem, memorization fraction, k-NN gap ratio", "The share of generated cells that are near-copies of a single training cell. Found by the gap ratio: distance to the nearest training cell divided by distance to the second nearest. A ratio below one third means the sample sits on top of one training cell and counts as a copy."),
 ("floor", "Noise floor", "What F_mem reads on a model with random, never-trained weights. Only values above it mean the trained model actually copied."),
 ("window", "Generalization window", "The stretch of training after the model has learned the structure of the data and before it starts copying individual cells."),
 ("gentheory", "Generalization theory", "Bonnaire et al.'s account of diffusion training: the model first learns the shared structure of the data and only later starts copying individual cells. The window between the two is what we study; memorization is its far edge."),
 ("taumem", "Memorization onset, τ_mem", "How many training steps pass before copying starts. Generalization theory shows it grows in proportion to the number of training cells, so bigger datasets stay in the window longer."),
 ("id", "Intrinsic dimension", "How many independent knobs it takes to describe the dataset, even though each cell holds millions of numbers. The pilot estimated about 33 by PCA."),
 ("pca", "PCA k95", "The number of straight-line directions needed to explain 95 percent of the dataset's spread. Overestimates when the data is curved."),
 ("mle", "Levina-Bickel MLE", "An estimate of intrinsic dimension from how nearest-neighbour distances grow with rank. It follows curvature, so it usually gives a smaller number than PCA."),
 ("typology", "Typology", "A group of cells that look alike, found by clustering. The building blocks for dataset mutants."),
 ("hdbscan", "HDBSCAN", "A clustering method that finds groups of any shape, leaves stragglers unassigned, and copes with dense and sparse regions at the same time. Needs no preset number of clusters."),
 ("subspace", "Subspace", "A flat, low-dimensional sheet inside the big space of all possible cells. Wang et al. assume each typology lies on its own sheet."),
 ("kcenter", "k-center greedy", "A way to pick a small set of cells that spread out over the whole dataset: repeatedly add the cell farthest from those already chosen."),
 ("mutant", "Mutant", "A training dataset made from the baseline by adding or removing whole typologies. Comparing models trained on mutants shows which structures survive."),
 ("moran", "Moran's I", "A number for spatial clumping. Near 1, neighbouring units look alike; near 0, there is no spatial pattern at all."),
 ("wasserstein", "Wasserstein distance", "How much material you would have to move to turn one histogram into another. A distance between two distributions."),
 ("jacobian", "Jacobian rank", "How many directions the trained denoiser actually responds to. A low rank means it has learned a low-dimensional structure rather than reacting to every pixel independently."),
 ("hutchinson", "Hutchinson estimator", "A trick to estimate that rank with random probes instead of computing the full matrix, which would be far too large."),
 ("umap", "UMAP and t-SNE", "Two ways to flatten many-number cells onto a 2-D map so that cells which resemble each other land close together. Good for seeing clusters and gaps, bad for measuring: distances on the map are not distances in the data. UMAP can place new points on a map fitted earlier; t-SNE must be refitted from scratch."),
 ("bimoran", "Bivariate Moran's I", "Moran's I between two grids: whether high values of one indicator sit next to high values of another in space. Near 1, buildings cluster around accessible units; near 0, the two grids ignore each other spatially."),
 ("smallmult", "Small multiples", "Many small panels with the same layout side by side, one per channel, so the eye compares them."),
 ("falsecolour", "False-colour composite", "Showing any three channels as red, green and blue. What remote sensing does with satellite bands."),
]
GKEYS = {k for k, _, _ in GLOSS}

TERMS = [  # phrase -> glossary key; longest first at apply time
 ("generalization theory","gentheory"),("memorization onset","taumem"),("bivariate Moran's I","bimoran"),("UMAP","umap"),("t-SNE","umap"),
 ("k-NN gap ratio","fmem"),("memorization fraction","fmem"),("F_mem","fmem"),("noise floor","floor"),
 ("descriptor","descriptor"),("Jacobian rank","jacobian"),("Hutchinson estimator","hutchinson"),
 ("Moran's I","moran"),("HDBSCAN","hdbscan"),("k-center greedy","kcenter"),("Levina-Bickel MLE","mle"),
 ("PCA k95","pca"),("intrinsic-dimension","id"),("intrinsic dimension","id"),("typologies","typology"),
 ("typology","typology"),("mutants","mutant"),("mutant","mutant"),("generalization window","window"),
 ("memorization time","taumem"),("subspaces","subspace"),("subspace","subspace"),
 ("standardization","standardize"),("standardize","standardize"),("small multiples","smallmult"),
 ("false-colour composites","falsecolour"),("Wasserstein","wasserstein"),
]
TERMS.sort(key=lambda t: -len(t[0]))
for _, k in TERMS: assert k in GKEYS, k

refs = [c for s in STAGES for r in s["rows"] for c in r]
assert sorted(refs) == sorted(N.keys()), set(refs) ^ set(N.keys())

def esc(s): return html.escape(s, quote=False)

def link_terms(text, done):
    """text is already escaped; link first occurrence of each term once per node."""
    for phrase, key in TERMS:
        if key in done: continue
        pat = re.compile(r'(?<![\w-])' + re.escape(esc(phrase)) + r'(?![\w-])')
        new, n = pat.subn(lambda m: f'<a class="term" href="#g-{key}">{m.group(0)}</a>', text, count=1)
        if n: text = new; done.add(key)
    return text

def html_node(code):
    n = N[code]; kind = n["kind"]; done = set()
    title = link_terms(esc(n["title"]), done)
    lis = "\n".join(f"        <li>{link_terms(esc(b), done)}</li>" for b in n["why"])
    return (f'    <article class="node {kind}" id="{code}">\n'
            f'      <header><span class="code">{code}</span><span class="kind">{KIND_LABEL[kind]}</span></header>\n'
            f'      <h3>{title}</h3>\n      <p class="why">Why</p>\n      <ul>\n{lis}\n      </ul>\n    </article>\n')

def html_row(row):
    if len(row) == 1:
        return f'  <div class="row single">\n    <div class="stub"></div>\n{html_node(row[0])}  </div>\n'
    return (f'  <div class="row parallel" style="--n:{len(row)}">\n    <div class="fork"><i></i></div>\n'
            f'    <div class="lanes">\n{"".join(html_node(c) for c in row)}    </div>\n    <div class="join"><i></i></div>\n  </div>\n')

GLOSS_HTML = '<dl class="gloss">\n' + "\n".join(
    f'  <dt id="g-{k}">{esc(t)}</dt>\n  <dd>{esc(d)}</dd>' for k, t, d in GLOSS) + '\n</dl>'

parts = []
for i, s in enumerate(STAGES):
    stub = "" if i == 0 else '    <div class="stub"></div>\n'
    parts.append(f'  <div class="stage">\n{stub}    <div class="stage-label"><span class="letter">STAGE {s["code"]}</span>'
                 f'<span class="name">{esc(s["title"])}</span><span class="ordinal">{esc(s["ordinal"])}</span></div>\n  </div>\n')
    parts += [html_row(r) for r in s["rows"]]
parts.append('  <div class="stub"></div>\n  <div class="end" aria-hidden="true"></div>\n')

n_nodes = len(N); n_gates = sum(1 for n in N.values() if n["kind"] == "gate")
MAP_HEAD = f'<title>Latent City Pipeline Map</title>\n<style>{CSS}</style>\n{FONTS}\n'
MAP_BODY = f'''<main class="page">
  <header class="masthead">
    <p class="eyebrow">DDPM urban vitality · pipeline v4 · decided 9 Sep 2026</p>
    <h1>Latent City Pipeline Map</h1>
    <p class="lede">One spine from raw vectors to the paper. Nodes side by side can run at the same time. Under every action sits the reason we chose it, so the argument travels with the task. Dotted words link to plain-language definitions at the bottom.</p>
    <ul class="legend">
      <li><span class="sw"></span>decided action</li>
      <li><span class="sw gate"></span>checkpoint that gates the next step</li>
      <li><span class="sw future"></span>later or optional</li>
      <li><span class="sw par"></span>fork: run in parallel</li>
    </ul>
    <p class="facts">zoom-15 tile = <b>1,201 m</b> at 10.78° N &nbsp;·&nbsp; 128 px → <b>9.4 m</b> &nbsp;·&nbsp; 256 → <b>4.7 m</b> &nbsp;·&nbsp; 512 → <b>2.35 m</b> &nbsp;·&nbsp; 1024 → <b>1.17 m, chosen</b> &nbsp;·&nbsp; {n_nodes} nodes, {n_gates} checkpoints</p>
    <p class="links">Companion page with the reasoning drawn out: <a href="{EXPLAINER_URL}">Cells, Channels, Units</a> &nbsp;·&nbsp; <a href="#terms">Terms in plain words</a></p>
  </header>
  <section class="flow" aria-label="Pipeline stages A to F">
{"".join(parts)}  </section>
  <section id="terms" class="terms-section">
    <h2>Terms in plain words</h2>
    {GLOSS_HTML}
  </section>
  <footer>
    <p>Distilled from the planning discussions of 9 Sep 2026 and the June v3 plan in <code>progress_260611.md</code>. This order supersedes <code>TODO.md</code>. Generated by <code>discussion/pipeline_map/gen.py</code>; a draw.io copy sits beside it.</p>
  </footer>
</main>
'''

def standalone(head, body):
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + head + '</head>\n<body>\n' + body + '</body>\n</html>\n')

def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: f.write(text)

write(os.path.join(ART_DIR, "latent-city-pipeline-map.html"), MAP_HEAD + MAP_BODY)
write(os.path.join(REPO_OUT, "pipeline_v4_map.html"), standalone(MAP_HEAD, MAP_BODY))

def scatter_svg():
    import random
    rnd = random.Random(7)
    def blob(cx, cy, r, n, cls):
        out = []
        for _ in range(n):
            a = rnd.uniform(0, 6.283); d = r * math.sqrt(rnd.uniform(0, 1))
            out.append(f'<circle cx="{cx + d*math.cos(a):.1f}" cy="{cy + d*math.sin(a):.1f}" r="3.2" class="{cls}"/>')
        return "".join(out)
    real = blob(100, 120, 40, 22, "f-pt") + blob(250, 90, 34, 20, "f-pt") + blob(215, 205, 30, 16, "f-pt") + blob(180, 150, 120, 10, "f-pt")
    gen = blob(100, 120, 38, 14, "f-pt-gen") + blob(212, 208, 5, 12, "f-pt-gen")
    return real + gen


# ---------------------------------------------------------------- descriptor
# Section 8 is built from real measurements, cached by
# make_descriptor_figures.py so this generator needs no raw data.
DESC_JSON = os.path.join(HERE, "descriptor_figures.json")


def descriptor_section():
    import json
    if not os.path.exists(DESC_JSON):
        return ""
    d = json.load(open(DESC_JSON, encoding="utf-8"))

    def esc(t):
        return html.escape(str(t))


    # -- 8a0: where the descriptor sits -----------------------------------
    def box(x, y, w, h, title, sub="", cls="f-box"):
        out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="{cls}" rx="3"/>',
               f'<text x="{x + w/2:.0f}" y="{y + (18 if sub else h/2 + 4):.0f}" '
               f'class="f-lab" text-anchor="middle">{title}</text>']
        if sub:
            for i, line in enumerate(sub.split("|")):
                out.append(f'<text x="{x + w/2:.0f}" y="{y + 34 + i*13:.0f}" '
                           f'class="f-sm f-mute" text-anchor="middle">{line}</text>')
        return "".join(out)

    def arrow(x1, y1, x2, y2, cls="f-edge"):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}" '
                f'marker-end="url(#ds-ar)"/>')

    landscape = f'''<figure><svg viewBox="0 0 742 398" role="img" aria-label="Real
      and generated cells feed two separate measurements: the gap ratio in
      standardized space, and the descriptor in data space. Their two onsets
      bracket the generalization window.">
      <defs><marker id="ds-ar" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="7" markerHeight="7" orient="auto">
        <path d="M0 0L10 5L0 10z" class="f-arrow"/></marker></defs>

      {box(14, 40, 120, 54, "real cells", "the training set|as stored", "f-box")}
      {box(14, 254, 120, 54, "generated cells", "one checkpoint|sampled from the model", "f-box-dash")}

      <text x="150" y="30" class="f-sm f-mute">both sides go through both
        measurements &mdash; that is the whole design</text>

      <line x1="74" y1="94" x2="74" y2="254" class="f-edge" stroke-dasharray="3 3"/>
      {arrow(134, 60, 176, 60)}
      {arrow(134, 274, 176, 130)}
      {arrow(134, 282, 176, 282)}
      {arrow(134, 74, 176, 236)}

      <rect x="170" y="42" width="404" height="112" class="f-a1" rx="4"/>
      <text x="182" y="60" class="f-mono f-acc">STANDARDIZED SPACE</text>
      <text x="182" y="76" class="f-sm f-mute">every channel weighs the same,
        because a distance</text>
      <text x="182" y="90" class="f-sm f-mute">over the whole cell is what is
        being asked for</text>
      {box(182, 100, 128, 44, "k-NN gap ratio", "d1 / d2 per sample")}
      {box(330, 100, 108, 44, "F_mem", "share below 1/3")}
      {box(458, 100, 104, 44, "above the", "D2 noise floor?", "f-box-acc")}
      {arrow(310, 122, 330, 122)}
      {arrow(438, 122, 458, 122)}

      <rect x="170" y="224" width="404" height="130" class="f-a1" rx="4"/>
      <text x="182" y="242" class="f-mono f-acc">DATA SPACE</text>
      <text x="182" y="258" class="f-sm f-mute">physical units, because
        &ldquo;zero fraction&rdquo; has to mean</text>
      <text x="182" y="272" class="f-sm f-mute">&ldquo;no building here&rdquo;
        rather than &ldquo;at the channel mean&rdquo;</text>
      {box(182, 282, 128, 58, "describe_cell", "one function|for both sides")}
      {box(330, 282, 108, 58, "one row of", "named numbers|per cell")}
      {box(458, 282, 104, 58, "distributions", "compared|E3")}
      {arrow(310, 311, 330, 311)}
      {arrow(438, 311, 458, 311)}

      <text x="182" y="176" class="f-sm">unpack (E1): the stored shift and scale,
        in reverse</text>
      {arrow(176, 190, 176, 224)}
      <line x1="176" y1="154" x2="176" y2="190" class="f-edge-acc"/>

      {box(586, 100, 142, 44, "memorization onset", "", "f-box-acc")}
      {box(586, 290, 142, 44, "structure onset", "", "f-box-acc")}
      {arrow(562, 122, 586, 122)}
      {arrow(562, 311, 586, 311)}
      <line x1="657" y1="144" x2="657" y2="180" class="f-edge"/>
      <line x1="657" y1="290" x2="657" y2="254" class="f-edge"/>
      {box(586, 180, 142, 74, "E4", "the generalization|window is the|stretch between|the two", "f-box-acc")}

      <text x="14" y="378" class="f-sm">The two measurements answer different
        questions and cannot substitute for each other: the descriptor is far
        too coarse to notice</text>
      <text x="14" y="392" class="f-sm">that one generated cell is a copy of one
        training cell, and the gap ratio says nothing about what the data looks
        like.</text>
    </svg>
    <figcaption>Where the descriptor sits. It is one of two measurements, run on
    both real and generated cells, and the pair of onsets they produce is what
    brackets the window the study is about. Note that they run in different
    spaces, and for opposite reasons: a distance needs every channel to weigh
    the same, so the gap ratio stays standardized, while the descriptor's
    numbers have to be readable in physical units, so a sample is unpacked
    first.</figcaption></figure>'''

    # -- 8a: the recipe, on real cells ------------------------------------
    cards = []
    for c in d["real_cells"]:
        cards.append(
            f'<div class="thumb">'
            f'<span class="cap">{esc(c["cell_id"])}</span>'
            f'<img src="{c["binary_png"]}" alt="Footprints of tile {esc(c["cell_id"])}">'
            f'<span class="val">built <b>{c["built_fraction"]:.1%}</b><br>'
            f'Moran I <b>{c["binary_morans_i"]:.3f}</b></span>'
            f'<img src="{c["density_png"]}" alt="Building density of tile {esc(c["cell_id"])}">'
            f'<span class="val">density <b>{c["density_mean"]:.3f}</b><br>'
            f'Moran I <b>{c["density_morans_i"]:.4f}</b><br>'
            f'cross <b>{c["cross"]:.2f}</b></span>'
            f'</div>')
    recipe = ('<figure><div class="thumbs">' + "".join(cards) + '</div>'
              '<figcaption>Three real zoom-15 tiles at 1024 px. Teal is the footprint '
              'grid, amber the 100 m density grid built from it, and under each pair '
              'are five of the numbers the descriptor returns for that cell. Every '
              'number here came from the raster alone, which is the constraint: a '
              'generated cell has no vectors behind it, so anything the descriptor '
              'needs must be readable off the grids.</figcaption></figure>')

    # -- 8b: the shuffle test ---------------------------------------------
    sh = d["shuffle"]
    rows = "".join(
        f'<tr><td>{esc(r["key"].replace("_", " "))}</td>'
        f'<td>{r["real"]:.4f}</td><td>{r["shuffled"]:.4f}</td>'
        f'<td class="{"moved" if r["moved"] else "same"}">'
        f'{"moved" if r["moved"] else "unchanged"}</td></tr>'
        for r in sh["rows"])
    shuffle = (
        '<figure><div class="pair">'
        f'<div class="thumb"><span class="cap">{esc(sh["cell_id"])} &middot; as it is</span>'
        f'<img src="{sh["real_png"]}" alt="Footprints of a real tile"></div>'
        f'<div class="thumb"><span class="cap">the same units, shuffled</span>'
        f'<img src="{sh["shuffled_png"]}" alt="The same units in random positions"></div>'
        '</div>'
        '<table class="t"><thead><tr><th>statistic</th><th>as it is</th>'
        '<th>shuffled</th><th></th></tr></thead><tbody>' + rows + '</tbody></table>'
        '<figcaption>The right-hand picture holds exactly the same units as the left, '
        'moved to random positions. Four of the five statistics cannot tell the '
        'difference, to four decimal places, because they are statements about how '
        'much there is. Moran&rsquo;s I falls from 0.797 to about zero, because it is '
        'the only one that is a statement about where it is.</figcaption></figure>')

    # -- 8c: Moran's I strip ----------------------------------------------
    strip = "".join(
        f'<div class="thumb"><img src="{p["png"]}" alt="{esc(p["name"])} pattern">'
        f'<span class="name">{esc(p["name"])}</span>'
        f'<span class="val">queen <b>{p["queen"]:+.3f}</b><br>'
        f'rook <b>{p["rook"]:+.3f}</b></span>'
        f'<span class="note">{esc(p["note"])}</span></div>'
        for p in d["morans_strip"])
    chequer = next(p for p in d["morans_strip"] if p["name"] == "chequer")
    morans = ('<figure><div class="strip">' + strip + '</div>'
              '<figcaption>Moran&rsquo;s I from &minus;1 to +1, on patterns whose '
              'arrangement is known by construction. Queen counts all eight '
              'neighbours, rook only the four that share an edge. They agree '
              'everywhere except on the chequer, where the diagonal neighbours are '
              f'the same colour: rook reads {chequer["rook"]:+.3f}, queen reads '
              f'{chequer["queen"]:+.3f}.'
              '</figcaption></figure>')


    # -- 8c2: what "next to it" means -------------------------------------
    CELL, GAP = 26, 2
    def grid5(ox, oy, fill_of):
        out = []
        for row in range(5):
            for col in range(5):
                x = ox + col * (CELL + GAP)
                y = oy + row * (CELL + GAP)
                out.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                           f'class="{fill_of(row, col)}"/>')
        return "".join(out)

    def rook_of(row, col):
        if (row, col) == (2, 2):
            return "f-probe"
        return "f-a3" if abs(row - 2) + abs(col - 2) == 1 else "f-panel"

    def queen_of(row, col):
        if (row, col) == (2, 2):
            return "f-probe"
        return "f-a3" if max(abs(row - 2), abs(col - 2)) == 1 else "f-panel"

    def chequer_of(row, col):
        return "f-bld" if (row + col) % 2 == 0 else "f-panel"

    ring = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        x = 512 + (2 + dc) * (CELL + GAP); y = 56 + (2 + dr) * (CELL + GAP)
        ring.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                    f'fill="none" stroke="var(--gate)" stroke-width="2.6"/>')
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        x = 512 + (2 + dc) * (CELL + GAP); y = 56 + (2 + dr) * (CELL + GAP)
        ring.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                    f'fill="none" stroke="var(--spine)" stroke-width="2.6" '
                    f'stroke-dasharray="4 3"/>')

    neighbours = f'''<figure><svg viewBox="0 0 742 300" role="img" aria-label="Rook
      neighbours are the four sharing an edge, queen neighbours add the four
      diagonals, and on a chequer the two rules disagree.">
      <text x="20" y="18" class="f-lab">rook &middot; 4 neighbours</text>
      <text x="20" y="36" class="f-sm f-mute">the ones sharing an edge</text>
      {grid5(20, 56, rook_of)}
      <text x="266" y="18" class="f-lab">queen &middot; 8 neighbours</text>
      <text x="266" y="36" class="f-sm f-mute">edges and corners</text>
      {grid5(266, 56, queen_of)}
      <text x="512" y="18" class="f-lab">why they disagree</text>
      <text x="512" y="36" class="f-sm f-mute">a chequer, centre unit dark</text>
      {grid5(512, 56, chequer_of)}
      {"".join(ring)}
      <text x="20" y="238" class="f-sm">Moran&rsquo;s I compares every unit with its
        neighbours and averages the result over the whole grid.</text>
      <text x="20" y="256" class="f-sm">One number, for one grid, inside one cell:
        how smooth is this picture.</text>
      <g>
        <rect x="452" y="230" width="14" height="10" fill="none"
          stroke="var(--gate)" stroke-width="2.6"/>
        <text x="472" y="239" class="f-sm">4 rook: all opposite &rarr; I = &minus;1.00</text>
        <rect x="452" y="252" width="14" height="10" fill="none"
          stroke="var(--spine)" stroke-width="2.6" stroke-dasharray="4 3"/>
        <text x="472" y="261" class="f-sm">4 diagonals: all the same &rarr; they cancel it</text>
        <text x="452" y="283" class="f-mono f-acc">queen reads &minus;0.03</text>
      </g>
    </svg>
    <figcaption>&ldquo;Next to it&rdquo; is a choice, and on fine patterns it changes the
    answer. A chequer&rsquo;s edge-neighbours are all its opposite, so rook weights
    read perfect dispersion; its corner-neighbours are all its own colour, and
    under queen weights those cancel the edges out to nearly nothing. At a
    1.17 m unit a row of houses separated by 3 m alleys is close to this
    pattern, so the rule has to be stated rather than defaulted
    into.</figcaption></figure>'''

    # -- 8f: what the resampling fix was ----------------------------------
    resampling = ""
    rs = d.get("resampling")
    if rs:
        PW, PH, PX0, PY0, GAPX = 216, 128, 20, 96, 26
        floor, ceiling = rs["coarse_min"], rs["coarse_max"]
        # Include zero in the range. The band between zero and the solved
        # minimum is the whole point of the figure, so it needs real height,
        # and showing zero says why a value under it is impossible rather than
        # merely unexpected.
        lo = min(0.0, rs["cubic_min"]) - 0.4
        hi = ceiling + 0.8

        # Show a detail window rather than the whole row. Across 1,201 m the
        # three curves are visually identical; the differences live in a few
        # hundred metres around a sharp edge, which is exactly where the
        # overshoot happens and where a corner would be read as a feature.
        cubic_full = rs["cubic"]
        centre = min(range(len(cubic_full)), key=lambda i: cubic_full[i])
        half = max(len(cubic_full) // 9, 12)
        w0 = max(0, min(centre - half, len(cubic_full) - 2 * half))
        w1 = min(len(cubic_full), w0 + 2 * half)
        frac0, frac1 = w0 / len(cubic_full), w1 / len(cubic_full)

        def window(curve):
            return curve[w0:w1]

        coarse_pts = [
            (i / (rs["n_coarse"] - 1), v) for i, v in enumerate(rs["coarse"])
            if frac0 <= i / (rs["n_coarse"] - 1) <= frac1
        ]

        def sx_frac(f, ox):
            return ox + (f - frac0) / (frac1 - frac0) * PW

        def sx(i, total, ox):
            return ox + (i / max(total - 1, 1)) * PW

        def sy(v):
            return PY0 + PH - (v - lo) / (hi - lo) * PH

        panels = []
        for k, (key, title, note) in enumerate([
            ("linear", "bilinear", "straight lines between the solved points"),
            ("cubic", "cubic", "smooth, but it overshoots"),
            ("clamped", "cubic, clamped", "held inside what was solved"),
        ]):
            ox = PX0 + k * (PW + GAPX)
            curve = window(d["resampling"][key])
            pts = " ".join(f"{sx(i, len(curve), ox):.1f},{sy(v):.1f}"
                           for i, v in enumerate(curve))
            dots = "".join(
                f'<circle cx="{sx_frac(f, ox):.1f}" cy="{sy(v):.1f}" r="2.6" '
                f'fill="var(--muted)"/>' for f, v in coarse_pts)
            # the band below anything that was actually solved
            band = (f'<rect x="{ox}" y="{sy(floor):.1f}" width="{PW}" '
                    f'height="{PY0 + PH - sy(floor):.1f}" fill="var(--gate)" '
                    f'fill-opacity=".13"/>')
            panels.append(
                f'<text x="{ox}" y="{PY0 - 30}" class="f-lab">{title}</text>'
                f'<text x="{ox}" y="{PY0 - 14}" class="f-sm f-mute">{note}</text>'
                f'{band}'
                f'<line x1="{ox}" y1="{sy(floor):.1f}" x2="{ox + PW}" '
                f'y2="{sy(floor):.1f}" stroke="var(--gate)" stroke-width="1" '
                f'stroke-dasharray="3 3"/>'
                f'<line x1="{ox}" y1="{sy(0):.1f}" x2="{ox + PW}" '
                f'y2="{sy(0):.1f}" class="f-edge"/>'
                + (f'<text x="{ox - 6}" y="{sy(floor) + 3:.1f}" class="f-mono f-mute" '
                   f'text-anchor="end">{floor:.0f}</text>'
                   f'<text x="{ox - 6}" y="{sy(0) + 3:.1f}" class="f-mono f-mute" '
                   f'text-anchor="end">0</text>' if k == 0 else "")
                + f'<polyline points="{pts}" fill="none" stroke="var(--spine)" '
                  f'stroke-width="2"/>{dots}')

        resampling = f'''<figure><svg viewBox="0 0 742 292" role="img"
          aria-label="One row of a ring count, interpolated three ways: bilinear
          leaves corners, cubic dips below the solved minimum, clamping fixes it.">
          <text x="20" y="18" class="f-lab">one row of {html.escape(rs["indicator"])},
            solved at {rs["n_coarse"]} points, stretched to {rs["target"]}
            &mdash; detail around one sharp edge</text>
          <text x="20" y="34" class="f-sm f-mute">grey dots are the values actually
            computed; the shaded band is below anything that was</text>
          {"".join(panels)}
          <text x="20" y="{PY0 + PH + 34}" class="f-sm">Solved range {floor:.0f} to
            {ceiling:.0f} POIs. Cubic alone reaches
            <tspan class="f-acc">{rs["cubic_min"]:.2f}</tspan>, under the solved
            minimum on {rs["below_share"]:.1%} of the row.</text>
        </svg>
        <figcaption>The indicator is computed at its own scale and then has to
        reach the shared 1024 px grid. Straight-line interpolation leaves a
        corner at every solved point, and a corner is a spatial feature the
        descriptor cannot tell from a real one. A cubic curve removes the
        corners but overshoots near a sharp edge, and pushed this ring count
        below the smallest value anywhere in the solve &mdash; on the entropy
        channel the same overshoot produced negative bits. Clamping to the
        range that was actually solved keeps the smooth curve and cannot
        invent a value, because no interior point was ever outside it.
        </figcaption></figure>'''

    # -- 8d: what the fine unit buys --------------------------------------
    W, H, L, R, T, B = 720, 300, 62, 150, 24, 52
    lo, hi = 1.0, 200.0

    def px(u):
        return L + (math.log10(u) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (W - L - R)

    def py(v):
        return T + (1.0 - v) * (H - T - B)

    parts = [f'<line x1="{L}" y1="{py(0)}" x2="{W-R}" y2="{py(0)}" class="f-edge"/>',
             f'<line x1="{L}" y1="{T}" x2="{L}" y2="{py(0)}" class="f-edge"/>']
    for v in (0, .25, .5, .75, 1.0):
        parts.append(f'<line x1="{L-4}" y1="{py(v):.1f}" x2="{L}" y2="{py(v):.1f}" class="f-edge"/>')
        parts.append(f'<text x="{L-8}" y="{py(v)+4:.1f}" class="f-mono f-mute" text-anchor="end">{int(v*100)}%</text>')
    for u, lab in ((1.17, "1.17"), (2.35, "2.35"), (4.7, "4.7"), (9.4, "9.4"),
                   (18.8, "18.8"), (37.6, "37.6"), (75, "75"), (150, "150")):
        parts.append(f'<line x1="{px(u):.1f}" y1="{py(0)}" x2="{px(u):.1f}" y2="{py(0)+4}" class="f-edge"/>')
        parts.append(f'<text x="{px(u):.1f}" y="{py(0)+18}" class="f-mono f-mute" text-anchor="middle">{lab}</text>')
    parts.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-14}" class="f-sm f-mute" text-anchor="middle">ground metres per unit</text>')
    parts.append(f'<text x="{L}" y="{T-8}" class="f-lab">share of the channel that survives being read on a coarser grid</text>')

    colours = ["f-acc", "f-mute"]
    for i, series in enumerate(d["resolution"]):
        pts = " ".join(f'{px(p["unit_m"]):.1f},{py(p["retained"]):.1f}'
                       for p in series["points"] if p["unit_m"] <= hi)
        stroke = "var(--spine)" if i == 0 else "var(--gate)"
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="2.2"/>')
        for p in series["points"]:
            if p["unit_m"] <= hi:
                parts.append(f'<circle cx="{px(p["unit_m"]):.1f}" cy="{py(p["retained"]):.1f}" r="3" fill="{stroke}"/>')
        last = [p for p in series["points"] if p["unit_m"] <= hi][-1]
        parts.append(f'<text x="{W-R+8}" y="{py(last["retained"])+4:.1f}" class="f-mono" fill="{stroke}">{esc(series["channel"])}</text>')

    # the pilot's unit, for reference
    parts.append(f'<line x1="{px(9.4):.1f}" y1="{T}" x2="{px(9.4):.1f}" y2="{py(0)}" stroke="var(--line)" stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{px(9.4)+6:.1f}" y="{T+14}" class="f-sm f-mute">pilot&rsquo;s 9.4 m</text>')

    curve = (f'<figure><svg viewBox="0 0 {W} {H}" role="img" aria-label="Retained '
             'variance against unit size for the footprint and density channels">'
             + "".join(parts) + '</svg>'
             '<figcaption>Pool each channel onto a coarser grid, interpolate back, and '
             'measure how much of its variance survives. The footprint channel loses '
             '9&nbsp;% of itself by 2.35 m and 31&nbsp;% by the pilot&rsquo;s 9.4 m. The '
             '100 m density channel is still at 99.5&nbsp;% on a 37.6 m grid: it only '
             'begins to lose anything near its own radius. This is the direct form of '
             'the question Moran&rsquo;s I only gestures at.</figcaption></figure>')

    # -- 8e: sparsity is a place ------------------------------------------
    sparsity = ""
    if d.get("sparsity"):
        cells = "".join(
            f'<div class="thumb"><span class="cap">{esc(s["cell_id"])}</span>'
            f'<img src="{s["png"]}" alt="Empty units of tile {esc(s["cell_id"])}">'
            f'<span class="val">empty <b>{s["zero_fraction"]:.1%}</b><br>'
            f'Moran I of the emptiness <b>{s["zero_morans_i"]:.3f}</b></span></div>'
            for s in d["sparsity"])
        sparsity = ('<figure><div class="thumbs">' + cells + '</div>'
                    '<figcaption>The same tiles again, now showing only which units are '
                    'empty. A zero fraction is a fact about a histogram and says nothing '
                    'about arrangement; the emptiness here has a Moran&rsquo;s I of its '
                    'own above 0.7, meaning the zeros are streets, yards and water '
                    'rather than scatter. Emptiness with a shape is something a model '
                    'can learn. Emptiness without one is a channel that is mostly '
                    'missing.</figcaption></figure>')

    return f'''  <section id="descriptor">
    <h2><span class="num">8</span>What the descriptor sees</h2>
    <p>The <b>descriptor</b> is one function that turns a cell into a short list of named numbers &mdash; a <b>cell</b> being one 1.2 km square of the city, the whole of what the model sees at once. It is run on real cells and on cells the model invented, through the same code, and every claim in this study about whether the two look alike is a comparison of two such lists.</p>
    <p>It has one hard constraint. It must be computable from the picture alone, because a generated cell has no buildings or roads behind it &mdash; only grids of numbers. So the recipe is: for each <b>channel</b> (one indicator&rsquo;s grid of values, six of them stacked to make a cell) take its average, its spread, the share of its <b>units</b> that are empty &mdash; a unit being one pixel, 1.17 m of ground &mdash; how lopsided its values are, and how clumped they are in space. Then the correlation between each pair of channels inside that cell. Then, once the units have been sorted into kinds, the mix of those kinds.</p>
    {landscape}
    <p>Inside that <code>describe_cell</code> box the recipe is fixed and short. Here it is on three real tiles.</p>
    {recipe}
    <p>Four of those five per-channel numbers are about <em>how much</em>. The average says how much building there is. The <b>spread</b> (its standard deviation) says how much the values vary from that average. The <b>zero fraction</b> says what share of the ground is empty. The <b>skewness</b> says how lopsided the values are &mdash; a few very high ones with a long tail of low ones, or the reverse.</p>
    <p>Only the fifth, <b>Moran&rsquo;s I</b>, is about <em>where</em>. The cleanest way to see that split is to shuffle a cell: move every unit to a random position, which leaves the collection of values exactly as it was and throws away every address.</p>
    {shuffle}
    <p>That is also what <b>Moran&rsquo;s I</b> means, if the name is new. It is a single number for one grid, and it asks one question: does a unit resemble the units next to it? +1 means always, 0 means no relation at all, &minus;1 means always the opposite. It knows nothing about the values themselves, only about how they are arranged. Which is why adding ten to every number in a grid, or doubling them all, leaves it unchanged &mdash; and why a shuffle destroys it.</p>
    {morans}
    <p>One thing has to be pinned down before that number means anything: which units count as &ldquo;next to&rdquo; a given unit. There are two conventional answers, borrowed from chess. <b>Rook</b> counts the four units sharing an edge. <b>Queen</b> counts those four plus the four corners, eight in all.</p>
    {neighbours}
    <p>Two cautions follow. First, the choice between rook and queen is not cosmetic on fine patterns: at a 1.17 m unit, a row of houses separated by 3 m alleys is close to a chequerboard, and the two rules would report it very differently. Whichever is used has to be stated.</p>
    <p>Second, a high Moran&rsquo;s I does not by itself mean a channel is <b>redundant</b> &mdash; that is, that a coarser grid would carry the same information. The footprint grid reads 0.80 to 0.93 and is full of detail, because buildings are blobs several units across. Smooth and redundant are different properties, and the second one can be measured directly: shrink the grid, blow it back up, and see how much of the original survives.</p>
    {curve}
    <p>Those two curves settle where the 1.17 m unit earns its cost. The footprint channel genuinely needs it: a third of that channel is gone by the pilot&rsquo;s 9.4 m unit, and that missing third is the alley-and-rowhouse grain the resolution was chosen for in the first place. The 100 m density channel does not: it could be computed on a 32&nbsp;&times;&nbsp;32 grid and lose half a percent.</p>
    <p>That is not an argument against the density channel, which measures something real. It is an argument that its cost should be counted against what it adds, since at 1024 px it is being sampled about eighty times finer than it has anything to say. The indicator specification table lists it as an open question.</p>
    {resampling}
    <p>One practical note belongs here, because it is a way the descriptor can be fooled by something that was never in the data. Most of these channels are not worked out at 1.17 m. A channel like &ldquo;how many shops within 300 m&rdquo; is calculated on a coarse grid &mdash; its meaning lives at 300 m, so there is no point solving it a million times per cell &mdash; and only then stretched onto the shared fine grid. <b>How</b> it is stretched decides whether the result carries features nobody measured.</p>
    {sparsity}
    <p>The last figure is about a number that is easy to over-read. A channel that is 87&nbsp;% zeros sounds like a channel that is mostly missing, and sometimes it is. But the zero fraction only describes the pile of values, not where they sit. Two channels with identical zero fractions can be completely different objects: one empty in the shape of a street network and a river, the other empty at random. Only the picture tells them apart, which is why sparsity belongs in the marginals check as a map and not only as a percentage.</p>
    <p>And one thing this whole section cannot do, which is why there is a second measurement beside it. Squashing six million numbers into forty-five loses almost everything: two genuinely different neighbourhoods can land on nearly the same forty-five. The descriptor compares what cells are like. Whether one generated cell is a <em>copy</em> of one training cell is a different question, asked by comparing the pixels themselves.</p>
  </section>
'''

EXP_HEAD = f'<title>Cells, Channels, Units</title>\n<style>{CSS}</style>\n{FONTS}\n'
EXP_BODY = (open(os.path.join(HERE, "explainer_body.html"), encoding="utf-8").read()
            .replace("{{DESCRIPTOR}}", descriptor_section()).replace("{{GLOSSARY}}", GLOSS_HTML).replace("{{MAP_URL}}", MAP_URL).replace("{{SCATTER}}", scatter_svg()))
write(os.path.join(ART_DIR, "cells-channels-units.html"), EXP_HEAD + EXP_BODY)
write(os.path.join(REPO_OUT, "cells_channels_units.html"),
      standalone(EXP_HEAD, EXP_BODY.replace(MAP_URL, "pipeline_v4_map.html")))

# ---------------------------------------------------------------- draw.io
W, GX, CX = 300, 40, 760
BAND_X, BAND_W = 40, 1440
COL = {"action": ("#1F6F8B", "#FFFFFF"), "gate": ("#B7791F", "#FBF2E0"), "future": ("#7F8D8F", "#F4F6F3")}

def est_h(n):
    t_lines = math.ceil(len(n["title"]) / 36)
    b_lines = sum(math.ceil(len(b) / 44) for b in n["why"])
    return int((10 + 12 + t_lines * 16 + 4 + 12 + b_lines * 14 + len(n["why"]) * 3 + 12) * 1.08)

def dio_label(code, n):
    items = "".join(f"<li>{html.escape(b)}</li>" for b in n["why"])
    return (f'<font style="font-size:9px;color:#5C6B6D;">{code} · {KIND_LABEL[n["kind"]].upper()}</font><br>'
            f'<b style="font-size:12px;">{html.escape(n["title"])}</b><br>'
            f'<font style="font-size:9px;color:#5C6B6D;">WHY</font><ul style="margin:2px 0 0 14px;padding:0;">{items}</ul>')

mxfile = ET.Element("mxfile", host="app.diagrams.net", compressed="false")
diagram = ET.SubElement(mxfile, "diagram", name="Pipeline v4", id="pipeline-v4")
model = ET.SubElement(diagram, "mxGraphModel", dx="1400", dy="900", grid="1", gridSize="10", guides="1", tooltips="1",
                      connect="1", arrows="1", fold="1", page="0", pageScale="1", pageWidth="1600", pageHeight="1200", math="0", shadow="0")
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", id="0"); ET.SubElement(root, "mxCell", id="1", parent="0")

def vertex(cid, value, style, x, y, w, h):
    c = ET.SubElement(root, "mxCell", id=cid, value=value, style=style, vertex="1", parent="1")
    ET.SubElement(c, "mxGeometry", x=str(int(x)), y=str(int(y)), width=str(int(w)), height=str(int(h)), **{"as": "geometry"})

def edge(eid, src, tgt):
    style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;"
             "strokeColor=#7C8B8D;strokeWidth=1.5;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;")
    c = ET.SubElement(root, "mxCell", id=eid, style=style, edge="1", parent="1", source=src, target=tgt)
    ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})

y = 40
vertex("title", '<b style="font-size:22px;">Latent City Pipeline Map</b><br><font style="font-size:11px;color:#5C6B6D;">DDPM urban vitality · pipeline v4 · 9 Sep 2026 · side-by-side nodes run in parallel · teal = action, amber = checkpoint, dashed = later</font>',
       "text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;strokeColor=none;fillColor=none;", BAND_X, y, BAND_W, 60)
y += 80
band_specs, node_specs, row_seq = [], [], []
for s in STAGES:
    band_y0 = y; y += 44
    for row in s["rows"]:
        n = len(row); total = n * W + (n - 1) * GX; x0 = CX - total / 2
        h = max(est_h(N[c]) for c in row)
        node_specs += [(c, x0 + i * (W + GX), y, W, h) for i, c in enumerate(row)]
        row_seq.append(row); y += h + 50
    band_specs.append((s, band_y0, y - 20 - band_y0)); y += 20
for s, by, bh in band_specs:
    vertex(f"band{s['code']}", f'<b style="font-size:12px;">STAGE {s["code"]} · {html.escape(s["title"])}</b> <font style="font-size:10px;color:#5C6B6D;">· {html.escape(s["ordinal"])}</font>',
           "rounded=0;whiteSpace=wrap;html=1;fillColor=#EAEFE9;strokeColor=none;align=left;verticalAlign=top;spacingLeft=12;spacingTop=8;fontColor=#1F6F8B;", BAND_X, by, BAND_W, bh)
for c, x, yy, w, h in node_specs:
    n = N[c]; stroke, fill = COL[n["kind"]]; dashed = "dashed=1;" if n["kind"] == "future" else ""
    vertex(c, dio_label(c, n), f"rounded=1;arcSize=4;whiteSpace=wrap;html=1;align=left;verticalAlign=top;spacing=6;spacingLeft=8;spacingRight=8;"
           f"fillColor={fill};strokeColor={stroke};strokeWidth=1.5;fontSize=11;fontColor=#1B2426;fontFamily=Helvetica;{dashed}", x, yy, w, h)
k = 0
for prev, nxt in zip(row_seq, row_seq[1:]):
    for a in prev:
        for b in nxt: edge(f"e{k}", a, b); k += 1
write(os.path.join(REPO_OUT, "pipeline_v4.drawio"), '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(mxfile, encoding="unicode"))
print(f"map: {n_nodes} nodes, {n_gates} checkpoints, {k} edges; glossary {len(GLOSS)} terms; explainer written")
