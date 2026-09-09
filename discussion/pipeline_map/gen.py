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
  {"code": "B", "title": "Build the dataset", "ordinal": "your third", "rows": [["B1"], ["B2"], ["B3", "B4", "B5"], ["B6"], ["B7"], ["B8", "B9"], ["B10"], ["B11"]]},
  {"code": "C", "title": "Pre-training analysis", "ordinal": "your fourth", "rows": [["C1", "C2"], ["C3"], ["C4", "C4b"], ["C5"], ["C6"], ["C7"]]},
  {"code": "D", "title": "Train", "ordinal": "your fifth", "rows": [["D1", "D2"]]},
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

EXP_HEAD = f'<title>Cells, Channels, Units</title>\n<style>{CSS}</style>\n{FONTS}\n'
EXP_BODY = (open(os.path.join(HERE, "explainer_body.html"), encoding="utf-8").read()
            .replace("{{GLOSSARY}}", GLOSS_HTML).replace("{{MAP_URL}}", MAP_URL).replace("{{SCATTER}}", scatter_svg()))
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
