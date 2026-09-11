# The April methodology, read against the v4 plan

A reflection, not an edit. The 0417 ISM report is not changed by this file. What
follows is what I would want to know before revising its Section 3, sorted into
what is now wrong, what survives and got stronger, and what the report does
better than the pipeline map does.

Written 10 Sep 2026, against `ISM report - Li Guanhong - 0417.docx`.

---

## The four HTML pages, and a warning about them

| page | what it is for | size |
|---|---|---:|
| `pipeline_v4_map.html` | the plan itself — 35 nodes, 5 checkpoints, the order and the reasons | 46 kB |
| `cells_channels_units.html` | the companion — eight questions answered with pictures, now including what the descriptor sees | 1.0 MB |
| `data_in_data_out.html` | the data path — what the model is handed and what it gives back | 2.2 MB |
| `stack_explorer.html` | interactive — break a cell on purpose and watch which numbers move | 1.0 MB |

The first two come from one generator and cannot drift apart. The last two have
their own generators, four in `discussion/pipeline_map/` altogether.

**The warning.** Sections 3 and 4 of the companion overlap heavily with sections
1, 2 and 4 of *Data in, data out*. Both explain what a channel is and what
unpacking does. That is tolerable now and becomes a liability the moment A1
fixes the real channel list, because a number will be updated in one and not the
other. Before the indicator table lands, one of the two should own that
explanation and the other should link to it.

---

## What is now wrong in Section 3

### 1. The F_mem definition is incorrect

Section 3.7.1 says:

> for each generated sample, its nearest neighbor in the training set (NN_train)
> and its nearest neighbor among other generated samples (NN_gen) are
> identified... The gap ratio is computed as d(sample, NN_train) / d(sample,
> NN_gen).

That is not what Bonnaire et al. do, and not what the code does. In their
`compute_fmem.py` the distance matrix is built against the **training set only**,
and the ratio is the **nearest** training image over the **second-nearest**
training image:

```python
dist = torch.norm(s - X, dim=2, p=2)      # X is the training set
knn  = dist.topk(k, dim=1, largest=False)
gap_ratio = distances_tensor_all[:, 0] / distances_tensor_all[:, 1]
```

This is not a wording slip. The denominator is doing different work in each
version, and the difference is not academic.

In the correct version, the second-nearest **training** distance is a property
of the training set's local spacing around that point. It does not depend on the
model at all. The ratio therefore asks one clean question: is this sample
sitting on one particular training cell, or between several?

In the report's version, the denominator is the distance to the nearest **other
generated** sample, which depends entirely on how varied the model's own output
is. The ratio mixes two independent properties — closeness to the training data,
and diversity of the samples — and cannot separate them afterwards.

The consequence is that the report's formula is blind to the worst case it
exists to detect. Both were run on four scenarios where the right answer is
known (40 training points in 8 dimensions, threshold 1/3):

| the model produces | truth | report's F_mem | correct F_mem |
|---|---|---:|---:|
| **the same training cell, 20 times** | memorised | **0.00** | **1.00** |
| 20 different training cells | memorised | 1.00 | 1.00 |
| 20 diverse novel cells | not memorised | 0.00 | 0.00 |
| 20 near-identical novel cells | not memorised | 0.00 | 0.00 |

Row one is the canonical memorization failure, and the report's formula scores
it zero — its median gap ratio is 1.07, nowhere near the 1/3 threshold. When
every sample is a copy of the same training cell, the distance to the nearest
*other generated sample* is also near zero, so the ratio is roughly nought over
nought and lands wherever the noise puts it. The numerator was right; the
denominator cancelled it.

The formula happens to work when the model copies many *different* training
cells (row two), which may be why the error went unnoticed. Our implementation
in `evaluation/gap_ratio.py` follows the reference.
**Section 3.7.1 needs correcting before anything else in the report.**

A note on terms, since the earlier draft of this file used one loosely.
*Mode collapse* is a model producing samples that are all alike — it has
collapsed onto one narrow region of the distribution instead of covering it.
That is a different failure from memorization, which is producing copies of
individual training examples. F_mem is only a measurement made at evaluation
and causes neither. The problem is that the report's denominator makes the
measurement *sensitive* to sample diversity, so a collapsed model hides its own
memorization from the metric.

### 2. The synergy analysis performs the operation the paper criticises

This is the most consequential item, because Section 3.7.4 is billed as "the
core epistemic probe" and as the thing that separates this work from
correlationist studies.

> For each generated image, the spatial mean of each channel is computed...
> yielding a three-dimensional point (B_mean, G_mean, R_mean). Across all
> generated images at a checkpoint, pairwise correlations between these spatial
> means are computed...

Three numbers per cell, correlated across cells. That is a correlation between
aggregate indicators — structurally the same move Section 2.2 objects to, run on
generated data rather than real data. It cannot see anything *inside* a cell. It
cannot distinguish a cell where the dense units sit on the accessible units from
one where they sit apart, because both have the same two means.

Measured on a real HCMC tile, the difference is not hypothetical: the same two
grids score +0.45, −0.00 or −0.43 on bivariate Moran's I depending only on how
one of them is arranged, while every cell-level mean and every histogram stays
identical.

The instinct in Section 2.4 is exactly right — *"the object of manifold learning
is not per-channel fidelity... it is the synergies between channels"*. The
instrument does not deliver it. Two nodes in the v4 plan now do:

- **E3c** — bivariate Moran's I between two grids inside one cell, plus
  co-location of unit classes and cross-correlograms by distance.
- **B10b** — the joint pattern of the six channels at unit level: which
  combinations of six numbers the city actually contains, measured on real cells
  and compared against generated ones.

Section 3.7.4 should be rebuilt on those. The classification scheme it proposes
(coupled / X-driven / fragmented, on |r| thresholds) goes with it — E4 already
notes that the |r| > 0.5 cut moved the boundary it was supposed to detect.

### 3. RGB fusion is presented as a principled choice; it was a container

Section 3.2.2 devotes a full subsection to RGB channel synthesis as a deliberate
epistemological decision, and Section 2.4 argues for it at length.

The argument underneath survives intact and is the strongest thing in the
chapter: refuse to pre-compute a composite score, hand the model raw unweighted
dimensions, make it discover the mixing logic. What does not survive is
**three**. Three was the number of channels a PNG holds. The U-Net reads its
channel count from a config; one or six or fourteen all work. Presenting three
as a considered choice invites a reviewer to ask why not four, and there is no
answer.

Rewrite so the epistemological claim is about *withholding weighting*, and the
channel count follows from the indicator table rather than from a file format.

### 4. Every number in Section 3.2 has moved

| | April | now |
|---|---|---|
| cell | 1 km fishnet, 8 436 cells | zoom-15 tile, 1 201 m, 1 881 tiles |
| non-empty | 731 | 617 with any building; 139 with ≥ 100 |
| calculation unit | 50 m, 20 × 20 per cell | 1.17 m, 1024 × 1024 |
| image | 128 × 128 RGB PNG | multi-channel tensor, count open |
| normalization | global 95th-percentile clip to [0,255] | per-channel transform + standardization, parameters stored |
| hardware | RTX 5090, 32 GB | RTX PRO 6000, 96 GB |

The dataset-size row matters most. 617 usable tiles is small, and generalization
theory has the memorization onset growing with sample count, so the window this
whole study is built to observe may be narrower than the report assumes.
Checkpoint **D3** now exists to measure it on one short run before the budget is
committed.

### 5. One sentence in 3.2.1 is the root of the 50 m unit, and it is a conflation

> a calculation unit that is too large blurs spatial detail, while one that is
> too small produces sparse attribute values.

The first half is right. The second half is not. A finer sampling unit does not
make an indicator sparser — it reads the same ring-based field at more points.
The pilot's 97.5 % zeros came from the **150 m buffer and the 50 m ring**, not
from the unit size. Confusing the two is what put the unit at 50 m and flattened
the building channel to Moran's I ≈ 0.

The distinction the report draws two sentences earlier — cell size versus
calculation unit — is the right one and is the direct ancestor of B2's
measurement scale versus sampling unit. It just needs the second half of that
sentence removed.

### 6. FID should go

Section 3.7.3 already hedges it into near-uselessness. With multi-channel
tensors there is no pretrained perceptual network that applies at all. Drop it
and say why in one line; the hedging paragraph is longer than the deletion would
be.

### 7. The Moran's I caveat is now too pessimistic

> Moran's I operates at the intra-cell pixel level (128 × 128), measuring
> clustering within a single 1 km cell. The urban spatial structure that
> ultimately matters... operates at a larger scale... a supplementary coherence
> signal rather than primary evidence.

True at a 50 m unit, where intra-cell texture was an interpolation artefact. At
1.17 m, intra-cell structure **is** urban form — alleys, rowhouse grain, block
interiors. Measured: the footprint channel reads 0.80 to 0.93 on real tiles
against the pilot's near-zero. Moran's I is promoted from supplementary to
primary, and the caveat should be replaced by the resolution argument.

---

## What survives, and got stronger

Worth stating, because the list above is one-directional and the chapter is not
in bad shape.

- **The supply/demand critique and the correlationist/connectionist frame.**
  Untouched. B10b and E3c turn the connectionist claim from a position into a
  measurement, which is what a reviewer will want.
- **Cell size versus calculation unit.** The report got there first. B2 sharpens
  it into measurement scale versus sampling unit and adds the floor argument.
- **"Not per-channel fidelity but the synergies between channels."** Correct
  instinct, wrong instrument. Keep the sentence, change what implements it.
- **Logarithmic checkpointing**, and its reasoning about early dynamics.
  Unchanged.
- **Discrete timesteps for comparability with Bonnaire et al.** Unchanged, and
  now load-bearing, since we inherit their metric exactly.
- **No augmentation, for observational honesty.** Still a live decision, and
  sharper now: with 617 tiles, Yoon et al.'s argument for inflating the
  effective dataset applies more strongly than it did at 731.
- **The "structure-preserving compressor" framing.** Unaffected by any of this.

---

## What the report does better than the pipeline map

This is the part worth carrying in the other direction.

### It says what each module is *for* before saying what it *does*

Section 3.1 opens by mapping every module to the epistemic frame — *"the
evaluation module probes what the model has learned through four complementary
lenses, each answering a distinct question"*. The map's stage labels are bare
nouns: "Build the dataset", "Sample and analyze". A reader of the map knows the
order and not the purpose.

**Borrow:** one line per stage saying what question that stage answers.

### It says what a measurement *cannot* see

Section 3.7.2 contains this, unprompted:

> It evaluates each channel independently... A model could reproduce each
> marginal accurately while failing to capture how the channels co-vary.

That is the discipline this project has been reinventing in conversation ever
since — the descriptor cannot detect a copy, the gap ratio cannot say what the
data looks like, a histogram has no addresses in it, cell averages cannot see
arrangement. The map has no slot for any of it.

**Borrow, and this is the important one:** every measurement node in the map
gets a *cannot see* line. Had that slot existed in April, item 2 above would
have been caught when Section 3.7.4 was written, because the author would have
had to fill in "this cannot see —" and the answer was "anything inside a cell".

### It records rejected alternatives with reasons

Cosine removed for numerical instability and for not being used by either
reference. EDM skipped as continuous-time. That is exactly what a reviewer
probes, and the map records decisions without the roads not taken.

**Borrow:** a rejected-alternatives line on nodes where a real fork existed —
pixel space over latent, raster over direct encoding, discrete over continuous.

### It is numbered finely enough to cite

3.7.1 through 3.7.4 can each be referenced from elsewhere in the paper. The map
has codes and uses them; the companion pages mention them unevenly.

---

## What to merge, concretely

1. **The map's Stage B replaces Section 3.2.** The tiling, the unit argument,
   the natural-scale-then-resample rule and the transform-and-store step are all
   more developed than the April text. Keep the report's prose framing; replace
   the content.
2. **Section 3.1's framing paragraph becomes the map's missing per-stage
   purpose line.** It is already written; it just lives in the wrong document.
3. **Section 3.7 restructures around four measurements, not four metrics:**
   F_mem with the corrected definition; the descriptor comparison; the joint
   channel pattern (B10b, E3c); irreducible characteristics across mutants. FID
   deleted.
4. **The indicator specification table becomes a table in the report**, not
   prose. It is the thing that makes the channel count defensible, and the April
   text has no equivalent.

---

## Open, and mine to say rather than decide

- Whether the report is revised now or after A1 lands. Most of Section 3.2's
  numbers cannot be fixed until the indicator list exists, so a revision now
  would be partial.
- Whether Section 2.4's RGB argument is rewritten or cut. It is good writing
  making a claim that has been superseded in its particulars but not in its
  substance, which is the hardest kind of paragraph to revise.

---

# Audit of the rest of the report

Same method as the F_mem finding above: check what can be checked against the
code, the reference repositories in this tree, and the report's own other
sections. Claims that cannot be checked are listed as such rather than waved
through.

## What verifies

Worth recording, because most of the report holds up and an audit that only
lists faults is misleading.

| claim | where | verdict |
|---|---|---|
| U-Net at ch=128 is ~157.8M parameters | 3.3 | **exact** — 157.84M |
| earlier ch=32 config was 9.88M | 3.3 | **exact** — 9.88M |
| gradient clipping at max norm 1.0 | 3.5 | **implemented**, `train.py:283` |
| four schedules implemented; cosine removed | 3.4 | **correct** — the enum has linear, fixed, vp, ve |
| 95th-percentile normalization | 3.2.2 | **correct** — `NORM_PERCENTILE = 95` |
| 50 m unit, 150 m buffer, 20 × 20, 128 px | 3.2 | **all correct** |
| gravity β = 0.3, 3 000 m cutoff | 3.2.1 | **correct** |
| 8 436 grid cells | 3.2.3 | **reproduced exactly** in the v4 rebuild |
| 584 + 73 + 74 = 731 at 80/10/10 | 3.2.3 | **arithmetic consistent** |
| DDIM 50-step is ~20× faster than DDPM 1000 | 3.6 | consistent |

## Errors in Sections 1 and 2

### 1. The channel assignment in 2.4 contradicts 3.2 and the code

Section 2.4:

> building density (physical mass) is encoded into the **Red** channel, road
> accessibility (connectivity infrastructure) into the **Green** channel, and
> Point of Interest diversity (functional intensity) into the **Blue** channel.

Section 3.2.2, and `datasets/dataset.py`, both say the opposite:
**R = accessibility, G = diversity, B = density.** The Introduction's
"(B), (G), (R)" ordering agrees with 3.2. Section 2.4 is the only place that
disagrees and is simply wrong.

### 2. CelebA is 32 × 32 grayscale, not 64 × 64, and it is Bonnaire's not Wang's

The Introduction:

> P. Wang et al. (2025) demonstrate the subspace clustering equivalence on
> CIFAR-10 (32×32 pixels) and CelebA (64×64 pixels)

Section 2.3 says something different: CelebA at "32×32 grayscale, downsampled",
attributed to **Bonnaire et al.** The Experiments section says CelebA is 32 × 32
again. So the report states this three times and disagrees with itself twice.

The reference code settles it. `Why-Diffusion-Models-Don-t-Memorize-main/
Experiments/src/Utils/cfg.py` sets `config.IMG_SHAPE = (1, 32, 32)` — one
channel, 32 × 32. Section 2.3 is right; the Introduction is wrong on both the
resolution and the attribution.

### 3. The Introduction understates its own references, weakening its gap

> Bonnaire et al. (2025) validate the generalization window on synthetic
> mixture-of-Gaussians distributions

Section 2.3 says they also used CelebA. The Introduction's motivating gap —
that the evidence "derives almost exclusively from controlled computational
settings" — is built on a description that the paper's own literature review
contradicts. A reviewer who knows the paper will notice, and the gap is real
without the overstatement: 32 × 32 grayscale faces are still nothing like a
1 024 px six-channel urban raster. Say that instead.

### 4. A stray quotation mark

Section 2.2: `" prevailing methods "operate under the assumptions of...`.
Doubled and unbalanced quoting. Cosmetic, but it is in a quoted passage.

## Further errors in Section 3

These are in addition to the F_mem and synergy findings above.

### 5. The R channel is not what 3.2.1 says it is

> Road accessibility (Red channel): Gravity-model distance decay **to the road
> network**, with decay parameter β = 0.3/km and a 3,000 m cutoff.

The code computes something else. `compute_accessibility(pois_utm, G, grid_utm)`
runs Dijkstra outward **from each POI** along the road graph and accumulates a
decayed score on the network nodes. It is gravity accessibility **to points of
interest, measured along the road network** — reachability of destinations, not
proximity to roads. Section 2.4 compounds it by glossing the channel as
"connectivity infrastructure", which fits the written description and not the
computation.

### 6. And that means two of the three channels are functions of the same layer

This is the most consequential item in Sections 1–3 that has not already been
raised.

- **G, POI diversity** — Shannon entropy of POI categories in a 150 m buffer.
- **R, accessibility** — gravity decay **from POIs** along the road network.

Both are computed from the same point layer. The road network enters R only as
the metric along which distance is measured; remove every POI and R is zero
everywhere.

The pilot measured a 0.88 correlation between these two channels and the report
treats that as a finding about the city. At least part of it is a property of
the encoding: two functions of one dataset, correlated.

Section 2.1 criticises Q. Li et al. (2021) in these exact terms — constructing a
POI-based dependent variable and regressing it on POI-based independent
variables, so that "the regression coefficient reflects spatial autocorrelation
of POI data". The same objection lands on the encoding, in weaker form. It does
not invalidate the approach, because the model is not being asked to regress one
channel on another. But the paper cannot claim the model discovered a
density–diversity–accessibility synergy when two of the three channels share a
source, and it must say so before a reviewer does.

The v4 indicator specification table is where this gets fixed: `poi_count`,
`poi_entropy` and `accessibility_gravity` all trace to the same POI layer and
the table's "source" column makes that visible at a glance, which is what the
column is for.

### 7. "Self-attention is disabled at all resolution levels" is not quite true

`DDPMUnet`'s `MiddleBlock` contains an `AttentionBlock` unconditionally. Passing
`is_attn=[False, False, False, False]` disables attention in the down and up
blocks only; the bottleneck always attends. The sentence should say so, and it
matters for v4: B2's argument for more levels at 1 024 px is precisely that the
bottleneck attention should see a small enough grid.

### 8. The updates-per-epoch figures are wrong

> 408 updates per epoch with bs=8 versus 102 with bs=32

With 584 training images: **73** and **18**. Both stated figures imply a dataset
of 3 264 images. The trade-off argument they support is sound; the numbers are
not.

## Claims and framing that need attention, not correction

### 9. The abstract describes a method the paper does not execute

> By curating minimal, intrinsic-dimension-aligned datasets via RGB channel
> synthesis, this research aims to isolate the robust, irreducible "essence"...

Section 3.2.3 says the pre-encoding analysis — PCA, intrinsic dimension,
typology — "has not yet been performed". There is therefore no
intrinsic-dimension-aligned curation anywhere in the paper. "aims to" is doing a
lot of work. Either the abstract drops the claim, or C4 and C6 of the v4 plan
have to run first.

### 10. The research question is not answerable by the analysis proposed

> Can a Denoising Diffusion Probabilistic Model learn the low-dimensional
> manifold of urban built environment from RGB-encoded geospatial data?

The three analytical dimensions listed under it are: whether a generalization
window emerges, what persists during it, and how generated distributions compare
to real ones. All three can come out positive while leaving the question
unanswered, because none of them distinguishes "learned the low-dimensional
manifold" from "learned some structure". A reviewer will ask what evidence would
tell those apart.

The v4 plan has two answers the April methodology does not contain:
**C5** — does each typology actually sit in a low-dimensional subspace — and
**E5** — the Jacobian rank of the trained denoiser, the model-side version of
the same test. Either would bear on the manifold claim directly. Without them,
the honest question is narrower: *does a DDPM trained on urban rasters exhibit a
generalization window, and what is stable within it?*

### 11. Three analytical dimensions, four evaluation lenses, never mapped

The Introduction promises three; Section 3.1 delivers four. Neither says how
they correspond.

## Unverifiable from here — flagged for a source check

None of these are claimed to be wrong. They cannot be checked against anything
in this repository, and each is load-bearing somewhere.

- Kadkhodaie et al. (2024) on "geometry-adaptive harmonic representations"
  (Introduction) versus "discover meaningful structural sub-organizations"
  (2.3). The second is a loose paraphrase of a specific technical result and
  should be checked against the paper.
- P. Wang et al. (2025) on which datasets, at which resolutions. The
  Introduction and 2.3 give different lists.
- Zhang et al. (2024) model reproducibility "primarily on CIFAR-10".
- Cole & Lu (2024) on sample complexity scaling linearly with intrinsic
  dimension — this underwrites the minimal-dataset argument in C6.
- Williams et al. (2019) on ghost cities.
- GlobalMapper / COHO "over 75% accuracy".
- F. Jiang et al. (2024), "120 generative urban design papers", "no discussion
  of analytical or epistemic uses" — an exclusivity claim, the kind most likely
  to be challenged.

## Sections 4 and 5

Both marked WIP and currently fragments. The one substantive line in Section 4
repeats the dataset-resolution facts a third time and disagrees with the
Introduction, so whichever version survives should be the one in 2.3.

## Order I would fix these in

1. F_mem definition (3.7.1) — the metric is wrong.
2. Synergy analysis (3.7.4) — the core probe measures the wrong thing.
3. Shared POI source across two channels (3.2.1, and disclose in 2.4).
4. Channel assignment in 2.4 — one sentence.
5. CelebA resolution and attribution in the Introduction — one sentence.
6. Updates-per-epoch numbers — two numbers.
7. Attention sentence in 3.3 — one clause.
8. Abstract's intrinsic-dimension claim, and the research question's scope.
