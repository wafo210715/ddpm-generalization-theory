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

This is not a wording slip; the two ratios measure different things. The
report's version compares a sample against its own siblings, which would
conflate memorization with mode collapse — a model producing many similar
samples would score as memorizing even if none of them resembled the training
data. Our implementation in `evaluation/gap_ratio.py` follows the reference.
**Section 3.7.1 needs correcting before anything else in the report.**

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
