# Publication roadmap

Three lanes run at once: the GPU, this machine, and the machine with the PDFs.
Each step lists what every lane does. A step ends when its gate is passed.

## Step 0. Close out (now)

- Other session finishes the audit. Commit, push.
- Share the new references with this session.

## Step 1. Build and train (days, GPU-bound)

| GPU | this machine | PDF machine |
|---|---|---|
| A1–A3, B-stage dataset build | D2: F_mem noise floor on an untrained model | Tier-1 bibliography: the 7 claims the audit flagged, then the new references at full text |
| D3: one short run until F_mem clears the floor, read off tau_mem | E-stage analysis code (E1–E4) ready before the first checkpoint lands | For each paper: what we cite it for, verified passage with page, what it does not claim, datasets and resolutions used, threat or connection |
| Mutant runs, budget set from D3 | Fix Section 3 of the report in the audit's order: F_mem definition, synergy rebuilt on E3c and B10b, shared POI source disclosed, channel assignment, CelebA, updates per epoch, attention clause | |

Gate: every mutant has checkpoints through the memorization onset. Tier-1 entries done.

## Step 2. Results and write-up (weeks 1–2 after training)

| GPU | this machine | PDF machine |
|---|---|---|
| E1 sampling at every checkpoint | E2–E4, F1: window per mutant, invariants vs correlationist baseline | Sections 4 and 5, then the abstract, claiming only what ran |
| | Figures from CSVs | Related-work paragraph updated from Tier-1 reading |

Gate: co-authors and supervisor sign off.

## Step 3. arXiv v1

- Primary cs.LG, cross-list cs.CY or physics.soc-ph. Get an endorser before the day.
- Frame: methods and first results.

## Step 4. v1 to v2 (parallel)

| GPU | this machine | PDF machine |
|---|---|---|
| E5 Jacobian rank, if there is time | Decide the CS paper from the E4 numbers: is tau_mem linear in n across the mutants? | Tier-2 reading: abstract triage, promote to full read only when a claim depends on it. New connections. |
| | | v2 text |

- Post v2.

## Step 5. Journal

- CEUS first, Environment and Planning B second. Check the sharing policy at submission.
- CS workshop paper only if the tau_mem-vs-n result is clean, and only after the urban submission is in.

## Rules that hold across steps

- The v3 pilot numbers are not preprint evidence.
- Nothing public before Step 2's gate.
- Bibliography entries live in `references/annotated/`, one file per citation key, so both machines use the same ones.

## Instructions for the reading session (the machine with the PDFs)

### What the two tiers are

Tier 1 is a paper one of our sentences depends on: the report attributes a
specific finding, number, quote, dataset or method to it, or we inherit a method
from it. If we misquote it, a reviewer catches it. Tier 1 gets a close read.

Tier 2 is everything else: background, foundational theory, standard references
cited only for a definition, and any paper that merely sits near our topic.
Tier 2 gets the abstract and one line.

### Four rules

1. **The audited claims come first and require a close read.** The audit in
   `ism_methodology_reflection.md` flagged seven claims it could not verify from
   the repo. Those papers are Tier 1 before anything else is opened:
   Kadkhodaie 2024; P. Wang 2025 (which datasets, which resolutions);
   Zhang 2024 (CIFAR-10 only?); Cole & Lu 2024 (linear in intrinsic dimension);
   Williams 2019 (ghost cities); L. He & Aliaga 2023 and 2024 (the 75% figure);
   F. Jiang 2024 (120 papers, "no analytical or epistemic uses").
2. **New papers start in Tier 2.** A new paper moves to Tier 1 only if its
   abstract shows a strong connection: it does something we claim is new, or it
   directly supports or contradicts a sentence we make. Otherwise it stays in
   Tier 2 with its one line. Most new papers will stay there. That is the point.
3. **Not every paper in the current draft needs a close read.** Jacobs, Whyte,
   Lynch, Maas, Montgomery, Landry, Chapelle, Vapnik, Ho 2020, Song 2021,
   Karras 2022, Rombach 2022, Dhariwal & Nichol 2021, Heusel 2018 are cited for
   what they are, not for a finding we lean on. They are Tier 2. The same goes
   for the urban papers cited only as examples of a data source or a method
   family (Kim 2020; Liu, Zhang & Long 2019; H. Liu 2022; Scepanovic 2021;
   Chen 2021; Gómez-Varo 2022; M. Wang 2025; Zeng 2018; Xia 2020; Qin 2025;
   Mouratidis 2020; Dogan 2024; Carlini 2023; Somepalli 2022; George 2025).
   Move one up only if a sentence turns out to rest on it.
4. **Every referenced paper gets an entry, but the depth differs.** Tier 1 gets a
   full entry with the five fields below. Tier 2 gets one line: what it does, and
   whether it touches our paper.

### Tier 1 in the current draft, beyond the seven audited claims

Bonnaire 2025 (F_mem, the window, tau_mem linear in n); Biroli 2024; Yoon 2023
(dummy data pushes past memorization); Y.-H. Wu 2025 (check: the report cites it
for "features constant in the window are the signal", which may be a loose
attribution); Rahaman 2019 and B. Wang & Pehlevan 2025 (spectral bias); S. Li 2024;
Doan & Zhang 2025 (several direct quotes); Q. Li 2021, X. Li 2022 and Pan 2021
(the three conflation examples make specific claims about their methods);
Y. Jiang 2023 (the POI quote); Tan 2026 (FAR 0.25 threshold); C. Wu 2023
("erroneous conclusions"); X. Liu 2024 (RGB fusion precedent); S.-Y. Huang 2025
(the connectionist frame); Q. Wang 2025, M. He 2026, Zhuang 2024 and Zhou 2024
(the "black box generators" claim rests on what they do and do not analyse).

### Entry format

Tier 1, one file each at `references/annotated/tier1/<author-year>.md`:

- **Cited for**: the report section and the sentence, copied verbatim.
- **What it says**: the supporting passage, with page or section number.
- **What it does not say**: where our sentence goes further than the paper.
- **Datasets and resolutions**: what they actually trained or tested on.
- **Threat or connection**: does it overlap our contribution, support it, or
  suggest a link we have not used.

Tier 2, one file for all at `references/annotated/tier2.md`, one line per paper:

`Author (year). What it does. Touches our paper: no / yes, how.`

### Order of work

Seven audited claims, then the rest of Tier 1 from the current draft, then the
new references (abstracts first, promote as rule 2 says), then Tier 2 lines for
the remainder. Stop at any point and the file is still useful.
