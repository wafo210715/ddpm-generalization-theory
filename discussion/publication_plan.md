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
