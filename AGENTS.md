# AGENTS.md

How we work in this workspace. Read this first, then the files it points to.
Maintained together with the user: when a way of working is agreed in conversation, add it here with the date; when one changes, keep the old line struck through or quoted with the date. Started 4 Oct 2026.

The workspace rules (1–7: answer first, ask before creating files, measure on this machine, no commit without asking, what `discussion/` may hold) are in `CLAUDE.md` § Workspace rules and take precedence. The rest of `CLAUDE.md` describes the code and is partly out of date.

## Index

| File | What it is | Who edits |
|---|---|---|
| `vitality_stance.md` | The user's own views on vitality, with dates and quotes. The baseline for calibrating against papers. § 3 holds the research logic (linear baseline → diffusion model → deductive loop-closing into CLD / SD) and why the per-paper CLDs exist (2026-10-04) | User confirms every change |
| `future_research.md` | The user's future-research stance: technical route (represent / understand / reason about space-time), problems to solve with their data needs, method-to-task table, open questions. Points to the Ch5 world-model notes in wafo-base (2026-10-04) | User confirms every change |
| `reading_method.md` | How to read the two literatures (vitality; diffusion generalization, later JEPA) | Agreed changes only |
| `discussion/hcmc_data_sources.html` | Ledger: data source in a paper read → its indicators → what HCMC can use → limits. Built-environment and human-centric tables | Add rows only for papers read |
| `discussion/vitality-cld/<Author-Year>.md` | Per-paper CLD extraction (template in `README.md` there). Committed only once the paper has been read together | Fix in place, quote old text |
| `discussion/vitality_indicators_review.md` | Factor tables (§4–5), anomaly types T1–T7, classification convention | Fix in place, quote old text |
| `discussion/indicator_specification.md` | Decided layer table (3 010 cells, 13 grids, 3 conditions), measured numbers | Fix in place, quote old text |
| `discussion/HANDOFF_diffusion_reading.md` | Threats T1–T3, supervisor questions, missing diffusion papers | |
| `references/annotated_bibliography.md` | Annotated bibliography | **User writes it**; agent only smooths and fact-checks, writes the file when told |
| `data-processing/` | Scripts and data-verification reports | |
| `~/Documents/wafo-base/literature/papers/` | Papers, pdf + txt (cite txt line numbers) | Read only |
| `archive/` | Superseded material, including the old `TODO.md` | |

## Reading sessions

The skill `paper-reading-session` holds the step-by-step procedure. In short:

1. One paper per turn. Read the full txt. Report briefly in plain Chinese: what the paper asks and finds, the four vitality points or the diffusion stance (see `reading_method.md`), where bias enters, and 1–3 new ideas.
2. Compare with the existing extraction and repo docs; fix errors in place, quoting the old text and date.
3. Add the paper's data sources to the ledger; tag indicators (候选 / 做不到), not sources.
4. Help with the user's annotated bibliography only when asked: ≤ 300 words (Chinese ≈ 600 characters), spelled out enough to be read a month later.
5. Calibrate: does the paper try to move anything in `vitality_stance.md`? Ask, do not edit the stance file on the user's behalf.

## How the agent should behave

- **Where to look first.** Literature: `~/Documents/wafo-base/literature/` (papers pdf + txt + bib, `_index.md`) and `~/Documents/wafo-base/research-vault/` (reading notes in `2-fleeting/papers/`, project notes in `1-project/`, diaries). Other conversations: use the `read_task` tool; it also reaches tasks outside this workspace by title or id (e.g. the JEPA sessions). Do not reconstruct from memory what one of these already holds (2026-10-04).
- **Increments, not bulk.** Output only what this session reviewed, so the user can see where the increment happened. Automated bulk output is hard to review and produces no new ideas (2026-09-30).
- **Quote, do not strengthen.** Claims about what an author said carry the original sentence and line number. Do not turn a paraphrase into a causal claim the text does not make. Keep "the authors wrote" apart from "our inference" and "the extractor's chaining" (2026-10-02, after Amer, Chen, Cai).
- **Say when unsure; verify.** External facts (datasets, methods) are checked on a page actually opened. Data availability: separate *exists* from *obtainable*; use a sub-session for such checks (2026-10-02).
- **Correct proactively.** When a document is found wrong while working, fix it without being asked; errors piling up unnoticed is the main risk (2026-09-30).
- **Concrete first.** Explain with an example or a small inline diagram; define jargon at first use. A method that needs pictures (e.g. SOM) gets a small interactive HTML in the session workbench, not in the repo (2026-10-02).
- **Do not decide for the user.** Open questions go to the user; the stance file lists them in § 5.
- **Commits:** only when asked; papers' extraction files only once read; unread ones stay untouched (2026-10-02).

## How the user learns (observed 30 Sep – 4 Oct 2026)

Guesses before reading the answer, by analogy with own past work; reads for the research, not to reconstruct the author; asks for the original sentence; needs a concrete example or picture before an abstraction lands; wants visible increments; links papers into lines of argument (symbolic reasoning, data is never raw, data-side vs model-side bias); writes the annotated bibliography and asks the agent to smooth it.

## Known out-of-date files

- `README.md` (root): describes the v3 pilot (4 182 cells, 128×128×3 RGB, 157.8M U-Net). Not yet updated.
- `CLAUDE.md` below the workspace rules: platform and pipeline descriptions predate the 3 010-cell design.
