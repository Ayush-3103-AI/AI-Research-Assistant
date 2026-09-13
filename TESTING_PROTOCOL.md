# Testing Protocol — live verification of the Trend Advisor and evidence-depth changes

**For:** a collaborator with a working Ollama + `qwen3.5:9b` setup, ideally on a machine with a real NVIDIA GPU.
**Branch:** `feature/trend-gap-advisor`
**Estimated time:** 4–8 hours of mostly unattended runs. Budget one evening.

---

## 1. Why you are being asked to run this

Two claims in this project are **built, unit-tested, and committed — but never once executed against a real language model.** Every automated test that covers them uses a fake `LanguageModel`. A fake model proves the wiring; it cannot prove a real model produces usable output.

They stayed unverified because the available machines had the wrong shape:

| Machine | Blocker |
|---|---|
| Original dev machine (RTX 4060) | Ollama was not running during the relevant work |
| Current machine | **No NVIDIA GPU** — Intel Iris Xe only, i5-1335U. A 9B model runs on CPU here |

You have the machine that can settle it. **Please do not skip the "what to record" sections** — the point of this exercise is a measured number, not a thumbs-up.

### The honesty rule for this protocol

This project's notes are deliberately strict about not overstating evidence, and this protocol inherits that:

- `gap_flagged: 0` means **not checked / none found**, not *no gaps exist*. Record which.
- If something fails, **record the failure**. A failed run is a result. Do not re-roll until it looks good and report only the good one.
- Report what you measured, never what the design predicted. Section 6 lists expectations **only** so you can see whether reality matched — not as a target to hit.

---

## 2. Prerequisites

```bash
ollama --version          # any recent version; 0.34.0 is known good
ollama list               # must show qwen3.5:9b
nvidia-smi                # record GPU name + VRAM; paste into the results template
python --version          # 3.11.x
```

If `qwen3.5:9b` is missing: `ollama pull qwen3.5:9b` (~6.6 GB).

Confirm the server answers:

```bash
curl http://localhost:11434/api/version
```

**Network is required.** Discovery, the advisor, and verification all hit live APIs (OpenAlex, Crossref, arXiv, Europe PMC). No API keys needed.

---

## 3. Setup

```bash
git clone https://github.com/TheIntruder007/AI-Research-Assistant.git
cd AI-Research-Assistant
git checkout feature/trend-gap-advisor

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements/base.txt
pip install -e .
```

Sanity-check before spending hours on it — the suite is fast and needs no model:

```bash
python -m pytest -q
```

Expected: **321 passed, 3 skipped**, roughly 2–3 minutes. If this does not pass, stop and report that instead; nothing below is meaningful on a broken checkout.

---

## 4. Protocol A — does the Trend Advisor's model half actually work?

### Background

The advisor is an optional Stage 0 that answers *"which topic should I even research?"* before the pipeline answers *"what does the literature say about it?"*. Its **data half is already verified live**: four real domain runs returned ranked topics with real DOIs from OpenAlex.

Its **model half has never run**: the gap-mining swarm, the converging chat, research-question drafting, and the pipeline hand-off. All four previous live runs recorded *"the local model was unavailable"*.

### A1. An arXiv domain

```bash
python scripts/smoke_test_trend_advisor.py CS_AI_ML
```

This drives the whole path through the CLI's own `_run_advisor` and **exits non-zero on any absorbed failure**, so a zero exit is meaningful. It finishes with a real four-stage pipeline run, so expect hours.

### A2. An OpenAlex-only domain — please do not skip this

```bash
python scripts/smoke_test_trend_advisor.py MECHANICAL
```

**Why both.** A code review found a structural issue affecting *only* the three arXiv domains (`CS_AI_ML`, `ECE`, `CLOUD_DEVOPS`):

> `merge_sources` interleaves arXiv papers into the example list, but full text is fetched only for the first two **OpenAlex** works. After merging, gap mining reads OpenAlex[0] (full text) + arXiv[0] (**abstract only**) and discards OpenAlex[1]'s already-fetched text. Because `MAX_PAPERS_PER_TOPIC == MIN_PAPERS_FLAGGING_A_GAP == 2`, `gap_signal` ends up hinging on an arXiv abstract — and abstracts rarely state future work.

So `gap_flagged: 0` from `CS_AI_ML` alone is **ambiguous**: the swarm could be broken, or this quirk could have starved it. `MECHANICAL` is OpenAlex-only, so both gap-mining slots get real full text. Running both tells us which. **This is the single most useful thing in this protocol.**

### What to record for Protocol A

For **each** of the two runs:

- [ ] Exit code (`echo $?` / `echo $LASTEXITCODE`)
- [ ] Did any stage log *"the local model was unavailable"*? (If yes, the model half still did not run — say so.)
- [ ] `gap_flagged` count, and **whether 0 means "checked, found none" or "never checked"**
- [ ] **Paste 3–5 actual gap statements verbatim.** This is the real question: are they specific enough to act on (*"no work evaluates X under Y conditions"*) or generic filler (*"more research is needed"*)? Only the text can answer this.
- [ ] Wall-clock time of the gap-mining fan-out, **per topic**
- [ ] Did the converging chat stay on task and converge to one topic?
- [ ] Was the drafted research question usable as-is, or would you have rewritten it?
- [ ] **Exactly one** run folder under `outputs/`? (Two is a known bug class — D-040 — and a regression if it reappears.)

Raw evidence lands in `outputs/<domain>_<run_id>/00_trend_advisor/result.json`. **Please attach both files.**

---

## 5. Protocol B — did forwarding Discussion text make papers longer?

### Background

A long investigation concluded the sole dominant limit on drafted paper length was **evidence depth**, not generation limits: Writing only ever received each paper's *abstract*. Across 4 real runs, **0 of 16 sections reached even half their own word budget** — the model ran out of supportable content, never out of room.

A change (D-039) now forwards Discovery's already-retrieved Discussion/Limitations/Future-work text through to Writing. The retrieval half is verified — it produced a 7,265-character evidence packet against the same paper's 176-character abstract-only packet.

**The effect on actual paper length has never been measured.** That is what you are measuring.

### B1. Cheap pre-check (no model, ~1 minute)

```bash
python scripts/smoke_test_evidence_depth.py
```

Confirms the richer packet is still retrieved. If this fails, stop — B2 cannot mean anything.

### B2. The measurement — 4 fresh topics

Use **four topics not run before** (avoid the project's existing topics). Pick real, current, well-published questions. Suggested shape:

```bash
python scripts/smoke_test_full_pipeline.py "How effective are graph neural networks for real-time traffic flow prediction in urban road networks?"
python scripts/smoke_test_full_pipeline.py "What are the durability limitations of geopolymer concrete in marine environments?"
python scripts/smoke_test_full_pipeline.py "How do solid-state electrolytes affect thermal runaway in lithium-ion batteries?"
python scripts/smoke_test_full_pipeline.py "What methods improve fatigue life prediction in additively manufactured titanium components?"
```

Run them **one at a time**, not in parallel — concurrent runs fight over VRAM and corrupt the timing numbers.

### The baseline you are comparing against

**Average draft length: 3,691 words** across 4 runs. Per-section, against each section's own maximum:

| Run | Background | Literature Review | Discussion | Limitations |
|---|---|---|---|---|
| A (IEEE/std) | 240/910 | 827/2240 | 369/1540 | 246/770 |
| B (IEEE/det) | 406/1365 | 1332/3360 | 699/2310 | 419/1155 |
| C (Springer/std) | 509/910 | 529/2240 | 497/1540 | 273/770 |
| D (Springer/det) | 581/1365 | 966/3360 | 585/2310 | 420/1155 |

**Zero of those 16 sections reached half its own maximum.** That is the number to beat, and the clearest single indicator: *did any section this time cross half its budget?*

### What to record for Protocol B

Per run:

- [ ] Total words in the rendered paper
- [ ] `draft_status` (`complete` / `partial` / `failed`)
- [ ] Per-section word count **and** that section's max (so it is comparable to the table above)
- [ ] `evidence_depth` per paper — `abstract_plus_discussion` or `abstract_only`
- [ ] **How many papers per corpus actually had retrievable open-access full text.** Historically a minority. If most papers still resolve to `abstract_only`, D-039 barely applied and a flat result says nothing about the idea — only about open-access coverage. **This determines whether the comparison is even valid.**
- [ ] Card-build time per paper — did it rise materially now the packet is larger?
- [ ] Total wall time and the per-stage `timings` block

Then the summary:

- [ ] **Average words across your 4 runs vs. the 3,691 baseline**
- [ ] **How many of your 16 sections reached half their own maximum, vs. 0 before**

---

## 6. What we expect — and why to ignore it while running

Listed **only** so the comparison is interesting, explicitly **not** a target:

- If D-039 worked, papers should get longer and some sections should finally clear half their budget.
- If open-access coverage is poor, most papers stay `abstract_only` and little changes — **an equally valid and useful result.**

A null result that is honestly measured is worth more than a good-looking number. If it did not work, that is the finding.

---

## 7. Known issues — do not re-report these as new

Found by code review on this branch, all **pre-existing**:

1. **`velocity.py:210`** — `_arxiv_candidate` catches `(httpx.HTTPError, ValueError)`, but `arxiv.search` raises `ET.ParseError` (a `SyntaxError`) on a non-XML 200 body. That escapes every guard and kills all of Stage 0 with a raw traceback instead of dropping just the secondary arXiv signal. **If an arXiv domain dies with a traceback, this is probably why.**
2. **`velocity.py:296`** — the per-topic `works_for_topic` gather sits outside the `HTTPError → VelocityUnavailable` guard. An OpenAlex 503 there surfaces as a raw error rather than the friendly failure path.
3. **`velocity.py:178`** — the arXiv/full-text interleaving issue described in §4.A2.
4. **`advisor_cli.py:237`** — `_research_question_for` / `_chat_turn` catch only `(ValueError, KeyError)`. A JSON `null` or array from the model raises `AttributeError`/`TypeError` and kills the session *after* the shortlist and chat are done. **If you lose a session right at question-drafting, this is why** — please record the traceback, since it confirms a real model can trigger it.

Also known: on a 9B local model, occasional section failures are expected and surface honestly through `draft_status` / `warnings` rather than being hidden. A `partial` is a normal outcome, not a crisis.

---

## 8. Reporting back

Please send:

1. This file with every checkbox filled in
2. Both `outputs/<domain>_<run_id>/00_trend_advisor/result.json` files from Protocol A
3. The four run directories from Protocol B (or at minimum each `metadata.json` and the rendered paper)
4. Your GPU model and VRAM, so the timings can be read in context
5. Anything that surprised you — especially anything that felt wrong but passed

### Results template

```
MACHINE: <GPU, VRAM, CPU, RAM>
OLLAMA:  <version>   MODEL: qwen3.5:9b
DATE:    <date>
SUITE:   <321 passed, 3 skipped?>

--- PROTOCOL A ---
CS_AI_ML    exit=<>  model_available=<y/n>  gap_flagged=<n>  (0 means: checked-none / never-checked)
MECHANICAL  exit=<>  model_available=<y/n>  gap_flagged=<n>  (0 means: checked-none / never-checked)
Gap statements (verbatim, 3-5):
  1.
  2.
  3.
Fan-out time per topic: CS=<>s  MECH=<>s
Converging chat converged? <y/n>   Question usable as-is? <y/n>
Exactly one run folder each? <y/n>

--- PROTOCOL B ---
Run 1 <topic>: words=<>  status=<>  full_text_papers=<n>/<corpus>  time=<>
Run 2 <topic>: words=<>  status=<>  full_text_papers=<n>/<corpus>  time=<>
Run 3 <topic>: words=<>  status=<>  full_text_papers=<n>/<corpus>  time=<>
Run 4 <topic>: words=<>  status=<>  full_text_papers=<n>/<corpus>  time=<>

AVERAGE WORDS: <>       (baseline 3,691)
SECTIONS >= HALF BUDGET: <>/16   (baseline 0/16)
Card-build time per paper rose? <y/n, numbers>

--- ANYTHING THAT BROKE ---
```
