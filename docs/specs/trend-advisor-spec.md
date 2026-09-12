# Spec: Trend & Gap Advisor (pre-Discovery research advisory stage)

## Problem Statement

Today, ResearchGenie's pipeline (Research Discovery → Research Writing → Citation Verification → Quality Assurance) only starts once a student, professor, or other university stakeholder already has a research question in hand. In practice, the harder problem for most students comes *before* that: they don't know which topics in their engineering domain currently have real research momentum, which topics are recurring, unmet gaps that multiple published papers flag as future work, or where a genuinely publishable project is likely to sit — across not just CS/AI/ML but ECE, EEE, Mechanical, Civil, and Cloud/DevOps as well. Today they're left to guess, browse arbitrarily, or rely on word-of-mouth, with no evidence-grounded way to gauge where the field is actually moving.

## Solution

A new **Trend & Gap Advisor** stage that runs *before* Research Discovery. A student or professor picks their domain from a fixed list, and the advisor produces a ranked shortlist of rising, evidence-backed topics using two real, keyless, verifiable signals: OpenAlex/arXiv publication-velocity analysis (which subfields are growing) and cross-paper future-work/gap-citation mining (which unmet needs multiple independent papers flag). The student can then refine that shortlist through a converging chat, ending in one locked-in topic that auto-fills the existing pipeline's `ResearchRequest` — so Discovery → Writing → Verification → QA then runs completely unchanged. The whole addition is built to independently demonstrate all seven of the course's skill areas (Agents, Skills, MCP, OpenCLA/OpenClaw, Swarms, Deep Research, Cloud SDLC), extending the same rows on the Team Worklet Canvas that the existing four services already fill.

## User Stories

1. As a student, I want to pick my engineering domain from a fixed list, so that I get topic suggestions relevant to my field without needing to know OpenAlex's taxonomy myself.
2. As a student, I want to see a ranked shortlist of rising topics in my domain, so that I can quickly identify where active research momentum exists.
3. As a student, I want each shortlisted topic to show its supporting evidence (publication growth numbers, example papers), so that I can trust the ranking isn't fabricated.
4. As a student, I want to see topics explicitly flagged as recurring gaps across multiple papers' future-work sections, so that I know where genuine unmet research needs exist, not just where publication volume is growing.
5. As a student, I want to distinguish "hot but crowded" topics (high volume) from "under-explored gap" topics (low volume, recurring gap mentions), so that I can choose between competing in a hot area or filling a quieter gap.
6. As a student, I want to ask follow-up questions about a specific shortlisted topic conversationally, so that I can narrow down or combine ideas before committing.
7. As a student, I want to say "go with #2" (or similar) at any point in the conversation, so that I can lock in a decision without a rigid, over-long dialogue.
8. As a student, I want my locked-in topic to auto-fill the research-question field of the existing ResearchGenie tool, so that I don't have to retype or reformat it.
9. As a student, I want the option to immediately chain into the existing Discovery→Writing→Verification→QA pipeline right after picking a topic, so that I don't have to manually re-invoke the tool separately.
10. As a student, I want the option to stop after getting the shortlist/topic pick without running the full pipeline immediately, so that I can think it over first.
11. As a student in a domain not on the fixed picklist, I want an "Other" option with free text, so that I'm not blocked from using the tool.
12. As a professor, I want to run the same advisor tool locally to explore what's active in a domain, so that I can guide students in an advising session without needing a separate interface.
13. As a professor, I want the shortlist's evidence to be independently verifiable (real papers, real counts), so that I can trust it enough to recommend a topic to a student.
14. As a stakeholder, I want the tool to work identically across CS/AI/ML, ECE, EEE, Mechanical, Civil, and Cloud/DevOps domains, so that it's useful university-wide, not just for CS students.
15. As a developer, I want the trend/gap agent's core logic exposed as one clean, testable service entrypoint (`run_trend_advisor()`), so that it can be tested and maintained the same way the existing four pipeline services are.
16. As a developer, I want the publication-velocity analysis and future-work/gap mining genuinely computed from live OpenAlex/arXiv data (never fabricated), so that the rigor of the rest of the pipeline extends to this new stage too.
17. As a developer, I want the future-work/gap-citation mining to run as a parallel fan-out swarm (one worker per candidate paper), so that this stage demonstrates the same Swarms pattern already proven in the Writing service, and stays fast on this hardware.
18. As a developer, I want a visible, honest failure (not a silent empty/fake result) when OpenAlex is unreachable or a domain yields too few results, so that the agent never presents fabricated confidence to a student.
19. As a developer, I want the domain→OpenAlex-concept mapping to be an explicit, hardcoded, verified table (not an LLM guess), so that results are reproducible and don't silently drift.
20. As a developer, I want this new stage wired into the orchestrator as an optional Stage 0 invoked only via a new CLI advisor flow (not forced onto the existing `researchgenie` entrypoint), so that the existing pipeline's behavior and tests are entirely unaffected.
21. As a course team member, I want each of the 7 skills visibly demonstrated by this new stage, so that the updated Team Worklet Canvas can show how the larger platform still covers all 7 skills.
22. As a course team member, I want a DECISIONS.md entry for each substantive design choice in the new stage, so that the documentation discipline already established in the repo is maintained for the new capability.
23. As a maintainer, I want the CLI advisor flow packaged the same way as the existing `researchgenie` entrypoint (pyproject.toml entry point), so that installation and usage stay consistent.
24. As a student picking "Other" domain, I want a clear message that free-text domains don't get a curated OpenAlex-concept mapping and results may be less precise, so that expectations are set correctly.
25. As a student, I want the chat refinement loop to stay responsive on this hardware, so that I'm not stuck waiting minutes per turn — the loop should use the minimum LLM calls per turn given `qwen3.5:9b`'s already-documented speed bottleneck (DECISIONS.md D-009).

## Implementation Decisions

- New service `services/trend-advisor/service.py`, exposing an async-generator entrypoint `run_trend_advisor(request: TrendAdvisorRequest) -> AsyncIterator[dict]` — same progress-event-then-final-result shape as the existing four services (Discovery/Writing/Verification/QA), so the orchestrator and any future consumer can treat it uniformly.
- New contract module `shared/contracts/trend_contract.py`:
  - `DomainChoice`: `CS_AI_ML | ECE | EEE | MECHANICAL | CIVIL | CLOUD_DEVOPS | OTHER`, with a `domain_other_name` field for free text when `OTHER` is chosen — mirrors the existing `WritingRequest.target_format`/`format_other_name` pattern exactly.
  - `TrendAdvisorRequest`: domain, optional `domain_other_name`, optional shortlist size (default e.g. 8, matching Discovery's own 6–9 corpus-size convention).
  - `TopicCandidate`: topic name, growth_metric, paper_count, example_papers (title + DOI), gap_signal (bool), gap_evidence (excerpts from future-work/limitations sections), source (`velocity` | `gap_mining` | `both`).
  - `TrendAdvisorResult`: domain, ranked shortlist of `TopicCandidate`, generated_at.
- **Step 1 (Agents, skill 1)** — `trend_advisor/velocity.py::find_rising_topics(domain) -> list[TopicCandidate]`: a standalone function with no dependency on chat/orchestrator/CLI. Maps domain to an OpenAlex concept ID via the hardcoded table, queries works-count-by-year for that concept's subfields over a recent window, computes year-over-year growth, returns the top candidates by growth. arXiv is layered in as a secondary signal only for CS-adjacent domains, reusing Discovery's existing arXiv access code path rather than a new client.
- **Step 5 (Swarms, skill 5)** — `trend_advisor/gap_mining.py`: a LangGraph fan-out (`Send`), one worker per candidate paper drawn from Step 1's evidence set (capped, matching D-009's `MAX_CANDIDATES_FOR_ANALYSIS` precedent), each extracting future-work/limitations text via the local LLM; a merge step aggregates recurring gap mentions across papers and annotates matching `TopicCandidate`s with `gap_signal`/`gap_evidence`.
- **Step 2 (Skills/Memory, skill 2)** — the service's full output is saved to its own timestamped run folder, matching the existing convention: `outputs/<slug>_<run_id>/00_trend_advisor/result.json`.
- **Step 3 (MCP, skill 3)** — reuses the existing OpenAlex/arXiv connector code already used by Discovery, extended (not duplicated) with a works-count-by-year query type.
- **Step 4 (OpenCLA/OpenClaw, skill 4)** — a new, separate CLI entrypoint (e.g. `researchgenie-advise`) calls `run_trend_advisor()`, then the converging chat loop, then — on convergence — offers to call `orchestrator.pipeline.run_pipeline()` directly with the auto-filled `ResearchRequest`. The existing `researchgenie` entrypoint and `orchestrator/pipeline.py`'s existing four stages are untouched; this is purely additive.
- **Converging chat loop**: seeded with the Step 1 + Step 5 shortlist as structured context (not just prose), tracks candidates as structured state so "go with #2" parses deterministically, one LLM call per user turn (no hidden multi-step reasoning per turn), ends only on the student's explicit confirmation of one topic.
- **Step 7 (Cloud SDLC, skill 7)** — new CLI entry point registered in `pyproject.toml`, alongside the existing `researchgenie` entry point.
- `build-process` canvas gets one additional sentence per row in the "HOW OUR WORKLET ADDRESSES IT" column once the build lands, reflecting the trend-advisor's contribution to each of the 7 skills.

## Testing Decisions

- A good test exercises external behavior at the `run_trend_advisor()` service boundary — matching `tests/services/test_discovery_service.py`'s existing pattern — not internal helper functions in isolation, except where noted below.
- Full rigor (root-cause diagnosis, DECISIONS.md entries, matching D-018/D-019's format) applies to `velocity.py` and `gap_mining.py`, since these are the actual novel reasoning this stage adds:
  - `velocity.py`: unit tests with mocked OpenAlex HTTP responses proving the growth-ranking math is correct, plus one live integration test against the real OpenAlex API, skipped without network — matching the existing `_ollama_available()`-style live-skip pattern already used three times in the repo (`tests/services/test_discovery_pipeline_integration.py` etc.).
  - `gap_mining.py`: unit tests with a fake `LanguageModel` recording prompts (no Ollama required), matching `tests/services/test_writing_literature_card_retry.py`'s pattern, proving the fan-out/merge aggregation logic against fixture papers with known future-work text.
- Lighter treatment (per the agreed rigor split) on the CLI advisor entrypoint and the orchestrator Stage-0 wiring: a smoke-test-style manual runner (matching `scripts/smoke_test_*.py`'s convention) confirming the CLI calls the service and correctly auto-fills a `ResearchRequest` — not a full pytest suite duplicating what the service-level tests already cover.
- No new tests duplicate existing Discovery/Writing coverage; only the new works-count-by-year query type on the shared OpenAlex/arXiv connector needs its own tests.

## Out of Scope

- Gartner reports, Google Trends, patent filings, and conference-CFP scraping as signal sources — considered and explicitly dropped (paywalled/unlicensed, low relevance to academic novelty, or not worth the added complexity for this pass).
- A web dashboard, multi-user accounts, authentication, or a persistent database — this stays a local CLI tool, matching the existing `researchgenie`/`terminal_app` pattern.
- Automatic topic selection without human confirmation — the student/professor always makes the final pick; the agent never auto-commits to a `ResearchRequest` without explicit confirmation.
- Bespoke connectors per engineering discipline beyond the six picklist domains + free-text "Other" — no attempt to integrate discipline-specific databases (e.g. IEEE Xplore, ASCE Library) in this pass.
- Open-ended tutoring/mentorship conversation — the chat loop's purpose is convergence to one topic pick, not general research advising beyond that scope.
- Any change to the existing four pipeline services' behavior, contracts, or tests — this addition is purely additive.

## Further Notes

- No hard deadline; soft target of finishing before next weekend (~2026-09-19/20) — workload assessed as light for this scope.
- University Agentic AI course project (KLE Technological University, Team 7, course code 26ECAC401); the deliverable should visibly map onto the course's 7-skill rubric per `build-process`, extending the existing Team Worklet Canvas rather than creating a second, separate one.
- Repository: `github.com/TheIntruder007/AI-Research-Assistant`.
- `qwen3.5:9b` via Ollama remains the default local model and known hardware bottleneck (DECISIONS.md D-009); all new LLM calls in this stage should follow the same `think=False`, tiered-context-window discipline already established there.
