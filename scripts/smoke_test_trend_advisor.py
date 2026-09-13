"""Manual smoke test for the Trend & Gap Advisor Service against live data.

Usage (from the project venv):
    python scripts/smoke_test_trend_advisor.py CS_AI_ML
    python scripts/smoke_test_trend_advisor.py Other "synthetic biology"

Streams each progress event, prints the ranked shortlist with its real
evidence, then drives the rest of the advisor CLI's own code path on it:
parse_pick on live shortlist data, _research_question_for, and both
hand-off branches — the second one running the real pipeline. Only the
Prompt/Confirm turns of _collect_domain and _converge stay manual.

Not part of the automated suite: needs network for OpenAlex/arXiv and a
running Ollama for the gap-mining swarm. Expect the pipeline hand-off to
take a long time on modest hardware (see smoke_test_full_pipeline.py).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestrator.pipeline import run_pipeline  # noqa: E402
from researchgenie.advisor_cli import (  # noqa: E402
    _research_question_for, parse_pick, with_chosen_topic,
)
from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from shared.contracts.trend_contract import TrendAdvisorRequest  # noqa: E402
from shared.utilities.llm_provider import LLMProviderError  # noqa: E402


def _load_service():
    path = ROOT / "services" / "trend-advisor" / "service.py"
    spec = importlib.util.spec_from_file_location("smoke_trend_advisor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def main(domain: str, other_name: str | None) -> None:
    service = _load_service()
    request = TrendAdvisorRequest(domain=domain, domain_other_name=other_name)

    result = None
    async for event in service.run_trend_advisor(request):
        if event["type"] == "result":
            result = event["result"]
            print(f"\nSaved to: {event['run_directory']}")
        else:
            print(f"[{event['stage']}/{event['state']}] {event['message']}")

    if result is None:
        raise SystemExit("\nFAILED: the advisor stream ended without a result event.")
    if not result.shortlist:
        raise SystemExit("\nFAILED: the advisor returned an empty shortlist.")

    print("\n=== SHORTLIST ===")
    for position, c in enumerate(result.shortlist, start=1):
        flag = "RECURRING GAP" if c.gap_signal else (
            "hot but crowded" if c.is_crowded else "under-explored")
        print(f"\n{position}. {c.topic}  [{flag}]")
        print(f"   growth {c.growth_metric}x  ({c.prior_papers:,} -> {c.paper_count:,} papers)")
        if c.arxiv_recent_count is None:
            print("   arXiv recent preprints among top hits: not checked")
        else:
            print(f"   arXiv recent preprints among top hits: {c.arxiv_recent_count}")
        for evidence in c.gap_evidence:
            print(f"   gap: {evidence}")
        for paper in c.example_papers[:3]:
            print(f"   paper: {paper.title}  (doi:{paper.doi})")

    for warning in result.warnings:
        print(f"\nWARNING: {warning}")

    position = parse_pick("go with #1", result.shortlist)
    if position is None:
        raise SystemExit("\nFAILED: parse_pick rejected \"go with #1\" on a live shortlist.")
    top = result.shortlist[position - 1]
    print(f"\n=== PICK PARSED ===\n\"go with #1\" -> {position}. {top.topic}")
    print("still manual: the domain picklist, the chat turns, and the "
          "\"Lock in ...?\" confirmation before this pick counts")

    try:
        question = (await _research_question_for(top) or "").strip()
    except LLMProviderError as error:
        raise SystemExit(f"\nFAILED: could not reach a model to draft a research "
                         f"question: {error}")
    if not question:
        raise SystemExit(f"\nFAILED: a model answered but returned no usable research "
                         f"question for \"{top.topic}\" (empty or unparseable reply).")
    print(f"\n=== RESEARCH QUESTION ===\n{question}")

    handed_off = with_chosen_topic(result, top)
    print(f"\n=== HAND-OFF, \"stop and review first\" ===\nchosen topic: "
          f"{handed_off.chosen_topic}; shortlist carried: {len(handed_off.shortlist)} "
          f"topic(s); the student re-runs `researchgenie` with the question above")
    print("\n=== HAND-OFF, \"run the pipeline now\" ===")
    completed = False
    async for event in run_pipeline(
        ResearchRequest(research_question=question, keywords=[top.topic]),
        trend_advisor=handed_off,
    ):
        if event["type"] == "result":
            completed = True
            print(f"\nrun directory: {event['result'].run_directory}")
        else:
            print(f"{event['emoji']} [{event['stage']}] {event['message']}")
    if not completed:
        raise SystemExit("\nFAILED: the pipeline hand-off ended without a result event.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
