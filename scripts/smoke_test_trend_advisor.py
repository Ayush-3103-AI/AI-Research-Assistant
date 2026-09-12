"""Manual smoke test for the Trend & Gap Advisor Service against live data.

Usage (from the project venv):
    python scripts/smoke_test_trend_advisor.py CS_AI_ML
    python scripts/smoke_test_trend_advisor.py Other "synthetic biology"

Streams each progress event, prints the ranked shortlist with its real
evidence, then shows the exact ResearchRequest the advisor CLI would
auto-fill from the top pick — the one piece of the CLI flow worth checking
by hand (the CLI's own pick parsing has unit tests; its prompts do not).

Not part of the automated suite: needs network for OpenAlex/arXiv and a
running Ollama for the gap-mining swarm.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from shared.contracts.trend_contract import TrendAdvisorRequest  # noqa: E402


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

    print("\n=== SHORTLIST ===")
    for position, c in enumerate(result.shortlist, start=1):
        flag = "RECURRING GAP" if c.gap_signal else (
            "hot but crowded" if c.is_crowded else "under-explored")
        print(f"\n{position}. {c.topic}  [{flag}]")
        print(f"   growth {c.growth_metric}x  ({c.prior_papers:,} -> {c.paper_count:,} papers)")
        if c.arxiv_recent_count is not None:
            print(f"   arXiv recent preprints among top hits: {c.arxiv_recent_count}")
        for evidence in c.gap_evidence:
            print(f"   gap: {evidence}")
        for paper in c.example_papers[:3]:
            print(f"   paper: {paper.title}  (doi:{paper.doi})")

    for warning in result.warnings:
        print(f"\nWARNING: {warning}")

    # The hand-off the CLI performs on convergence: the locked-in topic
    # auto-fills the existing pipeline's request, with nothing retyped.
    top = result.shortlist[0]
    auto_filled = ResearchRequest(research_question=top.topic)
    print("\n=== AUTO-FILLED ResearchRequest (top pick) ===")
    print(auto_filled.model_dump_json(indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
