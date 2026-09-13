"""Manual check for DECISIONS.md D-039: does a real retrieved Discussion section
actually reach Service 2, and how much bigger is the evidence packet?

Needs network but NO model — the full-text route (Europe PMC JATS, then an
open-access PDF) makes no LLM calls, so this reproduces the headline claim in
D-039 without Ollama running. Prints the real measured numbers rather than
asserting a fixed size, since the live sources can change what they return.

    python scripts/smoke_test_evidence_depth.py

Exits non-zero if no full text could be retrieved, so a silent network failure
cannot be mistaken for a verified result (D-036's precedent for unattended
runners).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))
sys.path.insert(0, str(ROOT / "services" / "research-writing"))

import httpx  # noqa: E402
from discovery import fulltext  # noqa: E402
from discovery.models import Paper  # noqa: E402
from discovery.pipeline import fulltext_done_event  # noqa: E402
from shared.contracts.discovery_contract import (  # noqa: E402
    DiscoveryRequest, DiscoveryResult, NoveltyAssessment, ResearchGap,
)
from writing.modules.document_registry import discover_documents  # noqa: E402
from writing_prep import write_literature_files  # noqa: E402

# Every pipeline service's entry point is named service.py — load by path under a
# unique name, as orchestrator/pipeline.py::_load_service() does.
_spec = importlib.util.spec_from_file_location(
    "discovery_service_smoke", ROOT / "services" / "research-discovery" / "service.py")
_discovery_service = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_discovery_service)

# A real PubMed Central open-access article.
PMCID = "PMC7284132"
DOI = "10.3389/fnins.2020.00001"
ABSTRACT = "A short abstract standing in for this paper's real abstract."


def _discovery_result(paper) -> DiscoveryResult:
    return DiscoveryResult(
        research_request=DiscoveryRequest(research_question="Q?", corpus_size=6),
        research_interpretation="i", search_queries={"keyword": "x"},
        sources_searched=["pubmed"], selected_papers=[paper],
        field_overview="o", limitations="n",
        research_gaps=[ResearchGap(
            title="G", gap_type="methodological", impact="high", description="d",
            evidence="e", supporting_paper_ids=[1], research_questions=["Q1?"])],
        novelty_analysis=NoveltyAssessment(
            novelty_summary="s", confidence="medium", caveats="c"),
        confidence_notes="n",
    )


async def main() -> int:
    paper = Paper(id=1, title="A real open-access article", pmcid=PMCID, doi=DOI,
                  abstract=ABSTRACT)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
        retrieved = await fulltext.fetch_future_sections(http, paper)

    if not retrieved:
        print(f"FAILED: no full text retrieved for {PMCID} — the source or the "
              "network is unavailable. Nothing is verified by this run.")
        return 1
    print(f"live full-text fetch: {len(paper.future_text)} chars retrieved")

    # The event the adapter reads, then the contract field it fills.
    event = fulltext_done_event([paper])
    meta = _discovery_service._paper_metadata(
        paper.to_client_dict(), event["excerpts"].get(paper.id))
    assert meta.discussion_excerpt == paper.future_text, "excerpt lost in the adapter"
    print(f"contract field:      {len(meta.discussion_excerpt)} chars")

    # The literature file Service 2 actually reads, vs the abstract-only packet.
    with tempfile.TemporaryDirectory() as directory:
        write_literature_files(_discovery_result(meta), Path(directory))
        document = discover_documents(directory)[0]
        packet = Path(document.source_path).read_text(encoding="utf-8-sig")

        meta_abstract_only = meta.model_copy(update={"discussion_excerpt": None})
        write_literature_files(_discovery_result(meta_abstract_only), Path(directory))
        before = next(Path(directory).glob("*.md")).read_text(encoding="utf-8-sig")

    print(f"evidence_depth:      {document.evidence_depth}")
    print(f"packet before:       {len(before)} chars (abstract only)")
    print(f"packet after:        {len(packet)} chars "
          f"({len(packet) / max(len(before), 1):.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
