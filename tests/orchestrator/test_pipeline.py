"""Fast unit tests for the orchestrator, with all four services faked out —
see tests/services/*_pipeline_integration.py and scripts/smoke_test_full_pipeline.py
for the real, slow, live-service coverage."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from shared.contracts.discovery_contract import (  # noqa: E402
    DiscoveryRequest, DiscoveryResult, NoveltyAssessment, PaperMetadata, ResearchGap,
)
from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from shared.contracts.qa_contract import QualityAssuranceResult, QualityScores  # noqa: E402
from shared.contracts.verification_contract import VerificationResult  # noqa: E402
from shared.contracts.writing_contract import DraftMetadata, WritingResult  # noqa: E402


def _fake_discovery_result() -> DiscoveryResult:
    return DiscoveryResult(
        research_request=DiscoveryRequest(research_question="Q?", corpus_size=6),
        research_interpretation="interp", search_queries={"keyword": "x"},
        sources_searched=["OpenAlex"],
        selected_papers=[PaperMetadata(id=1, title="Paper One", doi="10.1/x", has_abstract=True, abstract="abstract")],
        field_overview="overview", limitations="none",
        research_gaps=[ResearchGap(
            title="Gap", gap_type="methodological", impact="high",
            description="desc", evidence="evi", supporting_paper_ids=[1], research_questions=["Q1?"],
        )],
        novelty_analysis=NoveltyAssessment(novelty_summary="summary", confidence="medium", caveats="caveat"),
        confidence_notes="notes",
    )


def _fake_writing_result() -> WritingResult:
    return WritingResult(
        research_outline="# Q\n## Introduction", research_draft_markdown="draft body [1].",
        reference_candidates=['[1] A. Author, "Paper One," 2020, doi: 10.1/x.'],
        run_directory="/tmp/writing-run",
        draft_metadata=DraftMetadata(
            model_name="qwen3.5:9b", run_id="writing-run-1", generated_at="2026-08-27T00:00:00Z",
            sections_written=2, papers_cited=1,
        ),
    )


def _fake_verification_result() -> VerificationResult:
    return VerificationResult(
        validated_draft_markdown="draft body [1].",
        verified_references=[], invalid_references=[], unverifiable_references=[],
        validation_report="# Citation Verification Report\n",
    )


def _fake_qa_result() -> QualityAssuranceResult:
    return QualityAssuranceResult(
        final_draft_markdown="draft body [1].",
        final_references=['[1] A. Author, "Paper One," 2020, doi: 10.1/x.'],
        final_validation_report="# Citation Verification Report\n",
        quality_report_markdown="# Quality Assurance Report\n",
        scores=QualityScores(
            citation_integrity=5.0, claim_source_alignment=5.0,
            process_control=5.0, literature_coverage=5.0, overall=5.0,
        ),
    )


async def _fake_run_discovery(request):
    yield {"type": "status", "message": "searching"}
    yield {"type": "result", "result": _fake_discovery_result()}


async def _fake_run_writing(request, output_root=None):
    yield {"type": "status", "stage": "write", "state": "running", "message": "writing"}
    yield {"type": "result", "result": _fake_writing_result()}


async def _fake_run_verification(request):
    yield {"type": "status", "stage": "verify", "state": "running", "message": "verifying"}
    yield {"type": "result", "result": _fake_verification_result()}


async def _fake_run_qa(request):
    yield {"type": "status", "stage": "audit", "state": "running", "message": "auditing"}
    yield {"type": "result", "result": _fake_qa_result()}


@pytest.fixture(autouse=True)
def _patch_services(monkeypatch):
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery)
    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", _fake_run_verification)
    monkeypatch.setattr(orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)


def test_run_pipeline_produces_result_and_persists_artifacts(tmp_path):
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        events = []
        result = None
        async for event in orchestrator_pipeline.run_pipeline(request, output_root=tmp_path):
            if event["type"] == "result":
                result = event["result"]
            else:
                events.append(event)
        return events, result

    events, result = asyncio.run(run())

    assert result is not None
    assert result.quality_assurance.scores.overall == 5.0
    assert Path(result.run_directory).is_dir()
    assert (Path(result.run_directory) / "00_request.json").exists()
    assert (Path(result.run_directory) / "01_discovery" / "result.json").exists()
    assert (Path(result.run_directory) / "final" / "draft.md").exists()
    assert (Path(result.run_directory) / "final" / "draft.md").read_text(encoding="utf-8") == "draft body [1]."
    # LaTeX source is always generated (deterministic — see D-022); PDF
    # compilation is opportunistic and not asserted here since it depends
    # on a LaTeX toolchain being present on the machine running the test.
    tex_path = Path(result.run_directory) / "final" / "paper.tex"
    assert tex_path.exists()
    assert tex_path.read_text(encoding="utf-8").startswith("\\documentclass")

    metadata = json.loads((Path(result.run_directory) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["overall_quality_score"] == 5.0

    # Standardized progress events carry the required fields.
    stage_events = [e for e in events if e.get("stage") == "discovery"]
    assert stage_events
    for event in stage_events:
        assert set(event) >= {"run_id", "stage", "service", "status", "emoji", "title", "message", "details", "timestamp"}


def test_research_request_requires_format_other_name_when_format_is_other():
    with pytest.raises(ValueError):
        ResearchRequest(research_question="Q?", target_format="Other")
    # Valid when the name is supplied.
    ResearchRequest(research_question="Q?", target_format="Other", format_other_name="Custom Style")


def test_run_pipeline_raises_if_a_stage_yields_no_result(tmp_path, monkeypatch):
    async def _empty_discovery(request):
        yield {"type": "status", "message": "nothing"}
        return

    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", _empty_discovery)
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        async for _ in orchestrator_pipeline.run_pipeline(request, output_root=tmp_path):
            pass

    with pytest.raises(orchestrator_pipeline.PipelineError):
        asyncio.run(run())


# --- Optional Stage 0: Trend & Gap Advisor (DECISIONS.md D-031) -------------

def _fake_trend_result():
    from shared.contracts.trend_contract import TopicCandidate, TrendAdvisorResult
    return TrendAdvisorResult(
        domain="CS_AI_ML",
        shortlist=[TopicCandidate(
            topic="Edge Inference", topic_id="T1", growth_metric=3.0,
            paper_count=900, prior_papers=300, gap_signal=True,
            gap_evidence=["On-device latency is unmeasured."], source="both",
        )],
        generated_at="2026-09-12T00:00:00Z",
    )


def test_stage_zero_is_absent_by_default(tmp_path):
    """The advisor is optional: a plain run must be byte-for-byte what it
    was before this stage existed."""
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        result = None
        events = []
        async for event in orchestrator_pipeline.run_pipeline(request, output_root=tmp_path):
            if event["type"] == "result":
                result = event["result"]
            else:
                events.append(event)
        return events, result

    events, result = asyncio.run(run())

    assert result.trend_advisor is None
    assert not (Path(result.run_directory) / "00_trend_advisor").exists()
    assert not any(e.get("stage") == "trend_advisor" for e in events)
    metadata = json.loads((Path(result.run_directory) / "metadata.json").read_text(encoding="utf-8"))
    assert "chosen_topic" not in metadata


def test_stage_zero_is_persisted_into_the_same_run_folder(tmp_path):
    """Why a student chose this topic and what the pipeline then produced
    belong to one run, not two unrelated folders."""
    request = ResearchRequest(research_question="How does edge inference scale?", corpus_size=6)
    advisor = _fake_trend_result()

    async def run():
        result = None
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, trend_advisor=advisor,
        ):
            if event["type"] == "result":
                result = event["result"]
        return result

    result = asyncio.run(run())

    saved = Path(result.run_directory) / "00_trend_advisor" / "result.json"
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8"))["shortlist"][0]["topic"] == "Edge Inference"
    assert result.trend_advisor.shortlist[0].topic == "Edge Inference"


def test_stage_zero_is_reported_in_the_audit_trail(tmp_path):
    request = ResearchRequest(research_question="How does edge inference scale?", corpus_size=6)

    async def run():
        events = []
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, trend_advisor=_fake_trend_result(),
        ):
            if event["type"] != "result":
                events.append(event)
        return events

    events = asyncio.run(run())

    stage_zero = [e for e in events if e.get("stage") == "trend_advisor"]
    assert stage_zero, "Stage 0 produced no progress event"
    assert stage_zero[0]["status"] == "done"
    assert set(stage_zero[0]) >= {"run_id", "stage", "service", "status", "emoji",
                                  "title", "message", "details", "timestamp"}
    # Stage 0 is reported before Discovery, since it ran before it.
    assert events.index(stage_zero[0]) < min(
        i for i, e in enumerate(events) if e.get("stage") == "discovery")




def test_metadata_reads_the_named_pick_not_the_first_entry(tmp_path):
    """The chosen topic is whichever one the student named, wherever it sits
    in the shortlist's own ranking."""
    from shared.contracts.trend_contract import TopicCandidate
    advisor = _fake_trend_result()
    advisor = advisor.model_copy(update={
        "shortlist": [
            TopicCandidate(topic="Not Chosen", topic_id="T9", growth_metric=9.0,
                           paper_count=100),
            *advisor.shortlist,
        ],
        "chosen_topic": "Edge Inference",
    })
    request = ResearchRequest(research_question="How does edge inference scale?", corpus_size=6)

    async def run():
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, trend_advisor=advisor,
        ):
            if event["type"] == "result":
                return event["result"]

    result = asyncio.run(run())
    metadata = json.loads((Path(result.run_directory) / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["chosen_topic"] == "Edge Inference"
    assert metadata["chosen_topic_gap_flagged"] is True


def test_no_chosen_topic_recorded_when_the_advisor_result_names_none(tmp_path):
    """A shortlist saved without a pick (the student stopped to think it
    over) must not have a topic attributed to them."""
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, trend_advisor=_fake_trend_result(),
        ):
            if event["type"] == "result":
                return event["result"]

    result = asyncio.run(run())
    metadata = json.loads((Path(result.run_directory) / "metadata.json").read_text(encoding="utf-8"))

    assert "chosen_topic" not in metadata
    # The shortlist itself is still preserved as run provenance.
    assert (Path(result.run_directory) / "00_trend_advisor" / "result.json").exists()


def test_a_chosen_topic_absent_from_the_shortlist_is_not_reported_as_gap_flagged(tmp_path):
    """Defensive: the gap flag is looked up by topic name, so a name that
    matches nothing must report False rather than raise or guess."""
    advisor = _fake_trend_result().model_copy(update={"chosen_topic": "Some Other Topic"})
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, trend_advisor=advisor,
        ):
            if event["type"] == "result":
                return event["result"]

    result = asyncio.run(run())
    metadata = json.loads((Path(result.run_directory) / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["chosen_topic"] == "Some Other Topic"
    assert metadata["chosen_topic_gap_flagged"] is False
