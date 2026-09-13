"""Discovery's already-fetched Discussion/Limitations/Future-work text must
reach Writing, instead of being dropped at the contract boundary.

See DECISIONS.md D-029: Discovery fetches these sections for its own gap
mining and then forwards only `paper.abstract` onward, which that pass
quantified as the sole dominant bottleneck on paper length (0 of 16 real
sections ever reached even half their own word budget). These tests pin each
link of the chain: the contract field, Discovery's adapter, the literature
file Writing reads, the registry that parses it, the prompt that interprets
it, and the context-window cost of the larger card prompt.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-writing"))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))


def _load_service(name: str, directory: str):
    """Load one pipeline service's entry point under a unique module name.

    Every service's entry point is called `service.py`, so a bare
    `import service` resolves to whichever one another test file imported first
    in the same pytest session — a collision this repo hits repeatedly (see
    `tests/services/test_qa_service.py` and
    `orchestrator/pipeline.py::_load_service()`). Review caught this file doing
    exactly that: running it alongside `test_writing_prep.py` picked up the
    Writing service and failed, so the green full suite was down to collection
    order."""
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "services" / directory / "service.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


discovery_service = _load_service("discovery_service_evidence_depth", "research-discovery")
qa_service = _load_service("qa_service_evidence_depth", "quality-assurance")

from shared.contracts.discovery_contract import (  # noqa: E402
    DiscoveryRequest, DiscoveryResult, NoveltyAssessment, PaperMetadata, ResearchGap,
)
from shared.utilities.llm_provider import _context_window_for  # noqa: E402
from writing.modules.document_registry import (  # noqa: E402
    DocumentRegistryError, discover_documents,
)
from writing.modules.prompt_loader import load_prompt  # noqa: E402
from writing_prep import write_literature_files  # noqa: E402

EXCERPT = (
    "Discussion. Our effect sizes were measured only in a single cohort. "
    "Future work should replicate this across multiple sites."
)


def _discovery(papers: list[PaperMetadata]) -> DiscoveryResult:
    return DiscoveryResult(
        research_request=DiscoveryRequest(research_question="Q?", corpus_size=6),
        research_interpretation="interp",
        search_queries={"keyword": "x"},
        sources_searched=["OpenAlex"],
        selected_papers=papers,
        field_overview="overview",
        limitations="none",
        research_gaps=[ResearchGap(
            title="Gap", gap_type="methodological", impact="high",
            description="desc", evidence="evi", supporting_paper_ids=[1],
            research_questions=["Q1?"],
        )],
        novelty_analysis=NoveltyAssessment(
            novelty_summary="summary", confidence="medium", caveats="caveat",
        ),
        confidence_notes="notes",
    )


# ------------------------------------------------------- the contract boundary

def test_paper_metadata_carries_the_discussion_excerpt():
    paper = PaperMetadata(id=1, title="T", abstract="A.", discussion_excerpt=EXCERPT)
    assert paper.discussion_excerpt == EXCERPT
    # Round-trips, since the orchestrator persists this contract to disk.
    restored = PaperMetadata.model_validate_json(paper.model_dump_json())
    assert restored.discussion_excerpt == EXCERPT


def test_discussion_excerpt_defaults_to_none_so_existing_callers_are_unaffected():
    assert PaperMetadata(id=1, title="T").discussion_excerpt is None


# --------------------------------------------------- Discovery's own adapter

def test_discovery_adapter_attaches_the_excerpt_fetched_after_the_corpus_event():
    """The `corpus` event is emitted BEFORE the full-text stage runs, so the
    excerpts arrive separately on `fulltext_done` and must be merged in by id."""
    _paper_metadata = discovery_service._paper_metadata

    client_dict = {
        "id": 3, "title": "A Study of X", "authors": ["Alice Smith"], "year": 2022,
        "venue": "J. Examples", "citations": 12, "doi": "10.1000/xyz",
        "link": "https://doi.org/10.1000/xyz", "source": "OpenAlex",
        "has_abstract": True, "abstract": "This study examines X.",
    }
    assert _paper_metadata(client_dict).discussion_excerpt is None
    assert _paper_metadata(client_dict, EXCERPT).discussion_excerpt == EXCERPT


def test_fulltext_done_event_carries_the_text_not_only_the_paper_ids():
    """Without the text on this event the adapter has no way to reach it —
    `to_client_dict()` is built before the full-text stage populates it."""
    from discovery.models import Paper
    from discovery.pipeline import fulltext_done_event

    with_text = Paper(id=1, title="One", future_text=EXCERPT)
    without = Paper(id=2, title="Two")
    event = fulltext_done_event([with_text, without])
    assert event["type"] == "fulltext_done"
    assert event["paper_ids"] == [1]
    assert event["excerpts"] == {1: EXCERPT}


def test_run_discovery_merges_the_excerpt_onto_the_right_paper(monkeypatch):
    """The actual merge seam, driven through the real `run_discovery`.

    Testing `fulltext_done_event` and `_paper_metadata` separately cannot catch
    this: the excerpts arrive on a later event than the corpus and are matched
    back by paper id, so a wrong key, a wrong event name, or a lost id would
    still leave both unit tests green. The underlying pipeline is faked, so no
    network and no model are involved."""
    import asyncio

    async def fake_pipeline(*args, **kwargs):
        yield {"type": "queries", "queries": {"interpretation": "i", "keyword": "k"}}
        yield {"type": "corpus", "papers": [
            {"id": 1, "title": "Has Full Text", "authors": [], "year": 2020,
             "venue": "V", "citations": 1, "doi": "10.1/a", "link": None,
             "source": "pubmed", "has_abstract": True, "abstract": "Abstract A."},
            {"id": 2, "title": "Abstract Only", "authors": [], "year": 2021,
             "venue": "V", "citations": 2, "doi": "10.1/b", "link": None,
             "source": "pubmed", "has_abstract": True, "abstract": "Abstract B."},
        ]}
        yield {"type": "fulltext_done", "paper_ids": [1], "excerpts": {1: EXCERPT}}
        yield {"type": "report", "report": {
            "field_overview": "o", "themes": [], "limitations": "l",
            "gaps": [], "author_flagged_gaps": [], "confidence_notes": "c",
            "novelty_analysis": {"novelty_summary": "s", "supporting_gap_titles": [],
                                 "confidence": "medium", "caveats": "c"},
        }}

    monkeypatch.setattr(discovery_service, "run_pipeline", fake_pipeline)

    async def collect():
        async for event in discovery_service.run_discovery(
            DiscoveryRequest(research_question="Q?", corpus_size=6)
        ):
            if event["type"] == "result":
                return event["result"]
        return None

    result = asyncio.run(collect())
    by_id = {p.id: p for p in result.selected_papers}
    assert by_id[1].discussion_excerpt == EXCERPT
    assert by_id[2].discussion_excerpt is None


# ------------------------------------------- the literature file Writing reads

def test_literature_file_includes_the_excerpt_and_declares_the_deeper_depth(tmp_path):
    written = write_literature_files(_discovery([
        PaperMetadata(id=1, title="Paper One", abstract="Abstract one.",
                      has_abstract=True, discussion_excerpt=EXCERPT),
    ]), tmp_path)
    assert written == 1
    content = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert "evidence_depth: abstract_plus_discussion" in content
    assert "Abstract one." in content
    assert EXCERPT in content
    # The excerpt must be labelled, so the card builder never reads it as abstract text.
    assert "## Abstract" in content
    assert "## Discussion, Limitations, and Future Work" in content


def test_abstract_only_paper_still_declares_abstract_depth(tmp_path):
    """The deeper depth is claimed only where the text was genuinely retrieved."""
    write_literature_files(_discovery([
        PaperMetadata(id=1, title="Paper One", abstract="Abstract one.", has_abstract=True),
    ]), tmp_path)
    content = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert "evidence_depth: abstract" in content
    assert "abstract_plus_discussion" not in content


def test_paper_with_an_excerpt_but_no_abstract_is_no_longer_skipped(tmp_path):
    """Previously any paper without an abstract was dropped outright, discarding
    a real retrieved Discussion section along with it."""
    written = write_literature_files(_discovery([
        PaperMetadata(id=1, title="Paper One", has_abstract=False,
                      discussion_excerpt=EXCERPT),
    ]), tmp_path)
    assert written == 1
    assert EXCERPT in next(tmp_path.glob("*.md")).read_text(encoding="utf-8")


def test_paper_with_neither_abstract_nor_excerpt_is_still_skipped(tmp_path):
    assert write_literature_files(_discovery([
        PaperMetadata(id=1, title="Paper One", has_abstract=False),
    ]), tmp_path) == 0


# --------------------------------------------- the registry that parses it back

def test_registry_accepts_the_new_depth_end_to_end(tmp_path):
    write_literature_files(_discovery([
        PaperMetadata(id=1, title="Paper One", abstract="Abstract one.",
                      has_abstract=True, discussion_excerpt=EXCERPT),
    ]), tmp_path)
    documents = discover_documents(tmp_path)
    assert [d.evidence_depth for d in documents] == ["abstract_plus_discussion"]


def test_real_full_text_prose_survives_the_file_round_trip(tmp_path):
    """Real retrieved Discussion text is never plain ASCII — a live Europe PMC
    fetch returns en-dashes, non-breaking spaces, Greek letters and the like.
    The file is written as UTF-8 and read back as UTF-8-sig, so pin that."""
    messy = "Doses of 30–100 kDa RBAC raised IL‐6 and TNF‐α (p < 0.05)."
    write_literature_files(_discovery([
        PaperMetadata(id=1, title="Paper One", abstract="Abstract one.",
                      has_abstract=True, discussion_excerpt=messy),
    ]), tmp_path)
    document = discover_documents(tmp_path)[0]
    assert messy in Path(document.source_path).read_text(encoding="utf-8-sig")


def test_registry_still_rejects_an_unknown_depth(tmp_path):
    (tmp_path / "p.md").write_text("---\nevidence_depth: guesswork\n---\n\nBody.\n",
                                   encoding="utf-8")
    with pytest.raises(DocumentRegistryError):
        discover_documents(tmp_path)


# ------------------------------------------------ the prompt that reads the file

def test_card_prompt_tells_the_model_how_to_treat_the_new_depth():
    prompt = load_prompt("build_literature_card.md")
    assert "abstract_plus_discussion" in prompt
    # It must still forbid claiming the whole paper was reviewed — only the
    # abstract plus the authors' own closing sections are actually present.
    assert "reviewed in full text" in prompt


# ---------------------------------------- Service 4 must check the same evidence

def test_qa_evidence_profile_covers_the_excerpt_writing_was_given():
    """Service 4 flags a cited claim whose wording does not overlap its source's
    evidence profile. Once Service 2 can legitimately write from the Discussion
    excerpt, a profile built from the abstract alone would report those claims as
    unsupported — the same false-positive class as DECISIONS.md D-021."""
    from shared.contracts.verification_contract import ReferenceCheck, VerificationResult

    qa = qa_service
    paper = PaperMetadata(
        id=1, title="Fasting and Memory", doi="10.1/real", venue="Journal X",
        has_abstract=True,
        abstract="A randomized trial found intermittent fasting improved working memory.",
        discussion_excerpt=(
            "Our cohort was recruited from a single metropolitan hospital, so the "
            "generalisability of these effects to rural populations is unknown."
        ),
    )
    verification = VerificationResult(
        validated_draft_markdown="draft",
        verified_references=[ReferenceCheck(
            reference_entry="[1] A. Author, Fasting and Memory, 2020.",
            doi="10.1/real", verified=True, verification_source="Crossref",
        )],
        validation_report="# Citation Verification Report\n",
    )
    profiles = qa.build_reference_profiles(_discovery([paper]), verification)

    # A claim drawn from the Discussion section, not the abstract.
    claim = ("## Discussion\n\nGeneralisability to rural populations remains unknown "
             "for this single-hospital cohort [1].\n\n## References\n")
    assert qa.find_unsupported_claims(claim, profiles) == []

    # The check must still catch claims the evidence does not support. A wholly
    # unrelated claim is the easy case; the second is the one that matters — a
    # same-field claim about a population and outcome this paper never studied,
    # which a larger profile could plausibly have started waving through.
    off_topic = ("## Discussion\n\nQuantum computing will revolutionize cryptography "
                 "entirely [1].\n\n## References\n")
    assert len(qa.find_unsupported_claims(off_topic, profiles)) == 1

    plausible_but_unsupported = (
        "## Discussion\n\nSurgical weight-loss interventions reduced childhood asthma "
        "hospitalisations across Scandinavian registries [1].\n\n## References\n"
    )
    assert len(qa.find_unsupported_claims(plausible_but_unsupported, profiles)) == 1


# ------------------------------------------------------- the cost of doing this

def test_the_excerpt_fits_inside_the_card_calls_existing_context_tier():
    """D-030's precedent: prove the resource cost rather than assume it.

    What is pinned here is the MARGIN, not a guessed prompt size. The card
    call's payload includes tag definitions whose `description`/`include_when`
    prose is itself LLM-generated (`tag_semantics.py`), so there is no fixed
    constant to measure offline and no real card artifact in `outputs/` left to
    measure from. The margin is exact and checkable: at max_tokens=8000 the
    16,384 tier holds any prompt up to 33,536 chars, and the card's own system
    prompt plus a full-size excerpt (fulltext.MAX_SECTION_CHARS = 7000) spends
    only part of that — so the excerpt is paid for out of existing headroom,
    unlike D-030's rejected max_tokens increase, which moved the tier for every
    call. If the system prompt or the excerpt cap grows enough to eat the rest,
    this fails instead of silently slowing every card build."""
    # Found, not hand-derived: the largest prompt still selecting the 16384 tier.
    tier_limit = max(n for n in range(20_000, 40_000)
                     if _context_window_for(8000, n) == 16384)
    assert tier_limit == 33_539
    assert _context_window_for(8000, tier_limit + 1) == 24576

    fixed_cost = len(load_prompt("build_literature_card.md")) + 7_000
    payload_headroom = tier_limit - fixed_cost
    # Room left for tag definitions + metadata + the abstract, on top of a
    # full-size excerpt. Generous: a whole excerpt again would still fit.
    assert payload_headroom > 7_000
    assert _context_window_for(8000, fixed_cost + payload_headroom) == 16384
