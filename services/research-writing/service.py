"""Public entry point for the Research Writing Service.

Takes a DiscoveryResult (Service 1's output), builds an outline and one
literature file per selected paper (the abstract, plus the authors' own
Discussion/Limitations/Future-work text wherever Discovery retrieved a real
full text — see DECISIONS.md D-010 and D-039), runs the writing graph against
the local model, and adapts its output to the shared WritingResult contract.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.contracts.discovery_contract import DiscoveryResult  # noqa: E402
from shared.contracts.writing_contract import (  # noqa: E402
    CitationEntry, DraftMetadata, WritingRequest, WritingResult,
)
from shared.utilities import llm_provider  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from completeness import assess_completeness  # noqa: E402
from word_budget import classify_length  # noqa: E402
from writing.adapters.language_model import close_language_model  # noqa: E402
from writing.adapters.ollama_model import create_ollama_model  # noqa: E402
from writing.config import ReviewConfig  # noqa: E402
from writing.graph import run_review_async  # noqa: E402
from writing.schemas import ReviewInput  # noqa: E402
from writing_prep import (  # noqa: E402
    build_outline, render_research_gap_section, write_literature_files,
)

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / ".writing_runs"

# Single-pass generation: this hardware is slow for large-context calls (see
# DECISIONS.md D-009), and the graph's default revision/audit rounds would
# multiply an already-expensive per-call cost many times over. Revisit once
# a faster model/machine is available — see DECISIONS.md D-010.
#
# max_section_revision_rounds=1 (not 0, see DECISIONS.md D-020): with 0
# rounds, write_section_with_revisions() returns normally (no exception) the
# instant its FIRST audit fails — the outer max_section_attempts retry only
# triggers on an exception, so it never engages for an audit failure. With
# revisions fully disabled, a single audit issue (e.g. one citation
# placeholder not matching cited_paper_ids) permanently failed a section
# with zero recovery attempts of any kind. A real stress-test run showed
# this as the dominant failure mode (4 of 5 sections failing on the exact
# same audit-detectable, already-correctable issue). One bounded revision
# round — using the already-tested audit-driven revise_section() path, only
# for sections that actually fail their first audit — is a small, targeted
# cost for a large reliability gain.
FAST_REVIEW_CONFIG = ReviewConfig(
    max_section_revision_rounds=1,
    # Empty-content sections are as useless as a failed card (see the
    # SectionDraft min_length fix in writing/schemas.py) — one retry is
    # worth the extra time given the alternative is a blank section.
    max_section_attempts=2,
    max_full_revision_rounds=0,
    # Card extraction is the evidence source for the whole draft — a failed
    # card means an evidence-free section, not just lower polish, so this one
    # keeps a retry unlike the section/full-review settings above.
    max_card_attempts=2,
    max_concurrency=2,
    semantic_audits=False,
)

DISCLAIMER_BANNER = (
    "> **AI-generated research draft for human review.** Not a verified or "
    "published research paper. Every claim must be checked against its "
    "cited source before any external use.\n\n"
)


async def run_writing(request: WritingRequest,
                      output_root: Path | None = None) -> AsyncIterator[dict]:
    """Yields progress events, then a final {"type": "result", "result": WritingResult}
    event. Raises RuntimeError (via the writing graph) on unrecoverable failure."""
    discovery: DiscoveryResult = request.discovery
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    run_id = uuid4().hex
    run_scratch = output_root / run_id
    literature_dir = run_scratch / "literature"

    yield {"type": "status", "stage": "prepare", "state": "running",
           "message": "Building outline and literature files from discovery results…"}
    outline = build_outline(discovery)
    paper_count = write_literature_files(discovery, literature_dir)
    yield {"type": "status", "stage": "prepare", "state": "done",
           "message": f"Prepared outline and {paper_count} literature file(s)."}

    review_input = ReviewInput(
        review_question=discovery.research_request.research_question,
        outline=outline,
        literature_directory=str(literature_dir),
        output_directory=str(output_root),
        output_language=request.output_language,
        citation_style=request.citation_style,
        target_words=request.target_words,
    )

    model = create_ollama_model()
    yield {"type": "status", "stage": "write", "state": "running",
           "message": "Writing the evidence-grounded draft (this can take a while)…"}
    try:
        result = await run_review_async(
            review_input, model, config=FAST_REVIEW_CONFIG, run_id=run_id,
        )
    finally:
        await close_language_model(model)

    for warning in result.warnings:
        yield {"type": "source_error", "source": "Research Writing", "message": warning}
    yield {"type": "status", "stage": "write", "state": "done" if result.succeeded else "failed",
           "message": "Draft complete." if result.succeeded
           else "Draft generation finished with errors — see warnings/errors."}

    run_directory = Path(result.run_directory)
    draft_text = ""
    if result.final_review_path:
        draft_text = Path(result.final_review_path).read_text(encoding="utf-8")

    # The outline (writing_prep.build_outline) deliberately omits a
    # per-gap literature-review structure — see DECISIONS.md D-011 — so the
    # Research Gap / Proposed Novelty content is rendered directly from
    # Service 1's own synthesis and spliced in here rather than being
    # written (and evidence-gated) by the writing graph itself.
    gap_section = render_research_gap_section(discovery)
    insertion_marker = "\n## Limitations"
    reference_marker = "\n## References"
    if insertion_marker in draft_text:
        idx = draft_text.index(insertion_marker)
    elif reference_marker in draft_text:
        idx = draft_text.index(reference_marker)
    else:
        idx = len(draft_text)
    draft_text = draft_text[:idx] + "\n" + gap_section + "\n" + draft_text[idx:]

    draft_text = DISCLAIMER_BANNER + draft_text

    citation_mapping: dict[str, CitationEntry] = {}
    reference_candidates: list[str] = []
    references_path = run_directory / "references.json"
    if references_path.exists():
        refs = json.loads(references_path.read_text(encoding="utf-8"))
        reference_candidates = refs.get("references", [])

    total_sections = sum(1 for line in outline.splitlines() if line.startswith("#"))
    failed_sections = len(result.failed_tag_ids)

    # Real completeness validation (Fix 6, DECISIONS.md D-026): inspect the
    # ACTUAL RENDERED draft_text — the same markdown that becomes draft.md /
    # paper.tex / the final PDF — rather than trusting only the graph's own
    # internal tag-resolution signal, which cannot see a blank bookend
    # Introduction/Conclusion (see completeness.py's docstring for the real
    # run that proved this gap).
    completeness = assess_completeness(
        draft_text, outline, output_language=request.output_language,
    )
    # "failed" always wins from the rendered-text check; otherwise fall back
    # to "partial" if the graph itself reported an unresolved failure that
    # didn't happen to render as one of completeness.py's known placeholder
    # strings (belt-and-braces — never silently trust only one signal).
    if completeness.status == "failed":
        draft_status = "failed"
    elif completeness.status == "partial" or not result.succeeded or failed_sections:
        draft_status = "partial"
    else:
        draft_status = "complete"

    actual_words = len(draft_text.split())
    length_status = classify_length(actual_words, request.target_words)

    metadata = DraftMetadata(
        model_name=llm_provider.DEFAULT_MODEL,
        run_id=run_id,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        warnings=result.warnings,
        errors=result.errors,
        papers_cited=len(reference_candidates),
        sections_written=max(total_sections - failed_sections, 0),
        total_sections=total_sections,
        failed_sections=failed_sections,
        draft_status=draft_status,
        target_words=request.target_words,
        actual_words=actual_words,
        length_status=length_status,
        evidence_limited_sections=completeness.evidence_limited_sections,
        broken_sections=sorted(
            set(completeness.missing_sections)
            | set(completeness.blank_sections)
            | set(completeness.failed_sections)
        ),
        duplicate_sections=completeness.duplicate_sections,
    )

    writing_result = WritingResult(
        research_outline=outline,
        research_draft_markdown=draft_text,
        citation_mapping=citation_mapping,
        reference_candidates=reference_candidates,
        run_directory=str(run_directory),
        draft_metadata=metadata,
    )
    yield {"type": "result", "result": writing_result}
