"""Validated data exchanged between workflow modules."""

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base model that rejects unrecognised fields."""

    model_config = ConfigDict(extra="forbid")


def _normalize_paper_id(value: object) -> object:
    """Strip citation-placeholder decoration a model may echo into a
    structured cited_paper_ids field instead of the bare paper ID.

    Root cause (DECISIONS.md D-021): prose citations are written as
    `[@P001]` placeholders, and models writing this schema sometimes copy
    that decorated form (`"@P001"`, or occasionally `"[@P001]"`) into the
    separate `cited_paper_ids` list instead of the bare ID (`"P001"`) the
    rest of the pipeline expects. Every downstream comparison — against
    `allowed_paper_ids`, against the placeholder IDs extracted from prose —
    is a set-equality check, so an undecorated vs. decorated ID mismatch
    makes an otherwise fully valid, in-scope citation look like a forbidden
    one. Normalizing at the schema boundary makes every comparison
    deterministic regardless of which decorated form a model happens to use.
    """
    if not isinstance(value, str):
        return value
    return value.strip().strip("[]").lstrip("@").strip()


def _normalize_paper_id_list(values: object) -> object:
    if not isinstance(values, list):
        return values
    return [_normalize_paper_id(item) for item in values]


CitationStyle: TypeAlias = Literal[
    "elsevier-harvard",
    "apa-7",
    "chicago-author-date",
    "mla-9",
    "ieee",
    "vancouver",
]


class ReviewInput(StrictModel):
    """Inputs required to start a literature-review run."""

    review_question: str = Field(min_length=1)
    outline: str = Field(min_length=1)
    literature_directory: str = Field(min_length=1)
    output_directory: str = Field(min_length=1)
    output_language: str = Field(default="en", min_length=1)
    citation_style: CitationStyle = "elsevier-harvard"
    target_words: int | None = Field(default=None, gt=0)

    @field_validator("citation_style", mode="before")
    @classmethod
    def normalize_legacy_citation_style(cls, value: object) -> object:
        """Preserve existing request files while changing the named default."""

        return "elsevier-harvard" if value == "author-year" else value


class ParsedHeading(StrictModel):
    """A syntactically parsed heading before tag IDs are assigned."""

    level: int
    number: str | None = None
    title: str


class OutlineNode(StrictModel):
    """A node in the deterministic outline/tag tree."""

    tag_id: str
    number: str | None = None
    title: str
    normalized_label: str | None = None
    parent_id: str | None = None
    child_ids: list[str]
    ancestor_ids: list[str]
    depth: int
    order: int
    path_titles: list[str]


class TagDefinition(StrictModel):
    """Semantic meaning assigned to one immutable outline tag."""

    tag_id: str
    title: str
    normalized_label: str
    description: str
    include_when: list[str]
    exclude_when: list[str]
    parent_id: str | None = None
    child_ids: list[str]
    depth: int
    node_type: Literal["root", "container", "content", "mixed"]


class TagSemanticValue(StrictModel):
    """The only tag fields an LLM is allowed to decide."""

    tag_id: str
    normalized_label: str
    description: str
    include_when: list[str]
    exclude_when: list[str]
    node_type: Literal["root", "container", "content", "mixed"]


class TagSemanticBatch(StrictModel):
    """Structured model output for all outline tags."""

    tags: list[TagSemanticValue]


class PaperMetadata(StrictModel):
    """Paper-level bibliographic metadata."""

    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    year: int | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    paper_type: str | None = None


class PartialPaperMetadata(StrictModel):
    """Bibliographic fields known before optional model completion."""

    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    paper_type: str | None = None


EvidenceDepth = Literal["full_text", "abstract", "abstract_plus_discussion"]
"""How much of a source paper the evidence packet actually contains.

`abstract_plus_discussion` is the abstract plus the authors' own retrieved
Discussion/Limitations/Future-work sections — more than an abstract, and
genuinely less than the whole paper (see DECISIONS.md D-039). The card prompt
(`prompts/build_literature_card.md`) defines what each value permits; adding a
value here means teaching that prompt about it too."""


class PaperDocument(StrictModel):
    """A stable registry entry for one Markdown source document."""

    paper_id: str = Field(pattern=r"^P\d{3,}$")
    source_path: str
    relative_path: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_depth: EvidenceDepth = "full_text"
    metadata: PartialPaperMetadata


class CitationInfo(StrictModel):
    """Preformatted paper-level citation variants."""

    citation_key: str = Field(min_length=1)
    in_text_citation: str = Field(min_length=1)
    narrative_citation: str = Field(min_length=1)
    reference_entry: str = Field(min_length=1)


class TaggedPoint(StrictModel):
    """One faithful, paper-scoped claim available to later writing."""

    point_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    content_type: Literal[
        "definition",
        "theoretical_argument",
        "research_question",
        "method",
        "empirical_finding",
        "author_interpretation",
        "stated_limitation",
        "future_direction",
        "background",
    ]
    stance: Literal[
        "supporting",
        "contradicting",
        "mixed",
        "neutral",
        "not_applicable",
    ] = "neutral"
    certainty: Literal["established", "supported", "suggested", "speculative"]


class LiteratureCard(StrictModel):
    """All review-relevant points extracted from one paper."""

    paper_id: str = Field(pattern=r"^P\d{3,}$")
    metadata: PaperMetadata
    citation: CitationInfo
    points: dict[str, TaggedPoint] = Field(
        description="Keyed by each point's OWN point_id (e.g. 'P003-POINT-01'), "
                    "matching that point's point_id field exactly. Never key this "
                    "by a tag ID."
    )
    tag_values: dict[str, list[str]] = Field(
        description="Keyed by TAG ID (e.g. 'TAG-ROOT', 'TAG-1') — one entry for "
                    "every supplied tag, in the same order they were given. Each "
                    "value is a list of point_id strings that must all exist as "
                    "keys in `points`. Never key this by a point ID."
    )


class LiteratureCardContent(StrictModel):
    """Model-generated card fields before deterministic paper ID attachment."""

    metadata: PaperMetadata
    citation: CitationInfo
    points: dict[str, TaggedPoint] = Field(
        description="Keyed by each point's OWN point_id (e.g. 'P003-POINT-01'), "
                    "matching that point's point_id field exactly. Never key this "
                    "by a tag ID."
    )
    tag_values: dict[str, list[str]] = Field(
        description="Keyed by TAG ID (e.g. 'TAG-ROOT', 'TAG-1') — one entry for "
                    "every supplied tag, in the same order they were given. Each "
                    "value is a list of point_id strings that must all exist as "
                    "keys in `points`. Never key this by a point ID."
    )


class CardAudit(StrictModel):
    """Deterministic validation result for a literature card."""

    paper_id: str
    passed: bool
    errors: list[str]
    warnings: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)
    semantic_checked: bool = False
    attempts: int = Field(default=1, ge=1)


class CardSemanticAuditContent(StrictModel):
    """Structured semantic findings for one literature card."""

    passed: bool
    errors: list[str]
    revision_instructions: list[str]


class TagIndexItem(StrictModel):
    """One card point expanded under one directly assigned tag."""

    tag_id: str
    paper_id: str
    point_id: str
    content: str
    content_type: str
    stance: str
    certainty: str
    citation: CitationInfo


TagIndex: TypeAlias = dict[str, list[TagIndexItem]]


class TagCoverage(StrictModel):
    """Evidence availability for one tag, including descendants."""

    tag_id: str
    direct_point_count: int
    rollup_point_count: int
    paper_ids: list[str]
    insufficient_evidence: bool


class TagCoverageReport(StrictModel):
    """Coverage results for the complete immutable tag tree."""

    tags: dict[str, TagCoverage]
    insufficient_tag_ids: list[str]


class SectionSummary(StrictModel):
    """Compact, source-preserving summary of a completed child section."""

    tag_id: str
    summary: str
    cited_paper_ids: list[str]
    used_point_ids: list[str]
    # True when this section's evidence pool was too small to meaningfully
    # develop, so a short result is expected and not a generation failure —
    # see DECISIONS.md D-026. Distinguishes "genuinely thin evidence" from
    # "model under-generated despite having enough to work with".
    evidence_limited: bool = False


class SectionWritingContext(StrictModel):
    """The complete evidence and citation allowlist for one writing call."""

    tag_id: str
    title: str
    outline_path: list[str]
    depth: int
    writing_mode: Literal["leaf_section", "parent_intro", "subtree"]
    direct_points: list[TagIndexItem]
    ancestor_context: list[TagIndexItem]
    child_summaries: list[str]
    allowed_paper_ids: list[str]
    allowed_point_ids: list[str] = Field(default_factory=list)
    citations: dict[str, CitationInfo]
    sibling_titles: list[str]
    prohibited_topics: list[str]
    # target_words: the section's planned length (a target, not a hard
    # ceiling — see DECISIONS.md D-025/D-026). min_words/max_words bound it;
    # all three are None when no total paper length target was requested,
    # exactly preserving pre-D-025 behavior (no length guidance at all).
    target_words: int | None = None
    min_words: int | None = None
    max_words: int | None = None
    output_language: str = "en"


class SectionDraft(StrictModel):
    """One model-written section plus explicit provenance lists."""

    tag_id: str
    content: str = Field(min_length=1)
    cited_paper_ids: list[str]
    used_point_ids: list[str]
    summary: str | None = None

    @field_validator("cited_paper_ids", mode="before")
    @classmethod
    def _normalize_cited_paper_ids(cls, value: object) -> object:
        return _normalize_paper_id_list(value)


class SectionDraftContent(StrictModel):
    """Model-written fields before deterministic tag ID attachment."""

    content: str = Field(min_length=1)
    cited_paper_ids: list[str]
    used_point_ids: list[str]
    summary: str | None = None

    @field_validator("cited_paper_ids", mode="before")
    @classmethod
    def _normalize_cited_paper_ids(cls, value: object) -> object:
        return _normalize_paper_id_list(value)


class SectionAudit(StrictModel):
    """Deterministic or semantic audit result for one section."""

    tag_id: str
    passed: bool
    unsupported_claims: list[str]
    out_of_scope_content: list[str]
    invalid_paper_ids: list[str]
    missing_key_points: list[str]
    duplicated_child_content: list[str]
    revision_instructions: list[str]
    # True when this section's word count fell below its planned minimum
    # despite substantive, genuinely-evidenced content — a length problem,
    # tracked separately from out_of_scope_content/unsupported_claims
    # (correctness problems) so callers can tell "under-generated" apart
    # from "wrong". Still fails the audit (passed=False) so it drives the
    # existing bounded revision loop with a targeted "expand" instruction,
    # rather than a silent, permanent short section. See DECISIONS.md D-026.
    below_minimum_length: bool = False


class SectionAuditContent(StrictModel):
    """Structured semantic audit fields before deterministic tag attachment."""

    passed: bool
    unsupported_claims: list[str]
    out_of_scope_content: list[str]
    invalid_paper_ids: list[str]
    missing_key_points: list[str]
    duplicated_child_content: list[str]
    revision_instructions: list[str]


class SectionRunResult(StrictModel):
    """Final draft plus bounded audit history and unresolved issues."""

    draft: SectionDraft
    audits: list[SectionAudit]
    resolved: bool
    unresolved_issues: list[str]
    # True when, after every bounded revision round, the ONLY remaining
    # audit failure is below_minimum_length (no unsupported/out-of-scope/
    # invalid/duplicated content). This means the model genuinely ran out
    # of supported evidence to expand with, not that it produced something
    # wrong — see Fix 9 "Case B" (DECISIONS.md D-026). Such a section is
    # still `resolved=True` (its shorter content is kept, never blanked),
    # just flagged so callers/analytics can distinguish it from an
    # under-generation failure or a real correctness failure.
    evidence_limited: bool = False


class ReviewDocument(StrictModel):
    """Fully assembled review and its paper-level references."""

    markdown: str
    cited_paper_ids: list[str]
    references: list[str]


class VisualRecommendation(StrictModel):
    """A human-actionable figure or table placeholder requested by full audit."""

    tag_id: str = Field(min_length=1)
    visual_type: Literal["figure", "table"]
    prompt: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    placement_after: str = Field(min_length=1)
    source_point_ids: list[str] = Field(min_length=1)


class FullReviewAudit(StrictModel):
    """Deterministic audit of final review structure and provenance."""

    passed: bool
    missing_sections: list[str]
    heading_errors: list[str]
    missing_reference_paper_ids: list[str]
    duplicate_references: list[str]
    unresolved_placeholders: list[str]
    new_conclusion_paper_ids: list[str]
    new_conclusion_point_ids: list[str]
    errors: list[str]
    revision_instructions: list[str]
    semantic_checked: bool = False
    semantic_issues: list[str] = Field(default_factory=list)
    visual_recommendations: list[VisualRecommendation] = Field(default_factory=list)


class FullReviewSemanticAudit(StrictModel):
    """Structured LLM audit for cross-section semantic coherence."""

    passed: bool
    issues: list[str]
    revision_instructions: list[str]
    visual_recommendations: list[VisualRecommendation] = Field(default_factory=list)


class VisualRecommendationEvidenceAuditItem(StrictModel):
    """Evidence-support verdict for one indexed visual recommendation."""

    recommendation_index: int = Field(ge=0)
    supported: bool
    issues: list[str]


class VisualRecommendationEvidenceAudit(StrictModel):
    """Batch of fail-closed visual recommendation evidence verdicts."""

    items: list[VisualRecommendationEvidenceAuditItem]


class ReviewRevisionContent(StrictModel):
    """Editable prose fields for a structure-preserving full-review revision."""

    introduction: SectionDraftContent
    sections: dict[str, SectionDraftContent]
    conclusion: SectionDraftContent


class ReviewResult(StrictModel):
    """Public result of a completed or partially completed graph run."""

    run_id: str
    succeeded: bool
    run_directory: str
    final_review_path: str | None = None
    warnings: list[str]
    errors: list[str]
    failed_paper_ids: list[str]
    failed_tag_ids: list[str]
