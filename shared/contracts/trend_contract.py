"""Input/output contract for the Trend & Gap Advisor Service (pipeline stage 0).

Stage 0 runs *before* Research Discovery and is entirely optional: it turns
"which topic should I even research?" into a ranked, evidence-backed
shortlist, whose chosen entry then auto-fills a ResearchRequest. Defined
here — beside the four existing per-service contracts — so the advisor
service, the advisor CLI, and any future consumer depend on a stable shape
rather than on the service's internals.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DomainChoice = Literal[
    "CS_AI_ML", "ECE", "EEE", "MECHANICAL", "CIVIL", "CLOUD_DEVOPS", "Other",
]

CandidateSource = Literal["velocity", "gap_mining", "both"]

DEFAULT_SHORTLIST_SIZE = 8


class TrendAdvisorRequest(BaseModel):
    """Input to the Trend & Gap Advisor Service."""

    domain: DomainChoice
    # Mirrors WritingRequest.target_format/format_other_name exactly: a domain
    # outside the curated picklist is still usable, it just loses the verified
    # OpenAlex subfield mapping (see trend_advisor/velocity.py::DOMAIN_SUBFIELDS)
    # and falls back to a free-text OpenAlex search, which is less precise.
    domain_other_name: str | None = None
    # 6-9 matches DiscoveryRequest.corpus_size's own convention — the shortlist
    # is meant to be scannable in one screen, not exhaustive.
    shortlist_size: int = Field(default=DEFAULT_SHORTLIST_SIZE, ge=6, le=9)

    @model_validator(mode="after")
    def _require_domain_other_name(self) -> "TrendAdvisorRequest":
        if self.domain == "Other" and not self.domain_other_name:
            raise ValueError("domain_other_name is required when domain is 'Other'")
        return self

    @property
    def domain_label(self) -> str:
        return self.domain_other_name or self.domain


class ExamplePaper(BaseModel):
    """One real, independently checkable paper behind a topic's ranking."""

    title: str
    doi: str | None = None
    year: int | None = None
    citations: int | None = None
    # Carried so the gap-mining swarm can read each paper's own words without
    # re-fetching it; the CLI never prints these, it only prints title/DOI.
    abstract: str | None = None
    # The paper's own Discussion/Limitations/Future-research text, when it
    # could be retrieved from an open-access copy. This — not the abstract —
    # is where authors actually state what remains unsolved, so gap mining
    # prefers it and falls back to the abstract only when it is absent.
    future_text: str | None = None


# Threshold for the "hot but crowded" vs "under-explored gap" split (user
# story 5). Chosen from the real OpenAlex distribution probed while building
# this stage: within one engineering subfield pair over a 3-year window, the
# head topics run 30k-50k works and the long tail runs in the hundreds, so a
# few thousand separates "everyone is already here" from "quiet corner".
CROWDED_PAPER_COUNT = 3000


class TopicCandidate(BaseModel):
    """One shortlisted topic plus every number and quote it was ranked on.

    Every field here is computed from live OpenAlex/arXiv data — nothing is
    asserted by the model. A student must be able to re-run the same queries
    and get the same evidence (user stories 3 and 13).
    """

    topic: str
    # OpenAlex topic id (e.g. "T10028"), so the evidence is re-queryable.
    topic_id: str | None = None
    # recent_papers / max(prior_papers, 1) — >1.0 means the topic is growing.
    growth_metric: float
    # Works published in the recent window; the "crowded vs quiet" axis that
    # growth_metric alone cannot express (user story 5).
    paper_count: int
    # The same count for the preceding window. Together with paper_count this
    # is the whole growth calculation, shown so a student can redo the
    # division themselves.
    prior_papers: int = 0
    example_papers: list[ExamplePaper] = Field(default_factory=list)
    # True when independent papers' own future-work/limitations sections
    # repeatedly flag this as unmet — a different signal from publication
    # volume, and the one that marks a genuine gap rather than a crowd.
    gap_signal: bool = False
    # One concise statement per gap a paper raised, in the extracting model's
    # words rather than verbatim quotation (see gap_mining.GAP_SYSTEM_PROMPT).
    # A topic can contribute several from the same paper, so the count here is
    # not the number of papers that flagged it.
    gap_evidence: list[str] = Field(default_factory=list)
    # Secondary, CS-adjacent-domains-only signal; None when not applicable
    # or when arXiv was unreachable (never silently zero — see spec's
    # "visible, honest failure" requirement).
    arxiv_recent_count: int | None = None
    source: CandidateSource = "velocity"

    @property
    def is_crowded(self) -> bool:
        """A high-volume topic: competing here means competing with many."""
        return self.paper_count >= CROWDED_PAPER_COUNT


class TrendAdvisorResult(BaseModel):
    """Output of the Trend & Gap Advisor Service."""

    domain: DomainChoice
    domain_other_name: str | None = None
    shortlist: list[TopicCandidate]
    generated_at: str
    # Surfaced, not swallowed: an unreachable arXiv or a thin domain must be
    # visible to the student rather than presented as confident output.
    warnings: list[str] = Field(default_factory=list)
