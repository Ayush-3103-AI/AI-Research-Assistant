"""Step 1 — publication-velocity analysis: which subfields are actually growing.

A standalone, tool-using agent step (course skill 1): it decides its own
queries from a domain alone, calls real scholarly-database tools, and returns
evidence. It has no dependency on the chat loop, the service wrapper, the
orchestrator, or the CLI, and can be run and tested entirely on its own.

The whole ranking is arithmetic over live OpenAlex counts — the model is
never asked which topics are rising, because a model's answer to that is
unverifiable and drifts between runs. Likewise the domain -> subfield table
below is hardcoded and verified against the live OpenAlex taxonomy rather
than inferred at runtime (spec user story 19).
"""

from __future__ import annotations

import asyncio
import datetime
import re
import sys
from itertools import zip_longest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery import fulltext  # noqa: E402
from discovery.sources import arxiv, openalex  # noqa: E402
from shared.contracts.trend_contract import ExamplePaper, TopicCandidate  # noqa: E402

# Verified against https://api.openalex.org/fields/{17,21,22} on 2026-09-12 —
# every id below was read back from OpenAlex's own subfield list for its
# field, not guessed. Each domain lists the subfields a student in that
# department would recognize as theirs; overlaps (2208 in both ECE and EEE,
# 1712 in both CS and Cloud/DevOps) are deliberate, because the departments
# really do overlap there.
DOMAIN_SUBFIELDS: dict[str, list[str]] = {
    # Artificial Intelligence, Computer Vision, CS Applications, Software
    "CS_AI_ML": ["1702", "1707", "1706", "1712"],
    # Electrical & Electronic Eng, Computer Networks & Comms, Signal
    # Processing, Hardware & Architecture
    "ECE": ["2208", "1705", "1711", "1708"],
    # Electrical & Electronic Eng, Energy Eng & Power Technology, Renewable
    # Energy, Control & Systems Eng
    "EEE": ["2208", "2102", "2105", "2207"],
    # Mechanical Eng, Mechanics of Materials, Aerospace Eng, Industrial &
    # Manufacturing Eng
    "MECHANICAL": ["2210", "2211", "2202", "2209"],
    # Civil & Structural Eng, Building & Construction, Architecture, Ocean Eng
    "CIVIL": ["2205", "2215", "2216", "2212"],
    # Information Systems, Software, Computer Networks & Comms, Hardware &
    # Architecture
    "CLOUD_DEVOPS": ["1710", "1712", "1705", "1708"],
}

# Domains whose work genuinely preprints on arXiv in volume. Everything else
# gets no arXiv signal at all rather than a misleadingly thin one.
ARXIV_DOMAINS = frozenset({"CS_AI_ML", "ECE", "CLOUD_DEVOPS"})

WINDOW_YEARS = 3
# Below this many works in the recent window, a growth ratio is noise: three
# papers where there was one is a 3x "rise" that means nothing. Set well
# above that floor so the shortlist only contains topics a student could
# actually assemble a corpus in.
MIN_RECENT_WORKS = 200
# A domain that cannot produce at least this many rankable topics is a
# failure to report, not a short list to present (spec user story 18).
MIN_TOPICS_FOR_SHORTLIST = 3
EXAMPLE_PAPERS_PER_TOPIC = 5
# arXiv's Atom API ranks by relevance and has no count endpoint, so the
# preprint signal is a sample of this many top hits, not a census. That is
# why it never enters growth_metric — see _growth_key.
ARXIV_SEARCH_LIMIT = 50
# How many papers per topic get a full-text fetch for their Discussion /
# Limitations / Future-research sections. Matches gap_mining's
# MAX_PAPERS_PER_TOPIC — fetching text for papers the swarm will never read
# would be pure latency. Retrieval reuses Discovery's existing extractor.
GAP_TEXT_PAPERS_PER_TOPIC = 2
FULLTEXT_CONCURRENCY = 8


class VelocityUnavailable(RuntimeError):
    """OpenAlex was unreachable, or the domain yielded too little data to
    rank honestly. Raised instead of returning an empty or padded shortlist,
    so the advisor never shows a student confidence it does not have."""


def windows(current_year: int | None = None) -> tuple[tuple[int, int], tuple[int, int]]:
    """The (recent, prior) year windows to compare.

    The current year is excluded entirely: indexing lags, so a partial year
    always looks like a collapse in publication volume and would invert
    every ranking.
    """
    year = current_year or datetime.date.today().year
    last_complete = year - 1
    recent = (last_complete - WINDOW_YEARS + 1, last_complete)
    prior = (recent[0] - WINDOW_YEARS, recent[0] - 1)
    return recent, prior


def _normalize(name: str) -> str:
    """The cross-source matching key. OpenAlex and arXiv spell the same topic
    and the same paper title with different case, punctuation and hyphenation,
    so raw string equality would treat one thing as two."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _growth_key(candidate: TopicCandidate) -> tuple[float, int]:
    """Growth first, then how much recent work exists across *both* sources.

    arXiv only ever moves the volume half of the key. Its count comes from a
    relevance-ranked sample rather than a full corpus, so folding it into
    growth_metric would put a sampled number inside the one ratio a student is
    told they can re-derive from OpenAlex themselves.
    """
    return (candidate.growth_metric,
            candidate.paper_count + (candidate.arxiv_recent_count or 0))


def rank_by_growth(*, recent: dict[str, tuple[str, int]],
                   prior: dict[str, tuple[str, int]],
                   limit: int) -> list[TopicCandidate]:
    """Rank topics by how much faster they are published now than before.

    growth = recent / max(prior, 1). A topic absent from the prior window
    divides by 1, which makes a genuinely new topic rank by its own recent
    volume rather than by an infinity.
    """
    candidates = []
    for topic_id, (name, recent_count) in recent.items():
        if recent_count < MIN_RECENT_WORKS:
            continue
        prior_count = prior.get(topic_id, ("", 0))[1]
        candidates.append(TopicCandidate(
            topic=name or topic_id,
            topic_id=topic_id,
            growth_metric=round(recent_count / max(prior_count, 1), 2),
            paper_count=recent_count,
            prior_papers=prior_count,
            source="velocity",
        ))
    candidates.sort(key=_growth_key, reverse=True)
    return candidates[:limit]


def merge_sources(primary: list[TopicCandidate],
                  secondary: list[TopicCandidate]) -> list[TopicCandidate]:
    """Fold a second source's candidates into the first by normalized topic.

    A topic both sources found stays ONE shortlist entry. The primary keeps
    its own paper_count: OpenAlex counts a whole indexed corpus and arXiv a
    sample of preprints, so adding them would report a volume neither source
    measured. The secondary's count lands in arxiv_recent_count instead, where
    it is labelled for what it is. Example papers union, deduplicated by
    title, and interleaved rather than appended: gap mining reads only the
    first MAX_PAPERS_PER_TOPIC papers that have text, and every OpenAlex
    paper has one (works_for_topic filters on has_abstract), so appending
    put the preprints permanently out of the swarm's reach.
    """
    merged: dict[str, TopicCandidate] = {}
    for candidate in primary + secondary:
        existing = merged.get(_normalize(candidate.topic))
        if existing is None:
            merged[_normalize(candidate.topic)] = candidate
            continue
        if candidate.arxiv_recent_count is not None:
            existing.arxiv_recent_count = candidate.arxiv_recent_count
        seen = {_normalize(p.title) for p in existing.example_papers}
        added = []
        for paper in candidate.example_papers:
            if _normalize(paper.title) not in seen:
                seen.add(_normalize(paper.title))
                added.append(paper)
        existing.example_papers = [
            paper
            for pair in zip_longest(existing.example_papers, added)
            for paper in pair if paper is not None
        ]
    return list(merged.values())


async def _arxiv_candidate(http: httpx.AsyncClient, topic: str,
                           year_range: tuple[int, int]) -> TopicCandidate | None:
    """arXiv's own evidence for one topic: how many of its top hits fall in
    the same window OpenAlex was counted over, plus the preprints themselves
    as example papers. Reuses Discovery's existing arXiv connector rather
    than adding a second client. Returns None (never an empty candidate) when
    arXiv fails, so an outage is never mistaken for an absence of preprints.

    `year_range` is the OpenAlex recent window, upper bound included: the two
    counts are added in _growth_key, so an open-ended arXiv filter would let
    the current partial year through on one side of a sum whose other side
    deliberately excludes it.

    growth_metric stays 0.0 rather than being computed from the sample: a
    relevance-ranked top-50 cannot support a recent-vs-prior ratio, and
    inventing one would let a sampled number outrank counted ones.
    paper_count stays 0 for the same reason it is never summed in
    merge_sources: it is OpenAlex's measured volume, and arXiv has not
    measured it. The sample lands in arxiv_recent_count alone.
    """
    start, end = year_range
    try:
        papers = await arxiv.search(http, {"arxiv": topic},
                                    limit=ARXIV_SEARCH_LIMIT)
    except (httpx.HTTPError, ValueError):
        return None
    recent = [p for p in papers if p.year and start <= p.year <= end]
    return TopicCandidate(
        topic=topic,
        growth_metric=0.0,
        paper_count=0,
        arxiv_recent_count=len(recent),
        example_papers=[
            ExamplePaper(title=p.title, doi=p.doi, year=p.year,
                         citations=p.citations, abstract=p.abstract)
            for p in recent[:EXAMPLE_PAPERS_PER_TOPIC]
        ],
    )


async def _attach_future_work_text(http: httpx.AsyncClient,
                                   works_by_topic: list[list]) -> None:
    """Populate `paper.future_text` for the papers gap mining will read.

    Authors state what is still unsolved in a paper's Discussion /
    Limitations / Future-research sections, not in its abstract — so mining
    abstracts alone would weaken the one signal this stage exists to
    provide. Reuses Discovery's `fulltext` module (Europe PMC JATS, then an
    open-access PDF), which never raises and simply leaves `future_text`
    unset when no open-access copy is reachable; gap mining then falls back
    to the abstract for that paper.
    """
    targets = [
        paper
        for works in works_by_topic
        for paper in works[:GAP_TEXT_PAPERS_PER_TOPIC]
        if fulltext.is_candidate(paper)
    ]
    if not targets:
        return
    sem = asyncio.Semaphore(FULLTEXT_CONCURRENCY)

    async def one(paper) -> None:
        async with sem:
            await fulltext.fetch_future_sections(http, paper)

    await asyncio.gather(*(one(paper) for paper in targets))


async def find_rising_topics(
    domain: str, *, limit: int = 8, domain_other_name: str | None = None,
    transport: httpx.BaseTransport | None = None,
    current_year: int | None = None,
) -> list[TopicCandidate]:
    """Rank the fastest-growing topics in `domain`, each with real example papers.

    Raises VelocityUnavailable when OpenAlex cannot be reached or the domain
    is too thin to rank. `transport` is a test seam (httpx.MockTransport);
    production callers leave it None.
    """
    subfields = DOMAIN_SUBFIELDS.get(domain)
    label = domain_other_name or domain
    search = None if subfields else label
    recent_window, prior_window = windows(current_year)

    async with httpx.AsyncClient(timeout=60, transport=transport,
                                 follow_redirects=True) as http:
        try:
            recent, prior = await asyncio.gather(
                openalex.topic_counts(http, year_range=recent_window,
                                      subfield_ids=subfields, search=search),
                openalex.topic_counts(http, year_range=prior_window,
                                      subfield_ids=subfields, search=search),
            )
        except httpx.HTTPError as error:
            raise VelocityUnavailable(
                f"Could not reach OpenAlex to analyse publication velocity: {error}"
            ) from None

        ranked = rank_by_growth(recent=recent, prior=prior, limit=limit)
        if len(ranked) < MIN_TOPICS_FOR_SHORTLIST:
            raise VelocityUnavailable(
                f"OpenAlex returned too few rankable topics for '{label}' "
                f"({len(ranked)} with at least {MIN_RECENT_WORKS} recent works). "
                "No shortlist can be produced honestly for this domain."
            )

        # One round-trip per topic, issued together: these are independent
        # reads of the same API that already served the two count queries
        # above concurrently.
        per_topic_works = await asyncio.gather(*(
            openalex.works_for_topic(http, candidate.topic_id,
                                     year_range=recent_window,
                                     limit=EXAMPLE_PAPERS_PER_TOPIC)
            for candidate in ranked
        ))
        await _attach_future_work_text(http, per_topic_works)
        for candidate, works in zip(ranked, per_topic_works):
            candidate.example_papers = [
                ExamplePaper(title=w.title, doi=w.doi, year=w.year,
                             citations=w.citations, abstract=w.abstract,
                             future_text=getattr(w, "future_text", None))
                for w in works
            ]

        if domain in ARXIV_DOMAINS:
            # Deliberately sequential, unlike the OpenAlex reads above: arXiv
            # asks callers to space requests out, and this is a secondary
            # source not worth risking a rate-limit ban over.
            from_arxiv = []
            for candidate in ranked:
                found = await _arxiv_candidate(http, candidate.topic,
                                               recent_window)
                if found is not None:
                    from_arxiv.append(found)
            ranked = merge_sources(ranked, from_arxiv)
            ranked.sort(key=_growth_key, reverse=True)
            ranked = ranked[:limit]
            # Merging collapses topics whose names normalize alike, so the
            # shortlist can come out thinner than what was already checked
            # above. Re-check rather than under-deliver quietly.
            if len(ranked) < MIN_TOPICS_FOR_SHORTLIST:
                raise VelocityUnavailable(
                    f"Merging the OpenAlex and arXiv candidates for "
                    f"'{label}' left only {len(ranked)} "
                    "distinct topic(s). No shortlist can be produced honestly "
                    "for this domain."
                )

    return ranked
