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
import sys
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
    candidates.sort(key=lambda c: (c.growth_metric, c.paper_count), reverse=True)
    return candidates[:limit]


async def _arxiv_recent_count(http: httpx.AsyncClient, topic: str,
                              since_year: int) -> int | None:
    """Secondary signal: how many of arXiv's top hits for this topic are
    recent. Reuses Discovery's existing arXiv connector rather than adding a
    second client. Returns None (never 0) when arXiv fails, so an outage is
    never mistaken for an absence of preprints."""
    try:
        papers = await arxiv.search(http, {"arxiv": topic}, limit=50)
    except (httpx.HTTPError, ValueError):
        return None
    return sum(1 for p in papers if p.year and p.year >= since_year)


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
    search = None if subfields else (domain_other_name or domain)
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
            label = domain_other_name or domain
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
            # annotation not worth risking a rate-limit ban over.
            for candidate in ranked:
                candidate.arxiv_recent_count = await _arxiv_recent_count(
                    http, candidate.topic, recent_window[0],
                )

    return ranked
