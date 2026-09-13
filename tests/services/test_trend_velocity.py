"""Unit tests for the Trend & Gap Advisor's publication-velocity ranking.

The ranking math is the only genuinely novel reasoning in this step (the
HTTP layer is Discovery's already-tested OpenAlex connector), so it is
tested here against mocked OpenAlex responses with known counts — no
network, no model. The live counterpart lives in
test_trend_velocity_live.py and skips without network.
"""

import asyncio
import re
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))
sys.path.insert(0, str(ROOT / "services" / "trend-advisor"))

from shared.contracts.trend_contract import ExamplePaper, TopicCandidate  # noqa: E402
from trend_advisor.gap_mining import MAX_PAPERS_PER_TOPIC, mine_gaps  # noqa: E402
from trend_advisor.velocity import (  # noqa: E402
    DOMAIN_SUBFIELDS, MIN_RECENT_WORKS, MIN_TOPICS_FOR_SHORTLIST,
    VelocityUnavailable, _arxiv_candidate, _growth_key, find_rising_topics,
    merge_sources, rank_by_growth, windows,
)


def _groups(counts: dict[str, tuple[str, int]]) -> dict:
    return {"group_by": [
        {"key": f"https://openalex.org/{tid}", "key_display_name": name, "count": n}
        for tid, (name, n) in counts.items()
    ]}


def _work(title: str) -> dict:
    return {
        "display_name": title, "publication_year": 2024, "cited_by_count": 10,
        "doi": "https://doi.org/10.1000/x", "abstract_inverted_index": {"An": [0], "abstract": [1]},
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "primary_location": {"source": {"display_name": "A Journal"}},
    }


def _arxiv_feed(*entries: tuple[str, int]) -> str:
    body = "".join(
        f"<entry><title>{title}</title><summary>Preprint text.</summary>"
        f"<published>{year}-01-01T00:00:00Z</published>"
        f"<id>http://arxiv.org/abs/{year}.0000{i}</id></entry>"
        for i, (title, year) in enumerate(entries)
    )
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{body}</feed>'


def _transport(recent: dict, prior: dict,
               arxiv: tuple = (("A Preprint", 2024),),
               works: tuple = ("A Paper",)) -> httpx.MockTransport:
    """Answers the two group_by window queries by the year range in the
    filter, every per-topic works query with `works`, and the CS-adjacent
    domains' arXiv queries with a fixed preprint feed."""
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "arxiv.org" in str(request.url):
            return httpx.Response(200, text=_arxiv_feed(*arxiv))
        if "group_by" in params:
            start = params["filter"].split("publication_year:")[1].split("-")[0]
            body = _groups(recent) if int(start) >= 2023 else _groups(prior)
            return httpx.Response(200, json=body)
        return httpx.Response(200, json={"results": [_work(t) for t in works]})
    return httpx.MockTransport(handler)


# --- rank_by_growth: the ranking math itself -------------------------------

def test_growth_is_recent_over_prior():
    ranked = rank_by_growth(
        recent={"T1": ("Growing", 600)}, prior={"T1": ("Growing", 200)}, limit=5,
    )
    assert ranked[0].growth_metric == pytest.approx(3.0)
    assert ranked[0].paper_count == 600
    assert ranked[0].paper_count == 600 and ranked[0].prior_papers == 200


def test_ranks_by_growth_not_by_volume():
    """A huge, flat topic must lose to a smaller, fast-growing one —
    otherwise the shortlist just re-lists whatever is already biggest."""
    ranked = rank_by_growth(
        recent={"BIG": ("Crowded", 50_000), "FAST": ("Rising", 900)},
        prior={"BIG": ("Crowded", 49_000), "FAST": ("Rising", 150)},
        limit=5,
    )
    assert [c.topic for c in ranked] == ["Rising", "Crowded"]


def test_drops_topics_below_the_recent_volume_floor():
    """A topic with almost no recent work has an enormous but meaningless
    growth ratio (3 works vs 1) and must not top the shortlist."""
    ranked = rank_by_growth(
        recent={"NOISE": ("Noise", MIN_RECENT_WORKS - 1), "REAL": ("Real", 800)},
        prior={"NOISE": ("Noise", 1), "REAL": ("Real", 400)},
        limit=5,
    )
    assert [c.topic for c in ranked] == ["Real"]


def test_brand_new_topic_with_no_prior_work_is_ranked_not_divided_by_zero():
    ranked = rank_by_growth(
        recent={"NEW": ("Brand New", 900)}, prior={}, limit=5,
    )
    assert ranked[0].growth_metric == pytest.approx(900.0)
    assert ranked[0].prior_papers == 0


def test_shrinking_topic_keeps_its_real_sub_one_growth_metric():
    ranked = rank_by_growth(
        recent={"OLD": ("Fading", 500)}, prior={"OLD": ("Fading", 2000)}, limit=5,
    )
    assert ranked[0].growth_metric == pytest.approx(0.25)


def test_limit_truncates_to_the_requested_shortlist_size():
    recent = {f"T{i}": (f"Topic {i}", 500 + i * 100) for i in range(12)}
    prior = {f"T{i}": (f"Topic {i}", 300) for i in range(12)}
    assert len(rank_by_growth(recent=recent, prior=prior, limit=6)) == 6


def test_crowded_flag_separates_hot_from_quiet():
    ranked = rank_by_growth(
        recent={"BIG": ("Crowded", 40_000), "SMALL": ("Quiet", 700)},
        prior={"BIG": ("Crowded", 10_000), "SMALL": ("Quiet", 200)},
        limit=5,
    )
    by_topic = {c.topic: c for c in ranked}
    assert by_topic["Crowded"].is_crowded is True
    assert by_topic["Quiet"].is_crowded is False


# --- windows: which years get compared ------------------------------------

def test_windows_exclude_the_incomplete_current_year():
    recent, prior = windows(current_year=2026)
    assert recent == (2023, 2025)
    assert prior == (2020, 2022)
    assert recent[0] > prior[1]


# --- find_rising_topics: the step's public entry point ---------------------

def test_find_rising_topics_returns_candidates_with_example_papers():
    transport = _transport(
        recent={"T1": ("Rising", 900), "T2": ("Flat", 800), "T3": ("Steady", 600)},
        prior={"T1": ("Rising", 100), "T2": ("Flat", 790), "T3": ("Steady", 400)},
    )
    candidates = asyncio.run(find_rising_topics("MECHANICAL", limit=6, transport=transport))
    assert candidates[0].topic == "Rising"
    assert candidates[0].topic_id == "T1"
    assert candidates[0].example_papers[0].title == "A Paper"
    assert candidates[0].source == "velocity"
    # Mechanical is not CS-adjacent, so arXiv is never asked at all.
    assert all(c.arxiv_recent_count is None for c in candidates)


def test_every_curated_domain_maps_to_real_subfield_ids():
    for domain, subfields in DOMAIN_SUBFIELDS.items():
        assert subfields, f"{domain} has no subfield mapping"
        assert all(s.isdigit() and len(s) == 4 for s in subfields), domain


def test_unreachable_openalex_fails_loudly_rather_than_returning_nothing():
    """The spec's hard requirement: never present fabricated confidence."""
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("openalex unreachable", request=request)

    with pytest.raises(VelocityUnavailable):
        asyncio.run(find_rising_topics("CIVIL", limit=6,
                                       transport=httpx.MockTransport(refuse)))


def test_domain_yielding_too_few_topics_fails_loudly():
    transport = _transport(recent={}, prior={})
    with pytest.raises(VelocityUnavailable):
        asyncio.run(find_rising_topics("EEE", limit=6, transport=transport))


def test_other_domain_searches_free_text_instead_of_subfields():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "group_by" in params:
            seen.append(dict(params))
            start = params["filter"].split("publication_year:")[1].split("-")[0]
            counts = (
                {"T9": ("Bio Topic", 900), "T8": ("Other Bio", 700), "T7": ("More Bio", 500)}
                if int(start) >= 2023
                else {"T9": ("Bio Topic", 100), "T8": ("Other Bio", 600), "T7": ("More Bio", 480)}
            )
            return httpx.Response(200, json=_groups(counts))
        return httpx.Response(200, json={"results": [_work("A Paper")]})

    candidates = asyncio.run(find_rising_topics(
        "Other", limit=6, transport=httpx.MockTransport(handler),
        domain_other_name="synthetic biology",
    ))
    assert candidates[0].topic == "Bio Topic"
    assert all(q.get("search") == "synthetic biology" for q in seen)
    assert all("primary_topic.subfield.id" not in q["filter"] for q in seen)


# --- merge_sources: OpenAlex + arXiv without double-counting ---------------

def _arxiv_side(topic: str, count: int, *titles: str) -> TopicCandidate:
    return TopicCandidate(
        topic=topic, growth_metric=0.0, paper_count=0,
        arxiv_recent_count=count,
        example_papers=[ExamplePaper(title=t) for t in titles],
    )


def test_a_topic_found_in_both_sources_is_merged_not_double_counted():
    """The whole point of the merge: two sources finding the same topic must
    produce one shortlist entry whose paper_count is still OpenAlex's count,
    not the two corpora added together."""
    openalex = rank_by_growth(
        recent={"T1": ("Federated Learning", 900)},
        prior={"T1": ("Federated Learning", 300)}, limit=5,
    )
    merged = merge_sources(
        openalex, [_arxiv_side("federated-learning", 40, "A Preprint")],
    )

    assert len(merged) == 1
    assert merged[0].paper_count == 900
    assert merged[0].growth_metric == pytest.approx(3.0)
    assert merged[0].arxiv_recent_count == 40
    assert [p.title for p in merged[0].example_papers] == ["A Preprint"]


def test_the_same_paper_from_both_sources_is_listed_once():
    openalex = rank_by_growth(
        recent={"T1": ("Scaling", 900)}, prior={"T1": ("Scaling", 300)}, limit=5,
    )
    openalex[0].example_papers = [ExamplePaper(title="Scaling Laws for Models")]
    merged = merge_sources(
        openalex,
        [_arxiv_side("Scaling", 5, "scaling laws for models.", "A Second Preprint")],
    )

    assert [p.title for p in merged[0].example_papers] == [
        "Scaling Laws for Models", "A Second Preprint",
    ]


def test_a_topic_only_the_second_source_found_is_kept():
    openalex = rank_by_growth(
        recent={"T1": ("Known", 900)}, prior={"T1": ("Known", 300)}, limit=5,
    )
    merged = merge_sources(openalex, [_arxiv_side("Only On arXiv", 12)])

    assert {c.topic for c in merged} == {"Known", "Only On arXiv"}


def test_cs_domain_merges_arxiv_evidence_into_the_same_topic_entry():
    transport = _transport(
        recent={"T1": ("Rising", 900), "T2": ("Flat", 800), "T3": ("Steady", 600)},
        prior={"T1": ("Rising", 100), "T2": ("Flat", 790), "T3": ("Steady", 400)},
        arxiv=(("A Preprint", 2024), ("An Older Preprint", 2019)),
    )
    candidates = asyncio.run(find_rising_topics(
        "CS_AI_ML", limit=6, transport=transport, current_year=2026,
    ))

    assert [c.topic for c in candidates] == ["Rising", "Steady", "Flat"]
    top = candidates[0]
    assert top.paper_count == 900
    # Only the preprint inside the recent window counts.
    assert top.arxiv_recent_count == 1
    assert [p.title for p in top.example_papers] == ["A Paper", "A Preprint"]


def _arxiv_only(*entries: tuple[str, int],
                year_range: tuple[int, int] = (2023, 2025)) -> TopicCandidate:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=_arxiv_feed(*entries)))

    async def run():
        async with httpx.AsyncClient(transport=transport) as http:
            return await _arxiv_candidate(http, "Some Topic", year_range)

    return asyncio.run(run())


def test_arxiv_sample_is_counted_once_in_the_ranking_key():
    """Regression: paper_count carried the same sample as arxiv_recent_count,
    so _growth_key added an arXiv-only topic's volume to itself — and the
    sample was shown to the student as an OpenAlex 'Recent papers' count."""
    candidate = _arxiv_only(("One", 2024), ("Two", 2024))

    assert candidate.arxiv_recent_count == 2
    assert candidate.paper_count == 0
    assert _growth_key(candidate)[1] == 2
    assert candidate.is_crowded is False


def test_arxiv_window_matches_the_openalex_window_at_both_ends():
    """The two counts are summed, so the arXiv filter has to exclude the
    incomplete current year exactly as windows() does — otherwise one side
    of the sum spans four years and the other three."""
    candidate = _arxiv_only(("Too Old", 2022), ("In Window", 2024),
                            ("Current Year", 2026))

    assert candidate.arxiv_recent_count == 1
    assert [p.title for p in candidate.example_papers] == ["In Window"]


def test_merge_collapsing_topics_below_the_floor_fails_loudly():
    """Two OpenAlex topics that normalize to one key become one entry, which
    can push the shortlist under MIN_TOPICS_FOR_SHORTLIST after the merge —
    the guard that ran before it must run again."""
    names = ["Edge AI", "Edge-AI", "edge ai"]
    assert len(names) >= MIN_TOPICS_FOR_SHORTLIST
    transport = _transport(
        recent={f"T{i}": (n, 900 - i) for i, n in enumerate(names)},
        prior={f"T{i}": (n, 300) for i, n in enumerate(names)},
    )

    with pytest.raises(VelocityUnavailable):
        asyncio.run(find_rising_topics("CS_AI_ML", limit=6, transport=transport,
                                       current_year=2026))


class _PromptRecorder:
    def __init__(self):
        self.prompts: list[str] = []

    async def generate_structured(self, *, system_prompt, user_prompt, response_model):
        self.prompts.append(user_prompt)
        return response_model()


def test_an_arxiv_preprint_actually_reaches_a_gap_mining_worker():
    """D-035's claim, end to end. Gap mining reads only the first
    MAX_PAPERS_PER_TOPIC papers that have text, and every OpenAlex paper has
    text (works_for_topic filters on has_abstract), so appending the
    preprints put them permanently out of the swarm's reach."""
    transport = _transport(
        recent={"T1": ("Rising", 900), "T2": ("Flat", 800), "T3": ("Steady", 600)},
        prior={"T1": ("Rising", 100), "T2": ("Flat", 790), "T3": ("Steady", 400)},
        arxiv=(("A Preprint", 2024),),
        works=("A Paper", "Another Paper"),
    )
    candidates = asyncio.run(find_rising_topics(
        "CS_AI_ML", limit=6, transport=transport, current_year=2026,
    ))
    model = _PromptRecorder()
    asyncio.run(mine_gaps(candidates, model=model))

    mined = [p.split("Paper title: ")[1].split("\n")[0] for p in model.prompts]
    assert len(mined) == len(candidates) * MAX_PAPERS_PER_TOPIC
    # Every topic spent one of its two worker slots on the preprint.
    assert sorted(set(mined)) == ["A Paper", "A Preprint"]


def test_arxiv_volume_breaks_a_tie_between_equally_growing_topics():
    """arxiv_recent_count has to actually move the ranking, or the second
    source is decoration."""
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "arxiv.org" in str(request.url):
            entries = (("Beta Preprint", 2024), ("Another Beta Preprint", 2024))
            return httpx.Response(200, text=_arxiv_feed(
                *(entries if "Beta" in params["search_query"] else ()),
            ))
        if "group_by" in params:
            start = int(params["filter"].split("publication_year:")[1].split("-")[0])
            counts = (
                {"T1": ("Alpha", 600), "T2": ("Beta", 600), "T3": ("Gamma", 500)}
                if start >= 2023
                else {"T1": ("Alpha", 300), "T2": ("Beta", 300), "T3": ("Gamma", 480)}
            )
            return httpx.Response(200, json=_groups(counts))
        return httpx.Response(200, json={"results": [_work("A Paper")]})

    candidates = asyncio.run(find_rising_topics(
        "CS_AI_ML", limit=6, transport=httpx.MockTransport(handler),
        current_year=2026,
    ))

    by_topic = {c.topic: c for c in candidates}
    assert by_topic["Alpha"].growth_metric == by_topic["Beta"].growth_metric
    assert [c.topic for c in candidates][:2] == ["Beta", "Alpha"]


# --- the domain actually steers the query ----------------------------------

def _per_domain_transport() -> httpx.MockTransport:
    """OpenAlex, but every topic is named after the subfield it was asked
    for — so a domain whose mapping never reaches the wire comes back with
    another domain's topics, and the assertions below catch it."""
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "arxiv.org" in str(request.url):
            return httpx.Response(200, text=_arxiv_feed(("A Preprint", 2024)))
        if "group_by" in params:
            filters = params["filter"]
            subfields = re.findall(r"subfields/(\d{4})", filters)
            recent = int(filters.split("publication_year:")[1].split("-")[0]) >= 2023
            return httpx.Response(200, json=_groups({
                f"T{sid}": (f"Topic of subfield {sid}", (900 - i * 100) if recent else 300)
                for i, sid in enumerate(subfields)
            }))
        return httpx.Response(200, json={"results": [_work("A Paper")]})
    return httpx.MockTransport(handler)


@pytest.mark.parametrize("domain", sorted(DOMAIN_SUBFIELDS))
def test_every_curated_domain_returns_a_non_empty_ranked_shortlist(domain):
    candidates = asyncio.run(find_rising_topics(
        domain, limit=8, transport=_per_domain_transport()))

    assert candidates, f"{domain} produced no candidates"
    growths = [c.growth_metric for c in candidates]
    assert growths == sorted(growths, reverse=True), f"{domain} came back unranked"
    assert all(c.example_papers for c in candidates)
    # Every topic it returned came from a subfield this domain actually maps to.
    assert {c.topic_id.removeprefix("T") for c in candidates} <= set(DOMAIN_SUBFIELDS[domain])


def test_two_domains_produce_different_top_candidates():
    """The domain -> subfield table is only worth having if the domain
    changes the answer. A table that collapsed to one field, or a domain
    argument dropped on its way to the query, both survive every other test
    in this file."""
    transport = _per_domain_transport()
    mechanical = asyncio.run(find_rising_topics("MECHANICAL", limit=8, transport=transport))
    civil = asyncio.run(find_rising_topics("CIVIL", limit=8, transport=transport))

    assert mechanical[0].topic != civil[0].topic
    assert {c.topic for c in mechanical}.isdisjoint({c.topic for c in civil})
