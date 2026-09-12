"""Unit tests for the Trend & Gap Advisor's publication-velocity ranking.

The ranking math is the only genuinely novel reasoning in this step (the
HTTP layer is Discovery's already-tested OpenAlex connector), so it is
tested here against mocked OpenAlex responses with known counts — no
network, no model. The live counterpart lives in
test_trend_velocity_live.py and skips without network.
"""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))
sys.path.insert(0, str(ROOT / "services" / "trend-advisor"))

from trend_advisor.velocity import (  # noqa: E402
    DOMAIN_SUBFIELDS, MIN_RECENT_WORKS, VelocityUnavailable, find_rising_topics,
    rank_by_growth, windows,
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


def _transport(recent: dict, prior: dict) -> httpx.MockTransport:
    """Answers the two group_by window queries by the year range in the
    filter, and every per-topic works query with one fixed paper."""
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "group_by" in params:
            start = params["filter"].split("publication_year:")[1].split("-")[0]
            body = _groups(recent) if int(start) >= 2023 else _groups(prior)
            return httpx.Response(200, json=body)
        return httpx.Response(200, json={"results": [_work("A Paper")]})
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
