"""Direct tests for the works-count-by-year query the Trend Advisor added to
Discovery's shared OpenAlex connector (spec: "only the new works-count-by-year
query type on the shared connector needs its own tests"). The velocity tests
reach this code only through find_rising_topics; these pin the wire format
and the parsing on their own, so a change to the query can't hide behind a
mock that answers any request.
"""

import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery.sources import openalex  # noqa: E402


def _capture(body: dict) -> tuple[httpx.MockTransport, dict]:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler), seen


def _run(coro_factory, body):
    transport, seen = _capture(body)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            return await coro_factory(http)

    return asyncio.run(go()), seen["params"]


def test_topic_counts_groups_by_primary_topic_within_the_year_window():
    body = {"group_by": [
        {"key": "https://openalex.org/T10028", "key_display_name": "Edge Inference", "count": 900},
        {"key": "https://openalex.org/T10029", "key_display_name": "Digital Twins", "count": 700},
    ]}
    counts, params = _run(
        lambda http: openalex.topic_counts(http, year_range=(2023, 2025),
                                           subfield_ids=["1702", "1705"]),
        body,
    )

    assert counts == {"T10028": ("Edge Inference", 900), "T10029": ("Digital Twins", 700)}
    assert params["group_by"] == "primary_topic.id"
    assert "publication_year:2023-2025" in params["filter"]
    assert "primary_topic.subfield.id:subfields/1702|subfields/1705" in params["filter"]
    assert "search" not in params


def test_topic_counts_uses_free_text_search_for_an_unlisted_domain():
    _, params = _run(
        lambda http: openalex.topic_counts(http, year_range=(2023, 2025),
                                           search="synthetic biology"),
        {"group_by": []},
    )

    assert params["search"] == "synthetic biology"
    assert "primary_topic.subfield.id" not in params["filter"]


def test_topic_counts_returns_an_empty_dict_when_openalex_has_no_buckets():
    """Empty is the caller's cue to fail loudly, never 'nothing is growing'."""
    counts, _ = _run(
        lambda http: openalex.topic_counts(http, year_range=(2023, 2025), subfield_ids=["1702"]),
        {"meta": {"count": 0}},
    )
    assert counts == {}


def test_works_for_topic_filters_on_the_topic_and_requires_an_abstract():
    body = {"results": [{
        "display_name": "A Paper", "publication_year": 2024, "cited_by_count": 10,
        "doi": "https://doi.org/10.1000/x",
        "abstract_inverted_index": {"An": [0], "abstract": [1]},
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "primary_location": {"source": {"display_name": "A Journal"}},
    }]}
    papers, params = _run(
        lambda http: openalex.works_for_topic(http, "T10028", year_range=(2023, 2025), limit=3),
        body,
    )

    assert "primary_topic.id:T10028" in params["filter"]
    assert "publication_year:2023-2025" in params["filter"]
    assert "has_abstract:true" in params["filter"]
    assert params["sort"] == "cited_by_count:desc"
    assert params["per-page"] == "3"
    assert [p.title for p in papers] == ["A Paper"]
    assert papers[0].doi == "10.1000/x"
    assert papers[0].abstract == "An abstract"
