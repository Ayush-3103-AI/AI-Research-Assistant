"""Live integration test for the publication-velocity step against real OpenAlex.

The mocked tests in test_trend_velocity.py prove the ranking math; this one
proves the queries themselves still match OpenAlex's live API — the failure
mode a mock can never catch (a renamed filter, a changed group_by shape, a
retired subfield id). No local model is needed, only network, so it skips on
reachability rather than on Ollama, unlike the other live-skipped tests here.
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

from trend_advisor.velocity import find_rising_topics  # noqa: E402


def _openalex_available() -> bool:
    try:
        httpx.get("https://api.openalex.org/works", params={"per-page": 1},
                  timeout=8).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _openalex_available(), reason="OpenAlex not reachable from this machine",
)


def test_real_openalex_query_produces_a_ranked_shortlist_with_real_papers():
    candidates = asyncio.run(find_rising_topics("MECHANICAL", limit=6))

    assert len(candidates) == 6
    # Ranked, descending, by the growth metric the shortlist claims to rank on.
    growths = [c.growth_metric for c in candidates]
    assert growths == sorted(growths, reverse=True)

    for candidate in candidates:
        assert candidate.topic and candidate.topic_id.startswith("T")
        assert candidate.paper_count >= 200
        # The evidence a student is asked to trust must actually be there.
        assert candidate.example_papers, f"{candidate.topic} has no example papers"
        assert any(p.doi for p in candidate.example_papers)
        assert any(p.abstract for p in candidate.example_papers)
