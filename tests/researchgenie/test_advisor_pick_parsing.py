"""Tests for the advisor chat loop's deterministic topic-pick parsing.

The rest of the advisor CLI is covered by a manual smoke runner (see
scripts/smoke_test_trend_advisor.py, and the spec's rigor split), but this
one function is not presentation: if it mis-parses, the student silently
locks in the wrong topic and the whole pipeline runs on it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from researchgenie.advisor_cli import parse_pick  # noqa: E402
from shared.contracts.trend_contract import TopicCandidate  # noqa: E402

CANDIDATES = [
    TopicCandidate(topic=name, topic_id=f"T{i}", growth_metric=2.0, paper_count=500)
    for i, name in enumerate(["Solid-State Batteries", "Digital Twins", "Edge Inference"])
]


def test_go_with_hash_number_picks_that_entry():
    assert parse_pick("go with #2", CANDIDATES) == 2


def test_bare_number_picks_that_entry():
    assert parse_pick("2", CANDIDATES) == 2


def test_common_phrasings_all_resolve_to_the_same_entry():
    for phrasing in ["pick 3", "choose 3", "select 3", "option 3", "number 3",
                     "let's take 3", "I'll go with 3", "GO WITH #3"]:
        assert parse_pick(phrasing, CANDIDATES) == 3, phrasing


def test_exact_topic_name_picks_that_entry():
    assert parse_pick("Digital Twins", CANDIDATES) == 2
    assert parse_pick("  digital twins ", CANDIDATES) == 2


def test_out_of_range_number_is_not_a_pick():
    """Better to re-show the list than to lock in a topic that isn't there."""
    assert parse_pick("go with #9", CANDIDATES) is None
    assert parse_pick("0", CANDIDATES) is None


def test_a_question_mentioning_a_number_is_not_a_pick():
    assert parse_pick("how many papers does 2 have?", CANDIDATES) is None
    assert parse_pick("what's the difference between 1 and 3?", CANDIDATES) is None


def test_plain_question_is_not_a_pick():
    assert parse_pick("which of these is less crowded?", CANDIDATES) is None
