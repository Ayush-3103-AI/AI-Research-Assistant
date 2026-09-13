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

from researchgenie.advisor_cli import parse_pick, with_chosen_topic  # noqa: E402
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


def test_a_question_containing_a_pick_verb_is_not_a_pick():
    """Regression: "do 3" used to parse as a pick, so an ordinary question
    offered to lock in topic 3 and a reflexive Enter confirmed it."""
    for question in ["what do 3 papers say about this?",
                     "do 3 and 1 overlap?",
                     "should I pick 2 or 3?",
                     "is option 2 crowded?",
                     "would 2 take 3 years?",
                     "which of these do 2 groups both work on?",
                     "how long does option 1 take 2 write up?",
                     "can you compare 1 and 2?",
                     "tell me why 3 is growing"]:
        assert parse_pick(question, CANDIDATES) is None, question


def test_a_pick_phrased_as_a_statement_still_parses():
    """The question guard must not eat the phrasings students actually use."""
    for phrasing in ["go with #2", "let's do 2", "I'll take 2", "option 2",
                     "ok, pick 2 please"]:
        assert parse_pick(phrasing, CANDIDATES) == 2, phrasing


def test_plain_question_is_not_a_pick():
    assert parse_pick("which of these is less crowded?", CANDIDATES) is None


# --- The hand-off into the pipeline's optional Stage 0 ----------------------

def _result():
    from shared.contracts.trend_contract import TrendAdvisorResult
    return TrendAdvisorResult(
        domain="CS_AI_ML", shortlist=list(CANDIDATES),
        generated_at="2026-09-12T00:00:00Z",
    )


def test_the_pick_is_named_explicitly_not_implied_by_position():
    """The orchestrator records which topic the student chose. Naming it
    beats a positional convention the two modules could silently disagree
    about — a reordering on either side would otherwise report the wrong
    topic with no test failing."""
    result = _result()

    handed_off = with_chosen_topic(result, result.shortlist[2])

    assert handed_off.chosen_topic == "Edge Inference"


def test_the_shortlists_own_ranking_is_left_alone():
    """The advisor ranks gap-flagged topics first for a reason; the hand-off
    must not silently re-sort the evidence the student was shown."""
    result = _result()

    handed_off = with_chosen_topic(result, result.shortlist[2])

    assert [c.topic for c in handed_off.shortlist] == [c.topic for c in result.shortlist]


def test_a_value_equal_copy_of_the_pick_still_works():
    """Regression: the first version matched the pick by object identity, so
    a candidate that had been through a JSON round-trip (as any saved and
    reloaded result has) was not recognised."""
    from shared.contracts.trend_contract import TrendAdvisorResult
    result = _result()
    reloaded = TrendAdvisorResult.model_validate(result.model_dump())

    handed_off = with_chosen_topic(result, reloaded.shortlist[1])

    assert handed_off.chosen_topic == "Digital Twins"
    assert len(handed_off.shortlist) == len(result.shortlist)


def test_handoff_does_not_mutate_the_advisor_result():
    result = _result()

    with_chosen_topic(result, result.shortlist[2])

    assert result.chosen_topic is None
