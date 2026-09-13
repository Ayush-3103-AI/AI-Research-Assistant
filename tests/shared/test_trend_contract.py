import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.contracts.trend_contract import (  # noqa: E402
    TopicCandidate, TrendAdvisorRequest, TrendAdvisorResult,
)


@pytest.mark.parametrize("domain", ["Biotech", "cs_ai_ml", "OTHER"])
def test_unlisted_domain_is_rejected(domain):
    with pytest.raises(ValidationError):
        TrendAdvisorRequest(domain=domain)


def test_other_without_domain_other_name_is_rejected():
    with pytest.raises(ValidationError):
        TrendAdvisorRequest(domain="Other")


def test_other_with_domain_other_name_is_accepted():
    req = TrendAdvisorRequest(domain="Other", domain_other_name="Biotech")
    assert req.domain_label == "Biotech"


def test_fixed_domain_is_accepted():
    req = TrendAdvisorRequest(domain="MECHANICAL")
    assert req.domain_label == "MECHANICAL"


# --- Derived labels and lookups the CLI, smoke runner and orchestrator share ---

def _candidate(**kwargs) -> TopicCandidate:
    base = dict(topic="Edge Inference", topic_id="T1", growth_metric=2.0, paper_count=100)
    return TopicCandidate(**{**base, **kwargs})


def _result(**kwargs) -> TrendAdvisorResult:
    base = dict(domain="CS_AI_ML", generated_at="2026-09-13T00:00:00Z", shortlist=[])
    return TrendAdvisorResult(**{**base, **kwargs})


def test_a_gap_flagged_topic_reads_as_a_gap_even_when_it_is_also_crowded():
    """The gap signal is the reason this stage mines gaps at all, so it wins
    over raw volume in the one phrase the student is shown."""
    assert _candidate(gap_signal=True, paper_count=90_000).signal_label == "recurring gap"


def test_volume_alone_separates_a_crowded_topic_from_a_quiet_one():
    assert _candidate(paper_count=90_000).signal_label == "hot but crowded"
    assert _candidate(paper_count=10).signal_label == "under-explored"


def test_a_free_text_domain_labels_itself_by_its_own_name():
    result = _result(domain="Other", domain_other_name="synthetic biology")
    assert result.domain_label == "synthetic biology"
    assert _result().domain_label == "CS_AI_ML"


def test_gap_flagged_count_counts_topics_not_pieces_of_evidence():
    result = _result(shortlist=[
        _candidate(topic="A", gap_signal=True, gap_evidence=["one", "two", "three"]),
        _candidate(topic="B"),
    ])
    assert result.gap_flagged_count == 1


def test_the_chosen_candidate_is_found_by_name():
    result = _result(shortlist=[_candidate(topic="A"), _candidate(topic="B", gap_signal=True)],
                     chosen_topic="B")
    assert result.chosen_candidate.gap_signal is True


def test_a_chosen_topic_that_is_not_on_the_shortlist_resolves_to_nothing():
    """Never another topic's evidence: a name the shortlist doesn't carry
    must come back empty rather than falling through to a neighbour."""
    result = _result(shortlist=[_candidate(topic="A")], chosen_topic="Not Listed")
    assert result.chosen_candidate is None


def test_no_pick_resolves_to_nothing():
    assert _result(shortlist=[_candidate(topic="A")]).chosen_candidate is None
