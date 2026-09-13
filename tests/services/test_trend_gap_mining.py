"""Unit tests for the Trend & Gap Advisor's parallel gap-mining swarm.

Runs the real LangGraph fan-out/merge against a fake LanguageModel that
records every prompt it is given — the same approach as
test_writing_literature_card_retry.py, and for the same reason: the
aggregation logic is what can actually be wrong, and it must be testable
without a live Ollama server.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))
sys.path.insert(0, str(ROOT / "services" / "trend-advisor"))

from shared.contracts.trend_contract import ExamplePaper, TopicCandidate  # noqa: E402
from trend_advisor.gap_mining import (  # noqa: E402
    MAX_PAPERS_PER_TOPIC, MIN_PAPERS_FLAGGING_A_GAP, mine_gaps,
)


class _FakeModel:
    """Returns the gap statements keyed by paper title, and records prompts.

    Note the values are LISTS: the prompt allows a paper up to 3 statements,
    which is exactly the case that must not be mistaken for recurrence.
    """

    def __init__(self, gaps_by_title: dict[str, list[str]], fail_on: set[str] | None = None):
        self._gaps = gaps_by_title
        self._fail_on = fail_on or set()
        self.prompts: list[str] = []

    async def generate_structured(self, *, system_prompt, user_prompt, response_model):
        self.prompts.append(user_prompt)
        title = next((t for t in self._gaps if t in user_prompt), None)
        if title in self._fail_on:
            raise RuntimeError("model exploded on this paper")
        return response_model.model_validate({"gap_statements": self._gaps.get(title, [])})


def _candidate(topic: str, titles: list[str], topic_id: str = "T1") -> TopicCandidate:
    return TopicCandidate(
        topic=topic, topic_id=topic_id, growth_metric=2.0, paper_count=900, prior_papers=450,
        example_papers=[
            ExamplePaper(title=t, doi=f"10.1000/{t}", year=2024,
                         abstract=f"Abstract of {t}. Future work should continue.")
            for t in titles
        ],
    )


def test_topic_flagged_when_several_independent_papers_name_a_gap():
    candidates = [_candidate("Solid-State Batteries", ["Paper A", "Paper B"])]
    model = _FakeModel({
        "Paper A": ["Long-term cycling stability is not yet characterised."],
        "Paper B": ["No standard protocol exists for cycling-stability testing."],
    })

    result = asyncio.run(mine_gaps(candidates, model=model))

    assert result[0].gap_signal is True
    assert len(result[0].gap_evidence) == 2
    assert result[0].source == "both"


def test_one_paper_making_several_statements_is_still_only_one_paper():
    """Regression: the flag counted distinct STATEMENTS, so a single paper
    returning two future-work sentences set "flagged by multiple independent
    papers" on its own. "Recurring" counts papers."""
    candidates = [_candidate("Niche Topic", ["Paper A", "Paper B"])]
    model = _FakeModel({
        "Paper A": ["Cycling stability is uncharacterised.",
                    "No standard test protocol exists."],
        "Paper B": [],
    })

    result = asyncio.run(mine_gaps(candidates, model=model))

    assert result[0].gap_signal is False
    assert len(result[0].gap_evidence) == 2


def test_two_papers_repeating_the_same_statement_do_count_as_recurring():
    """The mirror case: identical wording from two papers is recurrence,
    and must not be collapsed by de-duplicating statement text."""
    candidates = [_candidate("Shared Topic", ["Paper A", "Paper B"])]
    same = "Long-term field validation is missing."
    model = _FakeModel({"Paper A": [same], "Paper B": [same]})

    result = asyncio.run(mine_gaps(candidates, model=model))

    assert result[0].gap_signal is True


def test_single_paper_mentioning_a_gap_is_not_a_recurring_gap():
    """One paper's future-work line is an opinion; the whole point of this
    step is that several independent papers said the same thing."""
    candidates = [_candidate("Niche Topic", ["Paper A", "Paper B"])]
    model = _FakeModel({"Paper A": ["Only this one paper mentions it."], "Paper B": []})

    result = asyncio.run(mine_gaps(candidates, model=model))

    assert result[0].gap_signal is False
    assert result[0].source == "velocity"
    # The one real excerpt is still kept — it is evidence, just not a signal.
    assert result[0].gap_evidence == ["Only this one paper mentions it."]


def test_papers_with_no_future_work_text_produce_no_fabricated_gap():
    candidates = [_candidate("Mature Topic", ["Paper A", "Paper B"])]
    model = _FakeModel({"Paper A": [], "Paper B": []})

    result = asyncio.run(mine_gaps(candidates, model=model))

    assert result[0].gap_signal is False
    assert result[0].gap_evidence == []


def test_each_topics_gaps_stay_on_that_topic():
    """Workers run concurrently and merge into shared state, so the merge
    must key evidence back to the right topic rather than pooling it."""
    candidates = [
        _candidate("Topic One", ["Paper A", "Paper B"], topic_id="T1"),
        _candidate("Topic Two", ["Paper C", "Paper D"], topic_id="T2"),
    ]
    model = _FakeModel({
        "Paper A": ["Gap one-A."], "Paper B": ["Gap one-B."],
        "Paper C": ["Gap two-C."], "Paper D": ["Gap two-D."],
    })

    result = asyncio.run(mine_gaps(candidates, model=model))
    by_topic = {c.topic: c for c in result}

    assert by_topic["Topic One"].gap_evidence == ["Gap one-A.", "Gap one-B."]
    assert by_topic["Topic Two"].gap_evidence == ["Gap two-C.", "Gap two-D."]


def test_one_failing_worker_does_not_lose_the_other_papers_evidence():
    candidates = [_candidate("Topic One", ["Paper A", "Paper B"])]
    model = _FakeModel(
        {"Paper A": ["Gap A."], "Paper B": ["Gap B."]}, fail_on={"Paper A"},
    )

    result = asyncio.run(mine_gaps(candidates, model=model))

    assert result[0].gap_evidence == ["Gap B."]
    assert result[0].gap_signal is False  # one surviving paper is not "recurring"


def test_worker_count_is_capped_per_topic():
    """Each LLM call is seconds on this hardware (DECISIONS.md D-009), so the
    swarm is bounded rather than fanning out over every example paper."""
    titles = [f"Paper {i}" for i in range(MAX_PAPERS_PER_TOPIC + 4)]
    model = _FakeModel({t: [] for t in titles})

    asyncio.run(mine_gaps([_candidate("Topic One", titles)], model=model))

    assert len(model.prompts) == MAX_PAPERS_PER_TOPIC


def test_worker_prompt_carries_the_papers_own_text():
    candidates = [_candidate("Topic One", ["Paper A", "Paper B"])]
    model = _FakeModel({"Paper A": [], "Paper B": []})

    asyncio.run(mine_gaps(candidates, model=model))

    assert all("Abstract of Paper" in prompt for prompt in model.prompts)


def test_papers_without_abstracts_are_not_sent_to_the_model():
    candidate = TopicCandidate(
        topic="Topic One", topic_id="T1", growth_metric=2.0, paper_count=900,
        example_papers=[ExamplePaper(title="No Abstract Paper", abstract=None)],
    )
    model = _FakeModel({})

    result = asyncio.run(mine_gaps([candidate], model=model))

    assert model.prompts == []
    assert result[0].gap_signal is False


def test_threshold_constant_matches_the_recurring_gap_claim():
    assert MIN_PAPERS_FLAGGING_A_GAP >= 2


def test_future_work_section_is_preferred_over_the_abstract():
    """Authors state unmet needs in Discussion/Limitations/Future-research,
    not in abstracts — so when velocity.py managed to retrieve that text, it
    is what the worker must read."""
    candidate = TopicCandidate(
        topic="Topic One", topic_id="T1", growth_metric=2.0, paper_count=900,
        example_papers=[ExamplePaper(
            title="Paper A", abstract="A short marketing-flavoured abstract.",
            future_text="Future work should validate this in the field.",
        )],
    )
    model = _FakeModel({"Paper A": []})

    asyncio.run(mine_gaps([candidate], model=model))

    assert "Future work should validate this in the field." in model.prompts[0]
    assert "marketing-flavoured" not in model.prompts[0]


def test_abstract_is_used_when_no_full_text_could_be_retrieved():
    candidate = TopicCandidate(
        topic="Topic One", topic_id="T1", growth_metric=2.0, paper_count=900,
        example_papers=[ExamplePaper(
            title="Paper A", abstract="The abstract is all we have.",
            future_text=None,
        )],
    )
    model = _FakeModel({"Paper A": []})

    asyncio.run(mine_gaps([candidate], model=model))

    assert "The abstract is all we have." in model.prompts[0]


class _BarrierModel:
    """Every worker must arrive before any is allowed to answer.

    A sequential implementation deadlocks on this: worker 1 waits for a
    worker 2 that is never started, and the test's own wait_for is what
    eventually fails it. So a fan-out that regresses into a `for` loop is
    caught here — which is the one claim the other tests in this file cannot
    make, since they all pass just as happily against a sequential loop.
    """

    def __init__(self, workers: int):
        self._barrier = asyncio.Barrier(workers)
        self.peak_in_flight = 0
        self._in_flight = 0

    async def generate_structured(self, *, system_prompt, user_prompt, response_model):
        self._in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        await self._barrier.wait()
        self._in_flight -= 1
        return response_model.model_validate({"gap_statements": []})


def test_workers_run_concurrently_rather_than_one_after_another():
    """Course skill 5 (Swarms) is the whole claim of this module: one worker
    per paper, all in flight together. Two topics x the per-topic cap, so the
    fan-out has to cross topics too, not just fan out within one."""
    candidates = [
        _candidate("Topic One", ["Paper A", "Paper B"], topic_id="T1"),
        _candidate("Topic Two", ["Paper C", "Paper D"], topic_id="T2"),
    ]
    expected_workers = 2 * MAX_PAPERS_PER_TOPIC
    model = _BarrierModel(expected_workers)

    async def run():
        # The barrier is the real assertion; the timeout only stops a
        # sequential implementation from hanging the suite forever.
        return await asyncio.wait_for(mine_gaps(candidates, model=model), timeout=10)

    asyncio.run(run())

    assert model.peak_in_flight == expected_workers
