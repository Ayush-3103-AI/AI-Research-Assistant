"""Tests for what the advisor CLI actually hands to the pipeline.

`main()` is interactive, so the prompts are faked and only the hand-off is
asserted — but it is asserted at the real call site. Without this, dropping
either the `keywords` or the `trend_advisor` argument from `main()` leaves
every other test in the suite passing while the run silently loses the
provenance of its own topic.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from researchgenie import advisor_cli  # noqa: E402
from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from shared.contracts.trend_contract import (  # noqa: E402
    TopicCandidate, TrendAdvisorRequest, TrendAdvisorResult,
)

CHOSEN = "Digital Twins"


def _advisor_result() -> TrendAdvisorResult:
    return TrendAdvisorResult(
        domain="CS_AI_ML", generated_at="2026-09-12T00:00:00Z",
        shortlist=[
            TopicCandidate(topic="Solid-State Batteries", topic_id="T0",
                           growth_metric=3.0, paper_count=900),
            TopicCandidate(topic=CHOSEN, topic_id="T1", growth_metric=2.0,
                           paper_count=700, gap_signal=True,
                           gap_evidence=["Field validation is missing."]),
            TopicCandidate(topic="Edge Inference", topic_id="T2",
                           growth_metric=1.5, paper_count=500),
        ],
    )


@pytest.fixture
def handoff(monkeypatch):
    """Runs main() with every prompt and network call faked, and returns
    whatever it passed to the pipeline."""
    captured: dict = {}

    async def fake_run_research(console, request, trend_advisor=None,
                                run_id=None, advisor_directory=None):
        captured["request"] = request
        captured["trend_advisor"] = trend_advisor
        captured["run_id"] = run_id
        captured["advisor_directory"] = advisor_directory

    def fake_run_advisor(console, request, run_id):
        captured["advisor_run_id"] = run_id
        return result, f"outputs/cs-ai-ml_{run_id}"

    async def fake_question(candidate):
        return "How do digital twins affect maintenance cost?"

    result = _advisor_result()
    captured["advisor_result"] = result

    monkeypatch.setattr(advisor_cli.rg_config, "is_setup_complete", lambda: True)
    monkeypatch.setattr(advisor_cli.rg_config, "load", lambda: object())
    monkeypatch.setattr(advisor_cli, "_ensure_ready", lambda console, config: True)
    monkeypatch.setattr(advisor_cli, "render_banner", lambda console: None)
    monkeypatch.setattr(advisor_cli, "_collect_domain",
                        lambda console: TrendAdvisorRequest(domain="CS_AI_ML"))
    monkeypatch.setattr(advisor_cli, "_run_advisor", fake_run_advisor)
    monkeypatch.setattr(advisor_cli, "_render_shortlist", lambda console, r: None)
    monkeypatch.setattr(advisor_cli, "_converge", lambda console, r: r.shortlist[1])
    monkeypatch.setattr(advisor_cli, "_research_question_for", fake_question)
    monkeypatch.setattr(advisor_cli, "_run_research", fake_run_research)
    monkeypatch.setattr(advisor_cli.Prompt, "ask",
                        staticmethod(lambda *a, **k: k.get("default", "")))
    monkeypatch.setattr(advisor_cli.Confirm, "ask", staticmethod(lambda *a, **k: True))
    return captured


def test_the_generated_question_reaches_the_pipeline(handoff):
    advisor_cli.main()
    assert handoff["request"].research_question == (
        "How do digital twins affect maintenance cost?")


def test_the_chosen_topic_travels_as_a_discovery_keyword(handoff):
    """D-018: Discovery scores candidate papers against user keywords as its
    highest-confidence terms, so the pick keeps steering the search even when
    the generated question is loosely worded."""
    advisor_cli.main()
    assert handoff["request"].keywords == [CHOSEN]


def test_the_advisor_result_travels_as_optional_stage_zero(handoff):
    advisor_cli.main()
    stage_zero = handoff["trend_advisor"]
    assert stage_zero is not None, "the pipeline was given no Stage 0 provenance"
    assert stage_zero.chosen_topic == CHOSEN
    # The topics not picked ride along as the context for the decision.
    assert len(stage_zero.shortlist) == 3


def test_the_advisor_run_id_and_folder_reach_the_pipeline(handoff):
    """One run, one folder: the pipeline must be told which run_id Stage 0
    already used and where it wrote, or the CLI leaves two directories per
    chained run (review finding against spec line 49)."""
    advisor_cli.main()
    assert handoff["run_id"] == handoff["advisor_run_id"]
    assert handoff["advisor_directory"] == f"outputs/cs-ai-ml_{handoff['run_id']}"


def test_declining_the_pipeline_run_starts_no_run(handoff, monkeypatch):
    """User story 10: stopping after the pick must not start the pipeline."""
    monkeypatch.setattr(advisor_cli.Confirm, "ask", staticmethod(lambda *a, **k: False))
    advisor_cli.main()
    assert "request" not in handoff


def test_quitting_the_chat_without_picking_starts_no_run(handoff, monkeypatch):
    monkeypatch.setattr(advisor_cli, "_converge", lambda console, r: None)
    advisor_cli.main()
    assert "request" not in handoff


# --- Surviving a provider blip after the long wait -------------------------

def _flat(capsys) -> str:
    return " ".join(capsys.readouterr().out.split())


def _scripted_prompt(monkeypatch, answers: list[str]) -> None:
    queued = list(answers)
    monkeypatch.setattr(advisor_cli.Prompt, "ask",
                        staticmethod(lambda *a, **k: queued.pop(0)))


def test_a_provider_blip_in_chat_does_not_discard_the_shortlist(monkeypatch, capsys):
    """The shortlist cost minutes of live queries and gap mining; one failed
    chat turn must not take it down with the process."""
    async def boom(question, context):
        raise advisor_cli.llm_provider.LLMProviderError("no model reachable")

    monkeypatch.setattr(advisor_cli, "_chat_turn", boom)
    _scripted_prompt(monkeypatch, ["which one is less crowded?", "quit"])

    assert advisor_cli._converge(advisor_cli.Console(legacy_windows=False),
                                 _advisor_result()) is None
    assert "no model reachable" in _flat(capsys)


def test_a_bare_enter_does_not_lock_in_a_topic(monkeypatch, capsys):
    """Confirm.ask's default IS what Enter returns. Locking in is an explicit
    human decision, so Enter must decline, not accept."""
    seen: dict = {}

    def fake_confirm(*args, **kwargs):
        seen["default"] = kwargs.get("default")
        return kwargs.get("default")

    monkeypatch.setattr(advisor_cli.Confirm, "ask", staticmethod(fake_confirm))
    _scripted_prompt(monkeypatch, ["go with #2", "quit"])

    assert advisor_cli._converge(advisor_cli.Console(legacy_windows=False),
                                 _advisor_result()) is None
    assert seen["default"] is False


def test_an_explicit_yes_still_locks_in(monkeypatch):
    monkeypatch.setattr(advisor_cli.Confirm, "ask", staticmethod(lambda *a, **k: True))
    _scripted_prompt(monkeypatch, ["go with #2"])

    chosen = advisor_cli._converge(advisor_cli.Console(legacy_windows=False),
                                   _advisor_result())

    assert chosen.topic == CHOSEN


def test_a_provider_blip_after_lock_in_falls_back_honestly(handoff, monkeypatch, capsys):
    """No traceback at the worst possible moment, and no topic name quietly
    presented as if the model had drafted a question from it."""
    async def boom(candidate):
        raise advisor_cli.llm_provider.LLMProviderError("no model reachable")

    monkeypatch.setattr(advisor_cli, "_research_question_for", boom)

    advisor_cli.main()

    printed = _flat(capsys)
    assert "no model reachable" in printed
    assert "not a research question" in printed
    assert handoff["request"].research_question == CHOSEN


def test_an_empty_drafted_question_announces_itself(handoff, monkeypatch, capsys):
    async def blank(candidate):
        return ""

    monkeypatch.setattr(advisor_cli, "_research_question_for", blank)

    advisor_cli.main()

    assert "not a research question" in _flat(capsys)
    assert handoff["request"].research_question == CHOSEN


def test_a_run_started_from_the_advisor_shows_the_stage_zero_row(monkeypatch):
    """cli._run_research decides whether the live view has a Stage 0 row;
    an advisor-started run must show the stage that produced its topic."""
    from researchgenie import cli
    seen: dict = {}

    async def fake_run_pipeline(request, trend_advisor=None, **kwargs):
        seen["trend_advisor"] = trend_advisor
        yield {"type": "progress", "stage": "trend_advisor", "status": "done",
               "message": "picked"}
        raise cli.PipelineError("stop here — the view is what's under test")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    captured_views = []
    real_view = cli.PipelineView
    monkeypatch.setattr(cli, "PipelineView",
                        lambda **kwargs: captured_views.append(real_view(**kwargs))
                        or captured_views[-1])

    import asyncio
    asyncio.run(cli._run_research(
        cli.Console(legacy_windows=False),
        ResearchRequest(research_question="Q?"),
        trend_advisor=_advisor_result(),
    ))

    assert "trend_advisor" in captured_views[0].stages
    assert seen["trend_advisor"] is not None


def test_a_plain_run_has_no_stage_zero_row(monkeypatch):
    from researchgenie import cli

    async def fake_run_pipeline(request, trend_advisor=None, **kwargs):
        yield {"type": "progress", "stage": "discovery", "status": "running",
               "message": "searching"}
        raise cli.PipelineError("stop here — the view is what's under test")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    captured_views = []
    real_view = cli.PipelineView
    monkeypatch.setattr(cli, "PipelineView",
                        lambda **kwargs: captured_views.append(real_view(**kwargs))
                        or captured_views[-1])

    import asyncio
    asyncio.run(cli._run_research(
        cli.Console(legacy_windows=False), ResearchRequest(research_question="Q?"),
    ))

    assert "trend_advisor" not in captured_views[0].stages
