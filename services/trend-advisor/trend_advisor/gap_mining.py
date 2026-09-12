"""Step 5 — cross-paper future-work/gap mining, as a parallel swarm.

A LangGraph fan-out (`Send`) with one worker agent per candidate paper, each
reading that paper's own words and extracting only the future-work /
limitations statements actually present in it, then a merge step that keeps a
topic's gap flag only where several *independent* papers said the same kind
of thing. Same Swarms pattern already proven in the Writing service's
literature-card fan-out (services/research-writing/writing/graph.py).

Why this is a separate signal from publication velocity: volume says where
people are already working; a recurring future-work mention says where they
have publicly admitted they have not. A topic with the second and not the
first is exactly the "under-explored gap" a student is looking for.
"""

from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from shared.contracts.trend_contract import TopicCandidate  # noqa: E402

# Each worker is one local-model call, and on this hardware those are the
# whole stage's cost (DECISIONS.md D-009's speed bottleneck). Bounded per
# topic, in the same spirit as Discovery's MAX_CANDIDATES_FOR_ANALYSIS cap,
# rather than fanning out over every example paper the velocity step found.
MAX_PAPERS_PER_TOPIC = 2
# "Recurring" means more than one paper: a single paper's future-work
# sentence is that paper's opinion, not evidence of an unmet field-wide need.
MIN_PAPERS_FLAGGING_A_GAP = 2
# Abstracts are short; a small budget keeps each worker fast and keeps the
# context window on the smallest tier (llm_provider._context_window_for).
WORKER_MAX_TOKENS = 800

GAP_SYSTEM_PROMPT = """You extract unmet research needs from a paper's own text.

Return ONLY statements the text itself makes about what remains unsolved,
unvalidated, untested, or left for future work — the paper's limitations and
future-research claims, in your own concise words.

Rules:
- If the text states no limitation and no future work, return an empty list.
  An empty list is the correct, expected answer for many papers.
- Never infer, guess, or generalise a gap the text does not state.
- Never restate what the paper achieved; only what it says is still missing.
- One sentence per statement, at most 3 statements."""


class _GapExtraction(BaseModel):
    """One worker's output: the gaps this single paper actually states."""

    gap_statements: list[str] = Field(default_factory=list)


class _MinedGap(TypedDict):
    topic_id: str
    # Which paper said it. Carried because "recurring" counts PAPERS, not
    # statements: the prompt lets one paper return up to 3 statements, so
    # counting statements would let a single paper's future-work paragraph
    # masquerade as a field-wide pattern.
    paper: str
    statement: str


class _SwarmState(TypedDict, total=False):
    papers: list[dict]
    # Workers run concurrently and each returns its own findings; operator.add
    # is what makes the fan-in a merge rather than a last-writer-wins clobber.
    mined: Annotated[list[_MinedGap], operator.add]
    # Per-worker inputs, set by Send.
    worker_paper: dict


def _worker_papers(candidates: list[TopicCandidate]) -> list[dict]:
    """The (paper, topic) pairs worth spending a model call on.

    Each paper contributes its Discussion/Limitations/Future-research text
    where an open-access copy was reachable (velocity.py fetches it), and
    its abstract otherwise — authors state unmet needs in the former, so it
    is strictly the better evidence when available. A paper with neither has
    no text to mine and is dropped here rather than sent to the model to
    hallucinate over.
    """
    papers = []
    for candidate in candidates:
        with_text = [
            (p, text) for p in candidate.example_papers
            if (text := (p.future_text or p.abstract or "").strip())
        ]
        for paper, text in with_text[:MAX_PAPERS_PER_TOPIC]:
            papers.append({
                "topic_id": candidate.topic_id or candidate.topic,
                "topic": candidate.topic,
                "title": paper.title,
                "text": text,
            })
    return papers


def _build_graph(model):
    def dispatch(state: _SwarmState) -> list[Send]:
        """One worker per paper. mine_gaps() returns early on an empty list,
        so there is always at least one paper to send here."""
        return [Send("mine_paper", {"worker_paper": paper})
                for paper in state["papers"]]

    async def mine_paper(state: _SwarmState) -> dict:
        paper = state["worker_paper"]
        user_prompt = (
            f"Paper title: {paper['title']}\n\n"
            f"Paper text:\n{paper['text']}\n\n"
            "List the unmet research needs this text states, if any."
        )
        try:
            extraction = await model.generate_structured(
                system_prompt=GAP_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=_GapExtraction,
            )
        except Exception:
            # Worker isolation, same rule as the Writing swarm: one paper's
            # failed call must not cost the whole stage the other papers'
            # findings. The topic simply ends up with less evidence, which
            # the recurrence threshold then treats honestly.
            return {"mined": []}
        return {"mined": [
            {"topic_id": paper["topic_id"], "paper": paper["title"],
             "statement": statement}
            for statement in extraction.gap_statements if statement.strip()
        ]}

    builder = StateGraph(_SwarmState)
    builder.add_node("mine_paper", mine_paper)
    # No dispatch or merge node: the fan-out is the conditional edge itself,
    # the fan-in is the operator.add reducer on `mined`, and the real merge
    # is annotate() below. Nodes that only forward state would be pure
    # ceremony.
    builder.add_conditional_edges(START, dispatch, ["mine_paper"])
    builder.add_edge("mine_paper", END)
    return builder.compile()


def annotate(candidates: list[TopicCandidate],
             mined: list[_MinedGap]) -> list[TopicCandidate]:
    """Attach each topic's mined evidence and decide its gap flag.

    Evidence is always attached when it exists; the flag is set only once
    MIN_PAPERS_FLAGGING_A_GAP independent papers contributed, which is the
    difference between "a paper mentioned this" and "the field keeps saying
    this is missing".
    """
    statements: dict[str, list[str]] = {}
    contributing_papers: dict[str, set[str]] = {}
    for entry in mined:
        key = entry["topic_id"]
        statements.setdefault(key, []).append(entry["statement"])
        contributing_papers.setdefault(key, set()).add(entry["paper"])

    for candidate in candidates:
        key = candidate.topic_id or candidate.topic
        evidence = statements.get(key, [])
        candidate.gap_evidence = evidence
        candidate.gap_signal = (
            len(contributing_papers.get(key, set())) >= MIN_PAPERS_FLAGGING_A_GAP
        )
        if candidate.gap_signal:
            candidate.source = "both"
    return candidates


async def mine_gaps(candidates: list[TopicCandidate], *, model) -> list[TopicCandidate]:
    """Run the swarm over `candidates` and return them annotated in place.

    `model` is any `LanguageModel` (writing.adapters.language_model's
    protocol): the advisor service passes the local Ollama adapter, tests
    pass a fake.
    """
    papers = _worker_papers(candidates)
    if not papers:
        return annotate(candidates, [])
    final = await _build_graph(model).ainvoke({"papers": papers, "mined": []})
    return annotate(candidates, final.get("mined", []))
