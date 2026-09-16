"""In-memory registry for asynchronous pipeline runs, backing
GET /research/runs/{run_id} and its SSE event stream (orchestrator/api.py).

Deliberately in-memory and single-process: this is an MVP addition with no
job queue or persistence layer of its own — a run's live status is only
queryable from the same server process that started it, and is lost on
restart (though its artifacts on disk under outputs/ persist regardless,
same as any run started through the existing synchronous POST /research).
See DECISIONS.md D-017.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from shared.contracts.pipeline_contract import PipelineResult, ResearchRequest

RunPipelineFn = Callable[..., Any]


@dataclass
class RunState:
    run_id: str
    status: str = "running"  # running | completed | failed
    events: list[dict[str, Any]] = field(default_factory=list)
    result: PipelineResult | None = None
    error: str | None = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)


_runs: dict[str, RunState] = {}


def get_run(run_id: str) -> RunState | None:
    return _runs.get(run_id)


def _publish(state: RunState, event: dict[str, Any]) -> None:
    state.events.append(event)
    for queue in list(state.subscribers):
        queue.put_nowait(event)


def subscribe(state: RunState) -> asyncio.Queue:
    """Returns a queue pre-loaded with every event so far, then kept live
    for any events published after subscribing — so a client connecting
    to the SSE stream after the run has already started still sees the
    full history, not just what happens from then on."""
    queue: asyncio.Queue = asyncio.Queue()
    for event in state.events:
        queue.put_nowait(event)
    state.subscribers.append(queue)
    return queue


def unsubscribe(state: RunState, queue: asyncio.Queue) -> None:
    if queue in state.subscribers:
        state.subscribers.remove(queue)


async def start_run(
    request: ResearchRequest, run_id: str, run_pipeline_fn: RunPipelineFn,
) -> RunState:
    """Registers a new run and launches it as a background task, returning
    immediately with the (running) RunState so the caller can hand the
    run_id back to its client without waiting for the pipeline itself."""
    state = RunState(run_id=run_id)
    _runs[run_id] = state
    asyncio.create_task(_drive_run(request, run_id, run_pipeline_fn, state))
    return state


async def _drive_run(
    request: ResearchRequest, run_id: str, run_pipeline_fn: RunPipelineFn, state: RunState,
) -> None:
    try:
        async for event in run_pipeline_fn(request, run_id=run_id):
            if event["type"] == "result":
                state.result = event["result"]
                state.status = "completed"
            _publish(state, event)
    except Exception as error:  # PipelineError, or any unexpected stage failure
        state.status = "failed"
        # Some exceptions (e.g. httpx timeouts) stringify to "" — never report a blank failure.
        state.error = str(error) or type(error).__name__
        _publish(state, {"type": "error", "message": state.error})
    finally:
        _publish(state, {"type": "end"})
