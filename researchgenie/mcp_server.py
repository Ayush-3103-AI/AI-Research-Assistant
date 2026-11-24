"""ResearchGenie as an MCP (Model Context Protocol) server.

Exposes the existing four-stage pipeline (Discovery -> Writing -> Citation
Verification -> Quality Assurance) to any MCP client: Claude Desktop,
Claude Code, Cursor, MCP Inspector, and so on. This module only translates
between MCP and the pipeline; the pipeline itself
(`orchestrator.pipeline.run_pipeline`) is used unchanged.

Why the tools are split into start / status / result: a full run takes
minutes, longer than most MCP clients wait for one tool call. So
`start_research` returns a run_id immediately and the run continues in the
background (via the same `orchestrator.run_registry` the HTTP API uses);
the client then polls `get_run_status` and fetches `get_research_result`.

Configuration (no interactive prompts are possible over MCP):
- Environment variables set in the MCP client's config win:
  AI_PROVIDER ("ollama" | "gemini"), LOCAL_AI_MODEL, GEMINI_MODEL,
  GEMINI_API_KEY, OLLAMA_HOST.
- Anything not set there falls back to the config saved by
  `researchgenie config` (~/.researchgenie/config.json).
- RESEARCHGENIE_OUTPUT_DIR sets where run folders are written
  (default: ~/ResearchGenie/outputs).

Run with:  researchgenie-mcp            (stdio, for local MCP clients)
           researchgenie-mcp --http     (streamable HTTP on 127.0.0.1:8765)
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ResourceError, ToolError
except ImportError as error:  # pragma: no cover - depends on the user's install
    raise SystemExit(
        "The MCP server needs the optional 'mcp' dependency. "
        "Install it with:  pip install \"researchgenie[mcp]\""
    ) from error

from researchgenie import __version__  # noqa: E402
from researchgenie import config as rg_config  # noqa: E402

LengthPreset = Literal["short", "standard", "detailed"]

_TERMINAL_STATUSES = {"completed", "failed"}
_MAX_WAIT_SECONDS = 50
_FINAL_FILES = {
    "draft": "draft.md",
    "references": "references.md",
    "quality_report": "quality_report.md",
    "validation_report": "validation_report.md",
    "latex": "paper.tex",
}


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def output_root() -> Path:
    configured = os.getenv("RESEARCHGENIE_OUTPUT_DIR")
    root = Path(configured).expanduser().resolve() if configured else Path.home() / "ResearchGenie" / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def apply_configuration() -> rg_config.Config:
    """Fill in provider settings the MCP client did not set explicitly, from
    the config saved by `researchgenie config`. Must run before the pipeline
    is imported: shared/utilities/llm_provider.py reads these variables at
    import time. The API key is only ever placed in the environment, never
    returned from a tool or written to output."""
    saved = rg_config.load()
    defaults = {
        "AI_PROVIDER": saved.provider,
        "LOCAL_AI_MODEL": saved.ollama_model,
        "GEMINI_MODEL": saved.gemini_model,
    }
    if saved.gemini_api_key:
        defaults["GEMINI_API_KEY"] = saved.gemini_api_key
    for name, value in defaults.items():
        if value and not os.environ.get(name):
            os.environ[name] = value
    return saved


def _provider() -> str:
    return os.environ.get("AI_PROVIDER", "ollama")


def _model() -> str:
    if _provider() == "gemini":
        return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return os.environ.get("LOCAL_AI_MODEL", "qwen3.5:9b")


@functools.lru_cache(maxsize=1)
def _length_presets() -> dict[str, int]:
    """word_budget.py is a standalone file inside services/research-writing/,
    loaded by path the same way researchgenie/cli.py does (DECISIONS.md D-010)."""
    path = ROOT / "services" / "research-writing" / "word_budget.py"
    spec = importlib.util.spec_from_file_location("researchgenie_mcp_word_budget", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return dict(module.LENGTH_PRESETS)


async def check_provider_ready(start_ollama: bool = True) -> tuple[bool, str]:
    """Mirrors the CLI's readiness check, without any terminal output."""
    provider = _provider()
    if provider == "gemini":
        if os.environ.get("GEMINI_API_KEY"):
            return True, f"Gemini ({_model()}) is configured."
        return False, ("No Gemini API key is configured. Set GEMINI_API_KEY in your MCP "
                       "client's config for this server, or run `researchgenie config` once.")

    from researchgenie import ollama_manager as om

    model = _model()
    if not await om.is_ollama_server_running():
        if not start_ollama or not om.is_ollama_on_path():
            return False, ("Ollama is not running. Start it (or install it from ollama.com), "
                           "or run `researchgenie config` once to set things up.")
        om.start_ollama_server()
        for _ in range(15):
            await asyncio.sleep(1)
            if await om.is_ollama_server_running():
                break
        else:
            return False, "Ollama did not start. Start it manually and try again."
    if not await om.is_model_installed(model):
        return False, (f"The Ollama model '{model}' is not downloaded. Run "
                       f"`ollama pull {model}` or `researchgenie config`, then try again.")
    return True, f"Ollama ({model}) is running and the model is installed."


# --------------------------------------------------------------------------
# Run bookkeeping
# --------------------------------------------------------------------------

def _find_run_directory(run_id: str) -> Path | None:
    if not run_id or not all(c.isalnum() for c in run_id):
        return None
    matches = sorted(output_root().glob(f"*_{run_id}"))
    return matches[0] if matches and matches[0].is_dir() else None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _summarize_event(event: dict[str, Any]) -> dict[str, Any]:
    return {key: event[key] for key in ("type", "stage", "status", "title", "message") if key in event}


def _run_status(run_id: str) -> dict[str, Any]:
    from orchestrator import run_registry

    state = run_registry.get_run(run_id)
    run_directory = _find_run_directory(run_id)
    if state is not None:
        progress = [_summarize_event(e) for e in state.events if e.get("type") not in {"result", "end"}]
        stages_done = [e["stage"] for e in progress if e.get("status") == "done" and "stage" in e]
        return {
            "run_id": run_id,
            "status": state.status,
            "current_stage": progress[-1].get("stage") if progress else "starting",
            "stages_completed": stages_done,
            "latest_update": progress[-1].get("message") if progress else "Run started.",
            "recent_events": progress[-8:],
            "error": state.error,
            "run_directory": str(run_directory) if run_directory else None,
        }
    if run_directory is not None:
        # Started in an earlier server process: only its files on disk remain.
        finished = (run_directory / "metadata.json").exists()
        return {
            "run_id": run_id,
            "status": "completed" if finished else "incomplete",
            "latest_update": ("Finished in an earlier session." if finished else
                              "Started in an earlier session that ended before the run finished."),
            "run_directory": str(run_directory),
        }
    return {"run_id": run_id, "status": "not_found",
            "error": f"No run with id '{run_id}'. Use list_research_runs to see existing runs."}


# --------------------------------------------------------------------------
# MCP server
# --------------------------------------------------------------------------

server = MCPServer(
    name="ResearchGenie",
    version=__version__,
    instructions=(
        "ResearchGenie turns a research question into an evidence-grounded, "
        "citation-verified, quality-audited paper draft. A full run takes several "
        "minutes: call start_research, then call get_run_status (with wait_seconds) "
        "until status is 'completed' or 'failed', then call get_research_result. "
        "Call check_setup first if a run fails to start."
    ),
)


@server.tool()
async def check_setup() -> dict[str, Any]:
    """Report which AI engine ResearchGenie will use and whether it is ready
    to run (Ollama running with its model downloaded, or a Gemini key set).
    Never reveals the API key itself."""
    ready, detail = await check_provider_ready(start_ollama=False)
    return {
        "ready": ready,
        "provider": _provider(),
        "model": _model(),
        "detail": detail,
        "output_directory": str(output_root()),
        "version": __version__,
    }


@server.tool()
async def start_research(
    research_question: str,
    corpus_size: int = 8,
    target_format: Literal["IEEE", "Springer"] = "IEEE",
    length: LengthPreset = "standard",
    publication_type: Literal["conference", "journal", "other"] = "conference",
    domain: str | None = None,
    keywords: list[str] | None = None,
    excluded_topics: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> dict[str, Any]:
    """Start a full ResearchGenie run in the background and return its run_id
    right away. The run discovers 6-9 relevant papers, identifies research
    gaps, writes a cited draft, verifies every citation against Crossref and
    OpenAlex, and audits the result. It takes several minutes; follow up with
    get_run_status and get_research_result.

    Args:
        research_question: The research question or topic, in plain language.
        corpus_size: How many papers to build the draft on (6-9).
        target_format: Citation and layout style of the paper.
        length: short, standard or detailed (target word counts as in the CLI).
        publication_type: Intended venue type.
        domain: Optional field, e.g. "computer science" or "civil engineering".
        keywords: Optional extra search keywords.
        excluded_topics: Optional topics to keep out of the literature search.
        year_from: Optional earliest publication year (needs year_to too).
        year_to: Optional latest publication year (needs year_from too).
    """
    from orchestrator import run_registry
    from orchestrator.pipeline import run_pipeline
    from shared.contracts.pipeline_contract import ResearchRequest

    if not research_question.strip():
        raise ToolError("research_question must not be empty.")
    if not 6 <= corpus_size <= 9:
        raise ToolError("corpus_size must be between 6 and 9.")
    if (year_from is None) != (year_to is None):
        raise ToolError("Give both year_from and year_to, or neither.")

    ready, detail = await check_provider_ready(start_ollama=True)
    if not ready:
        raise ToolError(detail)

    request = ResearchRequest(
        research_question=research_question.strip(),
        corpus_size=corpus_size,
        target_format=target_format,
        publication_type=publication_type,
        max_draft_length=_length_presets()[length],
        domain=domain,
        keywords=keywords or [],
        excluded_topics=excluded_topics or [],
        year_range=(year_from, year_to) if year_from is not None else None,
    )
    run_id = uuid4().hex
    await run_registry.start_run(
        request, run_id, functools.partial(run_pipeline, output_root=output_root()),
    )
    return {
        "run_id": run_id,
        "status": "running",
        "engine": f"{_provider()} · {_model()}",
        "next_step": "Call get_run_status with this run_id (wait_seconds up to "
                     f"{_MAX_WAIT_SECONDS}) until status is 'completed' or 'failed'.",
    }


@server.tool()
async def get_run_status(run_id: str, wait_seconds: int = 0) -> dict[str, Any]:
    """Get the progress of a run: its status (running, completed, failed),
    the stage it is in, and recent progress messages.

    Args:
        run_id: The id returned by start_research.
        wait_seconds: Optionally wait up to this many seconds (max 50) for the
            run to finish before answering, to poll less often.
    """
    wait = max(0, min(wait_seconds, _MAX_WAIT_SECONDS))
    status = _run_status(run_id)
    for _ in range(wait):
        if status["status"] not in {"running"}:
            break
        await asyncio.sleep(1)
        status = _run_status(run_id)
    return status


@server.tool()
async def get_research_result(run_id: str, include_draft: bool = True) -> dict[str, Any]:
    """Get the outcome of a finished run: the final draft (Markdown), the
    verified reference list, quality scores, citation-verification and
    quality reports, and the paths of every output file (including LaTeX,
    and the PDF when a LaTeX toolchain was available).

    Args:
        run_id: The id returned by start_research.
        include_draft: Set to false to get only scores, summaries and paths
            without the full draft text.
    """
    status = _run_status(run_id)
    if status["status"] == "running":
        return {**status, "detail": "The run is still in progress. Call get_run_status first."}
    if status["status"] in {"failed", "not_found"}:
        return status
    run_directory = _find_run_directory(run_id)
    final_dir = run_directory / "final" if run_directory else None
    if final_dir is None or not final_dir.is_dir():
        return {**status, "detail": "This run has no final output (it did not finish)."}

    metadata = _read_json(run_directory / "metadata.json") or {}
    files = {key: str(final_dir / name) for key, name in _FINAL_FILES.items() if (final_dir / name).exists()}
    pdf = final_dir / "paper.pdf"
    if pdf.exists():
        files["pdf"] = str(pdf)

    def read(key: str) -> str | None:
        return (final_dir / _FINAL_FILES[key]).read_text(encoding="utf-8") if key in files else None

    result: dict[str, Any] = {
        "run_id": run_id,
        "status": "completed",
        "research_question": metadata.get("research_question"),
        "overall_quality_score": metadata.get("overall_quality_score"),
        "draft_status": metadata.get("draft_status"),
        "target_words": metadata.get("target_words"),
        "actual_words": metadata.get("actual_words"),
        "timings_seconds": metadata.get("timings"),
        "references": read("references"),
        "quality_report": read("quality_report"),
        "citation_validation_report": read("validation_report"),
        "run_directory": str(run_directory),
        "files": files,
    }
    if include_draft:
        result["draft_markdown"] = read("draft")
    return result


@server.tool()
async def list_research_runs(limit: int = 20) -> list[dict[str, Any]]:
    """List recent ResearchGenie runs in the output folder, newest first,
    with their run_id, question, status and quality score."""
    runs = []
    directories = [p for p in output_root().iterdir() if p.is_dir() and "_" in p.name]
    directories.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for directory in directories[: max(1, min(limit, 100))]:
        run_id = directory.name.rsplit("_", 1)[-1]
        request = _read_json(directory / "00_request.json") or {}
        metadata = _read_json(directory / "metadata.json") or {}
        runs.append({
            "run_id": run_id,
            "research_question": request.get("research_question"),
            "status": _run_status(run_id)["status"],
            "overall_quality_score": metadata.get("overall_quality_score"),
            "run_directory": str(directory),
        })
    return runs


def _resource_text(run_id: str, key: str) -> str:
    run_directory = _find_run_directory(run_id)
    path = run_directory / "final" / _FINAL_FILES[key] if run_directory else None
    if path is None or not path.exists():
        raise ResourceError(f"No {key.replace('_', ' ')} for run '{run_id}' (unknown or unfinished run).")
    return path.read_text(encoding="utf-8")


@server.resource("researchgenie://runs/{run_id}/draft", mime_type="text/markdown",
                 description="Final paper draft (Markdown) of a finished run.")
def draft_resource(run_id: str) -> str:
    return _resource_text(run_id, "draft")


@server.resource("researchgenie://runs/{run_id}/references", mime_type="text/markdown",
                 description="Verified reference list of a finished run.")
def references_resource(run_id: str) -> str:
    return _resource_text(run_id, "references")


@server.resource("researchgenie://runs/{run_id}/quality-report", mime_type="text/markdown",
                 description="Quality-assurance report of a finished run.")
def quality_report_resource(run_id: str) -> str:
    return _resource_text(run_id, "quality_report")


@server.resource("researchgenie://runs/{run_id}/paper.tex", mime_type="application/x-tex",
                 description="LaTeX source of a finished run.")
def latex_resource(run_id: str) -> str:
    return _resource_text(run_id, "latex")


@server.prompt()
def research_paper(topic: str) -> str:
    """Write an evidence-grounded research paper on a topic with ResearchGenie."""
    return (
        f"Use the ResearchGenie tools to produce a research paper draft on: {topic}\n"
        "1. Call check_setup and stop with its advice if it is not ready.\n"
        "2. Call start_research with that topic.\n"
        "3. Call get_run_status with wait_seconds=50 until it is completed or failed, "
        "telling me which stage it is in as it goes.\n"
        "4. Call get_research_result and summarise the quality score, the main findings "
        "and any unverified citations, and give me the output folder path."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="researchgenie-mcp", description="Run ResearchGenie as an MCP server.",
    )
    parser.add_argument("--http", action="store_true",
                        help="Serve over streamable HTTP instead of stdio.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (with --http).")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (with --http).")
    args = parser.parse_args(argv)

    apply_configuration()
    if args.http:
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
