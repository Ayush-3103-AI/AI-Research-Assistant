"""Tests for the ResearchGenie MCP server, through a real in-process MCP
client (mcp.Client) with the pipeline and the engine check faked out — no
network, no LLM."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from orchestrator.pipeline import PipelineError  # noqa: E402
from researchgenie import mcp_server  # noqa: E402


async def _fake_pipeline_success(request, output_root=None, run_id=None):
    run_directory = output_root / f"fake-question_{run_id}"
    final = run_directory / "final"
    final.mkdir(parents=True)
    (run_directory / "00_request.json").write_text(json.dumps(request.model_dump(mode="json")))
    yield {"type": "progress", "stage": "discovery", "status": "running", "message": "Searching"}
    yield {"type": "progress", "stage": "discovery", "status": "done", "message": "Selected 6 papers"}
    (final / "draft.md").write_text("# Draft\nBody [1].")
    (final / "references.md").write_text("- [1] A. Author, 2024.\n")
    (final / "quality_report.md").write_text("# Quality\n")
    (final / "validation_report.md").write_text("# Validation\n")
    (final / "paper.tex").write_text("\\documentclass{article}")
    (run_directory / "metadata.json").write_text(json.dumps({
        "run_id": run_id, "research_question": request.research_question,
        "overall_quality_score": 4.5, "draft_status": "complete",
        "target_words": request.max_draft_length, "actual_words": 3000,
        "timings": {"discovery_seconds": 1.0},
    }))
    yield {"type": "result", "result": None}


async def _fake_pipeline_failure(request, output_root=None, run_id=None):
    yield {"type": "progress", "stage": "discovery", "status": "running", "message": "Searching"}
    raise PipelineError("Research Discovery Service produced no result.")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCHGENIE_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("RESEARCHGENIE_CONFIG_DIR", str(tmp_path / "config"))

    async def ready(start_ollama=True):
        return True, "ready"

    monkeypatch.setattr(mcp_server, "check_provider_ready", ready)
    monkeypatch.setattr(orchestrator_pipeline, "run_pipeline", _fake_pipeline_success)


def _data(result):
    assert not result.is_error, result.content
    if result.structured_content is not None:
        return result.structured_content.get("result", result.structured_content)
    return json.loads(result.content[0].text)


async def _until_finished(client, run_id):
    for _ in range(200):
        status = _data(await client.call_tool("get_run_status", {"run_id": run_id}))
        if status["status"] != "running":
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("run never finished")


def test_lists_expected_tools_resources_and_prompt():
    async def scenario():
        async with Client(mcp_server.server) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            templates = {t.uri_template for t in (await client.list_resource_templates()).resource_templates}
            prompts = {p.name for p in (await client.list_prompts()).prompts}
        return tools, templates, prompts

    tools, templates, prompts = asyncio.run(scenario())
    assert tools == {"check_setup", "start_research", "get_run_status",
                     "get_research_result", "list_research_runs"}
    assert "researchgenie://runs/{run_id}/draft" in templates
    assert "research_paper" in prompts


def test_full_run_through_mcp(tmp_path):
    async def scenario():
        async with Client(mcp_server.server) as client:
            started = _data(await client.call_tool("start_research", {
                "research_question": "How do LLMs affect code review?",
                "corpus_size": 6, "length": "short", "target_format": "Springer",
            }))
            run_id = started["run_id"]
            assert started["status"] == "running"
            status = await _until_finished(client, run_id)
            result = _data(await client.call_tool("get_research_result", {"run_id": run_id}))
            draft = await client.read_resource(f"researchgenie://runs/{run_id}/draft")
            runs = _data(await client.call_tool("list_research_runs", {}))
        return run_id, status, result, draft, runs

    run_id, status, result, draft, runs = asyncio.run(scenario())
    assert status["status"] == "completed"
    assert "discovery" in status["stages_completed"]
    assert result["overall_quality_score"] == 4.5
    assert result["draft_markdown"].startswith("# Draft")
    assert "references" in result and "[1]" in result["references"]
    assert set(result["files"]) >= {"draft", "references", "latex"}
    assert Path(result["run_directory"]).is_relative_to(tmp_path / "outputs")
    assert draft.contents[0].text.startswith("# Draft")
    assert runs[0]["run_id"] == run_id
    assert runs[0]["research_question"] == "How do LLMs affect code review?"


def test_failed_run_reports_error(monkeypatch):
    monkeypatch.setattr(orchestrator_pipeline, "run_pipeline", _fake_pipeline_failure)

    async def scenario():
        async with Client(mcp_server.server) as client:
            run_id = _data(await client.call_tool(
                "start_research", {"research_question": "Q?"}))["run_id"]
            return await _until_finished(client, run_id)

    status = asyncio.run(scenario())
    assert status["status"] == "failed"
    assert "no result" in status["error"]


def test_invalid_input_and_unready_engine_are_tool_errors(monkeypatch):
    async def not_ready(start_ollama=True):
        return False, "Ollama is not running."

    async def scenario():
        async with Client(mcp_server.server) as client:
            bad_size = await client.call_tool(
                "start_research", {"research_question": "Q?", "corpus_size": 3})
            monkeypatch.setattr(mcp_server, "check_provider_ready", not_ready)
            unready = await client.call_tool("start_research", {"research_question": "Q?"})
            unknown = _data(await client.call_tool("get_run_status", {"run_id": "doesnotexist"}))
            traversal = _data(await client.call_tool("get_run_status", {"run_id": "../etc"}))
        return bad_size, unready, unknown, traversal

    bad_size, unready, unknown, traversal = asyncio.run(scenario())
    assert bad_size.is_error
    assert unready.is_error and "Ollama is not running" in unready.content[0].text
    assert unknown["status"] == "not_found"
    assert traversal["status"] == "not_found"


def test_configuration_env_wins_over_saved_config(monkeypatch):
    from researchgenie import config as rg_config

    monkeypatch.setattr(rg_config, "load", lambda: rg_config.Config(
        setup_complete=True, provider="gemini", gemini_api_key="saved-key",
        ollama_model="saved-model"))
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.delenv("LOCAL_AI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    mcp_server.apply_configuration()
    import os
    assert os.environ["AI_PROVIDER"] == "ollama"          # explicit env kept
    assert os.environ["LOCAL_AI_MODEL"] == "saved-model"  # filled from saved config
    assert os.environ["GEMINI_API_KEY"] == "saved-key"


def test_check_setup_never_returns_the_api_key(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key-123")

    async def scenario():
        async with Client(mcp_server.server) as client:
            return await client.call_tool("check_setup", {})

    result = asyncio.run(scenario())
    assert "super-secret-key-123" not in json.dumps([c.model_dump() for c in result.content])
    assert _data(result)["provider"] == "gemini"
