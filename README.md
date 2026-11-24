# ResearchGenie

A local-first AI research assistant. Give it a research question — it discovers relevant literature, identifies genuine research gaps, drafts an evidence-grounded paper, verifies every citation against real scholarly databases, and independently audits its own quality. Everything runs on your own machine by default.

## 📦 Install

```
pip install researchgenie
```

## 🚀 Run

```
researchgenie
```

That's it. The first run walks you through a one-time setup; every run after that goes straight to research.

## ✨ What happens the first time

1. **Choose your AI engine**
   - 🟢 **Ollama** *(recommended)* — runs entirely on your machine, your research data never leaves your device, no API key needed.
   - 🔵 **Gemini** — a cloud alternative for machines without the hardware to comfortably run a local model. Requires your own [Gemini API key](https://aistudio.google.com/apikey).
2. If you chose Ollama, ResearchGenie checks whether it's installed and the model is downloaded — and does both automatically where it can be done safely on your operating system.
3. Your choice is saved locally. You won't be asked again.

Run `researchgenie config` any time to change your setup.

## 🔬 Doing research

Answer a few short questions — your research topic, how many papers to include (6–9), your preferred format (IEEE or Springer), and your target length (Short/Standard/Detailed) — confirm the summary, and watch live progress as each stage of the pipeline runs:

- 🔍 **Research Discovery** — searches multiple scholarly databases, filters for genuine topical relevance, and selects the strongest papers.
- ✍️ **Research Writing** — builds an evidence-grounded outline and draft, citing only the literature it actually found.
- 🔗 **Citation Verification** — checks every reference against real scholarly databases (Crossref, OpenAlex).
- 🛡️ **Quality Assessment** — an independent, fully deterministic audit of citation integrity and claim-to-evidence alignment.

## 📁 What you get

Every run saves a complete, timestamped project folder containing:

- The research draft (Markdown)
- LaTeX source, and a compiled PDF when a LaTeX toolchain is available on your machine
- The reference list
- The quality and citation-validation reports
- Full run metadata

ResearchGenie always tells you the exact folder path when a run finishes.

## 🔌 Use it from Claude, Cursor or any MCP client

ResearchGenie also runs as an [MCP](https://modelcontextprotocol.io) server, so an AI assistant can run research for you and read the results.

```
pip install "researchgenie[mcp]"
```

Run `researchgenie config` once to choose your AI engine (or set it in the `env` block below), then add this to your MCP client's config (for Claude Desktop, `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "researchgenie": {
      "command": "researchgenie-mcp",
      "env": { "RESEARCHGENIE_OUTPUT_DIR": "/path/to/ResearchGenie/outputs" }
    }
  }
}
```

With [uv](https://docs.astral.sh/uv/) you can skip the install step: use `"command": "uvx"` and `"args": ["--from", "researchgenie[mcp]", "researchgenie-mcp"]`. For Claude Code: `claude mcp add researchgenie -- researchgenie-mcp`.

| Tool | What it does |
|---|---|
| `check_setup` | Shows which AI engine will be used and whether it's ready |
| `start_research` | Starts a full run in the background and returns a `run_id` |
| `get_run_status` | Current stage and progress (can wait up to 50 s for completion) |
| `get_research_result` | Final draft, references, quality score, reports and file paths |
| `list_research_runs` | Previous runs in your output folder |

Finished runs can also be read as resources (`researchgenie://runs/{run_id}/draft`, `/references`, `/quality-report`, `/paper.tex`), and the `research_paper` prompt runs the whole flow for you.

Engine settings can be given in the `env` block instead of `researchgenie config`: `AI_PROVIDER` (`ollama` or `gemini`), `LOCAL_AI_MODEL`, `GEMINI_MODEL`, `GEMINI_API_KEY`, `OLLAMA_HOST`. Output goes to `RESEARCHGENIE_OUTPUT_DIR` (default `~/ResearchGenie/outputs`). To try it interactively: `npx @modelcontextprotocol/inspector researchgenie-mcp`. To serve over HTTP instead of stdio: `researchgenie-mcp --http --port 8765`.

## 🔒 Privacy

With the default Ollama engine, your research question, retrieved literature, and generated draft never leave your machine — there is no cloud AI call anywhere in the pipeline. Choosing the optional Gemini engine sends prompts to Google's API, same as using any cloud AI service; your API key is stored locally and never logged, committed, or embedded in generated output.

## 📊 Project status

See [TILL_NOW.md](TILL_NOW.md) for current progress and [DECISIONS.md](DECISIONS.md) for technical decisions and their reasoning. [analytics_and_results.md](analytics_and_results.md) documents the reliability work behind the pipeline, with real measured results.

## 🛠️ For developers

This repository *is* the ResearchGenie source — `pip install researchgenie` installs it directly from here. See `DECISIONS.md` for the internal architecture (a four-service pipeline behind one orchestrator) and `pyproject.toml` for the packaging setup.
