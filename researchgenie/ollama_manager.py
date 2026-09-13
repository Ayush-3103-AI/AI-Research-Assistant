"""Detect, launch, and prepare the local Ollama runtime and model.

Goal: a researchgenie user should never have to manually run `ollama serve`
or `ollama pull` when it can be done safely and automatically. Where it
cannot be done safely (no supported package manager present, or the only
available path is running an unreviewed remote script), this module gives
clear, exact manual instructions instead of pretending to automate it.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


def is_ollama_on_path() -> bool:
    return shutil.which("ollama") is not None


async def is_ollama_server_running() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as http:
            resp = await http.get(f"{OLLAMA_HOST}/api/tags")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def is_model_installed(model: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as http:
            resp = await http.get(f"{OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
            names = {entry["name"] for entry in resp.json().get("models", [])}
            return model in names or f"{model}:latest" in names
    except httpx.HTTPError:
        return False


def start_ollama_server() -> bool:
    """Best-effort: launch `ollama serve` detached in the background.
    Returns True only if the launch attempt itself succeeded — callers
    should poll is_ollama_server_running() to confirm it actually came up."""
    if not is_ollama_on_path():
        return False
    try:
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(["ollama", "serve"], **kwargs)
        return True
    except OSError:
        return False


# The Windows installer is a ~1.5GB download (OllamaSetup.exe v0.34.0 measured
# at 1501.3 MB on 2026-09-13), and Homebrew's bottle is the same order of
# magnitude. The old 600s value could not cover the download on an ordinary
# connection, let alone the install that follows it, so a real winget run on a
# clean machine was killed mid-download and reported as a flat failure. One
# hour is generous enough to be about the connection, not the clock.
INSTALL_TIMEOUT_SECONDS = 3600


@dataclass
class InstallAttempt:
    attempted: bool
    succeeded: bool
    message: str


def _timeout_message(tool: str, download_url: str) -> str:
    """A timeout is "not finished yet", not "this machine cannot install
    Ollama" — say so, or a user on a slow link gives up on a download that
    was progressing perfectly well."""
    return (
        f"{tool} install timed out after {INSTALL_TIMEOUT_SECONDS}s. The Ollama "
        "installer is a ~1.5GB download, so this usually means a slow connection "
        "rather than a broken install — it may well have been progressing fine. "
        f"Re-run to resume, or install it directly from {download_url}, then "
        "restart researchgenie."
    )


def attempt_automatic_install() -> InstallAttempt:
    """The safest, platform-appropriate automatic install path.

    Never attempts privilege escalation and never pipes a remote script
    into a shell — where the only documented path involves that (Linux),
    this returns clear manual instructions instead of pretending to
    automate an action this product should not perform unreviewed."""
    system = platform.system()

    if system == "Windows":
        if shutil.which("winget"):
            try:
                result = subprocess.run(
                    ["winget", "install", "--id", "Ollama.Ollama", "-e",
                     "--accept-package-agreements", "--accept-source-agreements"],
                    capture_output=True, text=True, timeout=INSTALL_TIMEOUT_SECONDS,
                )
                if result.returncode == 0:
                    return InstallAttempt(True, True, "Installed via winget.")
                return InstallAttempt(
                    True, False, f"winget install failed: {result.stderr.strip()[:300]}"
                )
            except subprocess.TimeoutExpired:
                return InstallAttempt(
                    True, False,
                    _timeout_message("winget", "https://ollama.com/download/windows"),
                )
            except OSError as error:
                return InstallAttempt(True, False, f"winget install failed: {error}")
        return InstallAttempt(
            False, False,
            "winget is not available on this system. Download and run the installer "
            "from https://ollama.com/download/windows, then restart researchgenie."
        )

    if system == "Darwin":
        if shutil.which("brew"):
            try:
                result = subprocess.run(
                    ["brew", "install", "ollama"], capture_output=True, text=True,
                    timeout=INSTALL_TIMEOUT_SECONDS,
                )
                if result.returncode == 0:
                    return InstallAttempt(True, True, "Installed via Homebrew.")
                return InstallAttempt(
                    True, False, f"brew install failed: {result.stderr.strip()[:300]}"
                )
            except subprocess.TimeoutExpired:
                return InstallAttempt(
                    True, False,
                    _timeout_message("brew", "https://ollama.com/download/mac"),
                )
            except OSError as error:
                return InstallAttempt(True, False, f"brew install failed: {error}")
        return InstallAttempt(
            False, False,
            "Homebrew is not available. Download the installer from "
            "https://ollama.com/download/mac, then restart researchgenie."
        )

    if system == "Linux":
        return InstallAttempt(
            False, False,
            "Automatic installation on Linux is not attempted for safety — the only "
            "officially documented path pipes a remote script into a shell, which this "
            "product will not do unreviewed. Run it yourself if you trust it:\n"
            "    curl -fsSL https://ollama.com/install.sh | sh\n"
            "then restart researchgenie. Alternatively use your distribution's package "
            "manager if it packages Ollama."
        )

    return InstallAttempt(
        False, False,
        f"Unrecognized platform ({system}). Install Ollama manually from "
        "https://ollama.com/download, then restart researchgenie."
    )


ProgressCallback = Callable[[str, int, int], None]


async def pull_model(model: str, on_progress: ProgressCallback | None = None) -> bool:
    """Streams Ollama's own /api/pull progress. on_progress, if given, is
    called with (status_text, bytes_completed, bytes_total) for each update."""
    try:
        async with httpx.AsyncClient(timeout=None) as http:
            async with http.stream(
                "POST", f"{OLLAMA_HOST}/api/pull", json={"name": model, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if on_progress:
                        on_progress(
                            chunk.get("status", ""),
                            chunk.get("completed", 0),
                            chunk.get("total", 0),
                        )
                    if chunk.get("error"):
                        return False
        return True
    except httpx.HTTPError:
        return False
