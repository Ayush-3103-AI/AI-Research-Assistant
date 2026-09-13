"""Tests for Ollama detection/installation logic — network and subprocess
calls are mocked so these run fast and hermetically."""

import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from researchgenie import ollama_manager as om  # noqa: E402


def test_is_ollama_on_path_reflects_shutil_which(monkeypatch):
    monkeypatch.setattr(om.shutil, "which", lambda name: "/usr/bin/ollama" if name == "ollama" else None)
    assert om.is_ollama_on_path() is True
    monkeypatch.setattr(om.shutil, "which", lambda name: None)
    assert om.is_ollama_on_path() is False


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json


class _FakeAsyncClient:
    def __init__(self, response=None, raise_error=None):
        self._response = response
        self._raise = raise_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, *args, **kwargs):
        if self._raise:
            raise self._raise
        return self._response


def test_is_ollama_server_running_true_when_reachable(monkeypatch):
    monkeypatch.setattr(om.httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(_FakeResponse(200)))
    assert asyncio.run(om.is_ollama_server_running()) is True


def test_is_ollama_server_running_false_when_unreachable(monkeypatch):
    monkeypatch.setattr(
        om.httpx, "AsyncClient",
        lambda *a, **k: _FakeAsyncClient(raise_error=httpx.ConnectError("no server")),
    )
    assert asyncio.run(om.is_ollama_server_running()) is False


def test_is_model_installed_true_when_present(monkeypatch):
    response = _FakeResponse(200, {"models": [{"name": "qwen3.5:9b"}]})
    monkeypatch.setattr(om.httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(response))
    assert asyncio.run(om.is_model_installed("qwen3.5:9b")) is True


def test_is_model_installed_false_when_absent(monkeypatch):
    response = _FakeResponse(200, {"models": [{"name": "llama3:8b"}]})
    monkeypatch.setattr(om.httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(response))
    assert asyncio.run(om.is_model_installed("qwen3.5:9b")) is False


def test_start_ollama_server_returns_false_when_not_on_path(monkeypatch):
    monkeypatch.setattr(om, "is_ollama_on_path", lambda: False)
    assert om.start_ollama_server() is False


def test_attempt_automatic_install_on_linux_never_pipes_a_remote_script(monkeypatch):
    """Explicit safety requirement: Linux has no safe automatic path, so
    this must never attempt to run curl|sh itself."""
    monkeypatch.setattr(om.platform, "system", lambda: "Linux")
    result = om.attempt_automatic_install()
    assert result.attempted is False
    assert "curl" in result.message  # instructs, does not execute


def test_attempt_automatic_install_on_windows_without_winget_gives_manual_instructions(monkeypatch):
    monkeypatch.setattr(om.platform, "system", lambda: "Windows")
    monkeypatch.setattr(om.shutil, "which", lambda name: None)
    result = om.attempt_automatic_install()
    assert result.attempted is False
    assert "ollama.com" in result.message


def test_attempt_automatic_install_on_windows_with_winget_runs_it(monkeypatch):
    monkeypatch.setattr(om.platform, "system", lambda: "Windows")
    monkeypatch.setattr(om.shutil, "which", lambda name: "winget.exe" if name == "winget" else None)

    class _Result:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(om.subprocess, "run", lambda *a, **k: _Result())
    result = om.attempt_automatic_install()
    assert result.attempted is True
    assert result.succeeded is True


# --- Install timeout (D-042) ------------------------------------------------
# These pin the one thing the mock-only tests above could never catch: the
# real Windows installer is ~1.5GB, so the timeout has to be big enough to
# actually download and install it. Measured 2026-09-13 against
# OllamaSetup.exe v0.34.0 (1501.3 MB) after the old 600s value killed a real
# winget install mid-download on a clean machine.


def _record_run(monkeypatch, returncode=0, raises=None):
    """Capture the kwargs attempt_automatic_install() actually passes to
    subprocess.run — the existing tests use `lambda *a, **k`, which silently
    discards exactly the argument that was wrong."""
    captured = {}

    class _Result:
        stderr = ""

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        if raises is not None:
            raise raises
        result = _Result()
        result.returncode = returncode
        return result

    monkeypatch.setattr(om.subprocess, "run", fake_run)
    return captured


def test_install_timeout_allows_for_a_multi_gigabyte_download():
    # 1.5GB at a conservative-but-realistic 1 MB/s is ~1500s of download
    # before Inno Setup has unpacked a single byte. Anything below this
    # floor cannot finish on an ordinary connection.
    assert om.INSTALL_TIMEOUT_SECONDS >= 1800


def test_windows_install_uses_the_shared_install_timeout(monkeypatch):
    monkeypatch.setattr(om.platform, "system", lambda: "Windows")
    monkeypatch.setattr(om.shutil, "which", lambda name: "winget.exe" if name == "winget" else None)
    captured = _record_run(monkeypatch)

    result = om.attempt_automatic_install()

    assert result.succeeded is True
    assert captured["kwargs"]["timeout"] == om.INSTALL_TIMEOUT_SECONDS


def test_macos_install_uses_the_shared_install_timeout(monkeypatch):
    monkeypatch.setattr(om.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(om.shutil, "which", lambda name: "brew" if name == "brew" else None)
    captured = _record_run(monkeypatch)

    result = om.attempt_automatic_install()

    assert result.succeeded is True
    assert captured["kwargs"]["timeout"] == om.INSTALL_TIMEOUT_SECONDS


def test_install_timeout_is_reported_as_a_timeout_not_a_flat_failure(monkeypatch):
    """A timeout means "not finished yet", which is not the same as "this
    machine cannot install Ollama" — the old message said only "failed",
    leaving a user on a slow link with no idea the download was fine."""
    monkeypatch.setattr(om.platform, "system", lambda: "Windows")
    monkeypatch.setattr(om.shutil, "which", lambda name: "winget.exe" if name == "winget" else None)
    _record_run(monkeypatch, raises=om.subprocess.TimeoutExpired(cmd="winget", timeout=1800))

    result = om.attempt_automatic_install()

    assert result.attempted is True
    assert result.succeeded is False
    assert "timed out" in result.message.lower()
    # Must point at the real cause (a big download that needs more time),
    # and give the user somewhere to go.
    assert "ollama.com" in result.message
