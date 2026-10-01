"""Tests for the `/llm` slash command."""

from __future__ import annotations

import io
from types import SimpleNamespace

from rich.console import Console

from iris_harness.cli import commands


def _capture_console(monkeypatch) -> io.StringIO:
    output = io.StringIO()
    monkeypatch.setattr(
        "iris_harness.cli.render.console",
        Console(file=output, force_terminal=False, width=120),
    )
    return output


def _state(mode: str = "active", *, pin: str | None = None, adaptive: bool = False) -> dict:
    return {
        "mode": mode,
        "auto_mode": mode,
        "pin": pin,
        "adaptive": adaptive,
        "snapshot": {
            "ram_free_gb": 18.4,
            "cpu_percent": 12.0,
            "cpu_speed_limit": 100,
            "thermal_throttled": False,
            "sampled_at": "2026-05-09T12:00:00+00:00",
        },
    }


def test_llm_status_calls_get_mode(monkeypatch) -> None:
    output = _capture_console(monkeypatch)
    calls: list[tuple[str, str, str, dict | None]] = []

    def fake_api_json(api_url, *, method, path, payload=None, timeout=30):
        calls.append((api_url, method, path, payload))
        return _state(adaptive=True)

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_llm(ctx, "") is True
    assert calls == [("http://testserver", "GET", "/llm/mode", None)]
    text = output.getvalue()
    assert "active" in text
    assert "ram_free=18.4GB" in text


def test_llm_status_explicit_subcommand(monkeypatch) -> None:
    _capture_console(monkeypatch)
    seen: list[tuple[str, str]] = []

    def fake_api_json(api_url, *, method, path, payload=None, timeout=30):
        seen.append((method, path))
        return _state()

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    assert commands._cmd_llm(SimpleNamespace(api_url="http://t"), "status") is True
    assert seen == [("GET", "/llm/mode")]


def test_llm_pressure_hits_pressure_endpoint(monkeypatch) -> None:
    _capture_console(monkeypatch)
    seen: list[tuple[str, str]] = []

    def fake_api_json(api_url, *, method, path, payload=None, timeout=30):
        seen.append((method, path))
        return _state()

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    assert commands._cmd_llm(SimpleNamespace(api_url="http://t"), "pressure") is True
    assert seen == [("GET", "/llm/pressure")]


def test_llm_mode_pin_sends_pin_payload(monkeypatch) -> None:
    _capture_console(monkeypatch)
    captured: dict = {}

    def fake_api_json(api_url, *, method, path, payload=None, timeout=30):
        captured.update({"method": method, "path": path, "payload": payload})
        return _state(mode="thermal", pin="thermal")

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    assert commands._cmd_llm(SimpleNamespace(api_url="http://t"), "mode thermal") is True
    assert captured == {"method": "POST", "path": "/llm/mode/pin", "payload": {"mode": "thermal"}}


def test_llm_unpin_sends_delete(monkeypatch) -> None:
    _capture_console(monkeypatch)
    seen: list[tuple[str, str]] = []

    def fake_api_json(api_url, *, method, path, payload=None, timeout=30):
        seen.append((method, path))
        return _state()

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    assert commands._cmd_llm(SimpleNamespace(api_url="http://t"), "unpin") is True
    assert seen == [("DELETE", "/llm/mode/pin")]


def test_llm_mode_rejects_unknown_value(monkeypatch) -> None:
    _capture_console(monkeypatch)
    called = False

    def fake_api_json(*_a, **_kw):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    assert commands._cmd_llm(SimpleNamespace(api_url="http://t"), "mode bogus") is True
    assert called is False


def test_llm_mode_requires_argument(monkeypatch) -> None:
    _capture_console(monkeypatch)

    def fake_api_json(*_a, **_kw):
        raise AssertionError("API should not be called when arg is missing")

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    assert commands._cmd_llm(SimpleNamespace(api_url="http://t"), "mode") is True


def test_llm_command_is_registered() -> None:
    cmd = commands._REGISTRY.get("/llm")
    assert cmd is not None
    assert cmd.handler is commands._cmd_llm
