"""Tests for the ``iris context-health`` CLI (renders GET /context-health)."""

from __future__ import annotations

from typing import Any

import httpx
from typer.testing import CliRunner

import iris_harness.cli.context_health as ch

runner = CliRunner()

_SNAPSHOT = {
    "available": True,
    "window": {
        "budget_tokens": 6144,
        "current_tokens": 5200,
        "fill_pct": 0.846,
        "compaction_ratio": 0.8,
        "near_full": True,
        "last_compaction": {
            "trigger": "tokens",
            "archived_count": 6,
            "tokens_before": 5790,
            "tokens_after": 2486,
        },
    },
    "budgets": {
        "transcript_budget": 2764,
        "memory_budget": 2150,
        "last_context_tokens": 4100,
        "last_transcript_evicted": 120,
    },
    "suppression": {"total_feedback": 4, "active_suppressions": 2, "by_subsystem": {"email": 2}},
}


def _serve(payload: dict[str, Any]) -> Any:
    def _handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/context-health"
        assert request.url.params["session_id"] == "default"
        return httpx.Response(200, json=payload)

    return _handle


def test_renders_snapshot(cli_api: Any) -> None:
    cli_api(_serve(_SNAPSHOT), ch)
    result = runner.invoke(ch.context_health_app, [])
    assert result.exit_code == 0
    assert "near full" in result.stdout
    assert "tokens trigger" in result.stdout
    assert "active" in result.stdout


def test_friendly_error_when_api_down(cli_api: Any) -> None:
    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    cli_api(_boom, ch)
    result = runner.invoke(ch.context_health_app, [])
    assert result.exit_code == 1
    assert "Is the stack up?" in result.stdout


def test_unavailable_runtime(cli_api: Any) -> None:
    cli_api(_serve({"available": False}), ch)
    result = runner.invoke(ch.context_health_app, [])
    assert result.exit_code == 0
    assert "unavailable" in result.stdout
