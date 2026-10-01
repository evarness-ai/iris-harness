"""Tests for the unified ``iris learning`` command group."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import iris_harness.cli.learning as learning_mod
from iris_harness.cli.learning import learning_app
from iris_harness.services.learning.store import LearningMetricsStore

runner = CliRunner()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "learning.db"
    store = LearningMetricsStore(db_path=path)
    store.ensure_schema()
    # Two behavior proposals: one approved, one rejected -> 50% acceptance.
    store.propose_behavior_pattern("p1", "reviews finances every morning", "high", [])
    store.resolve_behavior_proposal("p1", "approved")
    store.propose_behavior_pattern("p2", "skims ads", "low", [])
    store.resolve_behavior_proposal("p2", "rejected")
    return path


def test_status_renders_proposal_quality(db: Path) -> None:
    result = runner.invoke(learning_app, ["status", "--db-path", str(db)])
    assert result.exit_code == 0
    assert "proposals" in result.stdout
    assert "behaviors" in result.stdout
    assert "50%" in result.stdout  # 1 approved / 2 reviewed
    assert "no analysis yet" in result.stdout


def test_intelligence_renders_empty(db: Path) -> None:
    result = runner.invoke(learning_app, ["intelligence", "--db-path", str(db)])
    assert result.exit_code == 0
    assert "escalation precision" in result.stdout


def test_recommendations_none_yet(db: Path) -> None:
    result = runner.invoke(learning_app, ["recommendations", "--db-path", str(db)])
    assert result.exit_code == 0
    assert "no analysis yet" in result.stdout


def test_mounted_behaviors_subapp_reachable(db: Path) -> None:
    result = runner.invoke(learning_app, ["behaviors", "stats", "--db-path", str(db)])
    assert result.exit_code == 0
    assert "behaviors" in result.stdout
    assert "50%" in result.stdout  # 1 approved / 2 reviewed — mounted sub-app reads the same db


def test_preview_renders_from_api(cli_api: Any) -> None:
    payloads = {
        "/learning/behaviors/preview": {
            "available": True,
            "patterns": [{"text": "reviews finances every morning", "confidence": "high"}],
        },
        "/learning/intentions/preview": {"available": True, "intentions": []},
    }

    def _handle(request: httpx.Request) -> httpx.Response:
        data = payloads.get(request.url.path)
        if data is None:
            raise AssertionError(f"unexpected url {request.url}")
        return httpx.Response(200, json=data)

    cli_api(_handle, learning_mod)
    result = runner.invoke(learning_app, ["preview"])
    assert result.exit_code == 0
    assert "would-be behaviors (1)" in result.stdout
    assert "reviews finances every morning" in result.stdout
    assert "nothing was persisted" in result.stdout


def test_preview_friendly_error_when_api_down(cli_api: Any) -> None:
    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    cli_api(_boom, learning_mod)
    result = runner.invoke(learning_app, ["preview"])
    assert result.exit_code == 1
    assert "Is the stack up?" in result.stdout
