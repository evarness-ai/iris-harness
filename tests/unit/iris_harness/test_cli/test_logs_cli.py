"""``iris logs archive|restore`` — the encrypted session-log archive from the CLI."""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from iris_harness.cli import logs as logs_cli
from iris_harness.main import app

runner = CliRunner()


class _Retention:
    def __init__(self) -> None:
        self.restores: list[dict[str, Any]] = []

    def archived_logs(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "root": "/x/archive/logs",
            "months": [{"month": "2026-07", "files": 3, "stored_bytes": 20480, "members": []}],
            "stored_bytes": 20480,
        }

    def restore_logs(self, *, session: str | None = None, month: str | None = None) -> list[str]:
        self.restores.append({"session": session, "month": month})
        if month == "bad":
            raise ValueError("month must be YYYY-MM, got 'bad'")
        return ["session-web-a.jsonl"] if session == "web-a" else []


@pytest.fixture
def retention(monkeypatch: pytest.MonkeyPatch) -> _Retention:
    fake = _Retention()
    monkeypatch.setattr(logs_cli, "_retention", lambda db_path: fake)
    return fake


def test_archive_lists_months(retention: _Retention) -> None:
    result = runner.invoke(app, ["logs", "archive"])
    assert result.exit_code == 0
    assert "2026-07" in result.output and "20 KB" in result.output


def test_restore_a_session(retention: _Retention) -> None:
    result = runner.invoke(app, ["logs", "restore", "--session", "web-a"])
    assert result.exit_code == 0 and "restored session-web-a.jsonl" in result.output
    assert retention.restores == [{"session": "web-a", "month": None}]


def test_restore_needs_exactly_one_target(retention: _Retention) -> None:
    assert runner.invoke(app, ["logs", "restore"]).exit_code == 2
    both = runner.invoke(app, ["logs", "restore", "--session", "a", "--month", "2026-07"])
    assert both.exit_code == 2 and retention.restores == []
    assert runner.invoke(app, ["logs", "restore", "--month", "bad"]).exit_code == 2


def test_nothing_to_restore_says_so(retention: _Retention) -> None:
    result = runner.invoke(app, ["logs", "restore", "--session", "web-zzz"])
    assert result.exit_code == 0 and "nothing restored" in result.output
