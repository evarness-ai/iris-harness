"""Tests for the `iris cost summary` Typer subcommand (story 12.gov-3.7 Task 5)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.kernel.governance.cost import CostStore
from iris_harness.main import app


@pytest.fixture()
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[CostStore]:
    import iris_harness.kernel.governance.cost.store as store_mod

    db_path = tmp_path / "cost-ledger.db"
    monkeypatch.setattr(store_mod, "DEFAULT_COST_LEDGER_DB_PATH", db_path)
    yield CostStore(db_path=db_path)


def _record(store: CostStore, *, cost: float, tier: str = "tier_3") -> None:
    store.record(
        run_id="r1",
        agent_type="chat",
        tier=tier,
        prompt_tokens=10,
        completion_tokens=10,
        cost_usd=cost,
        user_id="local",
    )


def test_summary_empty(isolated_store: CostStore) -> None:
    result = CliRunner().invoke(app, ["cost", "summary"])
    assert result.exit_code == 0, result.output
    assert "$0.0000" in result.output
    assert "no rows" in result.output


def test_summary_text_shows_total_and_breakdown(isolated_store: CostStore) -> None:
    _record(isolated_store, cost=0.10, tier="tier_3")
    _record(isolated_store, cost=0.05, tier="tier_3")
    _record(isolated_store, cost=0.0, tier="tier_1")

    result = CliRunner().invoke(app, ["cost", "summary"])
    assert result.exit_code == 0, result.output
    assert "$0.1500" in result.output
    assert "tier_3" in result.output
    assert "tier_1" in result.output


def test_summary_json(isolated_store: CostStore) -> None:
    _record(isolated_store, cost=0.20, tier="tier_3")

    result = CliRunner().invoke(app, ["cost", "summary", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["user_id"] == "local"
    assert data["total_usd"] == pytest.approx(0.20)
    assert data["per_tier_usd"]["tier_3"] == pytest.approx(0.20)


def test_summary_since_filter_drops_old_rows(isolated_store: CostStore) -> None:
    from datetime import UTC, datetime, timedelta

    a_week_ago = datetime.now(UTC) - timedelta(days=7)
    isolated_store.record(
        run_id="old",
        agent_type="chat",
        tier="tier_3",
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=10.0,
        user_id="local",
        ts=a_week_ago,
    )
    _record(isolated_store, cost=0.10)

    # since=today should exclude the week-old row.
    today = datetime.now(UTC).date().isoformat()
    result = CliRunner().invoke(app, ["cost", "summary", "--since", today, "--format", "json"])
    data = json.loads(result.output)
    assert data["total_usd"] == pytest.approx(0.10)


def test_summary_user_filter(isolated_store: CostStore) -> None:
    isolated_store.record(
        run_id="x",
        agent_type="chat",
        tier="tier_3",
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=5.0,
        user_id="alice",
    )
    _record(isolated_store, cost=0.10)  # default user 'local'

    result = CliRunner().invoke(app, ["cost", "summary", "--user-id", "alice", "--format", "json"])
    data = json.loads(result.output)
    assert data["user_id"] == "alice"
    assert data["total_usd"] == pytest.approx(5.0)


def test_summary_invalid_since(isolated_store: CostStore) -> None:
    result = CliRunner().invoke(app, ["cost", "summary", "--since", "yesterday"])
    assert result.exit_code == 2
    assert "must be YYYY-MM-DD" in result.output


def test_summary_invalid_format(isolated_store: CostStore) -> None:
    result = CliRunner().invoke(app, ["cost", "summary", "--format", "xml"])
    assert result.exit_code == 2
    assert "unknown --format" in result.output.lower()
