"""Smoke tests for the `iris behaviors` CLI (HITL review of mined patterns)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from iris_harness.cli.behaviors import behaviors_app
from iris_harness.services.learning.store import LearningMetricsStore

runner = CliRunner()


def _seed(db: Path) -> LearningMetricsStore:
    s = LearningMetricsStore(db_path=db)
    s.ensure_schema()
    s.propose_behavior_pattern("abc123", "Asks for inbox each morning", "high", ["mon", "tue"])
    return s


def test_list_empty(tmp_path: Path) -> None:
    db = str(tmp_path / "learning.db")
    r = runner.invoke(behaviors_app, ["list", "--db-path", db])
    assert r.exit_code == 0, r.stdout
    assert "no pending" in r.stdout.lower()


def test_list_shows_proposal(tmp_path: Path) -> None:
    db = tmp_path / "learning.db"
    _seed(db)
    r = runner.invoke(behaviors_app, ["list", "--db-path", str(db)])
    assert r.exit_code == 0, r.stdout
    assert "abc123" in r.stdout and "inbox" in r.stdout


def test_approve_adds_to_episodic_and_clears(tmp_path: Path, monkeypatch) -> None:
    from iris_harness.memory.identity import loader

    episodic = tmp_path / "episodic.md"
    monkeypatch.setattr(loader, "EPISODIC_MD_PATH", episodic)
    monkeypatch.setattr(loader, "MEMORY_DIR", tmp_path)

    db = tmp_path / "learning.db"
    store = _seed(db)
    r = runner.invoke(behaviors_app, ["approve", "abc123", "--db-path", str(db)])
    assert r.exit_code == 0, r.stdout
    assert "episodic memory" in r.stdout
    # The pattern is now durable in episodic.md...
    assert "Asks for inbox each morning" in episodic.read_text()
    # ...and out of the pending queue.
    assert store.list_behavior_proposals(status="pending") == []
    assert len(store.list_behavior_proposals(status="approved")) == 1


def test_approve_unknown_id_exits_nonzero(tmp_path: Path) -> None:
    db = tmp_path / "learning.db"
    _seed(db)
    r = runner.invoke(behaviors_app, ["approve", "nope", "--db-path", str(db)])
    assert r.exit_code == 1


def test_reject_clears_and_blocks_requeue(tmp_path: Path) -> None:
    db = tmp_path / "learning.db"
    store = _seed(db)
    r = runner.invoke(behaviors_app, ["reject", "abc123", "--db-path", str(db)])
    assert r.exit_code == 0
    assert store.list_behavior_proposals(status="pending") == []
    # A re-mine of the same id must not re-queue (PK already present, now 'rejected').
    assert (
        store.propose_behavior_pattern("abc123", "Asks for inbox each morning", "high", []) is False
    )
