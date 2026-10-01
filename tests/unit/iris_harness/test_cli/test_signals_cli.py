"""Smoke tests for the read-only `iris signals` CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from iris_harness.cli.signals import signals_app
from iris_harness.services.learning.store import LearningMetricsStore

runner = CliRunner()


def _seed(db: Path) -> None:
    s = LearningMetricsStore(db_path=db)
    s.ensure_schema()
    s.record_user_behavior_signal("fact_corrected", subject="location", detail="New York")
    s.record_user_behavior_signal("pattern_confirmed", subject="Asks for inbox each morning")


def test_list_empty(tmp_path: Path) -> None:
    r = runner.invoke(signals_app, ["list", "--db-path", str(tmp_path / "learning.db")])
    assert r.exit_code == 0
    assert "no user-behavior signals" in r.stdout.lower()


def test_list_and_filter(tmp_path: Path) -> None:
    db = tmp_path / "learning.db"
    _seed(db)
    r = runner.invoke(signals_app, ["list", "--db-path", str(db)])
    assert r.exit_code == 0, r.stdout
    assert "location" in r.stdout and "inbox" in r.stdout

    filtered = runner.invoke(
        signals_app, ["list", "--db-path", str(db), "--kind", "fact_corrected"]
    )
    assert "location" in filtered.stdout and "inbox" not in filtered.stdout


def test_summary(tmp_path: Path) -> None:
    db = tmp_path / "learning.db"
    _seed(db)
    r = runner.invoke(signals_app, ["summary", "--db-path", str(db)])
    assert r.exit_code == 0, r.stdout
    assert "fact_corrected" in r.stdout and "pattern_confirmed" in r.stdout
