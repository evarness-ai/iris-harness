"""Tests for the user-behavior signal store + capture at the steering seams."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from iris_harness.cli.behaviors import behaviors_app
from iris_harness.cli.facts import facts_app
from iris_harness.memory.store import MemoryStore, UserFact
from iris_harness.services.learning.store import LearningMetricsStore

runner = CliRunner()


def _learning(tmp_path: Path) -> LearningMetricsStore:
    s = LearningMetricsStore(db_path=tmp_path / "learning.db")
    s.ensure_schema()
    return s


def test_store_record_list_summary(tmp_path: Path) -> None:
    s = _learning(tmp_path)
    s.record_user_behavior_signal("fact_corrected", subject="location", detail="NYC")
    s.record_user_behavior_signal("pattern_confirmed", subject="checks inbox AM")
    s.record_user_behavior_signal("fact_corrected", subject="blog", detail="x.com")
    assert s.user_behavior_summary() == {"fact_corrected": 2, "pattern_confirmed": 1}
    only = s.list_user_behavior_signals(kind="fact_corrected")
    assert [x.subject for x in only] == ["blog", "location"]  # newest first


def _seed_fact(db: Path) -> None:
    store = MemoryStore(db_path=db)
    store.ensure_schema()
    now = datetime.now(UTC)
    store.upsert_user_fact(
        UserFact(
            key="location",
            value="Berlin",
            confidence=0.9,
            source="test",
            first_seen=now,
            last_confirmed=now,
            times_confirmed=1,
        )
    )


def test_facts_cli_correct_emits_signal(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed_fact(db)
    r = runner.invoke(facts_app, ["correct", "location", "New York", "--db-path", str(db)])
    assert r.exit_code == 0, r.stdout
    sigs = LearningMetricsStore(db_path=tmp_path / "learning.db").list_user_behavior_signals()
    assert len(sigs) == 1
    assert sigs[0].kind == "fact_corrected" and sigs[0].subject == "location"


def test_facts_cli_forget_emits_signal(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed_fact(db)
    r = runner.invoke(facts_app, ["forget", "location", "--db-path", str(db), "--yes"])
    assert r.exit_code == 0
    sigs = LearningMetricsStore(db_path=tmp_path / "learning.db").list_user_behavior_signals()
    assert [s.kind for s in sigs] == ["fact_forgotten"]


def test_behaviors_cli_reject_emits_dismissed_signal(tmp_path: Path) -> None:
    db = tmp_path / "learning.db"
    s = _learning(tmp_path)
    s.propose_behavior_pattern("p1", "Checks weather often", "high", [])
    r = runner.invoke(behaviors_app, ["reject", "p1", "--db-path", str(db)])
    assert r.exit_code == 0
    sigs = s.list_user_behavior_signals(kind="pattern_dismissed")
    assert len(sigs) == 1 and sigs[0].subject == "Checks weather often"


def test_chat_memory_tool_correct_emits_signal(tmp_path: Path) -> None:
    """The conversational correction path (memory_correct tool) records a signal."""
    from iris_harness.runtime.react_tools import builtin_react_tools

    _seed_fact(tmp_path / "memory.db")
    ms = MemoryStore(db_path=tmp_path / "memory.db")
    ls = _learning(tmp_path)
    specs = {
        s.name: s
        for s in builtin_react_tools(
            semantic_index=None,
            wiki=None,
            repo_root=None,
            memory_store=ms,
            learning_store=ls,
        )
    }
    specs["memory_correct"].call({"key": "location", "value": "New York"})
    specs["memory_forget"].call({"key": "location"})
    kinds = [s.kind for s in ls.list_user_behavior_signals()]
    assert "fact_corrected" in kinds and "fact_forgotten" in kinds
