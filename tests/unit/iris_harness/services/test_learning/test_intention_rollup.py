"""Tests for the intention-rollup engine + intentions store."""

from __future__ import annotations

from pathlib import Path

from iris_harness.services.learning.intention_rollup import (
    Intention,
    intention_id_for,
    parse_intentions,
    rollup_intentions,
)
from iris_harness.services.learning.store import LearningMetricsStore

_GOOD = (
    '[{"intention":"Establish a consistent morning routine",'
    '"summary":"groups the morning brief + reminder habits",'
    '"supporting":["IRIS Morning Briefing routine","reminder habits"]},'
    '{"intention":"Stay on top of personal finances","summary":"bills + finance queries",'
    '"supporting":["bill reminders"]}]'
)


def test_parse_valid() -> None:
    out = parse_intentions(_GOOD)
    assert [i.title for i in out] == [
        "Establish a consistent morning routine",
        "Stay on top of personal finances",
    ]
    assert out[0].supporting[0].startswith("IRIS")


def test_parse_strips_fence_and_garbage() -> None:
    assert parse_intentions('```json\n[{"goal":"Learn Rust"}]\n```')[0].title == "Learn Rust"
    assert parse_intentions("not json") == []
    assert parse_intentions('{"intention":"x"}') == []  # object, not array


def test_id_stable_normalized() -> None:
    assert intention_id_for("Learn Rust") == intention_id_for("  learn   RUST ")


def test_rollup_requires_context() -> None:
    called = {"n": 0}

    def invoke(_s: str, _u: str) -> str:
        called["n"] += 1
        return _GOOD

    assert rollup_intentions("too short", invoke=invoke) == []
    assert called["n"] == 0


def test_rollup_returns_intentions() -> None:
    ctx = "Open tasks: call mom. Routines: Morning Briefing. Habits: reminders." * 3
    out = rollup_intentions(ctx, invoke=lambda _s, _u: _GOOD)
    assert all(isinstance(i, Intention) for i in out) and len(out) == 2


def test_rollup_swallows_failure() -> None:
    def boom(_s: str, _u: str) -> str:
        raise RuntimeError("down")

    assert rollup_intentions("x" * 100, invoke=boom) == []


# --- store ---


def _store(tmp_path: Path) -> LearningMetricsStore:
    s = LearningMetricsStore(db_path=tmp_path / "learning.db")
    s.ensure_schema()
    return s


def test_store_propose_list_resolve(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.propose_intention("i1", "Establish a morning routine", "groups X", ["a", "b"])
    assert s.propose_intention("i1", "Establish a morning routine", "dup", []) is False  # dedup
    [p] = s.list_intentions(status="proposed")
    assert p.title == "Establish a morning routine" and p.supporting == ("a", "b")

    assert s.resolve_intention("i1", "active") is True
    assert s.list_intentions(status="proposed") == []
    assert len(s.list_intentions(status="active")) == 1
    assert s.get_intention("i1").status == "active"


def test_resolve_missing_false(tmp_path: Path) -> None:
    assert _store(tmp_path).resolve_intention("nope", "active") is False


# --- CLI (HITL surface) ---


def test_cli_list_and_show(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from iris_harness.cli.intentions import intentions_app

    db = tmp_path / "learning.db"
    s = _store(tmp_path)
    s.propose_intention("i1", "Establish a morning routine", "groups morning brief", ["brief"])
    r = CliRunner()
    out = r.invoke(intentions_app, ["list", "--db-path", str(db)])
    assert out.exit_code == 0 and "Establish a morning routine" in out.stdout
    shown = r.invoke(intentions_app, ["show", "i1", "--db-path", str(db)])
    assert shown.exit_code == 0 and "brief" in shown.stdout


def test_cli_approve_writes_active_item(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from typer.testing import CliRunner

    from iris_harness.cli.intentions import intentions_app

    db = tmp_path / "learning.db"
    s = _store(tmp_path)
    s.propose_intention("i1", "Stay on top of finances", "bills", ["bill reminders"])

    # cmd_approve imports add_active_item lazily from iris_harness.memory.identity.loader; patch at source.
    from iris_harness.memory.identity import loader

    captured: dict[str, str] = {}
    monkeypatch.setattr(loader, "add_active_item", lambda text: captured.update(text=text))

    out = CliRunner().invoke(intentions_app, ["approve", "i1", "--db-path", str(db)])
    assert out.exit_code == 0, out.stdout
    assert "Stay on top of finances" in captured["text"]
    assert s.get_intention("i1").status == "active"
    assert s.list_intentions(status="proposed") == []
    assert s.list_user_behavior_signals(kind="intention_approved")[0].subject == (
        "Stay on top of finances"
    )


def test_cli_dismiss(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from iris_harness.cli.intentions import intentions_app

    db = tmp_path / "learning.db"
    s = _store(tmp_path)
    s.propose_intention("i1", "Learn Rust", "", [])
    out = CliRunner().invoke(intentions_app, ["dismiss", "i1", "--db-path", str(db)])
    assert out.exit_code == 0
    assert s.get_intention("i1").status == "dismissed"
    assert s.list_user_behavior_signals(kind="intention_dismissed")[0].subject == "Learn Rust"
