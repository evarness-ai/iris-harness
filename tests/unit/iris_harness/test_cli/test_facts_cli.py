"""Smoke tests for the `iris facts` CLI (list / show / audit / correct / forget / restore)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from iris_harness.cli.facts import facts_app
from iris_harness.memory.store import MemoryStore, UserFact

runner = CliRunner()


def _seed(db: Path) -> None:
    store = MemoryStore(db_path=db)
    store.ensure_schema()
    now = datetime.now(UTC)
    for key, value, conf in [
        ("hobby", "noise", 0.3),
        ("interest", "maybe", 0.5),
        ("name", "robin", 0.9),
    ]:
        store.upsert_user_fact(
            UserFact(
                key=key,
                value=value,
                confidence=conf,
                source="test",
                first_seen=now,
                last_confirmed=now,
                times_confirmed=1,
            )
        )


def test_list_empty(tmp_path: Path) -> None:
    db = str(tmp_path / "memory.db")
    r = runner.invoke(facts_app, ["list", "--db-path", db])
    assert r.exit_code == 0, r.stdout
    assert "no facts" in r.stdout.lower()


def test_list_shows_all_facts(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed(db)
    r = runner.invoke(facts_app, ["list", "--db-path", str(db)])
    assert r.exit_code == 0, r.stdout
    for key in ("hobby", "interest", "name"):
        assert key in r.stdout


def test_show_existing_and_missing(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed(db)
    ok = runner.invoke(facts_app, ["show", "name", "--db-path", str(db)])
    assert ok.exit_code == 0
    assert "robin" in ok.stdout

    missing = runner.invoke(facts_app, ["show", "nope", "--db-path", str(db)])
    assert missing.exit_code == 1


def test_audit_buckets_match_filter_thresholds(tmp_path: Path) -> None:
    """audit must classify by the live retriever defaults: <0.35 DROP, <0.6 UNCERTAIN."""
    db = tmp_path / "memory.db"
    _seed(db)
    r = runner.invoke(facts_app, ["audit", "--db-path", str(db)])
    assert r.exit_code == 0, r.stdout
    out = r.stdout
    assert "DROP 1" in out  # the 0.3 fact
    assert "UNCERTAIN 1" in out  # the 0.5 fact
    assert "CLEAN 1" in out  # the 0.9 fact
    # DROP/UNCERTAIN keys are listed; CLEAN hidden unless --show-clean.
    assert "hobby" in out and "interest" in out
    assert "name" not in out


def test_audit_show_clean_lists_high(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed(db)
    r = runner.invoke(facts_app, ["audit", "--db-path", str(db), "--show-clean"])
    assert r.exit_code == 0
    assert "name" in r.stdout


def test_correct_then_history_then_restore(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed(db)  # 'high' = robin @ 0.9
    c = runner.invoke(facts_app, ["correct", "name", "newvalue", "--db-path", str(db)])
    assert c.exit_code == 0, c.stdout
    assert MemoryStore(db_path=db).fetch_user_fact("name").value == "newvalue"

    h = runner.invoke(facts_app, ["history", "name", "--db-path", str(db)])
    assert h.exit_code == 0
    assert "correct" in h.stdout

    r = runner.invoke(facts_app, ["restore", "name", "--db-path", str(db)])
    assert r.exit_code == 0, r.stdout
    assert MemoryStore(db_path=db).fetch_user_fact("name").value == "robin"


def test_forget_with_yes_and_restore(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed(db)
    f = runner.invoke(facts_app, ["forget", "interest", "--db-path", str(db), "--yes"])
    assert f.exit_code == 0, f.stdout
    assert MemoryStore(db_path=db).fetch_user_fact("interest") is None

    r = runner.invoke(facts_app, ["restore", "interest", "--db-path", str(db)])
    assert r.exit_code == 0
    assert MemoryStore(db_path=db).fetch_user_fact("interest").value == "maybe"


def test_forget_missing_exits_nonzero(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed(db)
    f = runner.invoke(facts_app, ["forget", "nope", "--db-path", str(db), "--yes"])
    assert f.exit_code == 1


def test_forget_cancelled_keeps_fact(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed(db)
    # Decline the confirmation prompt.
    f = runner.invoke(facts_app, ["forget", "name", "--db-path", str(db)], input="n\n")
    assert f.exit_code == 0
    assert MemoryStore(db_path=db).fetch_user_fact("name") is not None


def _backdate(db: Path, key: str, days_ago: int) -> None:
    """A fact captured and forgotten ``days_ago`` (history is the statement chain now)."""
    from datetime import UTC, datetime, timedelta

    from iris_harness.foundation.persistence import sqlite_conn

    store = MemoryStore(db_path=db)
    now = datetime.now(UTC)
    store.upsert_user_fact(UserFact(key, "v", 0.9, "test", now, now, 1, True))
    store.delete_user_fact(key)
    ts = (now - timedelta(days=days_ago)).isoformat()
    with sqlite_conn(db) as conn:
        conn.execute(
            "UPDATE memris_statements SET recorded_at = ?, retracted_at = ? "
            "WHERE reason = 'forgot' AND recorded_at > ?",
            (ts, ts, (now - timedelta(seconds=60)).isoformat()),
        )


def test_retention_lists_candidates(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _backdate(db, "city", days_ago=200)
    _backdate(db, "hobby", days_ago=5)
    r = runner.invoke(facts_app, ["retention", "--db-path", str(db)])
    assert r.exit_code == 0, r.stdout
    assert "city" in r.stdout
    assert "hobby" not in r.stdout


def test_prune_requires_selector(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _backdate(db, "city", days_ago=200)
    r = runner.invoke(facts_app, ["prune", "--db-path", str(db)])
    assert r.exit_code == 2  # must pass --id or --older-than


def test_prune_older_than_bulk(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _backdate(db, "city", days_ago=200)
    _backdate(db, "email", days_ago=300)
    _backdate(db, "hobby", days_ago=5)
    r = runner.invoke(facts_app, ["prune", "--db-path", str(db), "--older-than", "180", "--yes"])
    assert r.exit_code == 0, r.stdout
    assert "pruned 2" in r.stdout  # two forgotten facts, each with its two events
    assert MemoryStore(db_path=db).count_fact_history() == 2  # only the recent one remains


def _one(key: str, value: str, conf: float) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=conf,
        source="test",
        first_seen=now,
        last_confirmed=now,
        times_confirmed=1,
    )


def test_contradictions_lists_and_ack_clears(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    store = MemoryStore(db_path=db)
    store.ensure_schema()
    store.upsert_user_fact(_one("city", "Berlin", 0.9))
    store.upsert_user_fact(_one("city", "New York", 0.9))  # superseding conflict
    cid = store.fetch_contradictions()[0].id

    r = runner.invoke(facts_app, ["contradictions", "--db-path", str(db)])
    assert r.exit_code == 0, r.stdout
    assert "city" in r.stdout and "superseded" in r.stdout

    a = runner.invoke(facts_app, ["ack", str(cid), "--db-path", str(db)])
    assert a.exit_code == 0
    assert "acknowledged 1" in a.stdout

    after = runner.invoke(facts_app, ["contradictions", "--db-path", str(db)])
    assert "no contradictions" in after.stdout.lower()
