"""Identity columns on the audit stores, and how a live database gets them (#134, stage 3).

Every database here is built in the shape the RELEASED version created (the stage-2 schema, no
identity columns, real rows in it), then opened by the new code. The migration must keep every
row as written, be a no-op the second time, survive several interpreters opening the same file
at the same instant (a real race, with real subprocesses, not a monkeypatch), and never rewrite
or backfill an old row (D6). The naive migration -- read ``table_info`` then ``ALTER`` inside a
deferred transaction -- is shown failing the same race, which is why the helper takes the write
lock first.
"""

# S603/S608: the subprocesses run this repo's own interpreter on fixed code, and the SQL
# interpolates table names from this module's own constant table.
# ruff: noqa: S603, S608
from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

import iris_harness
from iris_harness.foundation.persistence.sqlite import add_columns_if_missing
from iris_harness.kernel.governance.approvals.store import ApprovalStore
from iris_harness.kernel.governance.audit.log import (
    IDENTITY_COLUMNS as AUDIT_COLUMNS,
)
from iris_harness.kernel.governance.audit.log import AuditLog
from iris_harness.kernel.governance.side_effects.store import (
    IDENTITY_COLUMNS as LEDGER_COLUMNS,
)
from iris_harness.kernel.governance.side_effects.store import SideEffectLedger
from iris_harness.kernel.governor.audit import (
    IDENTITY_COLUMNS as GOVERNOR_COLUMNS,
)
from iris_harness.kernel.governor.audit import GovernorAuditLogger
from iris_harness.runtime.router_audit import (
    IDENTITY_COLUMNS as ROUTER_COLUMNS,
)
from iris_harness.runtime.router_audit import RouterAuditLogger

SRC = str(Path(iris_harness.__file__).resolve().parents[1])

# ---------------------------------------------------------------- the released schemas
_AUDIT_RELEASE = """
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, run_id TEXT NOT NULL,
    step_id INTEGER, agent_type TEXT NOT NULL, hook_point TEXT NOT NULL, plugin TEXT NOT NULL,
    decision TEXT NOT NULL, classification TEXT, tier TEXT, cost_usd REAL,
    severity TEXT NOT NULL, reason TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE INDEX idx_audit_run_ts ON audit_log(run_id, ts);
CREATE INDEX idx_audit_decision_ts ON audit_log(decision, ts);
INSERT INTO audit_log(ts, run_id, step_id, agent_type, hook_point, plugin, decision,
    severity, reason, payload_json)
VALUES ('2026-10-01T00:00:00+00:00', 'old-run', 1, 'a', 'pre_tool_use', 'p', 'allow', 'info',
    'old row', '{"session_id": "old-session", "call_id": "OLD-CALL"}'),
    ('2026-10-01T00:00:01+00:00', 'old-run', 2, 'a', 'post_tool_use', 'p', 'allow', 'info',
    'old row 2', '{}');
"""
_LEDGER_RELEASE = """
CREATE TABLE side_effect_ledger (
    side_effect_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, step_id INTEGER NOT NULL,
    tool TEXT NOT NULL, verification_probe TEXT NOT NULL,
    probe_metadata TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'pending',
    completed_at TEXT, error TEXT
);
CREATE INDEX idx_side_effect_run_status ON side_effect_ledger (run_id, status);
INSERT INTO side_effect_ledger(side_effect_id, run_id, step_id, tool, verification_probe)
VALUES ('old-run:1:OLD', 'old-run', 1, 'shred', '');
"""
_APPROVALS_RELEASE = """
CREATE TABLE approval_queue (
    approval_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, checkpoint_id TEXT, signal TEXT NOT NULL,
    context_summary TEXT NOT NULL, requested_at TEXT NOT NULL, channel TEXT NOT NULL DEFAULT 'cli',
    status TEXT NOT NULL DEFAULT 'pending', responded_at TEXT, response_actor TEXT,
    timeout_at TEXT NOT NULL, policy_on_timeout TEXT NOT NULL DEFAULT 'fail_closed',
    session_id TEXT, items_json TEXT, card_json TEXT, caller TEXT, executed_at TEXT, call_id TEXT
);
CREATE INDEX idx_approval_pending ON approval_queue(status, timeout_at);
INSERT INTO approval_queue(approval_id, run_id, signal, context_summary, requested_at, timeout_at,
    call_id)
VALUES ('old-approval', 'old-run', 's', 'c', '2026-10-01T00:00:00+00:00',
    '2026-10-01T00:10:00+00:00', 'OLD-CALL');
"""
_APPEND_ONLY = """
CREATE TRIGGER {t}_no_update BEFORE UPDATE ON {t}
BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
CREATE TRIGGER {t}_no_delete BEFORE DELETE ON {t}
BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
"""
_ROUTER_RELEASE = """
CREATE TABLE router_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, message_hash TEXT NOT NULL,
    message_length INTEGER NOT NULL, intent TEXT NOT NULL, agent_type TEXT NOT NULL,
    confidence REAL NOT NULL, source TEXT NOT NULL, is_multi_step INTEGER NOT NULL,
    router_model TEXT, channel TEXT, created_at TEXT NOT NULL
);
INSERT INTO router_decisions(session_id, message_hash, message_length, intent, agent_type,
    confidence, source, is_multi_step, created_at)
VALUES ('old-session', 'h', 3, 'general', 'system', 0.9, 'rule', 0, '2026-10-01T00:00:00+00:00');
""" + _APPEND_ONLY.format(t="router_decisions")
_GOVERNOR_RELEASE = """
CREATE TABLE governor_guard_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, route TEXT NOT NULL, action TEXT NOT NULL,
    allowed INTEGER NOT NULL, reason TEXT NOT NULL, matched_policy TEXT,
    requires_approval INTEGER NOT NULL, retry_after_seconds INTEGER,
    metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
);
INSERT INTO governor_guard_audit(route, action, allowed, reason, requires_approval,
    metadata_json, created_at)
VALUES ('coding/mcp', 'call_tool', 1, 'ok', 0, '{}', '2026-10-01T00:00:00+00:00');
""" + _APPEND_ONLY.format(t="governor_guard_audit")

# store -> (release DDL, table, identity columns, python that opens the store in a child)
STORES: dict[str, tuple[str, str, dict[str, str], str]] = {
    "audit_log": (
        _AUDIT_RELEASE,
        "audit_log",
        AUDIT_COLUMNS,
        "from iris_harness.kernel.governance.audit.log import AuditLog; AuditLog(db_path=P)",
    ),
    "side_effect_ledger": (
        _LEDGER_RELEASE,
        "side_effect_ledger",
        LEDGER_COLUMNS,
        "from iris_harness.kernel.governance.side_effects.store import SideEffectLedger; "
        "SideEffectLedger(P)",
    ),
    "approval_queue": (
        _APPROVALS_RELEASE,
        "approval_queue",
        {"step_id": "INTEGER", "turn_id": "TEXT"},
        "from iris_harness.kernel.governance.approvals.store import ApprovalStore; "
        "ApprovalStore(db_path=P)",
    ),
    "router_decisions": (
        _ROUTER_RELEASE,
        "router_decisions",
        ROUTER_COLUMNS,
        "from iris_harness.runtime.router_audit import RouterAuditLogger; "
        "assert RouterAuditLogger(P)._has_identity",
    ),
    "governor_guard_audit": (
        _GOVERNOR_RELEASE,
        "governor_guard_audit",
        GOVERNOR_COLUMNS,
        "from iris_harness.kernel.governor.audit import GovernorAuditLogger; "
        "g = GovernorAuditLogger(P); g._ensure_initialized(); assert g._has_identity",
    ),
}


def _release_db(path: Path, store: str) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(STORES[store][0])
    conn.commit()
    conn.close()
    return path


def _columns(path: Path, table: str) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def _open(store: str, path: Path) -> Any:
    return {
        "audit_log": lambda: AuditLog(db_path=path),
        "side_effect_ledger": lambda: SideEffectLedger(path),
        "approval_queue": lambda: ApprovalStore(db_path=path),
        "router_decisions": lambda: RouterAuditLogger(path),
        "governor_guard_audit": lambda: _governor(path),
    }[store]()


def _governor(path: Path) -> GovernorAuditLogger:
    logger = GovernorAuditLogger(path)
    logger._ensure_initialized()
    return logger


def _rows(path: Path, table: str, columns: list[str]) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"SELECT {', '.join(columns)} FROM {table} ORDER BY rowid").fetchall()
    finally:
        conn.close()


# ----------------------------------------------------------------------- upgrade and twice
@pytest.mark.parametrize("store", list(STORES))
def test_a_release_db_gains_the_columns_and_keeps_every_row_as_written(
    tmp_path: Path, store: str
) -> None:
    _, table, identity, _ = STORES[store]
    db = _release_db(tmp_path / "db.sqlite", store)
    old_columns = _columns(db, table)
    before = _rows(db, table, old_columns)
    assert before and not set(identity) & set(old_columns)  # a genuine release shape

    _open(store, db)

    new_columns = _columns(db, table)
    assert set(identity) <= set(new_columns)
    assert _rows(db, table, old_columns) == before  # every old value, untouched
    assert all(  # and no old row got an identity: never backfilled
        all(value is None for value in row) for row in _rows(db, table, list(identity))
    )


@pytest.mark.parametrize("store", list(STORES))
def test_the_migration_runs_twice_and_changes_nothing_the_second_time(
    tmp_path: Path, store: str
) -> None:
    _, table, identity, _ = STORES[store]
    db = _release_db(tmp_path / "db.sqlite", store)

    _open(store, db)
    first = (_columns(db, table), _rows(db, table, ["rowid"]))
    _open(store, db)
    assert (_columns(db, table), _rows(db, table, ["rowid"])) == first
    assert add_columns_if_missing(db, table, identity) == []  # nothing left to add


def test_a_missing_table_is_not_created_by_the_helper(tmp_path: Path) -> None:
    db = tmp_path / "empty.sqlite"
    sqlite3.connect(db).close()
    assert add_columns_if_missing(db, "nope", {"a": "TEXT"}) == []
    assert _columns(db, "nope") == []


def test_the_helper_refuses_a_name_that_is_not_a_plain_identifier(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        add_columns_if_missing(tmp_path / "x.sqlite", "t; DROP TABLE t", {"a": "TEXT"})
    with pytest.raises(ValueError):
        add_columns_if_missing(tmp_path / "x.sqlite", "t", {"a b": "TEXT"})


# ------------------------------------------------------------ the boundary (audit_meta)
def test_the_boundary_is_written_once_by_the_process_that_added_the_columns(
    tmp_path: Path,
) -> None:
    db = _release_db(tmp_path / "a.db", "audit_log")

    AuditLog(db_path=db)
    AuditLog(db_path=db)  # a second open adds nothing and writes nothing

    conn = sqlite3.connect(db)
    meta = conn.execute("SELECT key, value FROM audit_meta").fetchall()
    conn.close()
    assert [k for k, _ in meta] == ["identity"]
    boundary = json.loads(meta[0][1])
    assert boundary["schema"] == 2 and boundary["first_identity_row_id"] == 3  # 2 old rows
    # The old rows say "pre-identity era" by having no record_id, and still know their session.
    old, _ = AuditLog(db_path=db).query()
    assert old.record_id is None and old.session_id == "old-session"
    assert old.call_id is None  # not backfilled from the payload either


def test_a_fresh_database_has_no_pre_identity_era_and_writes_no_boundary(tmp_path: Path) -> None:
    db = tmp_path / "fresh.db"
    log = AuditLog(db_path=db)
    row_id = log.record(
        run_id="r",
        step_id=None,
        agent_type="a",
        hook_point="h",
        plugin="p",
        decision="allow",
        severity="info",
        reason="x",
    )
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM audit_meta").fetchone() == (0,)
    conn.close()
    assert log.query()[0].id == row_id and log.query()[0].record_id is not None


def test_record_id_is_minted_by_the_store_unique_and_never_taken_from_a_caller(
    tmp_path: Path,
) -> None:
    log = AuditLog(db_path=tmp_path / "a.db")
    ids = []
    for _ in range(3):
        log.record(
            run_id="r",
            step_id=None,
            agent_type="a",
            hook_point="h",
            plugin="p",
            decision="allow",
            severity="info",
            reason="x",
            payload={"record_id": "FORGED"},
        )
    ids = [row.record_id for row in log.query()]
    assert len(set(ids)) == 3 and "FORGED" not in ids
    conn = sqlite3.connect(log.db_path)
    with pytest.raises(sqlite3.IntegrityError):  # the unique partial index
        conn.execute(
            "INSERT INTO audit_log(ts, run_id, agent_type, hook_point, plugin, decision, "
            "severity, reason, payload_json, record_id) VALUES "
            f"('t', 'r', 'a', 'h', 'p', 'allow', 'info', 'x', '{{}}', '{ids[0]}')"
        )
    # Rows with no record_id (the pre-identity era) are not constrained.
    for _ in range(2):
        conn.execute(
            "INSERT INTO audit_log(ts, run_id, agent_type, hook_point, plugin, decision, "
            "severity, reason, payload_json) VALUES "
            "('t', 'r', 'a', 'h', 'p', 'allow', 'info', 'x', '{}')"
        )
    conn.close()


def test_the_identity_columns_hold_identifiers_only(tmp_path: Path) -> None:
    log = AuditLog(db_path=tmp_path / "a.db")
    log.record(
        run_id="r",
        step_id=None,
        agent_type="a",
        hook_point="h",
        plugin="p",
        decision="allow",
        severity="info",
        reason="x",
        payload={
            "call_id": "C1",
            "attempt": True,  # a bool is not a count
            "turn_id": 7,  # a number is not an id
            "parent_call_id": "",  # an empty string is no id
            "replay_of": ["x"],
            "session_id": "S1",
            "args": {"path": "SECRET-ARGUMENT-TEXT"},
        },
    )
    (row,) = log.query()
    assert (row.call_id, row.session_id) == ("C1", "S1")
    assert (row.attempt, row.turn_id, row.parent_call_id, row.replay_of) == (None,) * 4
    conn = sqlite3.connect(log.db_path)
    identity = conn.execute(f"SELECT {', '.join(AUDIT_COLUMNS)} FROM audit_log").fetchone()
    conn.close()
    assert "SECRET-ARGUMENT-TEXT" not in json.dumps(identity)


def test_record_many_keeps_the_good_rows_and_reports_the_bad_one_alone(tmp_path: Path) -> None:
    log = AuditLog(db_path=tmp_path / "a.db")
    base: dict[str, Any] = {
        "step_id": None,
        "agent_type": "a",
        "hook_point": "h",
        "plugin": "p",
        "decision": "allow",
        "severity": "info",
        "reason": "x",
    }

    ids = log.record_many(
        [
            {**base, "run_id": "r1"},
            {**base, "run_id": ["not", "a", "string"]},  # SQLite cannot bind it
            {**base, "run_id": "r3"},
        ]
    )

    assert ids[0] is not None and ids[1] is None and ids[2] is not None
    assert [row.run_id for row in log.query()] == ["r1", "r3"]


# ------------------------------------------------------ the real multi-interpreter race
_CHILD = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    P = Path(sys.argv[1]); start = float(sys.argv[2])
    {open_store}
    while time.time() < start:
        pass
    {open_store_again}
    print("ok")
""")


def _race(store: str, tmp_path: Path, n: int = 6) -> list[str]:
    """``n`` interpreters open the store on one release-shaped file at the same instant."""
    db = _release_db(tmp_path / "race.sqlite", store)
    opener = STORES[store][3]
    # Import inside the child BEFORE the barrier (an import takes seconds), open AFTER it.
    imports, call = opener.split("; ", 1)
    code = _CHILD.format(open_store=imports, open_store_again=call)
    start = time.time() + 6.0
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(db), str(start)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PYTHONPATH": SRC, "IRIS_AUTH_SECRET": "x", "PATH": "/usr/bin:/bin"},
        )
        for _ in range(n)
    ]
    out = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=120)
        out.append(stdout.strip() if proc.returncode == 0 else f"FAILED: {stderr.strip()[-300:]}")
    return out


@pytest.mark.parametrize("store", list(STORES))
def test_several_interpreters_opening_one_release_db_at_once_all_succeed(
    tmp_path: Path, store: str
) -> None:
    _, table, identity, _ = STORES[store]
    outcomes = _race(store, tmp_path)
    assert outcomes == ["ok"] * len(outcomes), outcomes
    columns = _columns(tmp_path / "race.sqlite", table)
    assert set(identity) <= set(columns)
    assert len(columns) == len(set(columns))  # each added exactly once
    if store == "audit_log":  # exactly one process wrote the boundary
        conn = sqlite3.connect(tmp_path / "race.sqlite")
        assert conn.execute("SELECT COUNT(*) FROM audit_meta").fetchone() == (1,)
        conn.close()


_NAIVE_CHILD = textwrap.dedent("""
    import sqlite3, sys, time
    db, start = sys.argv[1], float(sys.argv[2])
    while time.time() < start:
        pass
    conn = sqlite3.connect(db, isolation_level=None)  # busy timeout: python's default 5s
    try:
        conn.execute("BEGIN")  # deferred, as AuditLog._connect opens every transaction
        have = {r[1] for r in conn.execute("PRAGMA table_info(audit_log)")}
        for name in %r:
            if name not in have:
                try:
                    conn.execute("ALTER TABLE audit_log ADD COLUMN " + name + " TEXT")
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
        conn.execute("COMMIT")
        print("ok")
    except Exception as exc:
        print("ERR", exc)
""")


def test_the_naive_read_then_alter_in_a_deferred_transaction_fails_the_same_race(
    tmp_path: Path,
) -> None:
    """Why the helper takes the write lock first: a deferred transaction that reads
    ``table_info`` and then alters is refused with "database is locked" the moment another
    process commits in between, and the 5s busy timeout does not help
    (``SQLITE_BUSY_SNAPSHOT``). Measured at 37 of 40 rounds with 8 processes; the test allows
    a few rounds so a lucky one cannot make it flaky."""
    failures = 0
    for round_no in range(8):
        db = _release_db(tmp_path / f"naive-{round_no}.sqlite", "audit_log")
        code = _NAIVE_CHILD % (list(AUDIT_COLUMNS),)
        start = time.time() + 1.0
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", code, str(db), str(start)],
                stdout=subprocess.PIPE,
                text=True,
            )
            for _ in range(8)
        ]
        failures += sum("database is locked" in p.communicate()[0] for p in procs)
        if failures:
            break
    assert failures > 0


# ------------------------------------------------- an old release and a new one on one file
def test_an_old_release_writer_and_a_new_one_share_one_file(tmp_path: Path) -> None:
    """A rolling upgrade: the released code writes with its own column list (no identity
    columns) while the new code writes with them. Neither breaks the other."""
    db = _release_db(tmp_path / "a.db", "audit_log")
    new = AuditLog(db_path=db)  # migrates
    old_writer = textwrap.dedent("""
        import sqlite3, sys
        conn = sqlite3.connect(sys.argv[1])
        for n in range(50):
            conn.execute(
                "INSERT INTO audit_log(ts, run_id, step_id, agent_type, hook_point, plugin, "
                "decision, classification, tier, cost_usd, severity, reason, payload_json) "
                "VALUES ('t', 'old-writer', ?, 'a', 'h', 'p', 'allow', NULL, NULL, NULL, "
                "'info', 'x', '{}')", (n,))
            conn.commit()
        print("ok")
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", old_writer, str(db)], stdout=subprocess.PIPE, text=True
    )
    for n in range(50):
        new.record(
            run_id="new-writer",
            step_id=n,
            agent_type="a",
            hook_point="h",
            plugin="p",
            decision="allow",
            severity="info",
            reason="x",
            payload={"call_id": f"C{n}"},
        )
    assert proc.communicate(timeout=60)[0].strip() == "ok"
    rows = new.query()
    old = [r for r in rows if r.run_id == "old-writer"]
    mine = [r for r in rows if r.run_id == "new-writer"]
    assert len(old) == 50 and len(mine) == 50
    assert all(r.record_id is None for r in old)  # pre-identity by construction
    assert all(r.record_id and r.call_id for r in mine)


# -------------------------------------------------- ledger: key kept, events, resume lookups
def test_the_ledger_key_is_kept_and_run_lookups_still_work_on_a_migrated_db(
    tmp_path: Path,
) -> None:
    db = _release_db(tmp_path / "l.db", "side_effect_ledger")
    ledger = SideEffectLedger(db)
    key = ledger.record(
        side_effect_id="run-1:0:CALL1",
        run_id="run-1",
        step_id=0,
        tool="shred",
        verification_probe="",
        exclusive=True,
        call_id="CALL1",
        parent_call_id="PARENT",
        attempt=2,
        replay_of="HELD",
    )

    assert key == "run-1:0:CALL1"  # the key format is unchanged
    assert [r.side_effect_id for r in ledger.pending("run-1")] == [key]
    assert [r.side_effect_id for r in ledger.list_by_run("run-1")] == [key]
    assert [r.side_effect_id for r in ledger.pending("old-run")] == ["old-run:1:OLD"]  # old row
    assert ledger.finalize(key, status="completed")
    assert ledger.pending("run-1") == []  # settled: what `iris run resume` reads
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT call_id, parent_call_id, attempt, replay_of, record_id FROM side_effect_ledger "
        "WHERE side_effect_id = ?",
        (key,),
    ).fetchone()
    events = conn.execute(
        "SELECT status, call_id FROM side_effect_events WHERE side_effect_id = ? ORDER BY ts, "
        "event_id",
        (key,),
    ).fetchall()
    conn.close()
    assert row[:4] == ("CALL1", "PARENT", 2, "HELD") and row[4]
    assert events == [("pending", "CALL1"), ("completed", "CALL1")]  # the history is kept


def test_recording_a_taken_key_again_writes_no_second_event(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "l.db")
    for _ in range(2):  # INSERT OR IGNORE: the second is a no-op, and so is its event
        ledger.record(
            side_effect_id="k", run_id="r", step_id=0, tool="t", verification_probe="",
            call_id="C",
        )  # fmt: skip
    conn = sqlite3.connect(ledger.db_path)
    assert conn.execute("SELECT COUNT(*) FROM side_effect_events").fetchone() == (1,)
    conn.close()


# --------------------------------------------- append-only triggers, and a failed migration
@pytest.mark.parametrize("store", ["router_decisions", "governor_guard_audit"])
def test_the_append_only_triggers_still_fire_after_the_migration(
    tmp_path: Path, store: str
) -> None:
    _, table, _, _ = STORES[store]
    db = _release_db(tmp_path / "t.db", store)
    _open(store, db)
    conn = sqlite3.connect(db)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(f"UPDATE {table} SET record_id = 'x'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(f"DELETE FROM {table}")
    conn.close()


def _decision() -> Any:
    from iris_harness.kernel.governor.models import GovernorGuardDecision

    return GovernorGuardDecision(
        route="coding/mcp",
        action="call_tool",
        allowed=True,
        reason="ok",
        requires_approval=False,
        metadata={},
    )


def test_a_router_migration_that_fails_is_logged_as_an_error_and_decisions_still_land(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from iris_harness.agent.intent_router import IntentResult
    from iris_harness.runtime import router_audit

    def refuse(*_a: Any, **_k: Any) -> list[str]:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(router_audit, "add_columns_if_missing", refuse)
    db = _release_db(tmp_path / "r.db", "router_decisions")
    with caplog.at_level(logging.ERROR):
        logger = RouterAuditLogger(db)

    assert any(r.levelno == logging.ERROR for r in caplog.records)  # not a quiet warning
    logger.record(
        session_id="s",
        message="hello",
        result=IntentResult(intent="general", agent_type="system", confidence=0.9, source="rule"),
    )
    assert _rows(db, "router_decisions", ["session_id"])[-1] == ("s",)  # the old shape works
    assert "record_id" not in _columns(db, "router_decisions")


def test_a_governor_migration_that_fails_is_logged_as_an_error_and_decisions_still_land(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from iris_harness.kernel.governor import audit as governor_audit

    def refuse(*_a: Any, **_k: Any) -> list[str]:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(governor_audit, "add_columns_if_missing", refuse)
    db = _release_db(tmp_path / "g.db", "governor_guard_audit")
    logger = GovernorAuditLogger(db)
    with caplog.at_level(logging.ERROR):
        logger.record_decision(_decision(), identity={"call_id": "C"})

    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert _rows(db, "governor_guard_audit", ["route"])[-1] == ("coding/mcp",)
    assert "call_id" not in _columns(db, "governor_guard_audit")


def test_a_governor_decision_names_the_run_call_and_session_it_was_made_for(
    tmp_path: Path,
) -> None:
    db = _release_db(tmp_path / "g.db", "governor_guard_audit")
    logger = GovernorAuditLogger(db)

    logger.record_decision(
        _decision(),
        identity={"run_id": "R", "call_id": "C", "session_id": "S", "args": "SECRET"},  # type: ignore[dict-item]
    )

    row = _rows(db, "governor_guard_audit", ["run_id", "call_id", "session_id", "record_id"])[-1]
    assert row[:3] == ("R", "C", "S") and row[3]
    assert "SECRET" not in json.dumps(
        _rows(db, "governor_guard_audit", list(GOVERNOR_COLUMNS))
    )  # only the three keys are ever read
