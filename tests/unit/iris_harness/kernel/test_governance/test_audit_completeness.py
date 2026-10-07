"""The ledger can show what is missing (issue #134, stage 4a).

Every writer numbers its rows before writing them; a write the database refuses is spooled
(or, for a call that guards an effect, the call is refused when the spool cannot keep it
either); the next successful write records the gap; ``writer.start`` / ``writer.close`` bound
the writer's life. The failure is injected where it really happens -- the database connect --
and the multi-process cases use real interpreters.
"""

# S603: the subprocesses run this repo's own interpreter on fixed code.
# ruff: noqa: S603

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import iris_harness
from iris_harness.kernel.governance import GovernanceKernel
from iris_harness.kernel.governance.audit import AuditLog, sequence, spool
from iris_harness.kernel.governance.audit.write_health import write_health
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.services.health.governance import audit_writes_provider
from iris_harness.services.health.models import HealthState

SRC = str(Path(iris_harness.__file__).resolve().parents[1])


def _row(n: int = 0, **extra: Any) -> dict[str, Any]:
    return dict(
        run_id=f"run-{n}",
        step_id=None,
        agent_type="chat",
        hook_point="pre_tool_use",
        plugin="p",
        decision="allow",
        severity="info",
        reason=f"row {n}",
        payload={"n": n},
        **extra,
    )


class _DbDown:
    """Make every database connect of an AuditLog fail, the way a locked or full disk does."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.on = False
        original = AuditLog._connect

        def connect(log: AuditLog) -> Any:
            if self.on:
                raise sqlite3.OperationalError("database is locked")
            return original(log)

        monkeypatch.setattr(AuditLog, "_connect", connect)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "audit.db"


@pytest.fixture()
def down(monkeypatch: pytest.MonkeyPatch) -> _DbDown:
    return _DbDown(monkeypatch)


def _all(db: Path, where: str = "1=1") -> list[sqlite3.Row]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        sql = f"SELECT * FROM audit_log WHERE {where} ORDER BY id"  # noqa: S608
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


# ----------------------------------------------------------------- numbering and the writer
def test_every_row_carries_its_writer_and_a_contiguous_sequence(db: Path) -> None:
    log = AuditLog(db_path=db)
    for n in range(3):
        log.record(**_row(n))
    rows = _all(db, "kind IS NULL")
    assert [r["writer_seq"] for r in rows] == [1, 2, 3]
    assert len({r["writer_id"] for r in rows}) == 1


def test_writer_start_is_written_once_with_the_first_row(db: Path) -> None:
    log = AuditLog(db_path=db)
    log.record(**_row(1))
    log.record(**_row(2))
    starts = _all(db, "hook_point = 'writer.start'")
    assert len(starts) == 1 and starts[0]["writer_seq"] == 0 and starts[0]["kind"] == "writer"
    # A second AuditLog on the same file is the same writer: no second start, no restart at 1.
    AuditLog(db_path=db).record(**_row(3))
    assert len(_all(db, "hook_point = 'writer.start'")) == 1
    assert [r["writer_seq"] for r in _all(db, "kind IS NULL")] == [1, 2, 3]


def test_the_stores_own_rows_are_not_decisions(db: Path) -> None:
    log = AuditLog(db_path=db)
    log.record(**_row(1))
    assert log.count() == 1 and len(log.query()) == 1
    assert log.count(include_store_rows=True) == 2  # + writer.start
    assert {r.hook_point for r in log.query(include_store_rows=True)} == {
        "pre_tool_use",
        "writer.start",
    }


def test_a_clean_exit_writes_writer_close_with_the_last_number(db: Path) -> None:
    code = textwrap.dedent("""
        import sys
        from pathlib import Path
        from iris_harness.kernel.governance.audit import AuditLog
        log = AuditLog(db_path=Path(sys.argv[1]))
        for n in range(3):
            log.record(run_id="r", step_id=None, agent_type="chat", hook_point="pre_tool_use",
                       plugin="p", decision="allow", severity="info", reason="x")
    """)
    _run(code, db)
    [close] = _all(db, "hook_point = 'writer.close'")
    assert json.loads(close["payload_json"])["last_seq"] == 3
    assert close["kind"] == "writer"


def test_a_killed_process_leaves_a_start_and_no_close(db: Path) -> None:
    code = textwrap.dedent("""
        import os, sys
        from pathlib import Path
        from iris_harness.kernel.governance.audit import AuditLog
        log = AuditLog(db_path=Path(sys.argv[1]))
        log.record(run_id="r", step_id=None, agent_type="chat", hook_point="pre_tool_use",
                   plugin="p", decision="allow", severity="info", reason="pre",
                   payload={"call_id": "01CALL"})
        os._exit(9)  # killed between the call's PRE row and its POST row
    """)
    assert _run(code, db, check=False) == 9
    assert len(_all(db, "hook_point = 'writer.start'")) == 1
    assert _all(db, "hook_point = 'writer.close'") == []
    calls = _all(db, "call_id = '01CALL'")
    assert [r["hook_point"] for r in calls] == ["pre_tool_use"]  # an open call: PRE, no POST


# ------------------------------------------------------------------- failure: spool and gap
def test_a_refused_write_is_spooled_leaves_a_hole_and_the_next_write_records_the_gap(
    db: Path, down: _DbDown
) -> None:
    log = AuditLog(db_path=db)
    log.record(**_row(1))  # seq 1
    down.on = True
    assert log.record(**_row(2)) == 0  # seq 2: the spool took it
    assert log.record(**_row(3)) == 0  # seq 3
    down.on = False
    # The hole is in the data now: 1 present, 2 and 3 missing from the database.
    assert [r["writer_seq"] for r in _all(db, "kind IS NULL")] == [1]
    [line1, line2] = spool.read_valid(spool.spool_path_for(db))[0]
    assert (line1["writer_seq"], line2["writer_seq"]) == (2, 3)
    # The next write that lands records the gap and drains the spool.
    log.record(**_row(4))  # seq 4
    [gap] = _all(db, "kind = 'gap'")
    body = json.loads(gap["payload_json"])
    assert (body["first_missing"], body["last_missing"], body["count"]) == (2, 3, 2)
    assert body["causes"] == {"OperationalError": 2} and body["spooled"] == 2 and body["lost"] == 0
    assert "locked" not in gap["payload_json"]  # the class name only, never the message
    # And the drain put the spooled rows in, exactly once, under their own numbers.
    assert sorted(r["writer_seq"] for r in _all(db, "kind IS NULL")) == [1, 2, 3, 4]
    assert not spool.spool_path_for(db).exists()


def test_the_drain_is_idempotent(db: Path, down: _DbDown) -> None:
    log = AuditLog(db_path=db)
    down.on = True
    log.record(**_row(1))
    down.on = False
    path = spool.spool_path_for(db)
    saved = path.read_text()
    assert log.drain_spool() == 1
    path.write_text(saved)  # the same line comes back (a crash between replay and rewrite)
    log.drain_spool()
    ids = [r["record_id"] for r in _all(db, "kind IS NULL")]
    assert len(ids) == len(set(ids)) == 1


def test_a_row_neither_the_database_nor_the_spool_takes_is_lost_visibly(
    db: Path, down: _DbDown, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = AuditLog(db_path=db)
    log.record(**_row(1))
    down.on = True
    monkeypatch.setattr(spool, "append", _boom)
    with pytest.raises(sqlite3.OperationalError):
        log.record(**_row(2))
    down.on = False
    assert sequence.stats().lost == 1
    log.record(**_row(3))
    [gap] = _all(db, "kind = 'gap'")
    body = json.loads(gap["payload_json"])
    assert (body["first_missing"], body["lost"], body["spooled"]) == (2, 1, 0)
    assert gap["severity"] == "error"
    # The health row is red until the process restarts.
    [check] = audit_writes_provider()
    assert check.state is HealthState.RED and "could not be written" in check.detail


def _boom(*_a: Any, **_k: Any) -> None:
    raise OSError("spool unwritable")


def test_a_spooled_row_makes_health_yellow_until_it_drains(db: Path, down: _DbDown) -> None:
    os.environ["IRIS_GOVERNANCE_AUDIT_DB_PATH"] = str(db)
    try:
        log = AuditLog(db_path=db)
        down.on = True
        log.record(**_row(1))
        down.on = False
        [check] = audit_writes_provider()
        assert check.state is HealthState.YELLOW and "1 audit row(s) wait" in check.detail
        assert write_health(db)["spool_pending"] == 1
        log.drain_spool()
        assert audit_writes_provider() == []
    finally:
        os.environ.pop("IRIS_GOVERNANCE_AUDIT_DB_PATH", None)


def test_the_spool_is_drained_when_the_next_runtime_opens_the_ledger(
    db: Path, down: _DbDown
) -> None:
    log = AuditLog(db_path=db)
    down.on = True
    log.record(**_row(2))
    down.on = False
    assert spool.spool_path_for(db).exists()
    AuditLog(db_path=db)  # a new process / runtime start
    assert not spool.spool_path_for(db).exists()
    assert len(_all(db, "kind IS NULL")) == 1


# ----------------------------------------------------- the spool cannot forge or alter history
def test_a_spool_line_naming_an_existing_row_changes_nothing(db: Path) -> None:
    log = AuditLog(db_path=db)
    log.record(**_row(1))
    [real] = _all(db, "kind IS NULL")
    forged = {
        "record_id": real["record_id"],
        "writer_id": real["writer_id"],
        "writer_seq": 99,
        "kind": None,
        "ts": "2020-01-01T00:00:00+00:00",
        "run_id": "forged",
        "step_id": None,
        "agent_type": "chat",
        "hook_point": "pre_tool_use",
        "plugin": "p",
        "decision": "deny",
        "classification": None,
        "tier": None,
        "cost_usd": None,
        "severity": "critical",
        "reason": "rewritten",
        "payload": {},
    }
    spool.append(spool.spool_path_for(db), forged)
    log.drain_spool()
    [after] = _all(db, "kind IS NULL")
    assert dict(after) == dict(real)
    # The same holds for a line reusing an existing (writer_id, writer_seq).
    spool.append(spool.spool_path_for(db), {**forged, "record_id": "0" * 26, "writer_seq": 1})
    log.drain_spool()
    [after] = _all(db, "kind IS NULL")
    assert dict(after) == dict(real)


def test_malformed_spool_lines_are_set_aside_counted_and_never_crash_the_drain(db: Path) -> None:
    log = AuditLog(db_path=db)
    path = spool.spool_path_for(db)
    good = {
        "record_id": "01HZZZZZZZZZZZZZZZZZZZZZZZ",
        "writer_id": "01HYYYYYYYYYYYYYYYYYYYYYYY",
        "writer_seq": 7,
        "kind": None,
        "ts": "2026-10-07T00:00:00+00:00",
        "run_id": "r",
        "step_id": None,
        "agent_type": "chat",
        "hook_point": "pre_tool_use",
        "plugin": "p",
        "decision": "allow",
        "classification": None,
        "tier": None,
        "cost_usd": None,
        "severity": "info",
        "reason": "ok",
        "payload": {},
    }
    path.write_text(
        "\n".join(
            [
                "not json at all",
                json.dumps({"record_id": "x"}),  # missing fields
                json.dumps({**good, "extra": 1}),  # a field outside the closed set
                json.dumps({**good, "writer_seq": "7"}),  # wrong type
                json.dumps({**good, "record_id": "short"}),  # not a ULID
                json.dumps([1, 2]),
                json.dumps(good),
            ]
        )
        + "\n"
    )
    assert log.drain_spool() == 1
    assert [r["writer_seq"] for r in _all(db, "kind IS NULL")] == [7]
    rejected = path.with_name(path.name + ".rejected").read_text().splitlines()
    assert len(rejected) == 6
    assert spool.state(path).rejected == 6 and not path.exists()


def test_the_spool_and_the_ledger_share_a_directory_and_a_permission_model(
    db: Path, down: _DbDown
) -> None:
    log = AuditLog(db_path=db)
    down.on = True
    log.record(**_row(1))
    down.on = False
    path = spool.spool_path_for(db)
    assert path.parent == db.parent  # beside the database it belongs to
    assert (path.stat().st_mode & 0o777) == 0o600 == (db.stat().st_mode & 0o777)


# --------------------------------------------------------------------- the effectful tier
class _Allow:
    name = "allow_all"
    priority = 10

    def __init__(self, point: HookPoint) -> None:
        self.hook_point = point

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="ok")


def _kernel(log: AuditLog | None, point: HookPoint) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=log)
    kernel.register(_Allow(point))  # type: ignore[arg-type]
    kernel.init_lock()
    return kernel


def _ctx(point: HookPoint, **kw: Any) -> HookContext:
    return HookContext(hook_point=point, run_id="r", agent_type="chat", **kw)


EFFECTFUL = [
    (HookPoint.PRE_TOOL_USE, {"metadata": {"tool_effect": "write"}}),
    (HookPoint.PRE_TOOL_USE, {"metadata": {"tool_effect": "destructive"}}),
    (HookPoint.PRE_LLM_CALL, {"tier": "tier_3"}),
    (HookPoint.PRE_EGRESS, {}),
]
NOT_EFFECTFUL = [
    (HookPoint.PRE_TOOL_USE, {"metadata": {"tool_effect": "read"}}),
    (HookPoint.PRE_TOOL_USE, {}),
    (HookPoint.PRE_LLM_CALL, {"tier": "tier_1"}),
    (HookPoint.POST_TOOL_USE, {"metadata": {"tool_effect": "write"}}),
]


@pytest.mark.parametrize(("point", "kw"), EFFECTFUL)
async def test_an_effect_is_refused_when_neither_database_nor_spool_kept_its_row(
    db: Path, down: _DbDown, monkeypatch: pytest.MonkeyPatch, point: HookPoint, kw: dict[str, Any]
) -> None:
    kernel = _kernel(AuditLog(db_path=db), point)
    down.on = True
    monkeypatch.setattr(spool, "append", _boom)
    decision, _ = await kernel.fire(point, _ctx(point, **kw))
    assert decision.outcome == "deny" and decision.decided_by == "kernel"


@pytest.mark.parametrize(("point", "kw"), EFFECTFUL)
async def test_an_effect_proceeds_when_only_the_database_failed_and_the_spool_kept_the_row(
    db: Path, down: _DbDown, point: HookPoint, kw: dict[str, Any]
) -> None:
    kernel = _kernel(AuditLog(db_path=db), point)
    down.on = True
    decision, _ = await kernel.fire(point, _ctx(point, **kw))
    assert decision.outcome == "allow"
    [line] = spool.read_valid(spool.spool_path_for(db))[0]
    assert line["hook_point"] == point.value


@pytest.mark.parametrize(("point", "kw"), NOT_EFFECTFUL)
async def test_a_call_that_guards_no_effect_is_never_refused_for_a_lost_row(
    db: Path, down: _DbDown, monkeypatch: pytest.MonkeyPatch, point: HookPoint, kw: dict[str, Any]
) -> None:
    kernel = _kernel(AuditLog(db_path=db), point)
    down.on = True
    monkeypatch.setattr(spool, "append", _boom)
    decision, _ = await kernel.fire(point, _ctx(point, **kw))
    assert decision.outcome == "allow"


async def test_a_kernel_with_no_ledger_behaves_as_before() -> None:
    kernel = _kernel(None, HookPoint.PRE_TOOL_USE)
    decision, _ = await kernel.fire(
        HookPoint.PRE_TOOL_USE,
        _ctx(HookPoint.PRE_TOOL_USE, metadata={"tool_effect": "destructive"}),
    )
    assert decision.outcome == "allow"


async def test_kernel_rows_of_a_tool_call_and_a_capability_call_are_numbered(db: Path) -> None:
    log = AuditLog(db_path=db)
    kernel = _kernel(log, HookPoint.PRE_TOOL_USE)
    await kernel.fire(
        HookPoint.PRE_TOOL_USE, _ctx(HookPoint.PRE_TOOL_USE, metadata={"call_id": "c"})
    )
    await kernel.fire(
        HookPoint.PRE_TOOL_USE,
        _ctx(HookPoint.PRE_TOOL_USE, payload={"capability": "mail", "method": "send"}),
    )
    assert all(r["writer_id"] and r["writer_seq"] for r in _all(db, "kind IS NULL"))


# -------------------------------------------------------- processes: fork, and several writers
def test_a_forked_child_is_a_new_writer_and_does_not_disturb_the_parent(db: Path) -> None:
    log = AuditLog(db_path=db)
    log.record(**_row(1))
    parent_id = sequence.writer_for(db).writer_id
    pid = os.fork()
    if pid == 0:  # the child
        try:
            AuditLog(db_path=db).record(**_row(2))
            os._exit(0)
        except BaseException:  # noqa: BLE001
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    assert status == 0
    log.record(**_row(3))
    rows = {r["run_id"]: r for r in _all(db, "kind IS NULL")}
    assert rows["run-2"]["writer_id"] != parent_id and rows["run-2"]["writer_seq"] == 1
    assert (rows["run-1"]["writer_id"], rows["run-3"]["writer_id"]) == (parent_id, parent_id)
    assert (rows["run-1"]["writer_seq"], rows["run-3"]["writer_seq"]) == (1, 2)


_WRITER = textwrap.dedent("""
    import sys
    from pathlib import Path
    from iris_harness.kernel.governance.audit import AuditLog
    log = AuditLog(db_path=Path(sys.argv[1]))
    tag = sys.argv[2]
    for n in range(int(sys.argv[3])):
        log.record(run_id=f"{tag}-{n}", step_id=None, agent_type="chat",
                   hook_point="pre_tool_use", plugin="p", decision="allow",
                   severity="info", reason="x")
    print("ok")
""")


def test_several_writer_processes_each_keep_a_gapless_sequence(db: Path) -> None:
    AuditLog(db_path=db)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WRITER, str(db), tag, "30"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PYTHONPATH": SRC, "IRIS_AUTH_SECRET": "x", "PATH": "/usr/bin:/bin"},
        )
        for tag in ("a", "b", "c")
    ]
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert out.strip() == "ok", err[-400:]
    rows = _all(db, "kind IS NULL")
    by_writer: dict[str, list[int]] = {}
    for r in rows:
        by_writer.setdefault(r["writer_id"], []).append(r["writer_seq"])
    assert len(by_writer) == 3
    for seqs in by_writer.values():
        assert sorted(seqs) == list(range(1, 31))  # no hole, no duplicate
    assert len({r["record_id"] for r in rows}) == 90
    assert len(_all(db, "hook_point = 'writer.close'")) == 3


# --------------------------------------------------------------------------- the migration
def test_a_database_a_release_created_gets_the_columns_and_one_sequence_boundary(
    db: Path,
) -> None:
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, run_id TEXT NOT NULL,
            step_id INTEGER, agent_type TEXT NOT NULL, hook_point TEXT NOT NULL,
            plugin TEXT NOT NULL, decision TEXT NOT NULL, classification TEXT, tier TEXT,
            cost_usd REAL, severity TEXT NOT NULL, reason TEXT NOT NULL,
            payload_json TEXT NOT NULL);
        INSERT INTO audit_log(ts, run_id, agent_type, hook_point, plugin, decision, severity,
                              reason, payload_json)
        VALUES ('2026-01-01T00:00:00+00:00', 'old', 'chat', 'pre_turn', 'p', 'allow', 'info',
                'r', '{}');
    """)
    conn.commit()
    conn.close()
    AuditLog(db_path=db)
    AuditLog(db_path=db)  # twice: nothing more to add
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(audit_log)")}
    assert {"writer_id", "writer_seq", "kind", "record_id"} <= cols
    meta = dict(sqlite3.connect(db).execute("SELECT key, value FROM audit_meta").fetchall())
    boundary = json.loads(meta["sequence"])
    assert boundary["first_sequenced_row_id"] == 2  # the one old row is before the sequence
    old = AuditLog(db_path=db).query()[0]
    assert old.writer_seq is None and old.writer_id is None  # never backfilled


def test_an_older_release_writing_the_old_insert_shape_still_works(db: Path) -> None:
    AuditLog(db_path=db)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO audit_log(ts, run_id, agent_type, hook_point, plugin, decision, severity,"
        " reason, payload_json) VALUES ('2026-10-07T00:00:00+00:00','o','chat','pre_turn','p',"
        "'allow','info','r','{}')"
    )
    conn.execute(  # a second one: NULL writer columns do not collide on the unique index
        "INSERT INTO audit_log(ts, run_id, agent_type, hook_point, plugin, decision, severity,"
        " reason, payload_json) VALUES ('2026-10-07T00:00:01+00:00','o2','chat','pre_turn','p',"
        "'allow','info','r','{}')"
    )
    conn.commit()
    conn.close()
    assert AuditLog(db_path=db).count() == 2


def test_compaction_leaves_the_stores_own_rows_hot(db: Path) -> None:
    from datetime import UTC, datetime, timedelta

    from iris_harness.kernel.governance.audit.archive.compact import AuditCompactor
    from iris_harness.kernel.governance.audit.archive.writer import AuditArchive

    log = AuditLog(db_path=db)
    old = datetime.now(UTC) - timedelta(days=90)
    log.record(**_row(1, ts=old))
    compactor = AuditCompactor(
        audit_log=log, archive=AuditArchive(root=db.parent / "archive"), retention_days=30
    )
    compactor.compact()
    assert {r["hook_point"] for r in _all(db)} == {"writer.start"}  # the decision went cold


def _run(code: str, db: Path, *, check: bool = True) -> int:
    proc = subprocess.run(
        [sys.executable, "-c", code, str(db)],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PYTHONPATH": SRC, "IRIS_AUTH_SECRET": "x", "PATH": "/usr/bin:/bin"},
    )
    if check:
        assert proc.returncode == 0, proc.stderr[-500:]
    return proc.returncode


@pytest.fixture(autouse=True)
def _fresh_sequence_state() -> Iterator[None]:
    sequence.reset_for_tests()
    yield
    sequence.reset_for_tests()


# ------------------------------------------------ every surface that reads health shows it
@pytest.fixture()
def spooled_ledger(db: Path, down: _DbDown, monkeypatch: pytest.MonkeyPatch) -> AuditLog:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(db))
    log = AuditLog(db_path=db)
    _spool_one(log, down)
    monkeypatch.setattr(AuditLog, "drain_spool", lambda self: 0)
    return log


def _spool_one(log: AuditLog, down: _DbDown) -> None:
    down.on = True
    log.record(**_row(1))
    down.on = False


def test_get_governance_audit_and_state_report_the_waiting_spool(
    db: Path, down: _DbDown, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from iris_harness.foundation.auth import auth_headers
    from iris_harness.server.iris_api.main import create_app

    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(db))
    log = AuditLog(db_path=db)
    _spool_one(log, down)
    # Opening a ledger drains it when the database is up; here the drain is held off so the
    # screen is read while rows still wait (the database being down is the real case).
    monkeypatch.setattr(AuditLog, "drain_spool", lambda self: 0)
    with TestClient(create_app(auto_start_runtime=False), headers=auth_headers()) as client:
        audit = client.get("/governance/audit").json()
        state = client.get("/governance/state").json()
    for body in (audit, state):
        health = body["write_health"]
        assert health["ok"] is False and health["spool_pending"] == 1
        assert health["writes_spooled"] == 1 and health["writes_lost"] == 0


def test_iris_system_status_reports_the_waiting_spool(spooled_ledger: AuditLog) -> None:
    from typer.testing import CliRunner

    from iris_harness.main import app
    from iris_harness.services.system.status import iris_status

    assert iris_status().audit_writes["spool_pending"] == 1
    result = CliRunner().invoke(app, ["system", "status"])
    assert result.exit_code == 0, result.output
    assert "audit ledger" in result.output and "1 row(s) in the spool" in result.output


def test_a_healthy_ledger_adds_nothing_to_any_surface(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(db))
    AuditLog(db_path=db).record(**_row(1))
    assert write_health(db)["ok"] is True and audit_writes_provider() == []
