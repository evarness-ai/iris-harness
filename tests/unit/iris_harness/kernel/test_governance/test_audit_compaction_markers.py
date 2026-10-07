"""Compaction keeps what the ledger knows and accounts for what it moved (issue #134, stage 4b).

The crash is real (a subprocess exits between the fsynced chunks and the database transaction)
and so is the race (two interpreters compacting one database and one archive at once).
"""

# S603: the subprocesses run this repo's own interpreter on fixed code.
# ruff: noqa: S603, S608

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import iris_harness  # noqa: E402
from iris_harness.foundation.ids import new_ulid  # noqa: E402
from iris_harness.kernel.governance.audit import AuditLog, sequence  # noqa: E402
from iris_harness.kernel.governance.audit.archive import (  # noqa: E402
    AuditArchive,
    AuditCompactor,
    AuditQueryEngine,
    verify_archive,
)
from iris_harness.kernel.governance.audit.archive.markers import (  # noqa: E402
    marker_payload,
    read_markers,
)
from iris_harness.kernel.governance.audit.log import AuditRow, CompactionConflict  # noqa: E402

SRC = str(Path(iris_harness.__file__).resolve().parents[1])
LONG_AGO = datetime.now(UTC) - timedelta(days=90)
#: "now" for a run that should take everything but the marker rows.
FAR_FUTURE = datetime.now(UTC) + timedelta(days=400)


@pytest.fixture(autouse=True)
def _fresh_sequence_state() -> Iterator[None]:
    sequence.reset_for_tests()
    yield
    sequence.reset_for_tests()


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "audit.db"


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "archive"


def _row(n: int, **extra: Any) -> dict[str, Any]:
    return dict(
        run_id=f"run-{n}",
        step_id=n,
        agent_type="chat",
        hook_point="pre_tool_use",
        plugin="p",
        decision="allow",
        severity="info",
        reason=f"row {n}",
        payload={
            "session_id": f"s{n % 2}",
            "turn_id": f"t{n}",
            "call_id": f"c{n}",
            "parent_call_id": f"c{n - 1}" if n else None,
            "attempt": 1,
        },
        **extra,
    )


def _seed(db: Path, count: int = 5, *, ts: datetime = LONG_AGO) -> AuditLog:
    log = AuditLog(db_path=db)
    for n in range(count):
        log.record(**_row(n, ts=ts + timedelta(seconds=n)))
    return log


def _hot(db: Path, where: str = "1=1") -> list[sqlite3.Row]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        sql = f"SELECT * FROM audit_log WHERE {where} ORDER BY id"
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _compactor(log: AuditLog, root: Path, **kw: Any) -> AuditCompactor:
    return AuditCompactor(audit_log=log, archive=AuditArchive(root=root), retention_days=30, **kw)


def _view(db: Path, root: Path, sql: str, *, store: bool = True) -> list[tuple[Any, ...]]:
    engine = AuditQueryEngine(audit_db_path=db, archive_root=root, include_store_rows=store)
    return list(engine.query(sql).rows)


def _chunks(root: Path) -> list[Path]:
    return sorted(root.glob("year=*/month=*/audit-*.parquet"))


def _run(code: str, *args: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=180,
        env={"PYTHONPATH": SRC, "IRIS_AUTH_SECRET": "x", "PATH": "/usr/bin:/bin"},
    )
    if check:
        assert proc.returncode == 0, proc.stderr[-800:]
    return proc


# --------------------------------------------------------------------------- columns kept


def test_identity_and_sequence_columns_survive_compaction(db: Path, root: Path) -> None:
    log = _seed(db)
    before = {r["record_id"]: dict(r) for r in _hot(db)}
    assert any(r["call_id"] for r in before.values())

    _compactor(log, root).compact(now=FAR_FUTURE)

    cols = (
        "id, record_id, session_id, turn_id, call_id, parent_call_id, attempt, replay_of, "
        "resumed_from_run, writer_id, writer_seq, kind, run_id, step_id, hook_point, reason"
    )
    rows = _view(db, root, f"SELECT {cols} FROM audit_all WHERE kind IS DISTINCT FROM 'compaction'")
    got = {r[1]: r for r in rows}
    assert set(got) == set(before)
    names = [c.strip() for c in cols.split(",")]
    for record_id, orig in before.items():
        for name, value in zip(names, got[record_id], strict=True):
            assert value == orig[name], (record_id, name)


def test_the_stores_own_rows_archive_and_reconcile(db: Path, root: Path) -> None:
    log = _seed(db, 3)
    conn = sqlite3.connect(db)  # a gap row, as the writer would have written it
    conn.execute(
        "INSERT INTO audit_log(ts, run_id, agent_type, hook_point, plugin, decision, severity,"
        " reason, payload_json, record_id, writer_id, writer_seq, kind) VALUES (?, 'audit:w',"
        " 'audit', 'audit.gap', 'audit', 'allow', 'error', 'gap', '{}', ?, 'w-gap', 7, 'gap')",
        (LONG_AGO.isoformat(), new_ulid()),
    )
    conn.commit()
    conn.close()

    result = _compactor(log, root).compact(now=FAR_FUTURE)

    assert {r["kind"] for r in _hot(db)} == {"compaction"}  # only the marker stays hot
    (marker,) = read_markers(db)
    assert marker.payload["kinds"] == {"event": 3, "gap": 1, "writer": 1}
    assert marker.row_count == result.archived_rows == 5
    assert verify_archive(db, root).clean
    kinds = {r[0] for r in _view(db, root, "SELECT DISTINCT kind FROM audit_all")}
    assert kinds == {None, "gap", "writer", "compaction"}


def test_the_marker_is_never_archived(db: Path, root: Path) -> None:
    log = _seed(db, 3)
    compactor = _compactor(log, root)
    compactor.compact(now=FAR_FUTURE)
    assert compactor.compact(now=FAR_FUTURE).selected_rows == 0  # the marker is not a candidate
    assert len(_hot(db, "kind = 'compaction'")) == 1
    log.record(**_row(9, ts=LONG_AGO))
    compactor.compact(now=FAR_FUTURE)
    assert len(_hot(db, "kind = 'compaction'")) == 2


# --------------------------------------------------------------------------- reconciliation


def test_marker_counts_equal_chunk_rows_and_writer_ranges_reconcile(db: Path, root: Path) -> None:
    log = _seed(db, 6)
    conn = sqlite3.connect(db)  # a second writer's rows
    for seq in (1, 2, 4):  # seq 3 is a hole: lost, never stored
        conn.execute(
            "INSERT INTO audit_log(ts, run_id, agent_type, hook_point, plugin, decision, severity,"
            " reason, payload_json, record_id, writer_id, writer_seq) VALUES (?, 'r', 'chat',"
            " 'pre_tool_use', 'p', 'allow', 'info', 'x', '{}', ?, 'W-OTHER', ?)",
            ((LONG_AGO + timedelta(minutes=seq)).isoformat(), new_ulid(), seq),
        )
    conn.commit()
    conn.close()

    _compactor(log, root).compact(now=FAR_FUTURE)

    (marker,) = read_markers(db)
    in_chunks = sum(pq.read_metadata(c).num_rows for c in _chunks(root))
    assert marker.row_count == in_chunks == sum(c["rows"] for c in marker.chunks) == 6 + 1 + 3
    by_writer = {w["writer_id"]: w for w in marker.writers}
    assert by_writer["W-OTHER"] == {"writer_id": "W-OTHER", "min_seq": 1, "max_seq": 4, "count": 3}
    # the hole (3) is visible: the range is 4 wide, the count is 3
    assert by_writer["W-OTHER"]["max_seq"] - by_writer["W-OTHER"]["min_seq"] + 1 == 4
    assert marker.payload["id_range"][0] <= marker.payload["id_range"][1]
    assert verify_archive(db, root).clean


def test_verify_reports_a_tampered_chunk(db: Path, root: Path) -> None:
    log = _seed(db, 3)
    _compactor(log, root).compact(now=FAR_FUTURE)
    chunk = _chunks(root)[0]
    data = bytearray(chunk.read_bytes())
    data[len(data) // 2] ^= 0xFF
    chunk.write_bytes(bytes(data))
    report = verify_archive(db, root)
    assert not report.clean
    assert any("sha256" in p for p in report.problems) or report.unreadable


def test_marker_size_for_a_synthetic_month_of_writers() -> None:
    """A month of CLI runs is thousands of writers; the marker must stay small."""
    writers = 5000
    rows = [
        AuditRow(
            id=i + 1,
            ts="2026-09-01T00:00:00+00:00",
            run_id="r",
            step_id=None,
            agent_type="a",
            hook_point="h",
            plugin="p",
            decision="allow",
            classification=None,
            tier=None,
            cost_usd=None,
            severity="info",
            reason="x",
            payload_json="{}",
            record_id=new_ulid(),
            writer_id=f"W{i % writers:026d}",
            writer_seq=i // writers,
        )
        for i in range(writers * 3)
    ]
    payload = marker_payload(
        compaction_id=new_ulid(), cutoff_ts="x", rows=rows, chunks=(), adopted=False
    )
    size = len(json.dumps(payload))
    assert len(payload["writers"]) == writers
    assert size < 1_000_000, size  # ~1 MB is the documented line for a sidecar file


# --------------------------------------------------------------------------- atomicity


def test_a_changed_row_rolls_the_whole_transaction_back(db: Path) -> None:
    log = _seed(db, 3)
    rows = _hot(db, "kind IS NULL")
    keys = [(r["id"], r["record_id"]) for r in rows]
    keys[1] = (keys[1][0], "not-the-record-id")
    count = len(_hot(db))
    with pytest.raises(CompactionConflict):
        log.write_compaction({"compaction_id": "x"}, archived=keys)
    assert len(_hot(db)) == count  # nothing deleted
    assert not _hot(db, "kind = 'compaction'")  # and no marker


def test_a_row_that_lands_after_the_select_is_not_deleted_unarchived(db: Path, root: Path) -> None:
    """The delete is by id, not by time range: a late old-dated row survives."""
    log = _seed(db, 3)
    late = new_ulid()

    def land_late_row() -> None:
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO audit_log(ts, run_id, agent_type, hook_point, plugin, decision, severity,"
            " reason, payload_json, record_id) VALUES (?, 'late', 'chat', 'pre_tool_use', 'p',"
            " 'allow', 'info', 'late', '{}', ?)",
            (LONG_AGO.isoformat(), late),
        )
        conn.commit()
        conn.close()

    _compactor(log, root, _after_chunks=land_late_row).compact()
    assert [r["record_id"] for r in _hot(db, "run_id = 'late'")] == [late]
    (marker,) = read_markers(db)
    assert marker.row_count == 3  # the marker does not claim the late row


# --------------------------------------------------------------------------- crash recovery

_CRASH = """
    import os, sys
    from datetime import UTC, datetime, timedelta
    from pathlib import Path
    from iris_harness.kernel.governance.audit import AuditLog
    from iris_harness.kernel.governance.audit.archive import AuditArchive, AuditCompactor
    db, root = Path(sys.argv[1]), Path(sys.argv[2])
    log = AuditLog(db_path=db)
    old = datetime.now(UTC) - timedelta(days=90)
    for n in range(4):
        log.record(run_id=f"run-{n}", step_id=n, agent_type="chat", hook_point="pre_tool_use",
                   plugin="p", decision="allow", severity="info", reason="x",
                   payload={"call_id": f"c{n}"}, ts=old + timedelta(seconds=n))
    AuditCompactor(audit_log=log, archive=AuditArchive(root=root), retention_days=30,
                   _after_chunks=lambda: os._exit(3)).compact()
"""


def test_a_crash_between_the_chunks_and_the_transaction_loses_nothing(db: Path, root: Path) -> None:
    proc = _run(_CRASH, db, root, check=False)
    assert proc.returncode == 3  # it really died there
    assert len(_chunks(root)) == 1 and not read_markers(db)  # an orphan, no marker
    assert len(_hot(db, "kind IS NULL")) == 4  # nothing was deleted

    # the view counts each row once although it is in both tiers
    assert _view(db, root, "SELECT count(*) FROM audit_archive", store=False) == [(4,)]

    log = AuditLog(db_path=db)
    result = _compactor(log, root).compact()  # the next run adopts the orphan
    assert result.adopted_chunks == 1 and result.quarantined_chunks == 0
    (marker,) = read_markers(db)
    assert marker.payload["adopted"] is True and marker.row_count == 4
    assert not _hot(db, "kind IS NULL")
    assert _view(db, root, "SELECT count(*) FROM audit_archive", store=False) == [(4,)]
    assert verify_archive(db, root).clean


def test_a_duplicate_chunk_never_double_counts_and_is_quarantined_not_deleted(
    db: Path, root: Path
) -> None:
    log = _seed(db, 4)
    _compactor(log, root).compact(now=FAR_FUTURE)
    chunk = _chunks(root)[0]
    twin = chunk.with_name(f"audit-{new_ulid()}-0.parquet")  # the same rows in a second chunk
    twin.write_bytes(chunk.read_bytes())
    count = _view(db, root, "SELECT count(*) FROM audit_archive", store=False)
    assert count == [(4,)]  # collapsed by record_id
    assert verify_archive(db, root).orphans == (twin.relative_to(root).as_posix(),)

    result = _compactor(log, root).compact()  # its rows are not hot: do not trust it, move it
    assert result.quarantined_chunks == 1
    assert not twin.exists()
    moved = root / ".orphans" / twin.name.split("-")[1] / twin.relative_to(root)
    assert moved.read_bytes() == chunk.read_bytes()  # moved intact, reversible
    assert _view(db, root, "SELECT count(*) FROM audit_archive", store=False) == [(4,)]


def test_an_unreadable_orphan_run_chunk_is_moved_not_deleted(db: Path, root: Path) -> None:
    log = _seed(db, 2, ts=datetime.now(UTC))  # nothing old: only recovery has work
    archive = AuditArchive(root=root)
    cid = new_ulid()
    part = root / "year=2026" / "month=03"
    part.mkdir(parents=True)
    broken = part / f"audit-{cid}-0.parquet"
    broken.write_bytes(b"PAR1 torn")
    half = part / f".audit-{cid}-1.parquet.tmp"  # a chunk that never finished publishing
    half.write_bytes(b"half")
    del archive

    result = _compactor(log, root).compact()

    assert result.quarantined_chunks == 2
    assert not broken.exists() and not half.exists()
    assert (root / ".orphans" / cid / "year=2026" / "month=03" / broken.name).read_bytes() == (
        b"PAR1 torn"
    )


def test_a_legacy_chunk_is_never_touched_even_when_unreadable(db: Path, root: Path) -> None:
    log = _seed(db, 2)
    part = root / "year=2026" / "month=01"
    part.mkdir(parents=True)
    good = part / f"audit-{uuid.uuid4().hex}.parquet"  # the older shape
    _write_legacy_chunk(good)
    bad = part / f"audit-{uuid.uuid4().hex}.parquet"
    bad.write_bytes(b"not parquet at all")
    foreign = part / "notes.txt"
    foreign.write_text("keep me")
    snapshot = {p: p.read_bytes() for p in (good, bad, foreign)}

    _compactor(log, root).compact(now=FAR_FUTURE)

    assert {p: p.read_bytes() for p in snapshot} == snapshot  # not adopted, moved or removed
    assert not (root / ".orphans").exists()
    report = verify_archive(db, root)
    assert report.unreadable == (bad.relative_to(root).as_posix(),)
    assert not report.clean


def _write_legacy_chunk(path: Path) -> None:
    """A chunk as releases before stage 4b wrote them: 14 columns, no id, no identity."""
    dict_str = pa.dictionary(pa.int32(), pa.string())
    schema = pa.schema(
        [
            ("ts", pa.timestamp("ms", tz="UTC")),
            ("run_id", pa.string()),
            ("step_id", pa.int32()),
            ("agent_type", dict_str),
            ("hook_point", dict_str),
            ("plugin", dict_str),
            ("decision", dict_str),
            ("classification", dict_str),
            ("tier", dict_str),
            ("cost_usd", pa.float64()),
            ("severity", dict_str),
            ("reason", pa.string()),
            ("payload_json", pa.string()),
        ]
    )
    stamp = datetime(2026, 1, 5, tzinfo=UTC)
    cells: dict[str, list[Any]] = {
        "ts": [stamp],
        "run_id": ["legacy-run"],
        "step_id": [1],
        "agent_type": ["chat"],
        "hook_point": ["pre_tool_use"],
        "plugin": ["p"],
        "decision": ["allow"],
        "classification": [None],
        "tier": [None],
        "cost_usd": [None],
        "severity": ["info"],
        "reason": ["legacy"],
        "payload_json": ["{}"],
    }
    arrays = [pa.array(cells[f.name], type=f.type) for f in schema]
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)


def test_old_and_new_chunks_are_queryable_together(db: Path, root: Path) -> None:
    log = _seed(db, 3)
    part = root / "year=2026" / "month=01"
    part.mkdir(parents=True)
    _write_legacy_chunk(part / f"audit-{uuid.uuid4().hex}.parquet")
    _compactor(log, root).compact(now=FAR_FUTURE)

    rows = _view(
        db,
        root,
        "SELECT run_id, record_id IS NULL, id IS NULL FROM audit_archive ORDER BY run_id",
        store=False,
    )
    assert ("legacy-run", True, True) in rows  # NULL identity, read as is
    assert sum(1 for r in rows if r[1] is False) == 3  # the new chunk keeps its identity
    assert len(rows) == 4


# --------------------------------------------------------------------------- migration shapes


def _release_shaped_db(db: Path, rows: int) -> None:
    """A database as the release before call identity created it: 14 columns, no meta table."""
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, run_id TEXT NOT NULL,
            step_id INTEGER, agent_type TEXT NOT NULL, hook_point TEXT NOT NULL,
            plugin TEXT NOT NULL, decision TEXT NOT NULL, classification TEXT, tier TEXT,
            cost_usd REAL, severity TEXT NOT NULL, reason TEXT NOT NULL,
            payload_json TEXT NOT NULL);
    """)
    for n in range(rows):
        conn.execute(
            "INSERT INTO audit_log(ts, run_id, step_id, agent_type, hook_point, plugin, decision,"
            " severity, reason, payload_json) VALUES (?, ?, 1, 'chat', 'pre_tool_use', 'p',"
            " 'allow', 'info', 'old', ?)",
            ((LONG_AGO + timedelta(seconds=n)).isoformat(), f"old-{n}", '{"session_id": "s0"}'),
        )
    conn.commit()
    conn.close()


def test_the_view_reads_a_database_that_was_never_migrated(db: Path, root: Path) -> None:
    _release_shaped_db(db, 3)
    rows = _view(db, root, "SELECT run_id, record_id, kind FROM audit_archive ORDER BY run_id")
    assert [r[0] for r in rows] == ["old-0", "old-1", "old-2"]
    assert all(r[1] is None and r[2] is None for r in rows)


def test_a_release_shaped_database_compacts_twice_to_the_same_state(db: Path, root: Path) -> None:
    _release_shaped_db(db, 4)
    log = AuditLog(db_path=db)  # migrates in place
    compactor = _compactor(log, root)

    first = compactor.compact()
    assert first.archived_rows == 4 and first.deleted_rows == 4
    (marker,) = read_markers(db)
    assert marker.payload["pre_identity_rows"] == 4 and marker.payload["pre_sequence_rows"] == 4
    assert marker.payload["writers"] == []

    again = compactor.compact()
    assert again.selected_rows == 0 and len(read_markers(db)) == 1  # idempotent
    rows = _view(db, root, "SELECT id, run_id FROM audit_archive ORDER BY id", store=False)
    assert [r[1] for r in rows] == [f"old-{n}" for n in range(4)]
    assert [r[0] for r in rows] == [1, 2, 3, 4]  # the hot id survives in the chunk
    assert verify_archive(db, root).clean


_RACE = """
    import sys, time
    from pathlib import Path
    from iris_harness.kernel.governance.audit import AuditLog
    from iris_harness.kernel.governance.audit.archive import AuditArchive, AuditCompactor
    db, root, gate = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    log = AuditLog(db_path=db)
    comp = AuditCompactor(audit_log=log, archive=AuditArchive(root=root), retention_days=30)
    while not gate.exists():
        time.sleep(0.005)
    comp.compact()
"""


def test_two_interpreters_compacting_at_once_neither_lose_nor_duplicate(
    db: Path, root: Path, tmp_path: Path
) -> None:
    _release_shaped_db(db, 40)
    AuditLog(db_path=db)  # migrate once, as a running install would have
    gate = tmp_path / "go"
    env = {"PYTHONPATH": SRC, "IRIS_AUTH_SECRET": "x", "PATH": "/usr/bin:/bin"}
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(_RACE), str(db), str(root), str(gate)],
            env=env,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    gate.touch()
    for proc in procs:
        _, err = proc.communicate(timeout=180)
        assert proc.returncode == 0, err[-800:]

    assert not _hot(db, "kind IS NULL")
    markers = read_markers(db)
    assert sum(m.row_count for m in markers) == 40
    assert _view(
        db, root, "SELECT count(*), count(DISTINCT id) FROM audit_archive", store=False
    ) == [(40, 40)]
    assert verify_archive(db, root).clean
    assert not (root / ".orphans").exists()


# --------------------------------------------------------------------------- readers


def test_the_default_view_hides_store_rows_and_the_flag_shows_them(db: Path, root: Path) -> None:
    log = _seed(db, 2, ts=datetime.now(UTC))
    _compactor(log, root).compact(now=FAR_FUTURE)
    hidden = _view(db, root, "SELECT count(*) FROM audit_archive", store=False)
    shown = _view(db, root, "SELECT count(*) FROM audit_archive", store=True)
    every = _view(db, root, "SELECT count(*) FROM audit_all", store=False)
    assert hidden == [(2,)]
    assert shown == every and shown[0][0] > 2  # writer.start and the marker


def test_cli_query_export_and_verify(db: Path, root: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from iris_harness.main import app

    log = _seed(db, 2)
    _compactor(log, root).compact(now=FAR_FUTURE)
    runner = CliRunner()
    base = ["audit", "query", "SELECT count(*) AS n FROM audit_archive", "-f", "json"]
    where = ["--db", str(db), "--archive-root", str(root)]
    plain = runner.invoke(app, [*base, *where])
    store = runner.invoke(app, [*base, *where, "--include-store-rows"])
    assert plain.exit_code == 0 and json.loads(plain.stdout) == [{"n": 2}]
    assert json.loads(store.stdout)[0]["n"] > 2

    out = tmp_path / "e.jsonl"
    exported = runner.invoke(
        app, ["audit", "export", "--since", "2020-01-01", "--out", str(out), *where]
    )
    assert exported.exit_code == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 2 and all(line["kind"] is None for line in lines)
    assert all("writer_id" in line and "call_id" in line for line in lines)

    ok = runner.invoke(app, ["audit", "verify", *where])
    assert ok.exit_code == 0 and "reconciles" in ok.stdout
    chunk = _chunks(root)[0]
    chunk.write_bytes(chunk.read_bytes() + b"x")
    bad = runner.invoke(app, ["audit", "verify", *where])
    assert bad.exit_code == 1 and "sha256" in bad.stdout


def test_readers_of_the_archive_are_the_query_engine_only() -> None:
    """Every reader of the cold tier goes through ``AuditQueryEngine`` (the CLI and the demo).

    The API, the proof bundle and the trace builder read the hot ledger through ``AuditLog``,
    which already hides the store's rows; none of them opens Parquet.
    """
    src = Path(iris_harness.__file__).resolve().parent
    hits = sorted(
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if "read_parquet" in p.read_text(encoding="utf-8")
        or "audit-archive" in p.read_text(encoding="utf-8")
    )
    allowed = ("foundation/", "kernel/governance/audit/archive/")
    assert all(h.startswith(allowed) for h in hits), hits
