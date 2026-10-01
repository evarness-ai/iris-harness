from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from iris_harness.kernel.governance.audit.archive import (  # imported after optional-dep guard
    AuditArchive,
    AuditCompactor,
    AuditQueryEngine,
)
from iris_harness.kernel.governance.audit.log import AuditLog  # after guard


def _seed(log: AuditLog) -> None:
    log.record(
        run_id="old-run",
        step_id=1,
        agent_type="chat",
        hook_point="pre_llm_call",
        plugin="p",
        decision="allow",
        severity="info",
        reason="old",
        ts=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
    )
    log.record(
        run_id="new-run",
        step_id=2,
        agent_type="chat",
        hook_point="pre_llm_call",
        plugin="p",
        decision="allow",
        severity="info",
        reason="new",
        ts=datetime(2026, 5, 10, 12, 0, tzinfo=UTC),
    )


def test_compact_moves_old_rows_to_archive_and_deletes_from_sqlite(tmp_path: Path) -> None:
    audit = AuditLog(db_path=tmp_path / "audit.db")
    archive = AuditArchive(root=tmp_path / "archive")
    _seed(audit)

    compactor = AuditCompactor(audit_log=audit, archive=archive, retention_days=30)
    result = compactor.compact(now=datetime(2026, 5, 19, tzinfo=UTC))

    assert result.selected_rows == 1
    assert result.archived_rows == 1
    assert result.deleted_rows == 1
    assert audit.count() == 1
    assert list((tmp_path / "archive").glob("year=*/month=*/audit-*.parquet"))


def test_duckdb_query_spans_hot_and_cold_rows(tmp_path: Path) -> None:
    audit = AuditLog(db_path=tmp_path / "audit.db")
    archive = AuditArchive(root=tmp_path / "archive")
    _seed(audit)

    AuditCompactor(audit_log=audit, archive=archive, retention_days=30).compact(
        now=datetime(2026, 5, 19, tzinfo=UTC)
    )

    engine = AuditQueryEngine(
        audit_db_path=tmp_path / "audit.db",
        archive_root=tmp_path / "archive",
    )
    result = engine.query("SELECT run_id FROM audit_archive ORDER BY run_id ASC")
    assert result.columns == ("run_id",)
    assert result.rows == (("new-run",), ("old-run",))


def test_export_since_writes_jsonl_and_csv(tmp_path: Path) -> None:
    audit = AuditLog(db_path=tmp_path / "audit.db")
    archive = AuditArchive(root=tmp_path / "archive")
    _seed(audit)

    AuditCompactor(audit_log=audit, archive=archive, retention_days=30).compact(
        now=datetime(2026, 5, 19, tzinfo=UTC)
    )

    engine = AuditQueryEngine(
        audit_db_path=tmp_path / "audit.db",
        archive_root=tmp_path / "archive",
    )

    since = datetime(2026, 1, 1, tzinfo=UTC)
    jsonl = tmp_path / "audit.jsonl"
    csv_path = tmp_path / "audit.csv"

    json_count = engine.export_since(since=since, output=jsonl, fmt="jsonl")
    csv_count = engine.export_since(since=since, output=csv_path, fmt="csv")

    assert json_count == 2
    assert csv_count == 2

    lines = [line for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert "run_id" in payload

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "run_id" in csv_text.splitlines()[0]
