"""SQLite hot-tier compaction into Parquet cold-tier archive."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from iris_harness.kernel.governance.audit.archive.writer import AuditArchive
from iris_harness.kernel.governance.audit.log import AuditLog, AuditRow


@dataclass(frozen=True)
class CompactResult:
    selected_rows: int
    archived_rows: int
    deleted_rows: int
    partition_count: int
    cutoff_ts: str


class AuditCompactor:
    """Move old audit rows from SQLite to Parquet, then delete from SQLite."""

    def __init__(
        self,
        *,
        audit_log: AuditLog,
        archive: AuditArchive,
        retention_days: int = 30,
    ) -> None:
        if retention_days <= 0:
            raise ValueError("retention_days must be > 0")
        self._audit_log = audit_log
        self._archive = archive
        self._retention_days = retention_days

    def compact(self, *, now: datetime | None = None) -> CompactResult:
        current = (now or datetime.now(UTC)).astimezone(UTC)
        cutoff = current - timedelta(days=self._retention_days)
        cutoff_iso = cutoff.isoformat()

        rows = self._select_rows_older_than(cutoff_iso)
        if not rows:
            return CompactResult(
                selected_rows=0,
                archived_rows=0,
                deleted_rows=0,
                partition_count=0,
                cutoff_ts=cutoff_iso,
            )

        counts = self._archive.write(rows)
        deleted = self._delete_rows_by_id([row.id for row in rows])
        return CompactResult(
            selected_rows=len(rows),
            archived_rows=sum(counts.values()),
            deleted_rows=deleted,
            partition_count=len(counts),
            cutoff_ts=cutoff_iso,
        )

    def _select_rows_older_than(self, cutoff_iso: str) -> tuple[AuditRow, ...]:
        with closing(sqlite3.connect(self._audit_log.db_path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM audit_log
                WHERE ts < ?
                ORDER BY ts ASC, id ASC
                """,
                (cutoff_iso,),
            ).fetchall()

        return tuple(
            AuditRow(
                id=int(row["id"]),
                ts=str(row["ts"]),
                run_id=str(row["run_id"]),
                step_id=row["step_id"] if row["step_id"] is None else int(row["step_id"]),
                agent_type=str(row["agent_type"]),
                hook_point=str(row["hook_point"]),
                plugin=str(row["plugin"]),
                decision=str(row["decision"]),
                classification=(
                    row["classification"]
                    if row["classification"] is None
                    else str(row["classification"])
                ),
                tier=row["tier"] if row["tier"] is None else str(row["tier"]),
                cost_usd=row["cost_usd"] if row["cost_usd"] is None else float(row["cost_usd"]),
                severity=str(row["severity"]),
                reason=str(row["reason"]),
                payload_json=str(row["payload_json"]),
            )
            for row in rows
        )

    def _delete_rows_by_id(self, ids: list[int]) -> int:
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        # placeholders is only "?,?,…" bound params; ids are passed separately — no injection
        sql = f"DELETE FROM audit_log WHERE id IN ({placeholders})"  # noqa: S608
        with closing(sqlite3.connect(self._audit_log.db_path)) as conn, conn:
            cur = conn.execute(sql, ids)
            conn.commit()
            return int(cur.rowcount or 0)
