"""DuckDB query/export wrapper over hot (SQLite) + cold (Parquet) audit data."""

from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import duckdb

from iris_harness.kernel.governance.audit.archive.writer import default_archive_root
from iris_harness.kernel.governance.audit.log import _default_audit_db_path

ExportFormat = Literal["jsonl", "csv"]


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


class AuditQueryEngine:
    """Run DuckDB SQL against a unified ``audit_archive`` view."""

    def __init__(
        self,
        *,
        audit_db_path: Path | None = None,
        archive_root: Path | None = None,
    ) -> None:
        self._audit_db_path = (audit_db_path or _default_audit_db_path()).expanduser().resolve()
        self._archive_root = (archive_root or default_archive_root()).expanduser().resolve()

    def query(self, sql: str) -> QueryResult:
        normalized = sql.strip().lower()
        if not normalized:
            raise ValueError("query must not be empty")
        if ";" in normalized:
            raise ValueError("multiple SQL statements are not allowed")
        if not (normalized.startswith("select") or normalized.startswith("with")):
            raise ValueError("only SELECT/CTE queries are allowed")

        with duckdb.connect(database=":memory:") as conn:
            self._prepare_views(conn)
            rel = conn.sql(sql)
            rows = tuple(tuple(item for item in row) for row in rel.fetchall())
            cols = tuple(desc[0] for desc in rel.description)
            return QueryResult(columns=cols, rows=rows)

    def export_since(self, *, since: datetime, output: Path, fmt: ExportFormat) -> int:
        clause = since.astimezone(UTC).isoformat()
        # clause is a datetime.isoformat() string, not user input — no injection vector
        sql = (
            "SELECT * FROM audit_archive "  # noqa: S608
            f"WHERE ts >= '{clause}' "
            "ORDER BY ts ASC, id ASC"
        )
        result = self.query(sql)
        if fmt == "jsonl":
            with output.open("w", encoding="utf-8") as fh:
                for row in result.rows:
                    payload = dict(zip(result.columns, row, strict=True))
                    fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
            return len(result.rows)

        with output.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(result.columns)
            writer.writerows(result.rows)
        return len(result.rows)

    def _prepare_views(self, conn: duckdb.DuckDBPyConnection) -> None:
        hot_rows = self._read_hot_rows()
        conn.execute("""
            CREATE TEMP TABLE audit_hot (
                id BIGINT,
                ts TIMESTAMP,
                run_id VARCHAR,
                step_id INTEGER,
                agent_type VARCHAR,
                hook_point VARCHAR,
                plugin VARCHAR,
                decision VARCHAR,
                classification VARCHAR,
                tier VARCHAR,
                cost_usd DOUBLE,
                severity VARCHAR,
                reason VARCHAR,
                payload_json VARCHAR
            )
            """)
        if hot_rows:
            conn.executemany(
                """
                INSERT INTO audit_hot VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                hot_rows,
            )

        parquet_glob = str(self._archive_root / "year=*" / "month=*" / "audit-*.parquet")
        if any(self._archive_root.glob("year=*/month=*/audit-*.parquet")):
            # parquet_glob is an internal archive path, not user input — no injection vector
            conn.execute(
                "CREATE TEMP VIEW audit_cold AS "  # noqa: S608
                "SELECT NULL::BIGINT AS id, ts, run_id, step_id, agent_type, hook_point, "
                "plugin, decision, classification, tier, cost_usd, severity, reason, payload_json "
                f"FROM read_parquet('{parquet_glob}')"
            )
        else:
            conn.execute("CREATE TEMP VIEW audit_cold AS SELECT * FROM audit_hot WHERE 1=0")

        conn.execute("""
            CREATE TEMP VIEW audit_archive AS
            SELECT * FROM audit_hot
            UNION ALL
            SELECT * FROM audit_cold
            """)

    def _read_hot_rows(self) -> list[tuple[object, ...]]:
        if not self._audit_db_path.exists():
            return []
        with closing(sqlite3.connect(self._audit_db_path)) as conn, conn:
            rows = conn.execute("""
                SELECT id, ts, run_id, step_id, agent_type, hook_point, plugin, decision,
                       classification, tier, cost_usd, severity, reason, payload_json
                FROM audit_log
                """).fetchall()
        return [tuple(row) for row in rows]
