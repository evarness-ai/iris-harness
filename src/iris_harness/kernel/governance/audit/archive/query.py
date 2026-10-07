"""DuckDB query/export wrapper over hot (SQLite) + cold (Parquet) audit data."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import duckdb

from iris_harness.kernel.governance.audit.archive.writer import default_archive_root
from iris_harness.kernel.governance.audit.log import _default_audit_db_path

ExportFormat = Literal["jsonl", "csv"]

#: The columns of a row as the view shows them, in order, with their DuckDB types. The first
#: fourteen are the original archive; the rest arrived with call identity and the writer
#: sequence (issue #134) and read as NULL from a chunk or a database that predates them.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("id", "BIGINT"),
    ("ts", "TIMESTAMP"),
    ("run_id", "VARCHAR"),
    ("step_id", "INTEGER"),
    ("agent_type", "VARCHAR"),
    ("hook_point", "VARCHAR"),
    ("plugin", "VARCHAR"),
    ("decision", "VARCHAR"),
    ("classification", "VARCHAR"),
    ("tier", "VARCHAR"),
    ("cost_usd", "DOUBLE"),
    ("severity", "VARCHAR"),
    ("reason", "VARCHAR"),
    ("payload_json", "VARCHAR"),
    ("record_id", "VARCHAR"),
    ("session_id", "VARCHAR"),
    ("turn_id", "VARCHAR"),
    ("call_id", "VARCHAR"),
    ("parent_call_id", "VARCHAR"),
    ("attempt", "INTEGER"),
    ("replay_of", "VARCHAR"),
    ("resumed_from_run", "VARCHAR"),
    ("writer_id", "VARCHAR"),
    ("writer_seq", "BIGINT"),
    ("kind", "VARCHAR"),
)


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


class AuditQueryEngine:
    """Run DuckDB SQL against a unified ``audit_archive`` view.

    ``audit_archive`` holds the rows of hook firings, like ``AuditLog.query``; the store's own
    rows (writer start/close, gap, compaction: ``kind`` not NULL) are in it only with
    ``include_store_rows=True``. The ``audit_all`` view always holds every row.
    """

    def __init__(
        self,
        *,
        audit_db_path: Path | None = None,
        archive_root: Path | None = None,
        include_store_rows: bool = False,
    ) -> None:
        self._include_store_rows = include_store_rows
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
            "ORDER BY ts ASC, writer_id ASC, writer_seq ASC, id ASC"
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
        """``audit_all`` (every row, de-duplicated) and ``audit_archive`` (what readers see).

        A row can be in both tiers for a moment (a compaction wrote its chunk and crashed
        before deleting it), or in two chunks (an orphan re-selected by a later run). Rows are
        de-duplicated by ``record_id``; a row of the pre-identity era has none, so its ``id``
        (unique in the database, kept in the chunk since stage 4b) stands in. The hot copy wins.
        A chunk of the older shape has no ``id`` and no identity: it is read as is, never
        de-duplicated.
        """
        conn.execute("SET TimeZone = 'UTC'")
        hot_rows = self._read_hot_rows()
        defs = ", ".join(f"{name} {typ}" for name, typ in _COLUMNS)
        conn.execute(f"CREATE TEMP TABLE audit_hot ({defs})")
        if hot_rows:
            marks = ", ".join("?" for _ in _COLUMNS)
            conn.executemany(f"INSERT INTO audit_hot VALUES ({marks})", hot_rows)  # noqa: S608

        names = ", ".join(name for name, _ in _COLUMNS)
        glob = str(self._archive_root / "year=*" / "month=*" / "audit-*.parquet")
        if any(self._archive_root.glob("year=*/month=*/audit-*.parquet")):
            source = f"read_parquet('{glob}', union_by_name=true)"  # internal path, not user input
            describe = f"DESCRIBE SELECT * FROM {source}"  # noqa: S608 - internal path
            have = {row[0] for row in conn.execute(describe).fetchall()}
            select = ", ".join(
                name if name in have else f"NULL::{typ} AS {name}" for name, typ in _COLUMNS
            )
            conn.execute(
                f"CREATE TEMP VIEW audit_cold AS SELECT {select} FROM {source}"  # noqa: S608
            )
        else:
            conn.execute("CREATE TEMP VIEW audit_cold AS SELECT * FROM audit_hot WHERE 1=0")

        conn.execute(f"""
            CREATE TEMP VIEW audit_all AS
            WITH u AS (
                SELECT {names}, 0 AS src FROM audit_hot
                UNION ALL
                SELECT {names}, 1 AS src FROM audit_cold
            ), k AS (
                SELECT *, COALESCE(
                    record_id, CASE WHEN id IS NOT NULL THEN 'id:' || CAST(id AS VARCHAR) END
                ) AS dedup_key FROM u
            ), r AS (
                SELECT *, row_number() OVER (PARTITION BY dedup_key ORDER BY src, id) AS rn FROM k
            )
            SELECT {names} FROM r WHERE dedup_key IS NULL OR rn = 1
            """)  # noqa: S608 - fixed column list
        where = "" if self._include_store_rows else " WHERE kind IS NULL"
        view = f"CREATE TEMP VIEW audit_archive AS SELECT * FROM audit_all{where}"  # noqa: S608
        conn.execute(view)

    def _read_hot_rows(self) -> list[tuple[object, ...]]:
        if not self._audit_db_path.exists():
            return []
        with closing(sqlite3.connect(self._audit_db_path)) as conn, conn:
            present = {row[1] for row in conn.execute("PRAGMA table_info(audit_log)")}
            select = ", ".join(name if name in present else "NULL" for name, _ in _COLUMNS)
            rows = conn.execute(f"SELECT {select} FROM audit_log").fetchall()  # noqa: S608
        return [tuple(row) for row in rows]


def read_cold_rows(
    files: Sequence[Path],
    *,
    columns: Sequence[str],
    where: str = "TRUE",
    params: Sequence[object] = (),
    limit: int | None = None,
    timeout_s: float | None = None,
) -> tuple[list[dict[str, object]], bool]:
    """Rows of the given chunks (never the whole archive), and whether the read was cut short.

    ``columns`` that a chunk of the older shape lacks read as NULL. ``where`` is SQL written by
    the caller with ``?`` placeholders for ``params``; it may name any column of ``_COLUMNS``.
    A ``timeout_s`` interrupts the scan: the rows read so far are NOT returned (a partial scan
    is not an answer) and the flag is True.
    """
    import threading

    if not files:
        return [], False
    types = dict(_COLUMNS)
    with duckdb.connect(database=":memory:") as conn:
        conn.execute("SET TimeZone = 'UTC'")
        listing = ", ".join("'" + str(p).replace("'", "''") + "'" for p in files)
        source = f"read_parquet([{listing}], union_by_name=true)"  # internal paths
        describe = f"DESCRIBE SELECT * FROM {source}"  # noqa: S608 - internal paths
        have = {row[0] for row in conn.execute(describe).fetchall()}
        select = ", ".join(
            name if name in have else f"NULL::{types[name]} AS {name}" for name in types
        )
        conn.execute(f"CREATE TEMP VIEW cold AS SELECT {select} FROM {source}")  # noqa: S608
        picked = ", ".join(columns)
        tail = f" LIMIT {int(limit)}" if limit is not None else ""
        sql = f"SELECT {picked} FROM cold WHERE {where}{tail}"  # noqa: S608 - fixed names
        timer = None
        if timeout_s is not None:
            timer = threading.Timer(timeout_s, conn.interrupt)
            timer.start()
        try:
            cur = conn.execute(sql, list(params))
            rows = cur.fetchall()
        except duckdb.InterruptException:
            return [], True
        finally:
            if timer is not None:
                timer.cancel()
        names = [d[0] for d in cur.description]
    return [dict(zip(names, row, strict=True)) for row in rows], False
