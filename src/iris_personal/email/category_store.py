"""Cross-cutting categories table for IRIS (Phase 1 Track 1F).

Lives in ``data/iris.db`` per ADR-0017's cross-cutting design. The
``type`` column lets the same schema host categories for email,
finance, calendar, tasks, etc. — for Phase 1 the only value is
``email``.

Identity is path-style: ``<type>/<root>/<branch>/<leaf>`` per
ADR-0019 §2 (mirrors ADR-0016's ``provider:address`` slug shape).

Phase 1 Track 1F ships the Pydantic contract (this commit) plus a
``CategoryStore`` (next commit) plus the ``iris email
accept-categories`` CLI that converts an accepted
``CategoryProposal`` JSONL into rows on this table.

See ADR-0019 for the full schema + implementation shape, and
ADR-0020 for the drift-correction follow-up that this table's
companion ``categories_history`` enables (Path C self-learning).
"""

from __future__ import annotations

import builtins
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from iris_harness.sdk.persistence import data_path, sqlite_conn

# Allowed type values. Matches the categories.type column.
# Email is the only currently-shipped value; finance / calendar / tasks
# are reserved for later phases. Coercion-via-validator rather than
# Literal so adding a new type doesn't require a Pydantic version
# bump across consumers — values are enforced application-side.
ALLOWED_TYPES = ("email", "finance", "calendar", "tasks")

# Mirrors ALLOWED_ROOTS in iris_personal.plugins.email_workflows.discovery. Duplicated here
# (from when this module was core, ``iris_harness.foundation.data``) so the email domain
# does not import a plugin.
# When a new root is needed, both lists update together.
ALLOWED_ROOTS = (
    "shopping",
    "finance",
    "news",
    "social",
    "work",
    "personal",
    "learning",
    "travel",
    "jobs",
    "community",
    "automotive",
    "tools",
    "transactional",
    "other",
)

# Allowed sensitivity values per ADR-0002.
ALLOWED_SENSITIVITY = ("low", "medium", "high")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def make_path(type_: str, root: str, branch: str, leaf: str) -> str:
    """Compose the path-style primary key.

    Lowercased throughout so case-only differences don't create
    distinct rows — same posture as ``EmailAccount.id``.
    """
    parts = (
        type_.strip().lower(),
        root.strip().lower(),
        branch.strip().lower(),
        leaf.strip().lower(),
    )
    if not all(parts):
        raise ValueError(f"all path parts must be non-empty: got {parts!r}")
    return "/".join(parts)


class Category(BaseModel):
    """One accepted category in the cross-cutting taxonomy.

    Identity is the ``path`` (composed from type/root/branch/leaf at
    write time). Always carry the components as separate columns so
    downstream queries can filter by ``type`` and ``root`` without
    string-splitting.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    # ─── Identity ────────────────────────────────────────────────────
    path: str = Field(..., min_length=3)
    type: str = Field(..., min_length=1)
    root: str = Field(..., min_length=1)
    branch: str = Field(..., min_length=1)
    leaf: str = Field(..., min_length=1)

    # ─── Provenance ──────────────────────────────────────────────────
    # Soft FK to email_accounts.id; None = applies cross-account
    # (e.g. global category that any email account inherits).
    account_id: str | None = None
    # Cohesion at acceptance time, carried over from CategoryProposal.
    cohesion: float | None = Field(default=None, ge=0.0, le=1.0)

    # ─── Routing implication ─────────────────────────────────────────
    sensitivity: str = Field(default="low")  # per ADR-0002

    # ─── Lifecycle ───────────────────────────────────────────────────
    active: bool = True
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    # ─── Free-form ───────────────────────────────────────────────────
    # Snapshot of discovery params, top_domains, naming_rationale, etc.
    # Schema deliberately loose — Track 1J may add structured fields.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in ALLOWED_TYPES:
            raise ValueError(f"type must be one of {ALLOWED_TYPES}: got {v!r}")
        return v

    @field_validator("root")
    @classmethod
    def _check_root(cls, v: str) -> str:
        if v not in ALLOWED_ROOTS:
            raise ValueError(f"root must be one of {ALLOWED_ROOTS}: got {v!r}")
        return v

    @field_validator("sensitivity")
    @classmethod
    def _check_sensitivity(cls, v: str) -> str:
        if v not in ALLOWED_SENSITIVITY:
            raise ValueError(f"sensitivity must be one of {ALLOWED_SENSITIVITY}: got {v!r}")
        return v

    @field_validator("path")
    @classmethod
    def _check_path_lowercase(cls, v: str) -> str:
        if v != v.lower():
            raise ValueError(f"path must be lowercase: got {v!r}")
        return v


# ─── SQLite store ────────────────────────────────────────────────────────────


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass
class CategoryStore:
    """SQLite-backed CRUD over the cross-cutting categories table.

    Every mutation records a row in ``categories_history`` per
    ADR-0019 §3 + ADR-0020 Path C (the audit trail Track 1J's
    self-learning loop reads).

    Idempotent upsert by path per ADR-0019 §4: same-path
    ``upsert_if_new`` calls are no-ops (and no history row).
    """

    db_path: Path = field(default_factory=lambda: data_path("iris.db"))

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Commit on success, roll back on error -- what ``with sqlite3.connect()`` did --
        # and close the handle, which it did not (every call leaked one).
        with sqlite_conn(self.db_path, row_factory=sqlite3.Row) as conn:
            yield conn

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def upsert_if_new(
        self,
        category: Category,
        *,
        source: str = "accept-categories",
    ) -> bool:
        """Insert if path doesn't exist; reactivate if it does (archived
        → active). Never overwrites root/branch/leaf — see ADR-0019 §4.

        Returns True if a row was inserted or reactivated, False if
        the path already existed and was already active (no-op).

        Writes a ``categories_history`` row for every mutation (insert
        or reactivation), so the audit trail mirrors the table's
        change log exactly.
        """
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT path, active FROM categories WHERE path = ?",
                (category.path,),
            ).fetchone()
            now = _iso(_utc_now())

            if existing is None:
                conn.execute(_INSERT_SQL, _category_to_row(category))
                conn.execute(
                    _HISTORY_INSERT_SQL,
                    {
                        "category_path": category.path,
                        "op": "insert",
                        "old_path": None,
                        "new_path": category.path,
                        "payload": json.dumps(_category_to_payload(category)),
                        "edited_at": now,
                        "source": source,
                    },
                )
                return True

            if existing["active"] == 0:
                conn.execute(
                    "UPDATE categories SET active = 1, updated_at = ? WHERE path = ?",
                    (now, category.path),
                )
                conn.execute(
                    _HISTORY_INSERT_SQL,
                    {
                        "category_path": category.path,
                        "op": "update",
                        "old_path": category.path,
                        "new_path": category.path,
                        "payload": json.dumps({"reactivated": True}),
                        "edited_at": now,
                        "source": source,
                    },
                )
                return True

            return False

    def archive(self, path: str, *, source: str = "accept-categories") -> bool:
        """Mark a category inactive. No-op if already archived or
        missing. Records ``op='archive'`` in history when it acts."""
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT path, active FROM categories WHERE path = ?",
                (path,),
            ).fetchone()
            if existing is None or existing["active"] == 0:
                return False
            now = _iso(_utc_now())
            conn.execute(
                "UPDATE categories SET active = 0, updated_at = ? WHERE path = ?",
                (now, path),
            )
            conn.execute(
                _HISTORY_INSERT_SQL,
                {
                    "category_path": path,
                    "op": "archive",
                    "old_path": path,
                    "new_path": None,
                    "payload": None,
                    "edited_at": now,
                    "source": source,
                },
            )
            return True

    def record_correction(
        self,
        *,
        message_id: str,
        account_id: str,
        old_path: str | None,
        new_path: str,
        previous_classifier: str | None,
        reason: str | None = None,
    ) -> None:
        """Record a user-classification correction in categories_history.

        Per ADR-0024 §3: a correction logs ``op='update'`` with
        ``source='user-classification-correction'``. The payload
        carries the message + old/new paths so Track 1J's audit CLI
        and `--include-corrections` flow can reconstruct what the user
        changed without joining other tables.

        This method does NOT update the emails table — the caller is
        responsible for the matching ``EmailStore.mark_classified``
        write. Keeping the two separate lets the CLI handle exit-code
        semantics for "category missing" / "message missing" cases
        before any persistence happens.
        """
        import json

        payload: dict[str, Any] = {
            "message_id": message_id,
            "account_id": account_id,
            "old_path": old_path,
            "new_path": new_path,
            "previous_classifier": previous_classifier,
        }
        if reason:
            payload["reason"] = reason
        now = _iso(_utc_now())
        with self._connect() as conn:
            conn.execute(
                _HISTORY_INSERT_SQL,
                {
                    "category_path": new_path,
                    "op": "update",
                    "old_path": old_path,
                    "new_path": new_path,
                    "payload": json.dumps(payload),
                    "edited_at": now,
                    "source": "user-classification-correction",
                },
            )

    def list_corrections(
        self,
        *,
        account_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read user-classification corrections from categories_history.

        Returns the history rows where ``source=
        'user-classification-correction'``. Used by both the
        ``iris email corrections list`` audit CLI and the
        ``knn-gate --include-corrections`` measurement flow.
        """
        import json

        sql = "SELECT * FROM categories_history " "WHERE source = 'user-classification-correction'"
        params: list[Any] = []
        if since is not None:
            sql += " AND edited_at >= ?"
            params.append(_iso(since))
        sql += " ORDER BY history_id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            payload = json.loads(r["payload"] or "{}")
            if account_id is not None and payload.get("account_id") != account_id:
                continue
            out.append(
                {
                    "history_id": r["history_id"],
                    "edited_at": r["edited_at"],
                    "category_path": r["category_path"],
                    "old_path": r["old_path"],
                    "new_path": r["new_path"],
                    "payload": payload,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, path: str) -> Category | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM categories WHERE path = ?", (path,)).fetchone()
        return _row_to_category(row) if row else None

    def list(
        self,
        *,
        type: str | None = None,
        account_id: str | None = None,
        active_only: bool = True,
    ) -> list[Category]:
        sql = "SELECT * FROM categories WHERE 1=1"
        params: list[Any] = []
        if active_only:
            sql += " AND active = 1"
        if type is not None:
            sql += " AND type = ?"
            params.append(type)
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        sql += " ORDER BY path"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_category(r) for r in rows]

    def history(self, path: str) -> builtins.list[dict[str, Any]]:
        """Edit history for one category path. Most-recent first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM categories_history "
                "WHERE category_path = ? "
                "ORDER BY history_id DESC",
                (path,),
            ).fetchall()
        return [dict(r) for r in rows]


# ─── Row marshaling ──────────────────────────────────────────────────────────


def _category_to_row(c: Category) -> dict[str, Any]:
    return {
        "path": c.path,
        "type": c.type,
        "root": c.root,
        "branch": c.branch,
        "leaf": c.leaf,
        "account_id": c.account_id,
        "cohesion": c.cohesion,
        "sensitivity": c.sensitivity,
        "active": 1 if c.active else 0,
        "created_at": _iso(c.created_at),
        "updated_at": _iso(c.updated_at),
        "metadata": json.dumps(c.metadata),
    }


def _row_to_category(row: sqlite3.Row) -> Category:
    return Category(
        path=row["path"],
        type=row["type"],
        root=row["root"],
        branch=row["branch"],
        leaf=row["leaf"],
        account_id=row["account_id"],
        cohesion=row["cohesion"],
        sensitivity=row["sensitivity"],
        active=bool(row["active"]),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        metadata=json.loads(row["metadata"] or "{}"),
    )


def _category_to_payload(c: Category) -> dict[str, Any]:
    """Snapshot saved into categories_history.payload on insert.

    Carries enough of the Category that Track 1J's self-learning
    can reconstruct what was accepted, without re-joining tables.
    """
    return {
        "path": c.path,
        "account_id": c.account_id,
        "cohesion": c.cohesion,
        "sensitivity": c.sensitivity,
        "metadata": c.metadata,
    }


# ─── DDL ─────────────────────────────────────────────────────────────────────


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS categories (
    path             TEXT PRIMARY KEY,
    type             TEXT NOT NULL,
    root             TEXT NOT NULL,
    branch           TEXT NOT NULL,
    leaf             TEXT NOT NULL,
    account_id       TEXT,
    cohesion         REAL,
    sensitivity      TEXT NOT NULL DEFAULT 'low',
    active           INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    metadata         TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_categories_type_active ON categories(type, active);
CREATE INDEX IF NOT EXISTS idx_categories_account ON categories(account_id);

CREATE TABLE IF NOT EXISTS categories_history (
    history_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    category_path    TEXT NOT NULL,
    op               TEXT NOT NULL,
    old_path         TEXT,
    new_path         TEXT,
    payload          TEXT,
    edited_at        TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'accept-categories'
);

CREATE INDEX IF NOT EXISTS idx_history_category ON categories_history(category_path);
"""

_INSERT_SQL = """
INSERT INTO categories
  (path, type, root, branch, leaf, account_id, cohesion, sensitivity,
   active, created_at, updated_at, metadata)
VALUES
  (:path, :type, :root, :branch, :leaf, :account_id, :cohesion, :sensitivity,
   :active, :created_at, :updated_at, :metadata)
"""

_HISTORY_INSERT_SQL = """
INSERT INTO categories_history
  (category_path, op, old_path, new_path, payload, edited_at, source)
VALUES
  (:category_path, :op, :old_path, :new_path, :payload, :edited_at, :source)
"""
