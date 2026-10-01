"""The settings store: what the owner changed, on top of what the files ship (ADR-0120).

The harness reads its settings from files baked into the deployment — YAML under
``config/`` and the env file. On the cloud VM those are mounted read-only and a deploy
replaces them, so a change made from the app cannot be written back into them. It lands
here instead: one SQLite file under the data dir (``/srv/iris/data`` on the VM, the
volume the nightly backup already copies), holding

* **overrides** — the current value the owner chose, keyed by ``(section, key)``. The
  file value stays the default; clearing an override returns to it.
* **history** — every set and reset, with who made it and the value before and after,
  so the app can list changes and undo one.

The store knows nothing about what a section means. Each owner of a setting (the
heartbeat scheduler first) decides what a valid value is and applies it; this module
only keeps the record. Values are stored as JSON.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_harness.foundation.persistence import data_path, sqlite_conn, with_locked_retry

SETTINGS_DB_NAME = "settings.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS overrides (
    section    TEXT NOT NULL,
    key        TEXT NOT NULL,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    actor      TEXT NOT NULL,
    PRIMARY KEY (section, key)
);
CREATE TABLE IF NOT EXISTS history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    section   TEXT NOT NULL,
    key       TEXT NOT NULL,
    action    TEXT NOT NULL,
    old_json  TEXT,
    new_json  TEXT,
    actor     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS history_section ON history (section, id);
"""


@dataclass(frozen=True)
class SettingChange:
    """One row of the history: ``action`` is ``set`` or ``reset``.

    ``old`` and ``new`` are the effective values either side of the change, so a reset
    records the value it returned to, not ``None``.
    """

    id: int
    at: datetime
    section: str
    key: str
    action: str
    old: Any
    new: Any
    actor: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "at": self.at.isoformat(),
            "section": self.section,
            "key": self.key,
            "action": self.action,
            "old": self.old,
            "new": self.new,
            "actor": self.actor,
        }


# ``set``'s history defaults to the stored value; a caller storing only a diff passes the
# full value the setting now has instead.
_SAME = object()


def _loads(raw: str | None) -> Any:
    return None if raw is None else json.loads(raw)


class SettingsStore:
    """Overrides + history in one SQLite file. Safe to share across threads."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else data_path(SETTINGS_DB_NAME)
        self._lock = threading.Lock()
        self._ready = False

    def _ensure(self) -> None:
        # Lazy: a harness nobody has changed a setting on reads through an empty store,
        # and building the runtime should not create files it may never need.
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite_conn(self.db_path) as conn:
                conn.executescript(_SCHEMA)
            self._ready = True

    # -- reads -------------------------------------------------------------------
    def get(self, section: str, key: str) -> Any:
        """The override for ``(section, key)``, or ``None`` when the default applies."""
        if not self.db_path.exists():
            return None
        self._ensure()
        with sqlite_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT value_json FROM overrides WHERE section = ? AND key = ?",
                (section, key),
            ).fetchone()
        return None if row is None else _loads(row[0])

    def section(self, section: str) -> dict[str, Any]:
        """Every override in ``section``, by key."""
        if not self.db_path.exists():
            return {}
        self._ensure()
        with sqlite_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT key, value_json FROM overrides WHERE section = ? ORDER BY key",
                (section,),
            ).fetchall()
        return {key: _loads(raw) for key, raw in rows}

    def history(self, *, section: str | None = None, limit: int = 100) -> list[SettingChange]:
        """Newest first. ``limit`` <= 0 means every row."""
        if not self.db_path.exists():
            return []
        self._ensure()
        sql = "SELECT id, at, section, key, action, old_json, new_json, actor FROM history"
        args: list[Any] = []
        if section is not None:
            sql += " WHERE section = ?"
            args.append(section)
        sql += " ORDER BY id DESC"
        if limit > 0:
            sql += " LIMIT ?"
            args.append(limit)
        with sqlite_conn(self.db_path) as conn:
            rows = conn.execute(sql, args).fetchall()
        return [
            SettingChange(
                id=row[0],
                at=datetime.fromisoformat(row[1]),
                section=row[2],
                key=row[3],
                action=row[4],
                old=_loads(row[5]),
                new=_loads(row[6]),
                actor=row[7],
            )
            for row in rows
        ]

    # -- writes ------------------------------------------------------------------
    @with_locked_retry
    def set(  # the override plus both sides of its history row
        self,
        section: str,
        key: str,
        value: Any,
        *,
        old: Any,
        actor: str,
        new: Any = _SAME,
    ) -> SettingChange:
        """Store ``value`` as the override and record the change in one transaction.

        The history row's ``new`` is ``value`` unless ``new`` says otherwise: an owner
        that stores only the fields that differ from the default records the whole value
        the setting now has, so the history reads the same whichever fields changed.
        """
        recorded = value if new is _SAME else new
        self._ensure()
        now = datetime.now(UTC)
        value_json = json.dumps(value, sort_keys=True)
        with sqlite_conn(self.db_path) as conn:
            conn.execute(
                "INSERT INTO overrides (section, key, value_json, updated_at, actor) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (section, key) DO UPDATE SET "
                "value_json = excluded.value_json, updated_at = excluded.updated_at, "
                "actor = excluded.actor",
                (section, key, value_json, now.isoformat(), actor),
            )
            change_id = self._record(conn, now, section, key, "set", old, recorded, actor)
        return SettingChange(change_id, now, section, key, "set", old, recorded, actor)

    @with_locked_retry
    def clear(self, section: str, key: str, *, old: Any, new: Any, actor: str) -> SettingChange:
        """Drop the override (the default applies again) and record it as a reset.

        ``new`` is the default the setting returns to, supplied by the caller that knows
        it; the store only has the override.
        """
        self._ensure()
        now = datetime.now(UTC)
        with sqlite_conn(self.db_path) as conn:
            conn.execute("DELETE FROM overrides WHERE section = ? AND key = ?", (section, key))
            change_id = self._record(conn, now, section, key, "reset", old, new, actor)
        return SettingChange(change_id, now, section, key, "reset", old, new, actor)

    @staticmethod
    def _record(  # one history row
        conn: Any,
        now: datetime,
        section: str,
        key: str,
        action: str,
        old: Any,
        new: Any,
        actor: str,
    ) -> int:
        cursor = conn.execute(
            "INSERT INTO history (at, section, key, action, old_json, new_json, actor) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                now.isoformat(),
                section,
                key,
                action,
                json.dumps(old, sort_keys=True),
                json.dumps(new, sort_keys=True),
                actor,
            ),
        )
        return int(cursor.lastrowid or 0)
