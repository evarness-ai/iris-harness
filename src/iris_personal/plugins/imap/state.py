"""What the IMAP plugin remembers between runs, in its own SQLite file.

``imap_state.db`` under the data dir (``sdk.persistence.data_path``, so it follows
``IRIS_DATA_DIR`` / ``IRIS_HOME`` and a test never touches the owner's file). The sync
cursor itself lives with every provider's cursor in the email store; this keeps what
only IMAP needs:

* ``uid_map`` -- where each stored message is on the server: ``(folder, UIDVALIDITY,
  UID)`` plus its ``Message-ID``, so a write finds the message without a search, and a
  stale UID (UIDVALIDITY changed) is re-found by header. ``keywords`` is the IRIS
  keywords the last sync saw, which is how an owner's relabel is noticed.
* ``label_keywords`` -- the IRIS label name each keyword stands for.
* ``account_status`` -- the last login outcome, which System Health reports.

The approval that lets IRIS change a mailbox at all (R4) is not kept here: it is the
email library's, one gate for every provider (``iris_personal.email.write_approvals``).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from iris_harness.sdk.persistence import data_path, sqlite_conn

STATE_FILENAME = "imap_state.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS uid_map (
    account_id     TEXT NOT NULL,
    id             TEXT NOT NULL,
    folder         TEXT NOT NULL,
    uidvalidity    INTEGER NOT NULL,
    uid            INTEGER NOT NULL,
    rfc_message_id TEXT,
    home_folder    TEXT,
    keywords       TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (account_id, id)
);
CREATE INDEX IF NOT EXISTS idx_uid_map_uid ON uid_map(account_id, folder, uidvalidity, uid);

CREATE TABLE IF NOT EXISTS label_keywords (
    account_id TEXT NOT NULL,
    name       TEXT NOT NULL,
    keyword    TEXT NOT NULL,
    PRIMARY KEY (account_id, name)
);

CREATE TABLE IF NOT EXISTS account_status (
    account_id    TEXT PRIMARY KEY,
    last_ok_at    TEXT,
    last_error    TEXT,
    last_error_at TEXT,
    auth_failed   INTEGER NOT NULL DEFAULT 0
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Location:
    id: str
    folder: str
    uidvalidity: int
    uid: int
    rfc_message_id: str | None
    home_folder: str | None
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class AccountStatus:
    account_id: str
    last_ok_at: str | None
    last_error: str | None
    last_error_at: str | None
    auth_failed: bool


class ImapState:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        self._ready = False

    @property
    def db_path(self) -> Path:
        return self._db_path or data_path(STATE_FILENAME)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if not self._ready:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite_conn(self.db_path, row_factory=sqlite3.Row) as conn:
            if not self._ready:
                conn.executescript(_SCHEMA)
                self._ready = True
            yield conn

    # -- uid map -----------------------------------------------------------------

    def record(
        self,
        account_id: str,
        rows: Iterable[tuple[str, str, int, int, str | None, tuple[str, ...]]],
    ) -> None:
        """Upsert ``(id, folder, uidvalidity, uid, message-id, iris keywords)`` rows."""
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO uid_map (account_id, id, folder, uidvalidity, uid, rfc_message_id,"
                " keywords) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(account_id, id) DO UPDATE"
                " SET folder = excluded.folder, uidvalidity = excluded.uidvalidity,"
                " uid = excluded.uid, rfc_message_id = COALESCE(excluded.rfc_message_id,"
                " uid_map.rfc_message_id), keywords = excluded.keywords",
                [
                    (account_id, mid, folder, validity, uid, rfc, json.dumps(sorted(kw)))
                    for mid, folder, validity, uid, rfc, kw in rows
                ],
            )

    def moved(
        self,
        account_id: str,
        message_id: str,
        *,
        folder: str,
        uidvalidity: int,
        uid: int,
        home_folder: str | None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE uid_map SET folder = ?, uidvalidity = ?, uid = ?, home_folder = ?"
                " WHERE account_id = ? AND id = ?",
                (folder, uidvalidity, uid, home_folder, account_id, message_id),
            )

    def set_keywords(self, account_id: str, changes: dict[str, tuple[str, ...]]) -> None:
        with self._conn() as conn:
            conn.executemany(
                "UPDATE uid_map SET keywords = ? WHERE account_id = ? AND id = ?",
                [(json.dumps(sorted(kw)), account_id, mid) for mid, kw in changes.items()],
            )

    def locations(self, account_id: str, ids: Sequence[str]) -> dict[str, Location]:
        if not ids:
            return {}
        out: dict[str, Location] = {}
        with self._conn() as conn:
            for start in range(0, len(ids), 500):
                chunk = list(ids[start : start + 500])
                marks = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT * FROM uid_map WHERE account_id = ? AND id IN ({marks})",  # noqa: S608
                    (account_id, *chunk),
                ).fetchall()
                for row in rows:
                    out[row["id"]] = _location(row)
        return out

    def in_folder(self, account_id: str, folder: str, uidvalidity: int) -> dict[int, Location]:
        """``{uid: location}`` of every mapped message in one folder generation."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM uid_map WHERE account_id = ? AND folder = ? AND uidvalidity = ?",
                (account_id, folder, uidvalidity),
            ).fetchall()
        return {int(r["uid"]): _location(r) for r in rows}

    def with_keywords(self, account_id: str, folder: str) -> dict[str, Location]:
        """Mapped messages in ``folder`` that carried an IRIS keyword at the last sync."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM uid_map WHERE account_id = ? AND folder = ? AND keywords != '[]'",
                (account_id, folder),
            ).fetchall()
        return {r["id"]: _location(r) for r in rows}

    def forget_account(self, account_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM uid_map WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM label_keywords WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM account_status WHERE account_id = ?", (account_id,))

    # -- labels ------------------------------------------------------------------

    def remember_keywords(self, account_id: str, mapping: dict[str, str]) -> None:
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO label_keywords (account_id, name, keyword) VALUES (?, ?, ?)"
                " ON CONFLICT(account_id, name) DO UPDATE SET keyword = excluded.keyword",
                [(account_id, name, kw) for name, kw in mapping.items()],
            )

    def label_keywords(self, account_id: str) -> dict[str, str]:
        """``{keyword: label name}`` for the IRIS labels this account has used."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name, keyword FROM label_keywords WHERE account_id = ?", (account_id,)
            ).fetchall()
        return {r["keyword"]: r["name"] for r in rows}

    # -- login status ------------------------------------------------------------

    def login_ok(self, account_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO account_status (account_id, last_ok_at, auth_failed) VALUES (?, ?, 0)"
                " ON CONFLICT(account_id) DO UPDATE SET last_ok_at = excluded.last_ok_at,"
                " auth_failed = 0, last_error = NULL, last_error_at = NULL",
                (account_id, _now()),
            )

    def login_failed(self, account_id: str, error: str, *, auth: bool) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO account_status (account_id, last_error, last_error_at, auth_failed)"
                " VALUES (?, ?, ?, ?) ON CONFLICT(account_id) DO UPDATE SET"
                " last_error = excluded.last_error, last_error_at = excluded.last_error_at,"
                " auth_failed = excluded.auth_failed",
                (account_id, error, _now(), int(auth)),
            )

    def status(self, account_id: str) -> AccountStatus | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM account_status WHERE account_id = ?", (account_id,)
            ).fetchone()
        if row is None:
            return None
        return AccountStatus(
            account_id=account_id,
            last_ok_at=row["last_ok_at"],
            last_error=row["last_error"],
            last_error_at=row["last_error_at"],
            auth_failed=bool(row["auth_failed"]),
        )


def _location(row: sqlite3.Row) -> Location:
    return Location(
        id=row["id"],
        folder=row["folder"],
        uidvalidity=int(row["uidvalidity"]),
        uid=int(row["uid"]),
        rfc_message_id=row["rfc_message_id"],
        home_folder=row["home_folder"],
        keywords=tuple(json.loads(row["keywords"] or "[]")),
    )


__all__ = ["STATE_FILENAME", "AccountStatus", "ImapState", "Location"]
