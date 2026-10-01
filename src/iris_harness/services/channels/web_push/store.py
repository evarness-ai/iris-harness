"""Push subscriptions, one row per browser that agreed to be notified.

Keyed by the endpoint URL, which is what the push service issues and what
identifies the subscription — a browser that re-subscribes gets a new endpoint
and the old row becomes dead weight, so ``save`` upserts on it.

``device_id`` ties a subscription to the paired device it came from
(ADR-0117), so revoking a lost phone in Devices takes its notifications with
it. It is nullable because a subscription outliving its device row is not a
reason to drop the row on the floor.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from iris_harness.foundation.clock import utc_now_iso
from iris_harness.foundation.paths import governance_data_dir
from iris_harness.foundation.persistence.sqlite import connect

logger = logging.getLogger(__name__)

DEFAULT_DB_FILENAME = "push-subscriptions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS push_subscriptions (
    endpoint      TEXT PRIMARY KEY,
    p256dh        TEXT NOT NULL,
    auth          TEXT NOT NULL,
    device_id     TEXT,
    label         TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    last_sent_at  TEXT,
    failures      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_push_device ON push_subscriptions(device_id);
"""


@dataclass(frozen=True)
class PushSubscription:
    endpoint: str
    p256dh: str
    auth: str
    device_id: str | None = None
    label: str = ""
    created_at: str = ""
    last_sent_at: str | None = None
    failures: int = 0


class PushSubscriptionStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (governance_data_dir() / DEFAULT_DB_FILENAME)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.db_path, row_factory=sqlite3.Row)  # WAL + busy_timeout
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
        # An endpoint is a bearer capability: anyone holding it can ask the
        # push service to deliver. Same 0600 the other stores reassert.
        try:
            os.chmod(self.db_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:  # pragma: no cover - a read-only mount is not fatal
            pass

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def save(self, sub: PushSubscription) -> PushSubscription:
        """Insert, or refresh the keys of an endpoint already known.

        A browser may hand back the same endpoint with rotated keys; taking
        the new ones and resetting the failure count is the difference between
        a device that keeps working and one that silently stops.
        """
        created = sub.created_at or utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO push_subscriptions
                    (endpoint, p256dh, auth, device_id, label, created_at, failures)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(endpoint) DO UPDATE SET
                    p256dh = excluded.p256dh,
                    auth = excluded.auth,
                    device_id = excluded.device_id,
                    label = excluded.label,
                    failures = 0
                """,
                (sub.endpoint, sub.p256dh, sub.auth, sub.device_id, sub.label, created),
            )
        return PushSubscription(**{**sub.__dict__, "created_at": created})

    def list(self) -> tuple[PushSubscription, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM push_subscriptions ORDER BY created_at ASC"
            ).fetchall()
        return tuple(PushSubscription(**dict(row)) for row in rows)

    def get(self, endpoint: str) -> PushSubscription | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM push_subscriptions WHERE endpoint = ?", (endpoint,)
            ).fetchone()
        return PushSubscription(**dict(row)) if row else None

    def delete(self, endpoint: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))
        return cur.rowcount > 0

    def delete_for_device(self, device_id: str) -> int:
        """Revoking a device takes its notifications with it."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM push_subscriptions WHERE device_id = ?", (device_id,))
        return cur.rowcount

    def note_sent(self, endpoint: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE push_subscriptions SET last_sent_at = ?, failures = 0 WHERE endpoint = ?",
                (utc_now_iso(), endpoint),
            )

    def note_failure(self, endpoint: str) -> int:
        with self._connect() as conn:
            conn.execute(
                "UPDATE push_subscriptions SET failures = failures + 1 WHERE endpoint = ?",
                (endpoint,),
            )
            row = conn.execute(
                "SELECT failures FROM push_subscriptions WHERE endpoint = ?", (endpoint,)
            ).fetchone()
        return int(row["failures"]) if row else 0


__all__ = ["DEFAULT_DB_FILENAME", "PushSubscription", "PushSubscriptionStore"]
