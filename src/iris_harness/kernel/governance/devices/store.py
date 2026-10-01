"""DeviceStore — SQLite-backed paired devices and pairing codes (ADR-0117).

Holds hashes only. A device token or a pairing code is never written here in the
clear, so a copy of this file authenticates nobody: see ``service.py`` for what is
hashed and why a plain digest is enough for one and tolerable for the other.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from iris_harness.foundation.paths import governance_data_dir

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def default_devices_db_path() -> Path:
    """Beside the approvals store. Resolved per call, not at import, so a process
    that sets ``IRIS_HOME`` after importing this module still gets its own file."""
    return governance_data_dir() / "devices.db"


class DeviceNotFoundError(LookupError):
    """No device row with the given ID exists."""


@dataclass(frozen=True)
class DeviceRow:
    device_id: str
    name: str
    kind: str
    scope: str
    created_at: str
    last_seen_at: str | None
    revoked_at: str | None

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None


class DeviceStore:
    """Open-or-create the devices DB. Rows never carry the token hash outwards."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_devices_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if not self.db_path.exists():
            fd = os.open(str(self.db_path), os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        else:
            try:
                os.chmod(str(self.db_path), 0o600)
            except OSError as exc:  # pragma: no cover - non-POSIX or perms issue
                logger.warning("could not chmod %s to 0o600: %s", self.db_path, exc)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema)

    @staticmethod
    def _row(r: sqlite3.Row) -> DeviceRow:
        return DeviceRow(
            device_id=r["device_id"],
            name=r["name"],
            kind=r["kind"],
            scope=r["scope"],
            created_at=r["created_at"],
            last_seen_at=r["last_seen_at"],
            revoked_at=r["revoked_at"],
        )

    # ------------------------------------------------------------------
    # Pairing codes
    # ------------------------------------------------------------------

    def add_pairing_code(self, *, code_hash: str, scope: str, now: datetime, ttl: timedelta) -> str:
        """Store a fresh code and drop the dead ones. Returns the expiry (ISO)."""
        expires_at = (now + ttl).isoformat()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM pairing_codes WHERE used_at IS NOT NULL OR expires_at <= ?",
                (now.isoformat(),),
            )
            conn.execute(
                "INSERT INTO pairing_codes(code_hash, scope, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (code_hash, scope, now.isoformat(), expires_at),
            )
            conn.commit()
        return expires_at

    def claim(  # one transaction needs the code and the whole device
        self,
        *,
        code_hash: str,
        token_hash: str,
        name: str,
        kind: str,
        now: datetime,
        max_attempts: int,
    ) -> DeviceRow | None:
        """Spend a live code and create its device, in one transaction.

        ``None`` when the code is unknown, expired, already used, or voided by failed
        attempts. The UPDATE is the lock: of two claims racing on one code, exactly
        one sees ``rowcount == 1``.
        """
        now_iso = now.isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            spent = conn.execute(
                "UPDATE pairing_codes SET used_at = ? "
                "WHERE code_hash = ? AND used_at IS NULL AND expires_at > ? AND attempts < ?",
                (now_iso, code_hash, now_iso, max_attempts),
            )
            if spent.rowcount != 1:
                conn.rollback()
                return None
            scope = conn.execute(
                "SELECT scope FROM pairing_codes WHERE code_hash = ?", (code_hash,)
            ).fetchone()["scope"]
            device_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO devices(device_id, name, kind, scope, token_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (device_id, name, kind, scope, token_hash, now_iso),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,)).fetchone()
        return self._row(row)

    def record_failed_claim(self, *, now: datetime, max_attempts: int) -> int:
        """Charge a wrong guess to every live code; returns how many it just voided.

        A wrong code matches no row, so there is no "this code" to charge. Charging
        all live codes (normally one) is the fail-closed reading of "N attempts per
        code": guessing while a code is live kills the code, and the owner starts a
        new one.
        """
        now_iso = now.isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            # Counted before the increment: the codes on their last attempt are the
            # ones this guess voids. Counting `attempts = max` afterwards would
            # re-report a code voided by an earlier guess.
            voided = conn.execute(
                "SELECT COUNT(*) FROM pairing_codes "
                "WHERE used_at IS NULL AND expires_at > ? AND attempts = ?",
                (now_iso, max_attempts - 1),
            ).fetchone()[0]
            conn.execute(
                "UPDATE pairing_codes SET attempts = attempts + 1 "
                "WHERE used_at IS NULL AND expires_at > ? AND attempts < ?",
                (now_iso, max_attempts),
            )
            conn.commit()
        return int(voided)

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------

    def find_live(self, token_hash: str) -> DeviceRow | None:
        """The un-revoked device holding this token hash, if any."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM devices WHERE token_hash = ? AND revoked_at IS NULL",
                (token_hash,),
            ).fetchone()
        return self._row(row) if row is not None else None

    def touch(self, device_id: str, *, now: datetime, min_interval: timedelta) -> None:
        """Move ``last_seen_at`` forward, at most once per ``min_interval`` — a
        console polls every few seconds and each of those must not be a write."""
        stale_before = (now - min_interval).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE devices SET last_seen_at = ? "
                "WHERE device_id = ? AND (last_seen_at IS NULL OR last_seen_at < ?)",
                (now.isoformat(), device_id, stale_before),
            )
            conn.commit()

    def get(self, device_id: str) -> DeviceRow | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,)).fetchone()
        return self._row(row) if row is not None else None

    def list_devices(self) -> list[DeviceRow]:
        """Every device, revoked ones included (the list is also the history)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM devices ORDER BY created_at, device_id").fetchall()
        return [self._row(r) for r in rows]

    def revoke(self, device_id: str, *, now: datetime) -> tuple[DeviceRow, bool]:
        """Revoke a device. Returns the row and whether this call changed it, so a
        repeat revoke is a no-op rather than a second ledger entry."""
        with self._connect() as conn:
            changed = conn.execute(
                "UPDATE devices SET revoked_at = ? WHERE device_id = ? AND revoked_at IS NULL",
                (now.isoformat(), device_id),
            ).rowcount
            conn.commit()
            row = conn.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,)).fetchone()
        if row is None:
            raise DeviceNotFoundError(device_id)
        return self._row(row), changed == 1
