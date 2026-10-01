"""Which accounts the scheduled sweep waits on: an unfinished email setup's hold.

Owner decision (2026-09-30): an account whose email setup has not reached its "keep it
current" step (``enable_sweep``) is not swept on the schedule; that step turns the sweep
on. Setup does its own first fetch, so nothing the sweep lands can be judged, labelled
or digested ahead of the owner's review.

One row per account in ``email_sweep_gate`` (``email.db``, beside the mail and the
sweep's cursors), and three states:

* **no row** -- the sweep takes the account, exactly as before setup existed. The
  owner's accounts connected before ``iris email setup`` have none, so they never stop
  fetching;
* ``held`` -- setup began and has not turned the sweep on: the sweep passes the account
  by, names it in its run output and records the first skip once (log + audit row);
* ``on`` -- setup turned the sweep on (or found the account already being swept).

Email setup (``email_workflows/onboarding.py``) is the writer: it holds a new account
when its setup starts and releases it at ``enable_sweep``. A restart of setup drops
setup's state, not this row, so an account that was being swept keeps being swept.
The sweep (``email.sweep``) is the one reader that acts on it; the setup status and
System Health read it to say why an account is not swept.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from iris_harness.sdk.persistence import data_path, sqlite_conn

HELD = "held"
ON = "on"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS email_sweep_gate (
    account_id    TEXT PRIMARY KEY,
    state         TEXT NOT NULL,
    reason        TEXT NOT NULL DEFAULT '',
    updated_at    TEXT NOT NULL,
    skip_noted_at TEXT
);
"""


@dataclass(frozen=True)
class SweepGateEntry:
    account_id: str
    state: str
    reason: str
    updated_at: str
    # When the sweep first passed this held account by (logged and audited once).
    skip_noted_at: str | None = None

    @property
    def held(self) -> bool:
        return self.state == HELD


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SweepGate:
    """The ``email_sweep_gate`` table. ``db_path`` defaults to the data dir's
    ``email.db`` (the sweep passes its store's file; tests pass their own)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    @property
    def db_path(self) -> Path:
        return self._db_path or data_path("email.db")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite_conn(self.db_path, row_factory=sqlite3.Row) as conn:
            conn.executescript(_SCHEMA)
            yield conn

    def get(self, account_id: str) -> SweepGateEntry | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM email_sweep_gate WHERE account_id = ?", (account_id,)
            ).fetchone()
        return _entry(row) if row is not None else None

    def held(self) -> dict[str, SweepGateEntry]:
        """Held accounts by id."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM email_sweep_gate WHERE state = ? ORDER BY account_id", (HELD,)
            ).fetchall()
        return {row["account_id"]: _entry(row) for row in rows}

    def _set(self, account_id: str, state: str, reason: str) -> SweepGateEntry:
        if not account_id.strip():
            raise ValueError("an account id is required")
        entry = SweepGateEntry(account_id.strip(), state, reason, _now())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO email_sweep_gate (account_id, state, reason, updated_at, "
                "skip_noted_at) VALUES (?, ?, ?, ?, NULL) ON CONFLICT(account_id) DO UPDATE "
                "SET state = excluded.state, reason = excluded.reason, "
                "updated_at = excluded.updated_at, skip_noted_at = NULL",
                (entry.account_id, entry.state, entry.reason, entry.updated_at),
            )
        return entry

    def hold(self, account_id: str, reason: str) -> SweepGateEntry:
        """The sweep leaves ``account_id`` alone until :meth:`release`."""
        return self._set(account_id, HELD, reason)

    def release(self, account_id: str, reason: str) -> SweepGateEntry:
        """The sweep takes ``account_id`` again."""
        return self._set(account_id, ON, reason)

    def note_skip(self, account_id: str) -> bool:
        """Mark the first skip of a held account; True only that first time (so the
        sweep logs and audits it once, not every tick)."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE email_sweep_gate SET skip_noted_at = ? WHERE account_id = ? "
                "AND state = ? AND skip_noted_at IS NULL",
                (_now(), account_id, HELD),
            )
        return cur.rowcount > 0


def _entry(row: sqlite3.Row) -> SweepGateEntry:
    return SweepGateEntry(
        account_id=row["account_id"],
        state=row["state"],
        reason=row["reason"],
        updated_at=row["updated_at"],
        skip_noted_at=row["skip_noted_at"],
    )


def sweep_waits_for(account_id: str, *, gate: SweepGate | None = None) -> SweepGateEntry | None:
    """The hold that keeps the sweep off ``account_id``, or None when it is swept."""
    entry = (gate or SweepGate()).get(account_id)
    return entry if entry is not None and entry.held else None


__all__ = [
    "HELD",
    "ON",
    "SweepGate",
    "SweepGateEntry",
    "sweep_waits_for",
]
