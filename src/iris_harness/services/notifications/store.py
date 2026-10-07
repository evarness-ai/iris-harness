"""SQLite-backed store for reminders — the one reminder store (loop-proof D14).

Schema lives in ``data/tasks.db`` next to ``tasks`` and ``goals``. A row is "fire at
time T about target X" plus its own delivery lifecycle (see ``models``): the heartbeat
claims due rows, the channels layer sends them, and the store records whether any
channel accepted (``sent``) or the attempt failed (retried 3 times, 5 minutes apart,
then ``failed``). Repeating rows spawn their next occurrence as a new row of the same
``series_id``.

Time: every instant is stored as a UTC ISO string with a ``+00:00`` offset, so the
due query's string comparison is a time comparison. A naive datetime handed in is
read as UTC (as the ``iris reminder add`` CLI always has); callers that parse the
owner's words pass aware datetimes in ``IRIS_TZ``.

Bus: ``reminder.fired`` after an accepted send, ``reminder.completed`` on done,
``reminder.snoozed`` on a snooze, ``reminder.reopened`` on an undone Done (each with the
surface the owner used). Tests construct ``ReminderStore`` with ``bus=None`` for silent
operation.

Generated reminders (loop-proof PR 4): a ``dedupe_key`` is unique, so a generator that
runs every day (the bill schedule: ``bill:<due id>:t3d``) creates each row once, ever —
``create`` hands back the existing row instead of a second one. ``close_for_target``
ends every open row about one target at once (its bill was paid), with a reason that
starts ``closed:``; ``reopen_for_target`` re-arms the future ones when it is reopened.
"""

from __future__ import annotations

import builtins
import json
import logging
import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Any

from iris_harness.foundation.clock import as_utc, utc_now
from iris_harness.foundation.eventbus import EventBus
from iris_harness.foundation.persistence import connect, data_path
from iris_harness.foundation.persistence.sqlite import ensure_columns

from .bills import (
    DELIVERED_INFO_REASON,
    REOPENED_REASON,
    BillAlreadyPaid,
    asks_nothing,
    closed_as_paid,
    payable_after_end,
)
from .events import (
    REMINDER_ACKNOWLEDGED,
    REMINDER_COMPLETED,
    REMINDER_FIRED,
    REMINDER_REOPENED,
    REMINDER_SNOOZED,
    ReminderAcknowledgedPayload,
    ReminderCompletedPayload,
    ReminderFiredPayload,
    ReminderReopenedPayload,
    ReminderSnoozedPayload,
)
from .models import TERMINAL_STATUSES, Reminder, TargetKind
from .recurrence import next_occurrence_after, validate_rule

logger = logging.getLogger(__name__)

#: Sends per occurrence: the first at ``remind_at`` and three retries (D14).
MAX_ATTEMPTS = 4
#: Gap between two sends of one occurrence; also the lease on a claimed row, so a
#: process that dies mid-send leaves a row the next tick retries rather than loses.
RETRY_DELAY = timedelta(minutes=5)

_SNOOZABLE = frozenset({"pending", "sent", "failed"})

#: ``closed_reason`` of a row ended by ``close_for_target`` (its target was closed).
CLOSED_FOR_TARGET = "closed:"
#: ``closed_reason`` of a row the owner answered "Not yet" (acknowledged, PR 4).
NOT_YET = "not yet"
#: Ended rows ``reopen`` may bring back besides ``done``: the owner said "not paid"
#: after a payment email closed the bill, or took back a "Not yet".
_REOPENABLE = (CLOSED_FOR_TARGET, NOT_YET)
_DAY_STATUSES = ("pending", "sending", "sent")


def _iso(dt: datetime | None) -> str | None:
    return as_utc(dt).isoformat() if dt is not None else None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _row_to_reminder(row: sqlite3.Row) -> Reminder:
    keys = set(row.keys())

    def col(name: str, default: Any = None) -> Any:
        return row[name] if name in keys and row[name] is not None else default

    return Reminder(
        id=row["id"],
        target_kind=row["target_kind"],
        target_id=row["target_id"],
        remind_at=_parse_dt(row["remind_at"]) or utc_now(),
        channel=row["channel"] or "default",
        note=row["note"] or "",
        created_at=_parse_dt(row["created_at"]) or utc_now(),
        fired_at=_parse_dt(row["fired_at"]),
        dismissed_at=_parse_dt(row["dismissed_at"]),
        status=col("status", "pending"),
        attempts=int(col("attempts", 0)),
        next_attempt_at=_parse_dt(col("next_attempt_at")),
        last_error=col("last_error"),
        delivered_channels=tuple(json.loads(col("delivered_channels", "[]"))),
        message_refs=tuple(json.loads(col("message_refs", "[]"))),
        recurrence=col("recurrence"),
        recur_until=_parse_dt(col("recur_until")),
        series_id=col("series_id"),
        missed_digests=int(col("missed_digests", 0)),
        closed_reason=col("closed_reason"),
        dedupe_key=col("dedupe_key"),
        meta=_decode_meta(col("meta")),
    )


def _decode_meta(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except ValueError:
        return {}
    return {str(k): str(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}


@dataclass
class ReminderStore:
    """SQLite-backed reminder store.

    Pass ``bus=None`` (the default) for silent operation. ``tz`` is the owner's zone a
    repeating reminder keeps its wall time in; ``None`` reads ``IRIS_TZ`` when needed.
    """

    db_path: Path = field(default_factory=lambda: data_path("tasks.db"))
    bus: EventBus | None = None
    tz: tzinfo | None = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the table, or add the D14 columns to an older one (idempotent)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
        # The D14 columns, then the indexes that reference them, on a connection of their own
        # under BEGIN IMMEDIATE (#201): a read of ``table_info`` then an ``ALTER`` raised
        # "duplicate column name" for the process that lost a race to open an older file. The
        # one-time lifecycle backfill runs inside the transaction that adds ``status``, so only
        # the process that added it does it, once.
        ensure_columns(
            self.db_path,
            "notification_reminders",
            dict(_ADDED_COLUMNS),
            indexes=_LATE_INDEX_STATEMENTS,
            on_added=_backfill_when_status_added,
        )

    def _connect(self) -> sqlite3.Connection:
        conn = connect(self.db_path, row_factory=sqlite3.Row)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _zone(self) -> tzinfo:
        if self.tz is not None:
            return self.tz
        from iris_harness.services.digest.settings import iris_timezone

        return iris_timezone()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        target_kind: TargetKind,
        target_id: str,
        remind_at: datetime,
        channel: str = "default",
        note: str = "",
        recurrence: str | None = None,
        recur_until: datetime | None = None,
        series_id: str | None = None,
        dedupe_key: str | None = None,
        meta: dict[str, str] | None = None,
    ) -> Reminder:
        """A new row. With a ``dedupe_key`` already in the store, the row that holds
        it comes back unchanged — nothing is created and nothing raises (PR 4)."""
        if dedupe_key is not None:
            existing = self.get_by_dedupe_key(dedupe_key)
            if existing is not None:
                return existing
        rule = validate_rule(recurrence)
        reminder_id = str(uuid.uuid4())
        reminder = Reminder(
            id=reminder_id,
            target_kind=target_kind,
            target_id=target_id,
            remind_at=as_utc(remind_at),
            channel=channel,
            note=note,
            recurrence=rule,
            recur_until=as_utc(recur_until) if recur_until is not None else None,
            series_id=series_id or (reminder_id if rule else None),
            dedupe_key=dedupe_key,
            meta=dict(meta or {}),
        )
        try:
            with self._connect() as conn:
                conn.execute(_INSERT_SQL, _to_row(reminder))
        except sqlite3.IntegrityError:
            # Another writer took the key between the read and the insert.
            raced = self.get_by_dedupe_key(dedupe_key) if dedupe_key is not None else None
            if raced is None:
                raise
            return raced
        return reminder

    def get_by_dedupe_key(self, dedupe_key: str) -> Reminder | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notification_reminders WHERE dedupe_key = ?", (dedupe_key,)
            ).fetchone()
        return _row_to_reminder(row) if row else None

    def refresh(self, reminder_id: str, *, note: str, meta: dict[str, str]) -> bool:
        """New words for a row still waiting to be sent (a bill's amount changed).
        ``False`` — and nothing written — once it is being sent, was sent or ended."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE notification_reminders SET note = ?, meta = ? "
                "WHERE id = ? AND status = 'pending' AND fired_at IS NULL",
                (note, json.dumps(dict(meta)), reminder_id),
            )
            return cur.rowcount == 1

    def close_for_target(
        self,
        target_kind: TargetKind,
        target_id: str,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> int:
        """End every open row about one target (its bill was paid, PR 4): rows not
        yet sent (``pending``, ``sending``) are cancelled, delivered or undeliverable
        ones (``sent``, ``failed``) expire. ``reason`` is kept, prefixed ``closed:``
        when it is not, so ``reopen_for_target`` can find them. Returns the count."""
        why = reason if reason.startswith(CLOSED_FOR_TARGET) else f"{CLOSED_FOR_TARGET} {reason}"
        stamp = _iso(now or utc_now())
        with self._connect() as conn:
            cancelled = conn.execute(
                "UPDATE notification_reminders SET status = 'cancelled', dismissed_at = ?, "
                "next_attempt_at = NULL, closed_reason = ? "
                "WHERE target_kind = ? AND target_id = ? AND status IN ('pending', 'sending')",
                (stamp, why, target_kind, target_id),
            ).rowcount
            expired = conn.execute(
                "UPDATE notification_reminders SET status = 'expired', "
                "next_attempt_at = NULL, closed_reason = ? "
                "WHERE target_kind = ? AND target_id = ? AND status IN ('sent', 'failed')",
                (why, target_kind, target_id),
            ).rowcount
            # Rows that already ended while the target was still open (aged out
            # unanswered, a Not yet, a reopen) now say it closed: Paid on one of their
            # messages is "already marked paid", not a second close. Not counted.
            conn.execute(
                "UPDATE notification_reminders SET closed_reason = ? "
                "WHERE target_kind = ? AND target_id = ? AND status = 'expired' "
                "AND (closed_reason LIKE 'expired: delivered%' OR closed_reason LIKE ? "
                "OR closed_reason LIKE ?)",
                (why, target_kind, target_id, NOT_YET + "%", REOPENED_REASON + "%"),
            )
        return int(cancelled) + int(expired)

    def reopen_for_target(
        self, target_kind: TargetKind, target_id: str, *, now: datetime | None = None
    ) -> int:
        """Re-arm the rows ``close_for_target`` ended whose time is still ahead: back
        to ``pending`` with a fresh set of attempts. Past ones stay ended — a reopened
        bill never replays what it would have sent — but no longer say it closed
        (``expired: bill reopened``, and so does a past Done): Paid on one of their
        messages closes the reopened bill again. Returns the count re-armed."""
        cutoff = _iso(now or utc_now())
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE notification_reminders SET status = 'pending', dismissed_at = NULL, "
                "fired_at = NULL, attempts = 0, next_attempt_at = NULL, last_error = NULL, "
                "closed_reason = NULL, missed_digests = 0 "
                "WHERE target_kind = ? AND target_id = ? AND closed_reason LIKE ? "
                "AND status IN ('cancelled', 'expired') AND remind_at > ?",
                (target_kind, target_id, CLOSED_FOR_TARGET + "%", cutoff),
            )
            conn.execute(
                "UPDATE notification_reminders SET status = 'expired', "
                "next_attempt_at = NULL, closed_reason = ? "
                "WHERE target_kind = ? AND target_id = ? AND fired_at IS NOT NULL "
                "AND ((status = 'expired' AND closed_reason LIKE ?) OR status = 'done')",
                (REOPENED_REASON, target_kind, target_id, CLOSED_FOR_TARGET + "%"),
            )
            return int(cur.rowcount)

    def acknowledge(self, reminder_id: str, *, source: str = "api") -> Reminder:
        """The owner's "Not yet" on a bill's question (PR 4): the row ends as seen
        (``expired``, ``not yet: <source>``) — nothing is snoozed, the next question
        is its own row. Undone with ``reopen``. ``reminder.acknowledged`` is emitted,
        so the target's domain hears it (finance records the Not yet on the due)."""
        existing = self._require(reminder_id)
        if existing.status in TERMINAL_STATUSES:
            raise ValueError(f"reminder {reminder_id} already ended ({existing.status})")
        update = {
            "status": "expired",
            "next_attempt_at": None,
            "closed_reason": f"{NOT_YET}: {source}",
        }
        self._update(reminder_id, update)
        self._emit(
            REMINDER_ACKNOWLEDGED,
            ReminderAcknowledgedPayload(
                reminder_id=existing.id,
                target_kind=existing.target_kind,
                target_id=existing.target_id,
                source=source,
            ),
        )
        return existing.model_copy(update=update)

    def get(self, reminder_id: str) -> Reminder | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notification_reminders WHERE id = ?", (reminder_id,)
            ).fetchone()
        return _row_to_reminder(row) if row else None

    def _require(self, reminder_id: str) -> Reminder:
        existing = self.get(reminder_id)
        if existing is None:
            raise KeyError(f"reminder not found: {reminder_id}")
        return existing

    def list(
        self,
        *,
        target_kind: TargetKind | None = None,
        target_id: str | None = None,
        include_fired: bool = False,
        include_dismissed: bool = False,
        statuses: Iterable[str] | None = None,
        limit: int = 100,
    ) -> list[Reminder]:
        clauses: list[str] = []
        params: list[Any] = []
        if target_kind is not None:
            clauses.append("target_kind = ?")
            params.append(target_kind)
        if target_id is not None:
            clauses.append("target_id = ?")
            params.append(target_id)
        if not include_fired:
            clauses.append("fired_at IS NULL")
        if not include_dismissed:
            clauses.append("dismissed_at IS NULL")
        if statuses is not None:
            wanted = builtins.list(statuses)
            clauses.append(f"status IN ({','.join('?' * len(wanted))})")
            params.extend(wanted)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT * FROM notification_reminders {where} "  # noqa: S608
            "ORDER BY remind_at ASC LIMIT ?"
        )
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_reminder(r) for r in rows]

    def list_due(self, *, now: datetime | None = None, limit: int = 200) -> builtins.list[Reminder]:
        """Rows to send now, oldest first: pending rows whose time (``remind_at``, or
        the scheduled retry) has come, and claimed rows whose send lease ran out."""
        cutoff = _iso(now or utc_now())
        with self._connect() as conn:
            rows = conn.execute(
                _DUE_SQL + " ORDER BY remind_at ASC LIMIT ?", (cutoff, cutoff, limit)
            )
            found = rows.fetchall()
        return [_row_to_reminder(r) for r in found]

    def reschedule(self, reminder_id: str, *, remind_at: datetime) -> Reminder:
        """Move an unsent reminder. Raises once it has been sent or has ended (snooze
        moves a sent one; a new reminder replaces an ended one)."""
        existing = self._require(reminder_id)
        if existing.status != "pending" or existing.fired_at is not None:
            raise ValueError(
                f"reminder {reminder_id} is terminal or already sent "
                f"(status={existing.status}, fired_at={existing.fired_at})"
            )
        when = as_utc(remind_at)
        with self._connect() as conn:
            conn.execute(
                "UPDATE notification_reminders SET remind_at = ?, next_attempt_at = NULL "
                "WHERE id = ?",
                (_iso(when), reminder_id),
            )
        return existing.model_copy(update={"remind_at": when, "next_attempt_at": None})

    def cancel(self, reminder_id: str) -> Reminder:
        """Mark a reminder cancelled (terminal — never fires again).

        Idempotent: an ended reminder returns unchanged. A sent one is left as it is —
        delivery has already happened; the owner closes it with ``complete``.
        """
        existing = self._require(reminder_id)
        if existing.status in TERMINAL_STATUSES or existing.status == "sent":
            return existing
        stamp = utc_now()
        update = {
            "status": "cancelled",
            "dismissed_at": stamp,
            "next_attempt_at": None,
            "closed_reason": "cancelled",
        }
        self._update(reminder_id, update)
        return existing.model_copy(update=update)

    # ------------------------------------------------------------------
    # Lifecycle (D14 / D18)
    # ------------------------------------------------------------------

    def complete(
        self, reminder_id: str, *, source: str = "owner", now: datetime | None = None
    ) -> Reminder:
        """The owner's Done: status ``done``, ``reminder.completed`` emitted, and a
        repeating reminder's next occurrence exists afterwards. Idempotent on ``done``;
        raises on a cancelled or expired row — except a bill's (PR 4): Paid on a bill
        row that ended while its bill stayed open (aged out unanswered, a Not yet, a
        reopen) still closes the bill, and one whose bill is already closed as paid
        raises :class:`~.bills.BillAlreadyPaid`."""
        existing = self._require(reminder_id)
        if existing.status == "done":
            return existing
        if existing.status in TERMINAL_STATUSES:
            if closed_as_paid(existing):
                raise BillAlreadyPaid(reminder_id, existing.meta.get("entity", ""))
            if not payable_after_end(existing):
                raise ValueError(f"reminder {reminder_id} already ended ({existing.status})")
        update = {"status": "done", "next_attempt_at": None, "closed_reason": f"done: {source}"}
        self._update(reminder_id, update)
        done = existing.model_copy(update=update)
        self.ensure_next_occurrence(done, now=now)
        from .targets import reminder_title

        self._emit(
            REMINDER_COMPLETED,
            ReminderCompletedPayload(
                reminder_id=done.id,
                task=reminder_title(done, self.db_path),
                source=source,
                target_kind=done.target_kind,
                target_id=done.target_id,
            ),
        )
        return done

    def snooze(self, reminder_id: str, until: datetime, *, source: str = "api") -> Reminder:
        """Fire again at ``until``: back to ``pending`` with a fresh set of attempts, and
        ``reminder.snoozed`` emitted. Allowed from ``pending``, ``sent`` and ``failed``."""
        existing = self._require(reminder_id)
        if existing.status not in _SNOOZABLE:
            raise ValueError(f"reminder {reminder_id} cannot be snoozed ({existing.status})")
        update: dict[str, Any] = {
            "status": "pending",
            "remind_at": as_utc(until),
            "attempts": 0,
            "next_attempt_at": None,
            "last_error": None,
            "fired_at": None,
            "missed_digests": 0,
            "closed_reason": None,
        }
        self._update(reminder_id, update)
        snoozed = existing.model_copy(update=update)
        self._emit(
            REMINDER_SNOOZED,
            ReminderSnoozedPayload(
                reminder_id=snoozed.id,
                series_id=snoozed.series_id,
                from_at=existing.remind_at,
                until=snoozed.remind_at,
                source=source,
            ),
        )
        return snoozed

    def reopen(
        self, reminder_id: str, *, now: datetime | None = None, source: str = "api"
    ) -> Reminder:
        """Undo a Done: back to ``sent`` — or ``pending`` when its time is still ahead,
        so it fires then — and ``reminder.reopened`` emitted. A row its target's close
        ended (``closed: …``, a bill a payment email marked paid) or a "Not yet" comes
        back the same way (PR 4). An open row is returned unchanged; any other
        cancelled or expired one raises. A repeating series keeps the next occurrence
        Done left in place."""
        existing = self._require(reminder_id)
        reason = existing.closed_reason or ""
        reopenable = existing.status == "done" or (
            existing.status in TERMINAL_STATUSES and reason.startswith(_REOPENABLE)
        )
        if not reopenable:
            if existing.status in TERMINAL_STATUSES:
                raise ValueError(f"reminder {reminder_id} already ended ({existing.status})")
            return existing
        future = existing.remind_at > as_utc(now or utc_now())
        update: dict[str, Any] = {
            "status": "pending" if future or existing.fired_at is None else "sent",
            "next_attempt_at": None,
            "closed_reason": None,
            "dismissed_at": None,
        }
        if update["status"] == "pending":
            update.update({"attempts": 0, "fired_at": None, "last_error": None})
        self._update(reminder_id, update)
        reopened = existing.model_copy(update=update)
        self._emit(
            REMINDER_REOPENED,
            ReminderReopenedPayload(
                reminder_id=reopened.id,
                target_kind=reopened.target_kind,
                target_id=reopened.target_id,
                source=source,
                closed_reason=reason,
            ),
        )
        return reopened

    def unsnooze(
        self, reminder_id: str, remind_at: datetime, *, now: datetime | None = None
    ) -> Reminder:
        """Undo a Snooze: due at ``remind_at`` again. A time already past is recorded as
        delivered (``sent``) rather than fired a second time; a row that is no longer
        waiting (it fired again, or was closed) is returned unchanged."""
        existing = self._require(reminder_id)
        if existing.status != "pending":
            return existing
        when = as_utc(remind_at)
        stamp = as_utc(now or utc_now())
        update: dict[str, Any] = {"remind_at": when, "next_attempt_at": None}
        if when <= stamp:
            update.update({"status": "sent", "fired_at": when})
        self._update(reminder_id, update)
        return existing.model_copy(update=update)

    def expire(self, reminder_id: str, reason: str, *, now: datetime | None = None) -> Reminder:
        """Age a reminder out (D18): terminal, kept, with its reason. Ended rows are
        returned unchanged. A repeating series carries on with its next occurrence."""
        existing = self._require(reminder_id)
        if existing.status in TERMINAL_STATUSES:
            return existing
        update = {"status": "expired", "next_attempt_at": None, "closed_reason": reason}
        self._update(reminder_id, update)
        expired = existing.model_copy(update=update)
        self.ensure_next_occurrence(expired, now=now)
        return expired

    def find_by_message(self, channel: str, chat_id: str, message_id: str) -> Reminder | None:
        """The reminder a sent message belongs to (``message_refs``), newest first — so
        a reply to the reminder's message can act on it."""
        candidates = self._select(
            "WHERE message_refs LIKE ? ORDER BY remind_at DESC",
            (f"%{json.dumps(str(message_id))}%",),
        )
        for reminder in candidates:
            for ref in reminder.message_refs:
                if (
                    str(ref.get("channel", "")) == channel
                    and str(ref.get("chat_id", "")) == str(chat_id)
                    and str(ref.get("message_id", "")) == str(message_id)
                ):
                    return reminder
        return None

    def list_missed(self, now: datetime | None = None) -> builtins.list[Reminder]:
        """Reminders no channel accepted (``failed``, so not yet expired), oldest first.
        ``now`` is accepted for symmetry with the digest's other reads."""
        del now
        return self._select("WHERE status = 'failed' ORDER BY remind_at ASC", ())

    def list_sent_before(self, cutoff: datetime) -> builtins.list[Reminder]:
        """Delivered rows the owner never acted on, due before ``cutoff``, oldest first
        (the digest sweep expires them at the start of the owner's day, D18)."""
        return self._select(
            "WHERE status = 'sent' AND remind_at < ? ORDER BY remind_at ASC", (_iso(cutoff),)
        )

    def mark_shown_in_digest(self, ids: Iterable[str]) -> None:
        """Count one digest appearance for each reminder (D18: missed shows once)."""
        wanted = builtins.list(ids)
        if not wanted:
            return
        with self._connect() as conn:
            conn.executemany(
                "UPDATE notification_reminders SET missed_digests = missed_digests + 1 "
                "WHERE id = ?",
                [(i,) for i in wanted],
            )

    def list_for_day(self, day: date, tz: tzinfo | None = None) -> builtins.list[Reminder]:
        """Open reminders (pending, sending or sent) whose ``remind_at`` falls on the
        local ``day`` in ``tz`` (default ``IRIS_TZ``), by time."""
        zone = tz or self._zone()
        start = datetime.combine(day, time(0), tzinfo=zone)
        end = datetime.combine(day + timedelta(days=1), time(0), tzinfo=zone)
        marks = ",".join("?" * len(_DAY_STATUSES))
        return self._select(
            f"WHERE status IN ({marks}) AND remind_at >= ? AND remind_at < ? "
            "ORDER BY remind_at ASC",
            (*_DAY_STATUSES, _iso(start), _iso(end)),
        )

    # ------------------------------------------------------------------
    # Delivery (driven by services.notifications.channels.deliver_due)
    # ------------------------------------------------------------------

    def claim(self, reminder_id: str, *, now: datetime | None = None) -> Reminder | None:
        """Take a due row for one send: ``sending``, one more attempt, and a lease of
        ``RETRY_DELAY``. ``None`` when another tick took it first or it is not due."""
        when = now or utc_now()
        cutoff = _iso(when)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE notification_reminders SET status = 'sending', "  # noqa: S608
                "attempts = attempts + 1, next_attempt_at = ? "
                f"WHERE id = ? AND ({_DUE_WHERE})",
                (_iso(when + RETRY_DELAY), reminder_id, cutoff, cutoff),
            )
            claimed = cur.rowcount == 1
        return self.get(reminder_id) if claimed else None

    def mark_sent(
        self,
        reminder_id: str,
        *,
        delivered_channels: Iterable[str],
        message_refs: Iterable[dict[str, str]] = (),
        errors: str | None = None,
        now: datetime | None = None,
    ) -> tuple[Reminder, Reminder | None]:
        """A channel accepted the send: ``sent``, ``reminder.fired`` emitted, and for a
        repeating reminder the next occurrence created. Returns (sent row, next row)."""
        existing = self._require(reminder_id)
        fired_at = now or utc_now()
        if errors:
            logger.info("reminder %s sent; other channels: %s", reminder_id, errors)
        update: dict[str, Any] = {
            "status": "sent",
            "fired_at": fired_at,
            "next_attempt_at": None,
            # Accepted means delivered: a channel skipped on the way (no web-push
            # subscription, say) is not an error of this reminder — logged, not kept.
            "last_error": None,
            "delivered_channels": tuple(delivered_channels),
            "message_refs": tuple(message_refs),
        }
        if asks_nothing(existing):
            # A bill's "marked paid" confirmation: delivered is the end of it — never
            # swept later as "delivered, not acknowledged" (PR 4).
            update.update({"status": "expired", "closed_reason": DELIVERED_INFO_REASON})
        self._update(reminder_id, update)
        sent = existing.model_copy(update=update)
        self._emit(
            REMINDER_FIRED,
            ReminderFiredPayload(
                reminder_id=sent.id,
                target_kind=sent.target_kind,
                target_id=sent.target_id,
                remind_at=sent.remind_at,
                fired_at=fired_at,
                channel=sent.channel,
                note=sent.note,
            ),
        )
        return sent, self.ensure_next_occurrence(sent, now=fired_at)

    def mark_attempt_failed(
        self,
        reminder_id: str,
        *,
        error: str,
        now: datetime | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        retry_delay: timedelta = RETRY_DELAY,
    ) -> Reminder:
        """No channel accepted this send. Before ``max_attempts`` the row waits
        ``retry_delay`` as ``pending``; at it the row is ``failed`` (and a repeating
        series still gets its next occurrence)."""
        existing = self._require(reminder_id)
        when = now or utc_now()
        update: dict[str, Any]
        if existing.attempts >= max_attempts:
            update = {"status": "failed", "next_attempt_at": None, "last_error": error}
        else:
            update = {
                "status": "pending",
                "next_attempt_at": when + retry_delay,
                "last_error": error,
            }
        self._update(reminder_id, update)
        updated = existing.model_copy(update=update)
        if updated.status == "failed":
            self.ensure_next_occurrence(updated, now=when)
        return updated

    def ensure_next_occurrence(
        self, reminder: Reminder, *, now: datetime | None = None
    ) -> Reminder | None:
        """Create the series' next row after ``reminder`` unless one already waits.

        The next slot is the first after both this occurrence and ``now``, so a series
        that was stuck resumes at its next future time instead of firing a burst.
        """
        if not reminder.recurrence or not reminder.series_id:
            return None
        with self._connect() as conn:
            waiting = conn.execute(
                "SELECT 1 FROM notification_reminders WHERE series_id = ? AND id != ? "
                "AND status IN ('pending', 'sending') AND remind_at > ? LIMIT 1",
                (reminder.series_id, reminder.id, _iso(reminder.remind_at)),
            ).fetchone()
            first = conn.execute(
                "SELECT MIN(remind_at) FROM notification_reminders WHERE series_id = ?",
                (reminder.series_id,),
            ).fetchone()
        if waiting:
            return None
        zone = self._zone()
        anchor_at = _parse_dt(first[0]) if first and first[0] else reminder.remind_at
        anchor = (anchor_at or reminder.remind_at).astimezone(zone).date()
        not_before = max(as_utc(now or utc_now()), reminder.remind_at)
        nxt = next_occurrence_after(
            reminder.recurrence,
            reminder.remind_at,
            zone,
            not_before=not_before,
            until=reminder.recur_until,
            anchor=anchor,
        )
        if nxt is None:
            return None
        return self.create(
            target_kind=reminder.target_kind,
            target_id=reminder.target_id,
            remind_at=nxt,
            channel=reminder.channel,
            note=reminder.note,
            recurrence=reminder.recurrence,
            recur_until=reminder.recur_until,
            series_id=reminder.series_id,
        )

    # ------------------------------------------------------------------
    # Legacy direct fire (the `iris reminder tick` CLI; no channel send)
    # ------------------------------------------------------------------

    def fire(self, reminder_id: str) -> Reminder:
        """Mark a reminder sent and emit ``reminder.fired`` without sending anything.

        Idempotent on already-sent rows. Raises on cancelled rows. The heartbeat does
        not use this: it sends first and records only what a channel accepted.
        """
        existing = self._require(reminder_id)
        if existing.dismissed_at is not None or existing.status == "cancelled":
            raise ValueError(f"reminder {reminder_id} was dismissed; refusing to fire")
        if existing.fired_at is not None:
            return existing
        sent, _ = self.mark_sent(reminder_id, delivered_channels=())
        return sent

    def tick(self, *, now: datetime | None = None) -> builtins.list[Reminder]:
        """Mark every due reminder fired in one pass (CLI). Returns the fired rows."""
        fired: list[Reminder] = []
        for reminder in self.list_due(now=now):
            try:
                fired.append(self.fire(reminder.id))
            except Exception:  # per-row soft-fail
                logger.exception("reminder fire failed: id=%s", reminder.id)
        return fired

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select(self, tail: str, params: tuple[Any, ...]) -> builtins.list[Reminder]:
        with self._connect() as conn:
            # `tail` is always a constant clause from this module; values are bound.
            sql = f"SELECT * FROM notification_reminders {tail}"  # noqa: S608
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_reminder(r) for r in rows]

    def _update(self, reminder_id: str, values: dict[str, Any]) -> None:
        columns = builtins.list(values)
        encoded = [_encode(c, values[c]) for c in columns]
        assignments = ", ".join(f"{c} = ?" for c in columns)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE notification_reminders SET {assignments} WHERE id = ?",  # noqa: S608
                (*encoded, reminder_id),
            )

    def _emit(self, topic: str, payload: Any) -> None:
        if self.bus is None:
            return
        self.bus.emit_sync(topic, payload)


def _encode(column: str, value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso(value)
    if column in ("delivered_channels", "message_refs"):
        return json.dumps(builtins.list(value or ()))
    if column == "meta":
        return json.dumps(dict(value or {}))
    return value


def _backfill_lifecycle(conn: sqlite3.Connection) -> None:
    """One-time fill for rows written before D14: a status from the old stamps, and
    every instant rewritten as UTC so the due query compares like with like."""
    conn.execute(
        "UPDATE notification_reminders SET status = 'cancelled', closed_reason = 'cancelled' "
        "WHERE dismissed_at IS NOT NULL"
    )
    conn.execute(
        "UPDATE notification_reminders SET status = 'sent' "
        "WHERE dismissed_at IS NULL AND fired_at IS NOT NULL"
    )
    rows = conn.execute("SELECT id, remind_at FROM notification_reminders").fetchall()
    for row in rows:
        try:
            normalised = _iso(datetime.fromisoformat(row[1]))
        except (TypeError, ValueError):
            continue
        if normalised != row[1]:
            conn.execute(
                "UPDATE notification_reminders SET remind_at = ? WHERE id = ?",
                (normalised, row[0]),
            )


def _to_row(reminder: Reminder) -> dict[str, Any]:
    return {
        "id": reminder.id,
        "target_kind": reminder.target_kind,
        "target_id": reminder.target_id,
        "remind_at": _iso(reminder.remind_at),
        "channel": reminder.channel,
        "note": reminder.note,
        "created_at": _iso(reminder.created_at),
        "fired_at": _iso(reminder.fired_at),
        "dismissed_at": _iso(reminder.dismissed_at),
        "status": reminder.status,
        "attempts": reminder.attempts,
        "recurrence": reminder.recurrence,
        "recur_until": _iso(reminder.recur_until),
        "series_id": reminder.series_id,
        "dedupe_key": reminder.dedupe_key,
        "meta": json.dumps(dict(reminder.meta)) if reminder.meta else None,
    }


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS notification_reminders (
    id            TEXT PRIMARY KEY,
    target_kind   TEXT NOT NULL,
    target_id     TEXT NOT NULL,
    remind_at     TEXT NOT NULL,
    channel       TEXT NOT NULL DEFAULT 'default',
    note          TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    fired_at      TEXT,
    dismissed_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_notification_reminders_target
    ON notification_reminders(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_notification_reminders_due
    ON notification_reminders(remind_at)
    WHERE fired_at IS NULL AND dismissed_at IS NULL;
"""

# Added by loop-proof D14 on top of the Phase-2 table. New and old databases take the
# same ALTER path, so there is one schema however old the file is.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("status", "TEXT NOT NULL DEFAULT 'pending'"),
    ("attempts", "INTEGER NOT NULL DEFAULT 0"),
    ("next_attempt_at", "TEXT"),
    ("last_error", "TEXT"),
    ("delivered_channels", "TEXT"),
    ("recurrence", "TEXT"),
    ("recur_until", "TEXT"),
    ("series_id", "TEXT"),
    ("missed_digests", "INTEGER NOT NULL DEFAULT 0"),
    ("closed_reason", "TEXT"),
    ("message_refs", "TEXT"),
    ("dedupe_key", "TEXT"),
    ("meta", "TEXT"),
)

_STATUS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_notification_reminders_status
    ON notification_reminders(status, remind_at);
CREATE INDEX IF NOT EXISTS idx_notification_reminders_series
    ON notification_reminders(series_id) WHERE series_id IS NOT NULL;
"""

# PR 4: one row per key, ever (a generated reminder is created once).
_DEDUPE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_notification_reminders_dedupe
    ON notification_reminders(dedupe_key) WHERE dedupe_key IS NOT NULL;
"""

# The indexes over the columns added after the table first shipped, one statement each.
_LATE_INDEX_STATEMENTS: tuple[str, ...] = tuple(
    statement.strip()
    for statement in (_STATUS_INDEX_SQL + _DEDUPE_INDEX_SQL).split(";")
    if statement.strip()
)


def _backfill_when_status_added(conn: sqlite3.Connection, added: list[str]) -> None:
    """``add_columns_if_missing`` callback: the lifecycle backfill, only when ``status`` is new."""
    if "status" in added:
        conn.row_factory = sqlite3.Row
        _backfill_lifecycle(conn)


# Two cutoff parameters (both "now"). Pending rows wait for their retry when one is
# scheduled; a claimed row whose lease ran out is retried (its process died mid-send).
_DUE_WHERE = (
    "(status = 'pending' AND COALESCE(next_attempt_at, remind_at) <= ?) "
    "OR (status = 'sending' AND next_attempt_at <= ?)"
)
_DUE_SQL = f"SELECT * FROM notification_reminders WHERE {_DUE_WHERE}"  # noqa: S608

_INSERT_SQL = """
INSERT INTO notification_reminders (
    id, target_kind, target_id, remind_at, channel, note,
    created_at, fired_at, dismissed_at, status, attempts,
    recurrence, recur_until, series_id, dedupe_key, meta
) VALUES (
    :id, :target_kind, :target_id, :remind_at, :channel, :note,
    :created_at, :fired_at, :dismissed_at, :status, :attempts,
    :recurrence, :recur_until, :series_id, :dedupe_key, :meta
)
"""
