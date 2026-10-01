"""What a reminder is about, in words — the title of its target.

A reminder points at a row in another store: a task or goal in ``tasks.db``, an event
in ``calendar.db`` (ADR-0075: "remind me …" writes a calendar event plus a reminder
row). Both files sit in the data directory next to each other, so the paths are
derived from the reminder store's own ``tasks.db`` — never from the process's working
directory. Raw sqlite keeps this module free of the plugin that owns the calendar.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from iris_harness.foundation.persistence.sqlite import sqlite_conn

from .models import Reminder


def lookup_target_title(
    tasks_db: Path,
    target_kind: str,
    target_id: str,
    *,
    calendar_db: Path | None = None,
) -> str | None:
    """Best-effort title; ``None`` when the row is missing or the kind has no title here."""
    if target_kind == "event":
        db = calendar_db if calendar_db is not None else tasks_db.parent / "calendar.db"
        sql = "SELECT summary FROM calendar_events WHERE id = ?"
    elif target_kind == "task":
        db, sql = tasks_db, "SELECT title FROM tasks WHERE id = ?"
    elif target_kind == "goal":
        db, sql = tasks_db, "SELECT title FROM goals WHERE id = ?"
    else:
        return None
    if not db.exists():
        return None
    try:
        with sqlite_conn(db) as conn:
            row = conn.execute(sql, (target_id,)).fetchone()
    except sqlite3.Error:
        return None
    title = str(row[0]).strip() if row and row[0] is not None else ""
    return title or None


# A chat reminder is a calendar event titled "Reminder: <what>" (ADR-0075). The owner
# sees the "what": the ⏰ already says it is a reminder, and the finance hook matches
# the words against open dues.
_REMINDER_EVENT_PREFIX = "Reminder: "


def reminder_title(reminder: Reminder, tasks_db: Path, *, calendar_db: Path | None = None) -> str:
    """The words the owner sees: the target's title, else the note, else "Reminder"."""
    title = lookup_target_title(
        tasks_db, reminder.target_kind, reminder.target_id, calendar_db=calendar_db
    )
    if title and reminder.target_kind == "event" and title.startswith(_REMINDER_EVENT_PREFIX):
        title = title[len(_REMINDER_EVENT_PREFIX) :].strip() or title
    return title or reminder.note or "Reminder"


__all__ = ["lookup_target_title", "reminder_title"]
