"""The ``push_delivery`` health check (loop-proof D13/D14)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from iris_harness.services.health.models import HealthState
from iris_harness.services.notifications.health import (
    push_delivery_checks,
    push_delivery_provider,
)
from iris_harness.services.notifications.store import ReminderStore

CT = ZoneInfo("America/Chicago")
T = datetime(2026, 9, 28, 13, 0, tzinfo=UTC)


def _fail(store: ReminderStore, rid: str) -> None:
    for i in range(4):
        now = T + timedelta(minutes=5 * i)
        store.claim(rid, now=now)
        store.mark_attempt_failed(rid, error="telegram: failed", now=now)


def _store(tmp_path: Path) -> ReminderStore:
    store = ReminderStore(db_path=tmp_path / "tasks.db", tz=CT)
    store.ensure_schema()
    with sqlite3.connect(tmp_path / "calendar.db") as conn:
        conn.execute("CREATE TABLE calendar_events (id TEXT, summary TEXT)")
        conn.execute("INSERT INTO calendar_events VALUES ('ev', 'Take out the recycling')")
    return store


def test_no_database_is_green(tmp_path: Path) -> None:
    [check] = push_delivery_provider(tmp_path, lambda: CT)()
    assert check.target == "push_delivery" and check.state is HealthState.GREEN
    assert not (tmp_path / "tasks.db").exists()


def test_red_while_a_failed_reminder_is_unseen_then_green(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rid = store.create(target_kind="event", target_id="ev", remind_at=T).id
    assert push_delivery_checks(store, CT)[0].state is HealthState.GREEN

    _fail(store, rid)
    [red] = push_delivery_checks(store, CT)
    assert red.state is HealthState.RED
    assert red.detail == (
        "1 reminder couldn't be delivered: “Take out the recycling” (due Mon Sep 28 · 8:00 AM)"
    )

    store.mark_shown_in_digest([rid])  # the digest's "missed" line told the owner
    assert push_delivery_checks(store, CT)[0].state is HealthState.GREEN


def test_expired_failures_are_green(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rid = store.create(target_kind="event", target_id="ev", remind_at=T).id
    _fail(store, rid)
    store.expire(rid, "expired")
    assert push_delivery_checks(store, CT)[0].state is HealthState.GREEN


def test_several_failures_are_counted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for note in ("a", "b"):
        _fail(store, store.create(target_kind="bill", target_id="x", remind_at=T, note=note).id)
    [red] = push_delivery_checks(store, CT)
    assert red.detail.startswith("2 reminders couldn't be delivered: “a”")
