"""Tests for the ReminderStore (Phase 2 Track 2D; lifecycle from loop-proof D14/D18)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from iris_harness.foundation.eventbus import EventBus
from iris_harness.services.notifications.events import (
    REMINDER_COMPLETED,
    REMINDER_FIRED,
    ReminderCompletedPayload,
    ReminderFiredPayload,
)
from iris_harness.services.notifications.store import ReminderStore


@pytest.fixture
def store(tmp_path: Path) -> ReminderStore:
    s = ReminderStore(db_path=tmp_path / "tasks.db")
    s.ensure_schema()
    return s


def _at(offset_seconds: int) -> datetime:
    return datetime.now(UTC) + timedelta(seconds=offset_seconds)


def test_create_persists_reminder(store: ReminderStore) -> None:
    r = store.create(
        target_kind="task",
        target_id="task-123",
        remind_at=_at(60),
        note="ping me",
    )
    fetched = store.get(r.id)
    assert fetched is not None
    assert fetched.target_kind == "task"
    assert fetched.target_id == "task-123"
    assert fetched.channel == "default"
    assert fetched.note == "ping me"
    assert fetched.fired_at is None


def test_list_filters_by_target(store: ReminderStore) -> None:
    store.create(target_kind="task", target_id="t1", remind_at=_at(60))
    store.create(target_kind="task", target_id="t2", remind_at=_at(120))
    store.create(target_kind="goal", target_id="g1", remind_at=_at(180))

    assert len(store.list()) == 3
    assert len(store.list(target_kind="task")) == 2
    assert len(store.list(target_kind="task", target_id="t1")) == 1


def test_list_excludes_fired_and_dismissed_by_default(store: ReminderStore) -> None:
    r1 = store.create(target_kind="task", target_id="t1", remind_at=_at(-60))
    r2 = store.create(target_kind="task", target_id="t2", remind_at=_at(60))
    r3 = store.create(target_kind="task", target_id="t3", remind_at=_at(120))
    store.fire(r1.id)
    store.cancel(r2.id)

    active = store.list()
    assert {r.id for r in active} == {r3.id}

    with_fired = store.list(include_fired=True)
    assert {r.id for r in with_fired} == {r1.id, r3.id}


def test_list_due_returns_only_past_unfired(store: ReminderStore) -> None:
    past = store.create(target_kind="task", target_id="t1", remind_at=_at(-30))
    future = store.create(target_kind="task", target_id="t2", remind_at=_at(60))
    dismissed = store.create(target_kind="task", target_id="t3", remind_at=_at(-60))
    store.cancel(dismissed.id)

    due = store.list_due()
    assert {r.id for r in due} == {past.id}
    assert future.id not in {r.id for r in due}
    assert dismissed.id not in {r.id for r in due}


def test_fire_marks_row_and_emits_event() -> None:
    bus = EventBus()
    received: list[ReminderFiredPayload] = []

    def _capture(payload: object) -> None:
        assert isinstance(payload, ReminderFiredPayload)
        received.append(payload)

    bus.on(REMINDER_FIRED, _capture)

    store = ReminderStore(db_path=Path(":memory:"), bus=bus)
    # :memory: doesn't survive across new connections; use a real file via tmp.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db") as tmpf:
        store = ReminderStore(db_path=Path(tmpf.name), bus=bus)
        store.ensure_schema()
        r = store.create(target_kind="task", target_id="t1", remind_at=_at(-30), channel="telegram")
        fired = store.fire(r.id)
        assert fired.fired_at is not None
        assert len(received) == 1
        assert received[0].reminder_id == r.id
        assert received[0].target_id == "t1"
        assert received[0].channel == "telegram"


def test_fire_is_idempotent(store: ReminderStore) -> None:
    r = store.create(target_kind="task", target_id="t1", remind_at=_at(-30))
    first = store.fire(r.id)
    second = store.fire(r.id)
    assert first.fired_at == second.fired_at


def test_fire_refuses_dismissed(store: ReminderStore) -> None:
    r = store.create(target_kind="task", target_id="t1", remind_at=_at(-30))
    store.cancel(r.id)
    with pytest.raises(ValueError, match="dismissed"):
        store.fire(r.id)


def test_cancel_is_idempotent(store: ReminderStore) -> None:
    r = store.create(target_kind="task", target_id="t1", remind_at=_at(60))
    first = store.cancel(r.id)
    second = store.cancel(r.id)
    assert first.dismissed_at == second.dismissed_at


def test_cancel_fired_reminder_is_no_op(store: ReminderStore) -> None:
    r = store.create(target_kind="task", target_id="t1", remind_at=_at(-30))
    fired = store.fire(r.id)
    after_cancel = store.cancel(r.id)
    assert after_cancel.dismissed_at is None
    assert after_cancel.fired_at == fired.fired_at


def test_reschedule_updates_remind_at(store: ReminderStore) -> None:
    r = store.create(target_kind="task", target_id="t1", remind_at=_at(60))
    new_time = _at(120)
    updated = store.reschedule(r.id, remind_at=new_time)
    assert updated.remind_at == new_time
    assert store.get(r.id).remind_at == new_time  # type: ignore[union-attr]


def test_reschedule_rejects_terminal(store: ReminderStore) -> None:
    r = store.create(target_kind="task", target_id="t1", remind_at=_at(-30))
    store.fire(r.id)
    with pytest.raises(ValueError, match="terminal"):
        store.reschedule(r.id, remind_at=_at(120))


def test_tick_fires_all_due(store: ReminderStore) -> None:
    a = store.create(target_kind="task", target_id="t1", remind_at=_at(-60))
    b = store.create(target_kind="task", target_id="t2", remind_at=_at(-30))
    _future = store.create(target_kind="task", target_id="t3", remind_at=_at(60))

    fired = store.tick()
    fired_ids = {r.id for r in fired}
    assert fired_ids == {a.id, b.id}
    # And the rows are persisted as fired.
    assert store.get(a.id).fired_at is not None  # type: ignore[union-attr]
    assert store.get(b.id).fired_at is not None  # type: ignore[union-attr]


# ── the one reminder store (loop-proof D14 / D18) ─────────────────────────

CT = ZoneInfo("America/Chicago")
T = datetime(2026, 9, 28, 13, 0, tzinfo=UTC)  # Mon 8:00 AM CDT


@pytest.fixture
def ct_store(tmp_path: Path) -> ReminderStore:
    s = ReminderStore(db_path=tmp_path / "tasks.db", tz=CT)
    s.ensure_schema()
    return s


_OLD_TABLE = """
CREATE TABLE notification_reminders (
    id TEXT PRIMARY KEY, target_kind TEXT NOT NULL, target_id TEXT NOT NULL,
    remind_at TEXT NOT NULL, channel TEXT NOT NULL DEFAULT 'default',
    note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
    fired_at TEXT, dismissed_at TEXT
);
"""


def test_an_old_table_gains_the_lifecycle_columns_with_backfill(tmp_path: Path) -> None:
    db = tmp_path / "tasks.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(_OLD_TABLE)
        rows = [
            ("a", "2026-09-28T08:00:00-05:00", None, None),  # local offset: normalised
            ("b", "2026-09-20T13:00:00+00:00", "2026-09-20T13:00:05+00:00", None),
            ("c", "2026-09-21T13:00:00+00:00", None, "2026-09-21T12:00:00+00:00"),
        ]
        for rid, at, fired, dismissed in rows:
            conn.execute(
                "INSERT INTO notification_reminders (id, target_kind, target_id, remind_at,"
                " created_at, fired_at, dismissed_at) VALUES (?, 'event', 'e', ?, ?, ?, ?)",
                (rid, at, at, fired, dismissed),
            )

    store = ReminderStore(db_path=db)
    store.ensure_schema()
    store.ensure_schema()  # idempotent

    a, b, c = (store.get(i) for i in "abc")
    assert a is not None and b is not None and c is not None
    assert (a.status, b.status, c.status) == ("pending", "sent", "cancelled")
    with sqlite3.connect(db) as conn:
        stored = conn.execute("SELECT remind_at FROM notification_reminders WHERE id='a'")
        assert stored.fetchone()[0] == "2026-09-28T13:00:00+00:00"
    assert [r.id for r in store.list_due(now=T)] == ["a"]


def test_create_stores_utc_and_a_series(ct_store: ReminderStore) -> None:
    local = datetime(2026, 9, 28, 8, 0, tzinfo=CT)
    r = ct_store.create(target_kind="event", target_id="e1", remind_at=local, recurrence="weekly")
    assert r.remind_at == T and r.remind_at.tzinfo == UTC
    assert r.series_id == r.id and r.status == "pending"
    with sqlite3.connect(ct_store.db_path) as conn:
        raw = conn.execute("SELECT remind_at FROM notification_reminders").fetchone()[0]
    assert raw.endswith("+00:00")
    with pytest.raises(ValueError):
        ct_store.create(target_kind="event", target_id="e", remind_at=T, recurrence="hourly")


def test_claim_is_taken_once(ct_store: ReminderStore) -> None:
    r = ct_store.create(target_kind="event", target_id="e1", remind_at=T)
    claimed = ct_store.claim(r.id, now=T)
    assert claimed is not None and claimed.status == "sending" and claimed.attempts == 1
    assert ct_store.claim(r.id, now=T) is None  # another tick got there first
    assert ct_store.list_due(now=T + timedelta(minutes=4)) == []
    # The process died mid-send: the lease runs out and the row is due again.
    assert [x.id for x in ct_store.list_due(now=T + timedelta(minutes=5))] == [r.id]


def test_retry_schedule_then_failed(ct_store: ReminderStore) -> None:
    r = ct_store.create(target_kind="event", target_id="e1", remind_at=T)
    seen: list[tuple[str, str]] = []
    now = T
    for _ in range(4):
        assert ct_store.claim(r.id, now=now) is not None
        updated = ct_store.mark_attempt_failed(r.id, error="telegram: failed", now=now)
        seen.append((now.strftime("%H:%M"), updated.status))
        if updated.next_attempt_at is not None:
            assert ct_store.list_due(now=updated.next_attempt_at - timedelta(seconds=1)) == []
            now = updated.next_attempt_at
    assert seen == [
        ("13:00", "pending"),
        ("13:05", "pending"),
        ("13:10", "pending"),
        ("13:15", "failed"),
    ]
    failed = ct_store.get(r.id)
    assert failed is not None and failed.attempts == 4 and failed.last_error == "telegram: failed"
    assert [m.id for m in ct_store.list_missed()] == [r.id]


def test_mark_sent_emits_fired_and_spawns_the_next_occurrence(tmp_path: Path) -> None:
    bus = EventBus()
    fired: list[ReminderFiredPayload] = []
    bus.on(REMINDER_FIRED, fired.append)
    store = ReminderStore(db_path=tmp_path / "tasks.db", bus=bus, tz=CT)
    store.ensure_schema()
    r = store.create(target_kind="event", target_id="e1", remind_at=T, recurrence="weekly")
    store.claim(r.id, now=T)

    sent, nxt = store.mark_sent(
        r.id,
        delivered_channels=["telegram"],
        message_refs=[{"channel": "telegram", "chat_id": "42", "message_id": "7"}],
        now=T,
    )

    assert sent.status == "sent" and sent.fired_at == T
    assert [p.reminder_id for p in fired] == [r.id]
    again = store.get(r.id)
    assert again is not None and again.delivered_channels == ("telegram",)
    assert again.message_refs == ({"channel": "telegram", "chat_id": "42", "message_id": "7"},)
    assert nxt is not None and nxt.series_id == r.id and nxt.status == "pending"
    assert nxt.remind_at.astimezone(CT) == datetime(2026, 10, 5, 8, 0, tzinfo=CT)
    # Done on the sent row does not make a second next occurrence.
    store.complete(r.id)
    series = [x for x in store.list(include_fired=True) if x.series_id == r.id]
    assert len(series) == 2


def test_complete_emits_completed_with_the_title(tmp_path: Path) -> None:
    bus = EventBus()
    done: list[ReminderCompletedPayload] = []
    bus.on(REMINDER_COMPLETED, done.append)
    store = ReminderStore(db_path=tmp_path / "tasks.db", bus=bus, tz=CT)
    store.ensure_schema()
    r = store.create(target_kind="bill", target_id="b1", remind_at=T, note="pay the card")

    closed = store.complete(r.id, source="telegram")

    assert closed.status == "done" and closed.closed_reason == "done: telegram"
    assert done == [
        ReminderCompletedPayload(
            reminder_id=r.id,
            task="pay the card",
            source="telegram",
            target_kind="bill",
            target_id="b1",
        )
    ]
    assert store.complete(r.id).status == "done"  # idempotent, no second event
    assert len(done) == 1


def test_completing_an_unsent_repeating_reminder_moves_the_series_on(
    ct_store: ReminderStore,
) -> None:
    r = ct_store.create(target_kind="event", target_id="e", remind_at=T, recurrence="daily")
    ct_store.complete(r.id, now=T - timedelta(hours=1))
    [nxt] = ct_store.list(statuses=["pending"])
    assert nxt.remind_at == T + timedelta(days=1)


def test_snooze_from_sent_and_failed(ct_store: ReminderStore) -> None:
    r = ct_store.create(target_kind="event", target_id="e", remind_at=T)
    ct_store.claim(r.id, now=T)
    ct_store.mark_sent(r.id, delivered_channels=["telegram"], now=T)
    later = T + timedelta(hours=1)
    snoozed = ct_store.snooze(r.id, later)
    assert snoozed.status == "pending" and snoozed.remind_at == later and snoozed.attempts == 0
    assert ct_store.list_due(now=later - timedelta(seconds=1)) == []
    assert [x.id for x in ct_store.list_due(now=later)] == [r.id]

    for i in range(4):
        ct_store.claim(r.id, now=later + timedelta(minutes=5 * i))
        ct_store.mark_attempt_failed(r.id, error="x", now=later + timedelta(minutes=5 * i))
    assert ct_store.get(r.id).status == "failed"  # type: ignore[union-attr]
    ct_store.mark_shown_in_digest([r.id])
    back = ct_store.snooze(r.id, later + timedelta(days=1))
    assert back.status == "pending" and back.missed_digests == 0 and back.last_error is None

    ct_store.complete(r.id)
    with pytest.raises(ValueError, match="cannot be snoozed"):
        ct_store.snooze(r.id, later)


def test_expire_and_missed_digests(ct_store: ReminderStore) -> None:
    r = ct_store.create(target_kind="event", target_id="e", remind_at=T)
    for i in range(4):
        ct_store.claim(r.id, now=T + timedelta(minutes=5 * i))
        ct_store.mark_attempt_failed(r.id, error="x", now=T + timedelta(minutes=5 * i))
    ct_store.mark_shown_in_digest([r.id])
    ct_store.mark_shown_in_digest([])
    assert ct_store.get(r.id).missed_digests == 1  # type: ignore[union-attr]

    expired = ct_store.expire(r.id, "expired: shown as missed in the digest")
    assert expired.status == "expired"
    assert ct_store.list_missed() == []
    assert ct_store.expire(r.id, "again").closed_reason == expired.closed_reason
    with pytest.raises(ValueError):
        ct_store.complete(r.id)


def test_list_for_day_uses_the_local_day(ct_store: ReminderStore) -> None:
    # 23:30 CT on Sep 28 is 04:30Z on Sep 29: it belongs to the 28th locally.
    late = ct_store.create(
        target_kind="event", target_id="e", remind_at=datetime(2026, 9, 28, 23, 30, tzinfo=CT)
    )
    early = ct_store.create(target_kind="event", target_id="e", remind_at=T)
    next_day = ct_store.create(
        target_kind="event", target_id="e", remind_at=datetime(2026, 9, 29, 0, 30, tzinfo=CT)
    )
    gone = ct_store.create(target_kind="event", target_id="e", remind_at=T)
    ct_store.cancel(gone.id)

    assert [r.id for r in ct_store.list_for_day(date(2026, 9, 28), CT)] == [early.id, late.id]
    assert [r.id for r in ct_store.list_for_day(date(2026, 9, 29))] == [next_day.id]


def test_cancel_stops_a_retrying_reminder(ct_store: ReminderStore) -> None:
    r = ct_store.create(target_kind="event", target_id="e", remind_at=T)
    ct_store.claim(r.id, now=T)
    ct_store.mark_attempt_failed(r.id, error="x", now=T)
    cancelled = ct_store.cancel(r.id)
    assert cancelled.status == "cancelled" and cancelled.dismissed_at is not None
    assert ct_store.list_due(now=T + timedelta(hours=1)) == []


# ── PR 3b: Done / Snooze from every surface ─────────────────────────────────


def test_snooze_emits_snoozed_with_where_it_came_from(tmp_path: Path) -> None:
    from iris_harness.services.notifications.events import (
        REMINDER_SNOOZED,
        ReminderSnoozedPayload,
    )

    bus = EventBus()
    heard: list[ReminderSnoozedPayload] = []
    bus.on(REMINDER_SNOOZED, heard.append)
    store = ReminderStore(db_path=tmp_path / "tasks.db", bus=bus, tz=CT)
    store.ensure_schema()
    r = store.create(target_kind="event", target_id="e", remind_at=T, recurrence="weekly")
    store.mark_sent(r.id, delivered_channels=["telegram"], now=T)

    store.snooze(r.id, T + timedelta(hours=1), source="push")

    assert heard == [
        ReminderSnoozedPayload(
            reminder_id=r.id,
            series_id=r.id,
            from_at=T,
            until=T + timedelta(hours=1),
            source="push",
        )
    ]


def test_reopen_undoes_a_done(ct_store: ReminderStore) -> None:
    sent = ct_store.create(target_kind="task", target_id="t", remind_at=T)
    ct_store.mark_sent(sent.id, delivered_channels=["telegram"], now=T)
    ct_store.complete(sent.id, source="sheet")
    back = ct_store.reopen(sent.id, now=T + timedelta(minutes=3))
    assert back.status == "sent" and back.closed_reason is None and back.fired_at == T
    assert ct_store.get(sent.id).status == "sent"  # type: ignore[union-attr]
    assert ct_store.list_due(now=T + timedelta(minutes=3)) == []  # not fired twice

    ahead = ct_store.create(target_kind="task", target_id="t2", remind_at=T + timedelta(days=1))
    ct_store.complete(ahead.id)
    assert ct_store.reopen(ahead.id, now=T).status == "pending"
    assert ct_store.reopen(ahead.id, now=T).status == "pending"  # open: unchanged

    gone = ct_store.create(target_kind="task", target_id="t3", remind_at=T)
    ct_store.cancel(gone.id)
    with pytest.raises(ValueError, match="already ended"):
        ct_store.reopen(gone.id)


def test_reopen_keeps_a_series_to_one_next_occurrence(ct_store: ReminderStore) -> None:
    r = ct_store.create(target_kind="event", target_id="e", remind_at=T, recurrence="weekly")
    ct_store.mark_sent(r.id, delivered_channels=["telegram"], now=T)
    ct_store.complete(r.id)
    ct_store.reopen(r.id, now=T)
    ct_store.complete(r.id)
    waiting = [x for x in ct_store.list(statuses=["pending"]) if x.series_id == r.id]
    assert [x.remind_at for x in waiting] == [T + timedelta(days=7)]


def test_unsnooze_puts_the_time_back_without_firing_twice(ct_store: ReminderStore) -> None:
    r = ct_store.create(target_kind="task", target_id="t", remind_at=T)
    ct_store.mark_sent(r.id, delivered_channels=["telegram"], now=T)
    ct_store.snooze(r.id, T + timedelta(hours=1))

    back = ct_store.unsnooze(r.id, T, now=T + timedelta(minutes=2))
    assert back.status == "sent" and back.remind_at == T
    assert ct_store.list_due(now=T + timedelta(minutes=2)) == []

    later = ct_store.create(target_kind="task", target_id="t2", remind_at=T + timedelta(hours=5))
    ct_store.snooze(later.id, T + timedelta(hours=6))
    moved = ct_store.unsnooze(later.id, T + timedelta(hours=5), now=T)
    assert moved.status == "pending" and moved.remind_at == T + timedelta(hours=5)
    # Once it has fired again (or closed), undo leaves it alone.
    ct_store.complete(later.id)
    assert ct_store.unsnooze(later.id, T, now=T).status == "done"


def test_find_by_message_matches_channel_chat_and_message(ct_store: ReminderStore) -> None:
    a = ct_store.create(target_kind="task", target_id="a", remind_at=T)
    b = ct_store.create(target_kind="task", target_id="b", remind_at=T)
    ct_store.mark_sent(
        a.id,
        delivered_channels=["telegram"],
        message_refs=[{"channel": "telegram", "chat_id": "77", "message_id": "12"}],
    )
    ct_store.mark_sent(
        b.id,
        delivered_channels=["telegram"],
        message_refs=[{"channel": "telegram", "chat_id": "12", "message_id": "77"}],
    )
    assert ct_store.find_by_message("telegram", "77", "12").id == a.id  # type: ignore[union-attr]
    assert ct_store.find_by_message("telegram", "12", "77").id == b.id  # type: ignore[union-attr]
    assert ct_store.find_by_message("web_push", "77", "12") is None
    assert ct_store.find_by_message("telegram", "77", "13") is None
    # A snooze keeps the refs: a second reply to the same message still finds it.
    ct_store.snooze(a.id, T + timedelta(hours=1))
    assert ct_store.find_by_message("telegram", "77", "12").id == a.id  # type: ignore[union-attr]


# ── generated reminders: dedupe, close / reopen by target (loop-proof PR 4) ──


def test_a_dedupe_key_is_created_once_ever(ct_store: ReminderStore) -> None:
    first = ct_store.create(
        target_kind="bill",
        target_id="d1",
        remind_at=T,
        note="first",
        dedupe_key="bill:d1:t3d",
        meta={"entity": "Example Card", "step": "t3d"},
    )
    again = ct_store.create(
        target_kind="bill",
        target_id="d1",
        remind_at=T + timedelta(days=1),
        note="second",
        dedupe_key="bill:d1:t3d",
    )
    assert again == first
    assert ct_store.get_by_dedupe_key("bill:d1:t3d") == first
    assert first.meta == {"entity": "Example Card", "step": "t3d"}
    assert len(ct_store.list(include_fired=True, include_dismissed=True)) == 1
    # Even after it ended: the key never comes back as a second row.
    ct_store.cancel(first.id)
    assert (
        ct_store.create(
            target_kind="bill", target_id="d1", remind_at=T, dedupe_key="bill:d1:t3d"
        ).status
        == "cancelled"
    )
    # No key: as many rows as asked for.
    ct_store.create(target_kind="task", target_id="t", remind_at=T)
    ct_store.create(target_kind="task", target_id="t", remind_at=T)
    assert len(ct_store.list(include_fired=True, include_dismissed=True)) == 3


def test_the_dedupe_index_refuses_a_second_row_under_one_key(ct_store: ReminderStore) -> None:
    ct_store.create(target_kind="bill", target_id="d", remind_at=T, dedupe_key="k")
    with sqlite3.connect(ct_store.db_path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO notification_reminders (id, target_kind, target_id, remind_at, "
            "created_at, dedupe_key) VALUES ('x', 'bill', 'd', ?, ?, 'k')",
            (T.isoformat(), T.isoformat()),
        )


def test_an_old_table_gains_dedupe_key_and_meta(tmp_path: Path) -> None:
    db = tmp_path / "tasks.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE notification_reminders (id TEXT PRIMARY KEY, target_kind TEXT NOT "
            "NULL, target_id TEXT NOT NULL, remind_at TEXT NOT NULL, channel TEXT NOT NULL "
            "DEFAULT 'default', note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, "
            "fired_at TEXT, dismissed_at TEXT)"
        )
        conn.execute(
            "INSERT INTO notification_reminders VALUES ('old', 'task', 't', ?, 'default', "
            "'n', ?, NULL, NULL)",
            (T.isoformat(), T.isoformat()),
        )
    store = ReminderStore(db_path=db, tz=CT)
    store.ensure_schema()
    store.ensure_schema()  # idempotent
    old = store.get("old")
    assert old is not None and old.dedupe_key is None and old.meta == {}
    assert (
        store.create(target_kind="bill", target_id="d", remind_at=T, dedupe_key="k").dedupe_key
        == "k"
    )


def _bill_rows(store: ReminderStore) -> dict[str, str]:
    """dedupe key -> a row in each state close_for_target meets."""
    keys = {}
    for key in ("pending", "sending", "sent", "failed", "done", "other"):
        target = "d2" if key == "other" else "d1"
        keys[key] = store.create(
            target_kind="bill", target_id=target, remind_at=T, dedupe_key=f"bill:{target}:{key}"
        ).id
    store.claim(keys["sending"], now=T)
    store.mark_sent(keys["sent"], delivered_channels=["telegram"], now=T)
    store.mark_sent(keys["done"], delivered_channels=["telegram"], now=T)
    store.complete(keys["done"], source="telegram")
    for _ in range(4):
        store.claim(keys["failed"], now=T + timedelta(hours=1))
        store.mark_attempt_failed(keys["failed"], error="down", now=T)
    return keys


def test_close_for_target_ends_every_open_row_of_that_target(ct_store: ReminderStore) -> None:
    ids = _bill_rows(ct_store)
    assert ct_store.get(ids["failed"]).status == "failed"  # type: ignore[union-attr]

    closed = ct_store.close_for_target("bill", "d1", reason="closed: paid (payment_email)")

    assert closed == 4
    status = {k: ct_store.get(v) for k, v in ids.items()}
    assert status["pending"].status == "cancelled"  # type: ignore[union-attr]
    assert status["sending"].status == "cancelled"  # type: ignore[union-attr]
    assert status["sent"].status == "expired"  # type: ignore[union-attr]
    assert status["failed"].status == "expired"  # type: ignore[union-attr]
    assert status["done"].status == "done"  # type: ignore[union-attr] — the owner's Paid stays
    assert status["other"].status == "pending"  # type: ignore[union-attr] — another bill
    assert {status[k].closed_reason for k in ("pending", "sent")} == {  # type: ignore[union-attr]
        "closed: paid (payment_email)"
    }
    assert ct_store.list_due(now=T + timedelta(days=1)) == [ct_store.get(ids["other"])]
    # A bare reason is prefixed, so reopen_for_target finds it.
    ct_store.close_for_target("bill", "d2", reason="paid")
    assert ct_store.get(ids["other"]).closed_reason == "closed: paid"  # type: ignore[union-attr]


def test_reopen_for_target_rearms_only_rows_still_ahead(ct_store: ReminderStore) -> None:
    past = ct_store.create(target_kind="bill", target_id="d", remind_at=T, dedupe_key="a")
    ahead = ct_store.create(
        target_kind="bill", target_id="d", remind_at=T + timedelta(days=2), dedupe_key="b"
    )
    owner = ct_store.create(
        target_kind="bill", target_id="d", remind_at=T + timedelta(days=3), dedupe_key="c"
    )
    ct_store.cancel(owner.id)  # the owner's own cancel is not the bill's close
    ct_store.close_for_target("bill", "d", reason="closed: paid (finance)")

    assert ct_store.reopen_for_target("bill", "d", now=T + timedelta(days=1)) == 1

    back = ct_store.get(ahead.id)
    assert back is not None and back.status == "pending" and back.closed_reason is None
    assert back.dismissed_at is None
    assert ct_store.get(past.id).status == "cancelled"  # type: ignore[union-attr]
    assert ct_store.get(owner.id).status == "cancelled"  # type: ignore[union-attr]


def test_acknowledge_is_not_yet_and_reopen_takes_it_back(tmp_path: Path) -> None:
    from iris_harness.services.notifications.events import (
        REMINDER_REOPENED,
        ReminderReopenedPayload,
    )

    bus = EventBus()
    reopened: list[ReminderReopenedPayload] = []
    bus.on(REMINDER_REOPENED, reopened.append)
    store = ReminderStore(db_path=tmp_path / "tasks.db", bus=bus, tz=CT)
    store.ensure_schema()
    r = store.create(target_kind="bill", target_id="d", remind_at=T, dedupe_key="bill:d:ask1")
    store.mark_sent(r.id, delivered_channels=["telegram"], now=T)

    seen = store.acknowledge(r.id, source="telegram")
    assert (seen.status, seen.closed_reason) == ("expired", "not yet: telegram")
    with pytest.raises(ValueError):
        store.acknowledge(r.id)

    back = store.reopen(r.id, now=T + timedelta(hours=1), source="sheet")
    assert back.status == "sent" and back.closed_reason is None
    assert reopened == [
        ReminderReopenedPayload(
            reminder_id=r.id,
            target_kind="bill",
            target_id="d",
            source="sheet",
            closed_reason="not yet: telegram",
        )
    ]


def test_a_bill_a_payment_email_closed_can_be_reopened(ct_store: ReminderStore) -> None:
    sent = ct_store.create(target_kind="bill", target_id="d", remind_at=T, dedupe_key="s")
    ct_store.mark_sent(sent.id, delivered_channels=["telegram"], now=T)
    ahead = ct_store.create(
        target_kind="bill", target_id="d", remind_at=T + timedelta(days=1), dedupe_key="f"
    )
    ct_store.close_for_target("bill", "d", reason="closed: paid (payment_email)")

    assert ct_store.reopen(sent.id, now=T + timedelta(hours=2)).status == "sent"
    again = ct_store.reopen(ahead.id, now=T + timedelta(hours=2))
    assert again.status == "pending" and again.dismissed_at is None
    # Any other ended row still refuses.
    gone = ct_store.create(target_kind="task", target_id="t", remind_at=T)
    ct_store.expire(gone.id, "expired: delivered, not acknowledged")
    with pytest.raises(ValueError):
        ct_store.reopen(gone.id)


def test_refresh_rewords_only_a_row_still_waiting(ct_store: ReminderStore) -> None:
    waiting = ct_store.create(target_kind="bill", target_id="d", remind_at=T, note="old")
    sent = ct_store.create(target_kind="bill", target_id="d", remind_at=T, note="old")
    ct_store.mark_sent(sent.id, delivered_channels=["telegram"], now=T)

    assert ct_store.refresh(waiting.id, note="new", meta={"amount": "$70.00"}) is True
    assert ct_store.refresh(sent.id, note="new", meta={}) is False
    fresh = ct_store.get(waiting.id)
    assert fresh is not None and fresh.note == "new" and fresh.meta == {"amount": "$70.00"}
    assert ct_store.get(sent.id).note == "old"  # type: ignore[union-attr]


# ── Paid from ANY of a bill's messages (PR 4 demo, 2026-09-25) ────────────────
#
# The demo: after the 07:00 sweep expired a delivered reminder ("expired: delivered, not
# acknowledged") or after Not yet, ✅ Paid on that Telegram message answered "That
# reminder isn't open any more" and the bill stayed unpaid.


def _sent_bill(store: ReminderStore, step: str, *, target: str = "d") -> str:
    row = store.create(
        target_kind="bill",
        target_id=target,
        remind_at=T,
        dedupe_key=f"bill:{target}:{step}",
        meta={"entity": "Example Card", "step": step, "due": "2026-10-12"},
    )
    store.mark_sent(row.id, delivered_channels=["telegram"], now=T)
    return row.id


def _bus_store(tmp_path: Path) -> tuple[ReminderStore, list[ReminderCompletedPayload]]:
    bus = EventBus()
    done: list[ReminderCompletedPayload] = []
    bus.on(REMINDER_COMPLETED, done.append)
    store = ReminderStore(db_path=tmp_path / "tasks.db", bus=bus, tz=CT)
    store.ensure_schema()
    return store, done


@pytest.mark.parametrize(
    "ended_by",
    ["expired: delivered, not acknowledged", "not yet", "expired: bill reopened"],
)
def test_paid_on_a_bill_row_that_ended_while_the_bill_was_open_closes_it(
    tmp_path: Path, ended_by: str
) -> None:
    store, done = _bus_store(tmp_path)
    rid = _sent_bill(store, "ask1")
    if ended_by == "not yet":
        store.acknowledge(rid, source="telegram")
    else:
        store.expire(rid, ended_by)

    closed = store.complete(rid, source="telegram")

    assert (closed.status, closed.closed_reason) == ("done", "done: telegram")
    assert [(p.reminder_id, p.target_kind, p.target_id, p.source) for p in done] == [
        (rid, "bill", "d", "telegram")
    ]


def test_paid_on_a_bill_already_closed_as_paid_says_so(tmp_path: Path) -> None:
    from iris_harness.services.notifications.bills import BillAlreadyPaid

    store, done = _bus_store(tmp_path)
    rid = _sent_bill(store, "t3d")
    store.close_for_target("bill", "d", reason="closed: paid (chat)")

    with pytest.raises(BillAlreadyPaid) as raised:
        store.complete(rid, source="telegram")
    assert raised.value.entity == "Example Card"
    assert done == []


def test_other_ended_rows_still_refuse_paid(ct_store: ReminderStore) -> None:
    from iris_harness.services.notifications.bills import BillAlreadyPaid

    task = ct_store.create(target_kind="task", target_id="t", remind_at=T)
    ct_store.expire(task.id, "expired: delivered, not acknowledged")
    cancelled = ct_store.create(target_kind="bill", target_id="d", remind_at=T)
    ct_store.cancel(cancelled.id)
    for rid in (task.id, cancelled.id):
        with pytest.raises(ValueError) as raised:
            ct_store.complete(rid)
        assert not isinstance(raised.value, BillAlreadyPaid)


def test_a_close_marks_rows_that_already_ended_as_paid_too(ct_store: ReminderStore) -> None:
    """A row the sweep aged out before the bill was paid in chat: Paid on its message is
    "already marked paid", never a second close."""
    from iris_harness.services.notifications.bills import BillAlreadyPaid

    swept = _sent_bill(ct_store, "t3d")
    ct_store.expire(swept, "expired: delivered, not acknowledged")
    said = _sent_bill(ct_store, "ask1")
    ct_store.acknowledge(said, source="push")
    other = _sent_bill(ct_store, "ask1", target="e")
    ct_store.expire(other, "expired: delivered, not acknowledged")

    ct_store.close_for_target("bill", "d", reason="closed: paid (chat)")

    for rid in (swept, said):
        assert ct_store.get(rid).closed_reason == "closed: paid (chat)"  # type: ignore[union-attr]
        with pytest.raises(BillAlreadyPaid):
            ct_store.complete(rid)
    # Another bill's row is untouched.
    assert ct_store.get(other).closed_reason == "expired: delivered, not acknowledged"  # type: ignore[union-attr]


def test_a_reopened_bill_takes_paid_on_its_past_messages_again(ct_store: ReminderStore) -> None:
    paid_here = _sent_bill(ct_store, "t3d")
    ct_store.complete(paid_here, source="telegram")
    closed = _sent_bill(ct_store, "dayof")
    ct_store.close_for_target("bill", "d", reason="closed: paid (telegram)")

    ct_store.reopen_for_target("bill", "d", now=T + timedelta(days=1))

    for rid in (paid_here, closed):
        row = ct_store.get(rid)
        assert row is not None and (row.status, row.closed_reason) == (
            "expired",
            "expired: bill reopened",
        )
        assert ct_store.complete(rid, source="sheet").status == "done"


def test_the_paid_confirmation_ends_once_delivered(ct_store: ReminderStore) -> None:
    """ "✅ … marked paid" asks nothing: the sweep must never call it unanswered."""
    paid = ct_store.create(
        target_kind="bill",
        target_id="d",
        remind_at=T,
        dedupe_key="bill:d:paid",
        meta={"entity": "Example Card", "step": "paid"},
    )
    sent, _ = ct_store.mark_sent(paid.id, delivered_channels=["telegram"], now=T)

    assert (sent.status, sent.closed_reason) == ("expired", "delivered: nothing to answer")
    assert ct_store.get(paid.id).status == "expired"  # type: ignore[union-attr]
    assert ct_store.list_sent_before(T + timedelta(days=3)) == []


def test_not_yet_emits_acknowledged_with_the_target(tmp_path: Path) -> None:
    from iris_harness.services.notifications.events import (
        REMINDER_ACKNOWLEDGED,
        ReminderAcknowledgedPayload,
    )

    bus = EventBus()
    heard: list[ReminderAcknowledgedPayload] = []
    bus.on(REMINDER_ACKNOWLEDGED, heard.append)
    store = ReminderStore(db_path=tmp_path / "tasks.db", bus=bus, tz=CT)
    store.ensure_schema()
    rid = _sent_bill(store, "ask2")

    store.acknowledge(rid, source="push")

    assert heard == [
        ReminderAcknowledgedPayload(
            reminder_id=rid, target_kind="bill", target_id="d", source="push"
        )
    ]
