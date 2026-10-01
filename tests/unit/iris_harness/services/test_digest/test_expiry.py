"""The one expiry mechanism: every dated item ends closed or expired (owner rule 2026-09-25)."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from iris_harness.services.digest.expiry import (
    EXPIRY_RANGES,
    UNACKNOWLEDGED_REMINDER_REASON,
    ExpiryPolicy,
    expires_at,
    expiry_reason,
    expiry_view,
    is_expired,
    is_task_expired,
    parse_expiry,
    sweep_expired_tasks,
    sweep_missed_reminders,
    sweep_unacknowledged_reminders,
    task_expiry_kind,
)
from iris_harness.services.digest.settings import load_defaults
from iris_harness.services.tasks.models import TaskAction
from iris_harness.services.tasks.store import TaskStore

REPO_CONFIG = Path(__file__).resolve().parents[5] / "config"
CHICAGO = ZoneInfo("America/Chicago")
# Fri Sep 25 2026, 07:00 in Chicago (the digest's hour).
NOW = datetime(2026, 9, 25, 7, 0, tzinfo=CHICAGO)
POLICY = ExpiryPolicy()


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    s = TaskStore(db_path=tmp_path / "tasks.db")
    s.ensure_schema()
    return s


# -- the policy -------------------------------------------------------------------


def test_the_shipped_file_carries_the_owner_approved_defaults() -> None:
    """The core's keys; a plugin's own kinds are its manifest's (test_expiry_kinds.py)."""
    policy = load_defaults(REPO_CONFIG).settings.expiry
    view = expiry_view(policy)
    assert {key: view[key] for key in EXPIRY_RANGES} == {
        "prep_after_event": 0,
        "event_after_end": 0,
        "task_overdue_days": 3,
        "bill_ask_days": 3,
        "bill_overdue_digest_days": 7,
        "inbox_notice_days": 3,
        "reminder_missed_digests": 1,
    }


def test_a_partial_expiry_keeps_the_other_defaults() -> None:
    policy = parse_expiry({"task_overdue_days": "5"})
    assert policy.task_overdue_days == 5
    assert policy.bill_overdue_digest_days == 7


@pytest.mark.parametrize(
    "raw",
    [
        {"task_overdue_days": -1},
        {"task_overdue_days": 400},
        {"prep_after_event": 2000},
        {"task_overdue_days": "soon"},
        {"task_overdue_days": True},
        {"tasks_overdue": 3},
        ["task_overdue_days"],
    ],
)
def test_an_expiry_that_does_not_fit_is_refused(raw: object) -> None:
    with pytest.raises(ValueError, match="expiry"):
        parse_expiry(raw)


def test_a_bad_expiry_in_the_file_warns_and_keeps_the_built_in_policy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "digest.yaml").write_text("expiry:\n  task_overdue_days: -4\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        settings = load_defaults(tmp_path).settings
    assert settings.expiry == ExpiryPolicy()
    assert "expiry" in caplog.text


# -- when an item expires ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("due_day", "expired"),
    [
        (25, False),  # due today
        (22, False),  # overdue since Tue: shown Sep 23, 24, 25
        (21, True),  # its three days were Sep 22-24
        (1, True),
    ],
)
def test_a_task_is_overdue_for_three_local_days_then_expires(due_day: int, expired: bool) -> None:
    due = datetime(2026, 9, due_day, 17, 0, tzinfo=CHICAGO)
    assert is_expired("task", due, NOW, POLICY, tz=CHICAGO) is expired


def test_a_task_expires_at_the_start_of_the_owners_day() -> None:
    due = datetime(2026, 9, 22, 23, 30, tzinfo=CHICAGO)  # already Sep 23 in UTC
    assert expires_at("task", due, POLICY, tz=CHICAGO) == datetime(2026, 9, 26, tzinfo=CHICAGO)


def test_a_prep_task_expires_when_its_meeting_ends() -> None:
    start = datetime(2026, 9, 25, 6, 30, tzinfo=CHICAGO)
    end = datetime(2026, 9, 25, 7, 30, tzinfo=CHICAGO)
    assert not is_expired("prep", start, NOW, POLICY, end_at=end)  # meeting in progress
    assert is_expired("prep", start, NOW + timedelta(minutes=30), POLICY, end_at=end)
    # Its end unknown (no calendar at hand), the start is the line.
    assert is_expired("prep", start, NOW, POLICY)
    late = ExpiryPolicy(prep_after_event=60)
    assert not is_expired("prep", start, NOW + timedelta(minutes=30), late, end_at=end)


def test_an_event_leaves_the_digest_once_it_has_ended() -> None:
    start = datetime(2026, 9, 25, 5, 0, tzinfo=CHICAGO)
    assert is_expired("event", start, NOW, POLICY, end_at=start + timedelta(hours=1))
    assert not is_expired("event", start, NOW, POLICY, end_at=start + timedelta(hours=3))


def test_a_bill_stays_overdue_for_its_own_days() -> None:
    # Unpaid: asked about for bill_ask_days (3), then in the digest for
    # bill_overdue_digest_days (7) more — 10 days past its due date (owner, PR 4).
    assert not is_expired("bill", datetime(2026, 9, 15, tzinfo=CHICAGO), NOW, POLICY, tz=CHICAGO)
    assert is_expired("bill", datetime(2026, 9, 14, tzinfo=CHICAGO), NOW, POLICY, tz=CHICAGO)


def test_a_bill_due_oct_13_is_asked_then_shown_then_expires_oct_24() -> None:
    due = date(2026, 10, 13)
    last_shown = datetime(2026, 10, 23, 23, 59, tzinfo=CHICAGO)
    assert not is_expired("bill", due, last_shown, ExpiryPolicy(), tz=CHICAGO)
    assert is_expired("bill", due, last_shown + timedelta(minutes=1), ExpiryPolicy(), tz=CHICAGO)
    shorter = ExpiryPolicy(bill_ask_days=0, bill_overdue_digest_days=7)
    assert is_expired("bill", due, datetime(2026, 10, 21, tzinfo=CHICAGO), shorter, tz=CHICAGO)


def test_undated_items_never_expire() -> None:
    assert not is_expired("task", None, NOW, POLICY)
    assert not is_expired("prep", None, NOW, POLICY)


def test_which_tasks_are_dated_items(store: TaskStore) -> None:
    due = datetime(2026, 9, 1, tzinfo=UTC)
    assert task_expiry_kind(store.create(title="Todo", due_at=due)) == "task"
    assert task_expiry_kind(store.create(title="Undated")) is None
    prep = store.create(title="Prep: Sync", due_at=due, source_kind="calendar-prep")
    assert task_expiry_kind(prep) == "prep"
    bill = store.create(title="Pay card", due_at=due, source_kind="finance-bills")
    assert task_expiry_kind(bill) == "bill"
    # A system-raised pending action keeps its own lifecycle (ADR-0073/0118).
    action = TaskAction(kind="review", label="Approve", target_id="x")
    pending = store.create(title="Approve invite", due_at=due, action=action)
    assert task_expiry_kind(pending) is None
    assert not is_task_expired(pending, NOW, POLICY)


# -- the sweep ----------------------------------------------------------------------------


def test_the_sweep_expires_any_past_item_and_keeps_it(store: TaskStore) -> None:
    old_prep = store.create(
        title="Prep: Swimming - Stage 3",
        due_at=datetime(2026, 9, 22, 22, 0, tzinfo=UTC),
        source_kind="calendar-prep",
        dedup_key="meeting-prep:ics:swim-3",
    )
    stale = store.create(title="Renew registration", due_at=datetime(2026, 8, 1, tzinfo=UTC))
    recent = store.create(title="Call bank", due_at=datetime(2026, 9, 24, 15, tzinfo=UTC))
    undated = store.create(title="Someday")

    expired = sweep_expired_tasks(store, now=NOW, policy=POLICY, tz=CHICAGO)

    assert {t.id for t in expired} == {old_prep.id, stale.id}
    kept = store.get(old_prep.id)
    assert kept is not None and kept.status == "expired"  # closed, never deleted
    assert kept.closed_reason == expiry_reason("prep")
    assert store.get(stale.id).closed_reason == "expired: overdue past 3 day(s)"  # type: ignore[union-attr]
    assert store.get(recent.id).status == "open"  # type: ignore[union-attr]
    assert store.get(undated.id).status == "open"  # type: ignore[union-attr]
    # Idempotent: nothing left to expire.
    assert sweep_expired_tasks(store, now=NOW, policy=POLICY, tz=CHICAGO) == []


def test_an_expired_task_keeps_its_dedup_key_so_it_is_never_raised_again(store: TaskStore) -> None:
    key = "meeting-prep:ics:swim-3"
    first = store.upsert(
        dedup_key=key,
        title="Prep: Swimming",
        due_at=datetime(2026, 9, 22, 22, 0, tzinfo=UTC),
        source_kind="calendar-prep",
    )
    sweep_expired_tasks(store, now=NOW, policy=POLICY, tz=CHICAGO)
    again = store.upsert(dedup_key=key, title="Prep: Swimming", source_kind="calendar-prep")
    assert again.id == first.id and again.status == "expired"


def test_the_sweep_asks_for_a_prep_tasks_real_meeting_end(store: TaskStore) -> None:
    prep = store.create(
        title="Prep: Long workshop",
        due_at=datetime(2026, 9, 25, 6, 0, tzinfo=CHICAGO),
        source_kind="calendar-prep",
        source_id="evt-1",
    )
    ends = {"evt-1": datetime(2026, 9, 25, 9, 0, tzinfo=CHICAGO)}
    swept = sweep_expired_tasks(
        store, now=NOW, policy=POLICY, tz=CHICAGO, event_end=lambda t: ends.get(t.source_id or "")
    )
    assert swept == []  # still running at 07:00
    assert store.get(prep.id).status == "open"  # type: ignore[union-attr]


def test_a_reopened_task_forgets_why_it_expired(store: TaskStore) -> None:
    task = store.create(title="Renew registration", due_at=datetime(2026, 8, 1, tzinfo=UTC))
    store.expire(task.id, "expired: overdue past 3 day(s)")
    reopened = store.update(task.id, status="open")
    assert reopened.closed_reason is None


def test_a_bare_due_date_counts_local_days() -> None:
    """A bill's due date is a date, already the owner's local day."""
    chicago = ZoneInfo("America/Chicago")
    due = date(2026, 9, 10)
    last_shown = datetime(2026, 9, 20, 23, 59, tzinfo=chicago)  # due + 3 asks + 7 days
    assert not is_expired("bill", due, last_shown, ExpiryPolicy(), tz=chicago)
    assert is_expired("bill", due, last_shown + timedelta(minutes=1), ExpiryPolicy(), tz=chicago)


def test_an_all_day_event_given_as_a_date_ends_with_its_local_day() -> None:
    chicago = ZoneInfo("America/Chicago")
    day = date(2026, 9, 25)
    assert not is_expired("event", day, datetime(2026, 9, 25, 23, 59, tzinfo=chicago), tz=chicago)
    assert is_expired("event", day, datetime(2026, 9, 26, 0, 0, tzinfo=chicago), tz=chicago)


# -- reminders: missed ones are carried into N digests, then expire (D18) --------------


def _failed_reminder(db: Path, note: str) -> str:
    """A reminder whose delivery gave up (stream A's retries set status ``failed``)."""
    import sqlite3

    from iris_harness.services.notifications.store import ReminderStore

    rstore = ReminderStore(db_path=db)
    rstore.ensure_schema()
    row = rstore.create(
        target_kind="task",
        target_id=note,
        remind_at=datetime(2026, 9, 28, 13, tzinfo=UTC),
        note=note,
    )
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE notification_reminders SET status = 'failed' WHERE id = ?", (row.id,))
    return row.id


def test_a_missed_reminder_is_shown_in_one_digest_then_expires(tmp_path: Path) -> None:
    from iris_harness.services.notifications.store import ReminderStore

    db = tmp_path / "tasks.db"
    rid = _failed_reminder(db, "Take out the recycling")
    rstore = ReminderStore(db_path=db)

    expired, shown = sweep_missed_reminders(rstore, now=NOW, policy=POLICY)
    assert (expired, shown) == ([], [rid])  # Tuesday's digest lists it
    assert [r.id for r in rstore.list_missed(NOW)] == [rid]

    expired, shown = sweep_missed_reminders(rstore, now=NOW, policy=POLICY)
    assert [r.id for r in expired] == [rid] and shown == []  # gone before Wednesday's
    assert expired[0].status == "expired"
    assert expired[0].closed_reason == "expired: missed, shown in 1 digest(s)"
    assert rstore.list_missed(NOW) == []


def test_reminder_missed_digests_counts_digests(tmp_path: Path) -> None:
    from iris_harness.services.notifications.store import ReminderStore

    db = tmp_path / "tasks.db"
    rid = _failed_reminder(db, "Call the bank")
    rstore = ReminderStore(db_path=db)
    twice = ExpiryPolicy(reminder_missed_digests=2)

    assert sweep_missed_reminders(rstore, now=NOW, policy=twice)[1] == [rid]
    assert sweep_missed_reminders(rstore, now=NOW, policy=twice)[1] == [rid]
    assert [r.id for r in sweep_missed_reminders(rstore, now=NOW, policy=twice)[0]] == [rid]

    other = _failed_reminder(db, "Never shown")
    never = ExpiryPolicy(reminder_missed_digests=0)
    assert [r.id for r in sweep_missed_reminders(rstore, now=NOW, policy=never)[0]] == [other]


def test_reminders_count_digests_not_days() -> None:
    assert expires_at("reminder", NOW, POLICY) is None
    assert not is_expired("reminder", NOW - timedelta(days=30), NOW, POLICY)


# -- reminders: a delivered one nobody answered is listed once as missed, then expires (PR 3b)


def _sent_reminder(db: Path, note: str, remind_at: datetime, **kw: object) -> str:
    from iris_harness.services.notifications.store import ReminderStore

    rstore = ReminderStore(db_path=db, tz=CHICAGO)
    rstore.ensure_schema()
    row = rstore.create(
        target_kind="task", target_id=note, remind_at=remind_at, note=note, **kw  # type: ignore[arg-type]
    )
    rstore.mark_sent(row.id, delivered_channels=["telegram"], now=remind_at)
    return row.id


def test_a_delivered_reminder_from_before_today_is_listed_once_then_expires(
    tmp_path: Path,
) -> None:
    from iris_harness.services.notifications.store import ReminderStore

    db = tmp_path / "tasks.db"
    # Thu Sep 24 9:00 PM Chicago = Fri 02:00 UTC — a UTC date of "today", still yesterday.
    old = _sent_reminder(db, "Call the dentist", datetime(2026, 9, 25, 2, 0, tzinfo=UTC))
    # Fri Sep 25 6:30 AM Chicago — delivered today, before the digest: stays open.
    fresh = _sent_reminder(db, "Water the plants", datetime(2026, 9, 25, 6, 30, tzinfo=CHICAGO))
    failed = _failed_reminder(db, "Take out the recycling")
    rstore = ReminderStore(db_path=db, tz=CHICAGO)

    # Friday's digest: nothing expires yet — the delivered one is counted into this digest.
    assert sweep_unacknowledged_reminders(rstore, now=NOW, policy=POLICY, tz=CHICAGO) == []
    row = rstore.get(old)
    assert row is not None and row.status == "sent" and row.missed_digests == 1
    assert rstore.get(fresh).missed_digests == 0  # type: ignore[union-attr]
    assert rstore.get(failed).status == "failed"  # type: ignore[union-attr]

    # Saturday's digest: shown once already, so it expires with its reason.
    saturday = NOW + timedelta(days=1)
    expired = sweep_unacknowledged_reminders(rstore, now=saturday, policy=POLICY, tz=CHICAGO)
    assert old in [r.id for r in expired]
    row = rstore.get(old)
    assert row is not None and row.status == "expired"
    assert row.closed_reason == UNACKNOWLEDGED_REMINDER_REASON
    assert row.closed_reason == "expired: delivered, not acknowledged"
    # Friday's 6:30 AM one is Saturday's to list (its own day is over now).
    assert rstore.get(fresh).missed_digests == 1  # type: ignore[union-attr]


def test_zero_missed_digests_expires_a_delivered_reminder_unlisted(tmp_path: Path) -> None:
    from iris_harness.services.notifications.store import ReminderStore

    db = tmp_path / "tasks.db"
    old = _sent_reminder(db, "Call the dentist", datetime(2026, 9, 24, 9, 0, tzinfo=CHICAGO))
    rstore = ReminderStore(db_path=db, tz=CHICAGO)
    never = ExpiryPolicy(reminder_missed_digests=0)

    [expired] = sweep_unacknowledged_reminders(rstore, now=NOW, policy=never, tz=CHICAGO)

    assert expired.id == old and expired.missed_digests == 0


def test_an_unacknowledged_repeating_reminder_keeps_its_series(tmp_path: Path) -> None:
    from iris_harness.services.notifications.store import ReminderStore

    db = tmp_path / "tasks.db"
    monday = datetime(2026, 9, 21, 8, 0, tzinfo=CHICAGO)
    rid = _sent_reminder(db, "Take out the recycling", monday, recurrence="weekly")
    rstore = ReminderStore(db_path=db, tz=CHICAGO)
    [nxt] = rstore.list(statuses=["pending"])

    assert sweep_unacknowledged_reminders(rstore, now=NOW, policy=POLICY, tz=CHICAGO) == []
    [expired] = sweep_unacknowledged_reminders(
        rstore, now=NOW + timedelta(days=1), policy=POLICY, tz=CHICAGO
    )

    assert expired.id == rid and expired.status == "expired"
    # The next Monday already existed and is the only waiting row (no duplicate).
    assert [r.id for r in rstore.list(statuses=["pending"])] == [nxt.id]
    assert nxt.remind_at == (monday + timedelta(days=7)).astimezone(UTC)
