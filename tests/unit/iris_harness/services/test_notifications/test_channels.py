"""Reminder delivery — accept-then-fire (loop-proof D14).

Every reminder goes to Telegram and web push; delivered = at least one channel said
SENT; a web push with no browser (SKIPPED) is not a failure; with none accepted the
send is retried at +5, +10, +15 minutes and then the row is ``failed``.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from iris_harness.foundation.eventbus import EventBus
from iris_harness.services.channels import ChannelGateway, ChannelMessage, DeliveryStatus
from iris_harness.services.channels.models import DeliveryReceipt
from iris_harness.services.notifications.channels import (
    DEFAULT_REMINDER_CHANNELS,
    deliver_due,
    load_reminder_channels,
    resolve_channels,
)
from iris_harness.services.notifications.events import REMINDER_FIRED, ReminderFiredPayload
from iris_harness.services.notifications.store import ReminderStore
from iris_harness.services.tasks.store import TaskStore

CT = ZoneInfo("America/Chicago")
T = datetime(2026, 9, 28, 13, 0, tzinfo=UTC)  # Mon Sep 28, 8:00 AM CDT
BOTH = ["telegram", "web_push"]


class _Channel:
    """A connector that answers with a scripted status and records what it got."""

    def __init__(self, name: str, status: DeliveryStatus, *, chat_id: str = "") -> None:
        self.name = name
        self.status = status
        self.sent: list[ChannelMessage] = []
        if chat_id:
            self._default_chat_id = chat_id

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        self.sent.append(message)
        mid = str(100 + len(self.sent)) if self.status is DeliveryStatus.SENT else ""
        error = "" if self.status is DeliveryStatus.SENT else f"{self.name} {self.status.value}"
        return DeliveryReceipt(channel=self.name, status=self.status, message_id=mid, error=error)


def _gateway(
    telegram: DeliveryStatus = DeliveryStatus.SENT, push: DeliveryStatus = DeliveryStatus.SENT
) -> tuple[ChannelGateway, _Channel, _Channel]:
    gateway = ChannelGateway()
    tg = _Channel("telegram", telegram, chat_id="4242")
    wp = _Channel("web_push", push)
    gateway.register(tg)
    gateway.register(wp)
    return gateway, tg, wp


def _calendar(data_dir: Path, event_id: str, summary: str) -> None:
    with sqlite3.connect(data_dir / "calendar.db") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS calendar_events (id TEXT, summary TEXT)")
        conn.execute("INSERT INTO calendar_events VALUES (?, ?)", (event_id, summary))


@pytest.fixture
def store(tmp_path: Path) -> ReminderStore:
    s = ReminderStore(db_path=tmp_path / "tasks.db", tz=CT)
    s.ensure_schema()
    _calendar(tmp_path, "ev-1", "Take out the recycling")
    return s


def _weekly(store: ReminderStore) -> str:
    return store.create(target_kind="event", target_id="ev-1", remind_at=T, recurrence="weekly").id


def test_both_channels_get_it_and_the_row_is_sent(
    store: ReminderStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(store.db_path.parent.parent)  # calendar.db is found by data dir
    gateway, tg, wp = _gateway()
    rid = _weekly(store)

    report = deliver_due(store, gateway, channels=BOTH, tz=CT, now=T)

    assert report.sent == [rid] and report.ok
    assert tg.sent[0].body == (
        "⏰ <b>Take out the recycling</b>\nMon Sep 28 · 8:00 AM · repeats every Monday"
    )
    assert tg.sent[0].metadata["parse_mode"] == "HTML"
    push = wp.sent[0]
    assert push.subject == "⏰ Take out the recycling"
    assert push.body == "8:00 AM · repeats every Monday"
    assert push.metadata["tag"] == f"reminder:{rid}"
    assert push.metadata["url"] == f"/reminders/{rid}"
    row = store.get(rid)
    assert row is not None and row.status == "sent" and row.fired_at == T
    assert row.delivered_channels == ("telegram", "web_push")
    assert row.message_refs[0] == {"channel": "telegram", "chat_id": "4242", "message_id": "101"}
    # The next Monday exists straight away (same series).
    [nxt] = [r for r in store.list(statuses=["pending"]) if r.series_id == rid]
    assert nxt.remind_at == T + timedelta(days=7)
    assert report.created_next == [nxt.id]


def test_two_reminders_never_share_a_push_tag(store: ReminderStore) -> None:
    gateway, _, wp = _gateway()
    store.create(target_kind="event", target_id="ev-1", remind_at=T)
    store.create(target_kind="event", target_id="ev-1", remind_at=T)
    deliver_due(store, gateway, channels=BOTH, tz=CT, now=T)
    tags = {m.metadata["tag"] for m in wp.sent}
    assert len(tags) == 2


def test_a_reminder_push_offers_done_and_snooze(store: ReminderStore) -> None:
    gateway, tg, wp = _gateway()
    rid = _weekly(store)
    deliver_due(store, gateway, channels=BOTH, tz=CT, now=T)
    meta = wp.sent[0].metadata
    assert meta["actions"] == [
        {"action": "done", "title": "Done"},
        {"action": "snooze_1h", "title": "Snooze 1h"},
    ]
    assert meta["reminder_id"] == rid and meta["url"] == f"/reminders/{rid}"
    # The push buttons are the web worker's; Telegram's keyboard is its own.
    assert "actions" not in tg.sent[0].metadata


def test_no_browser_subscribed_is_not_a_failure(store: ReminderStore) -> None:
    gateway, _, _ = _gateway(push=DeliveryStatus.SKIPPED)
    rid = _weekly(store)
    report = deliver_due(store, gateway, channels=BOTH, tz=CT, now=T)
    assert report.sent == [rid]
    row = store.get(rid)
    assert row is not None and row.delivered_channels == ("telegram",)
    # A skipped channel on an accepted send is not this reminder's error.
    assert row.last_error is None


def test_push_alone_is_a_delivery(store: ReminderStore) -> None:
    gateway, _, _ = _gateway(telegram=DeliveryStatus.FAILED)
    rid = _weekly(store)
    report = deliver_due(store, gateway, channels=BOTH, tz=CT, now=T)
    row = store.get(rid)
    assert report.sent == [rid]
    assert row is not None and row.status == "sent" and row.delivered_channels == ("web_push",)
    assert row.last_error is None  # delivered: the failed channel is logged, not kept


def test_retries_at_5_10_15_then_failed(store: ReminderStore) -> None:
    """The owner's timeline: attempts at 8:00, 8:05, 8:10, 8:15, then failed."""
    gateway, tg, wp = _gateway(telegram=DeliveryStatus.FAILED, push=DeliveryStatus.SKIPPED)
    rid = _weekly(store)
    timeline: list[str] = []
    for minute in range(0, 21):  # the heartbeat runs every minute
        now = T + timedelta(minutes=minute)
        report = deliver_due(store, gateway, channels=BOTH, tz=CT, now=now)
        if report.retrying or report.failed:
            state = "retrying" if report.retrying else "failed"
            timeline.append(f"{now.astimezone(CT):%H:%M} {state}")

    assert timeline == [
        "08:00 retrying",
        "08:05 retrying",
        "08:10 retrying",
        "08:15 failed",
    ]
    assert len(tg.sent) == 4 and len(wp.sent) == 4
    row = store.get(rid)
    assert row is not None and row.status == "failed" and row.attempts == 4
    assert "telegram: failed" in (row.last_error or "")
    # A failed occurrence does not end the series.
    assert [r.remind_at for r in store.list(statuses=["pending"])] == [T + timedelta(days=7)]


def test_a_channel_that_comes_back_delivers_the_retry(store: ReminderStore) -> None:
    gateway, tg, _ = _gateway(telegram=DeliveryStatus.FAILED, push=DeliveryStatus.SKIPPED)
    rid = store.create(target_kind="event", target_id="ev-1", remind_at=T).id
    deliver_due(store, gateway, channels=BOTH, tz=CT, now=T)
    tg.status = DeliveryStatus.SENT
    report = deliver_due(store, gateway, channels=BOTH, tz=CT, now=T + timedelta(minutes=5))
    assert report.sent == [rid]
    row = store.get(rid)
    assert row is not None and row.status == "sent" and row.attempts == 2
    assert row.last_error is None  # the first attempt's failure is cleared by the send


def test_fired_is_emitted_only_after_a_channel_accepted(tmp_path: Path) -> None:
    bus = EventBus()
    fired: list[ReminderFiredPayload] = []
    bus.on(REMINDER_FIRED, fired.append)
    store = ReminderStore(db_path=tmp_path / "tasks.db", bus=bus, tz=CT)
    store.ensure_schema()
    rid = store.create(target_kind="task", target_id="t", remind_at=T, note="call").id

    down, _, _ = _gateway(telegram=DeliveryStatus.FAILED, push=DeliveryStatus.FAILED)
    deliver_due(store, down, channels=BOTH, tz=CT, now=T)
    assert fired == []

    up, _, _ = _gateway()
    deliver_due(store, up, channels=BOTH, tz=CT, now=T + timedelta(minutes=5))
    assert [p.reminder_id for p in fired] == [rid]


def test_nothing_due_sends_nothing(store: ReminderStore) -> None:
    gateway, tg, wp = _gateway()
    store.create(target_kind="event", target_id="ev-1", remind_at=T + timedelta(minutes=1))
    report = deliver_due(store, gateway, channels=BOTH, tz=CT, now=T)
    assert report.ok and report.sent == [] and tg.sent == [] and wp.sent == []


def test_no_channel_registered_is_a_failed_attempt(store: ReminderStore) -> None:
    rid = store.create(target_kind="event", target_id="ev-1", remind_at=T).id
    report = deliver_due(store, ChannelGateway(), channels=[], tz=CT, now=T)
    assert report.retrying == [rid]
    row = store.get(rid)
    assert row is not None and row.last_error == "no reminder channel is registered"


def test_a_task_reminder_reads_its_title_and_note(tmp_path: Path) -> None:
    tasks_db = tmp_path / "tasks.db"
    task = TaskStore(db_path=tasks_db)
    task.ensure_schema()
    t = task.create(title="Renew passport")
    store = ReminderStore(db_path=tasks_db, tz=CT)
    store.ensure_schema()
    store.create(target_kind="task", target_id=t.id, remind_at=T, note="starts in 15 min")
    gateway = ChannelGateway()
    console = _Channel("console", DeliveryStatus.SENT)
    gateway.register(console)

    deliver_due(store, gateway, channels=["console"], tz=CT, now=T)

    assert console.sent[0].body == "⏰ Renew passport\nMon Sep 28 · 8:00 AM · starts in 15 min"


def test_a_late_push_names_the_day(store: ReminderStore) -> None:
    gateway, _, wp = _gateway()
    store.create(target_kind="event", target_id="ev-1", remind_at=T)
    deliver_due(store, gateway, channels=BOTH, tz=CT, now=T + timedelta(days=1))
    assert wp.sent[0].body == "Mon Sep 28 · 8:00 AM"


def test_resolve_channels_keeps_registered_and_falls_back() -> None:
    gateway, _, _ = _gateway()
    assert resolve_channels(gateway, ["telegram", "sms", "web_push"]) == BOTH
    console_only = ChannelGateway()
    console_only.register(_Channel("console", DeliveryStatus.SENT))
    assert resolve_channels(console_only, BOTH, default_channel="console") == ["console"]
    assert resolve_channels(console_only, BOTH, default_channel=None) == []


def test_reminder_channels_config(tmp_path: Path) -> None:
    assert load_reminder_channels(tmp_path) == DEFAULT_REMINDER_CHANNELS
    (tmp_path / "notifications.yaml").write_text("reminder_channels: [telegram]\n")
    assert load_reminder_channels(tmp_path) == ("telegram",)
    (tmp_path / "notifications.yaml").write_text(": not yaml [")
    assert load_reminder_channels(tmp_path) == DEFAULT_REMINDER_CHANNELS
    shipped = Path(__file__).resolve().parents[5] / "config"
    assert load_reminder_channels(shipped) == ("telegram", "web_push")


def test_the_telegram_reminder_carries_done_and_snooze_buttons(store: ReminderStore) -> None:
    """PR 3b: four buttons, each a command under Telegram's 64-byte callback limit; the
    push message carries none (its actions are the service worker's)."""
    gateway, tg, wp = _gateway()
    rid = _weekly(store)

    deliver_due(store, gateway, channels=BOTH, tz=CT, now=T)

    keyboard = tg.sent[0].metadata["inline_keyboard"]
    assert keyboard == [
        [
            {"text": "✅ Done", "callback_data": f"/done {rid}"},
            {"text": "⏰ 1 hour", "callback_data": f"/snooze {rid} 1h"},
        ],
        [
            {"text": "10 min", "callback_data": f"/snooze {rid} 10m"},
            {"text": "Tomorrow 9am", "callback_data": f"/snooze {rid} tomorrow_9am"},
        ],
    ]
    for row in keyboard:  # type: ignore[union-attr]
        for button in row:
            assert len(button["callback_data"].encode("utf-8")) <= 64
    assert "inline_keyboard" not in wp.sent[0].metadata


def test_a_bill_reminder_is_worded_from_its_row_with_paid_buttons(store: ReminderStore) -> None:
    """PR 4: a bill's reminder says its own headline and line, with ✅ Paid + Not yet on
    its question — the same words on Telegram and the push."""
    gateway, tg, wp = _gateway()
    row = store.create(
        target_kind="bill",
        target_id="due-1",
        remind_at=T,
        note="❓ Did you pay Example Card $35.00?",
        dedupe_key="bill:due-1:ask1",
        meta={
            "entity": "Example Card",
            "amount": "$35.00",
            "statement": "$1,284.50",
            "due": "2026-09-27",
            "step": "ask1",
        },
    )

    deliver_due(store, gateway, channels=BOTH, tz=CT, now=T)

    message = tg.sent[0]
    assert message.body == (
        "<b>❓ Did you pay Example Card $35.00?</b>\n"
        "It was due Sun Sep 27 · no payment email seen"
    )
    assert message.metadata["inline_keyboard"] == [
        [
            {"text": "✅ Paid", "callback_data": f"/paid {row.id}"},
            {"text": "Not yet", "callback_data": f"/notyet {row.id}"},
        ]
    ]
    push = wp.sent[0]
    assert push.subject == "❓ Did you pay Example Card $35.00?"
    assert push.body == "It was due Sun Sep 27 · no payment email seen"
    assert push.metadata["actions"] == [
        {"action": "paid", "title": "Paid"},
        {"action": "not_yet", "title": "Not yet"},
    ]
    assert push.metadata["url"] == f"/reminders/{row.id}"
    assert store.get(row.id).status == "sent"  # type: ignore[union-attr]


def test_a_hand_made_bill_row_keeps_the_plain_reminder_message(store: ReminderStore) -> None:
    gateway, tg, _wp = _gateway()
    store.create(target_kind="bill", target_id="b", remind_at=T, note="pay the card")
    deliver_due(store, gateway, channels=BOTH, tz=CT, now=T)
    assert tg.sent[0].body.startswith("⏰ <b>pay the card</b>")
