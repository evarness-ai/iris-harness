"""Done and Snooze on a reminder from Telegram (loop-proof D14, PR 3b).

A tap on the reminder's buttons, a typed ``/done`` / ``/snooze``, or a reply to the
reminder's message ("snooze 1h", "done") — end to end at the poller, over the real
reminder store written in process (``LocalReminderBackend``). Approvals still answer
first; anything that is not a Done or a Snooze on a reminder is chat, unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from iris_harness.foundation.eventbus import EventBus
from iris_harness.foundation.paths import repo_root
from iris_harness.services.channels import ChannelMessage, DeliveryReceipt, DeliveryStatus
from iris_harness.services.channels.approval_commands import CommandReply
from iris_harness.services.channels.connectors.telegram_poller import TelegramPoller
from iris_harness.services.channels.reminder_commands import (
    GONE,
    NOT_UNDERSTOOD,
    PAID,
    ChainedCommands,
    LocalReminderBackend,
    ReminderCommands,
    ReminderOutcome,
    done_text,
    not_yet_text,
    snooze_text,
)
from iris_harness.services.notifications.events import (
    REMINDER_COMPLETED,
    REMINDER_SNOOZED,
)
from iris_harness.services.notifications.store import ReminderStore

CT = ZoneInfo("America/Chicago")
CHAT = "555"


class _FakeConnector:
    def __init__(self) -> None:
        self.sent: list[ChannelMessage] = []
        self.answered: list[str] = []
        self.cleared: list[tuple[str, str]] = []

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        self.sent.append(message)
        return DeliveryReceipt(channel="telegram", status=DeliveryStatus.SENT)

    def answer_callback(self, callback_query_id: str, *, text: str = "") -> None:
        self.answered.append(callback_query_id)

    def clear_inline_keyboard(self, chat_id: str, message_id: str) -> None:
        self.cleared.append((chat_id, message_id))


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_TZ", "America/Chicago")
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(repo_root() / "config"))


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def store(tmp_path: Path) -> ReminderStore:
    s = ReminderStore(db_path=tmp_path / "tasks.db", tz=CT)
    s.ensure_schema()
    return s


def _sent(store: ReminderStore, *, message_id: str = "70", recurrence: str | None = None) -> str:
    due = datetime.now(UTC) - timedelta(minutes=2)
    r = store.create(
        target_kind="task", target_id="t", remind_at=due, note="Call", recurrence=recurrence
    )
    store.mark_sent(
        r.id,
        delivered_channels=["telegram"],
        message_refs=[{"channel": "telegram", "chat_id": CHAT, "message_id": message_id}],
    )
    return r.id


def _poller(
    conn: _FakeConnector, store: ReminderStore, chat: list[str], bus: EventBus | None = None
) -> TelegramPoller:
    reminders = ReminderCommands(
        LocalReminderBackend(data_dir=store.db_path.parent, bus=bus), allowed_users=frozenset()
    )
    approvals_seen: list[str] = []

    def approvals(text: str, user_id: str) -> CommandReply | None:
        approvals_seen.append(text)
        return CommandReply("approved") if text.startswith("/approve") else None

    return TelegramPoller(
        bot_token="t",
        connector=conn,  # type: ignore[arg-type]
        chat_handler=lambda text, sid, _audience: chat.append(text) or "chat reply",
        command_handler=ChainedCommands(approvals, reminders),
        allowed_chat_ids=frozenset({CHAT}),
    )


def _tap(poller: TelegramPoller, data: str, *, message_id: int = 70) -> None:
    poller._handle_update(
        {
            "callback_query": {
                "id": "cbq",
                "data": data,
                "from": {"id": 42},
                "message": {"message_id": message_id, "chat": {"id": int(CHAT)}},
            }
        }
    )


def _say(poller: TelegramPoller, text: str, *, reply_to: int | None = None) -> None:
    message: dict[str, Any] = {"text": text, "from": {"id": 42}, "chat": {"id": int(CHAT)}}
    if reply_to is not None:
        message["reply_to_message"] = {"message_id": reply_to, "text": "⏰ Call"}
    poller._handle_update({"message": message})


def test_done_button_closes_it_and_clears_the_buttons(store: ReminderStore, bus: EventBus) -> None:
    completed: list[Any] = []
    bus.on(REMINDER_COMPLETED, completed.append)
    conn, chat = _FakeConnector(), []
    rid = _sent(store, recurrence="weekly")
    poller = _poller(conn, store, chat, bus)
    try:
        _tap(poller, f"/done {rid}")
    finally:
        poller.stop()
    assert chat == []
    assert conn.cleared == [(CHAT, "70")]
    assert store.get(rid).status == "done"  # type: ignore[union-attr]
    [nxt] = [r for r in store.list(statuses=["pending"]) if r.series_id == rid]
    local = nxt.remind_at.astimezone(CT)
    assert conn.sent[-1].body == (
        f"✅ Marked done. Next: {local:%a %b} {local.day} "
        f"{local.hour % 12 or 12}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}."
    )
    assert [p.source for p in completed] == ["telegram"]


def test_snooze_buttons_move_it(store: ReminderStore, bus: EventBus) -> None:
    snoozed: list[Any] = []
    bus.on(REMINDER_SNOOZED, snoozed.append)
    conn, chat = _FakeConnector(), []
    rid = _sent(store)
    poller = _poller(conn, store, chat, bus)
    before = datetime.now(UTC)
    try:
        _tap(poller, f"/snooze {rid} 1h")
    finally:
        poller.stop()
    row = store.get(rid)
    assert row is not None and row.status == "pending"
    assert before + timedelta(minutes=59) < row.remind_at < before + timedelta(minutes=61)
    assert conn.sent[-1].body.startswith("⏰ Snoozed — I'll remind you ")
    assert [(p.reminder_id, p.source) for p in snoozed] == [(rid, "telegram")]


def test_a_reply_to_the_reminder_acts_on_it(store: ReminderStore) -> None:
    conn, chat = _FakeConnector(), []
    rid = _sent(store, message_id="71")
    other = _sent(store, message_id="72")
    poller = _poller(conn, store, chat)
    try:
        _say(poller, "snooze 10 min", reply_to=71)
        _say(poller, "done", reply_to=72)
    finally:
        poller.stop()
    assert chat == []
    assert store.get(rid).status == "pending"  # type: ignore[union-attr]
    assert store.get(other).status == "done"  # type: ignore[union-attr]
    assert [m.body for m in conn.sent][-1] == "✅ Marked done."


def test_other_replies_and_messages_are_chat_unchanged(store: ReminderStore) -> None:
    conn, chat = _FakeConnector(), []
    rid = _sent(store, message_id="71")
    poller = _poller(conn, store, chat)
    try:
        _say(poller, "thanks, what's next today?", reply_to=71)  # not a Done / Snooze
        _say(poller, "done", reply_to=99)  # not a reminder's message
        _say(poller, "snooze 1h")  # not a reply at all
    finally:
        poller.stop()
    assert chat == ["thanks, what's next today?", "done", "snooze 1h"]
    assert store.get(rid).status == "sent"  # type: ignore[union-attr]


def test_approvals_still_answer_first(store: ReminderStore) -> None:
    conn, chat = _FakeConnector(), []
    poller = _poller(conn, store, chat)
    try:
        _tap(poller, "/approve 3f9c0d2a-0000-4000-8000-000000000001")
    finally:
        poller.stop()
    assert conn.sent[-1].body == "approved"


def test_an_ended_or_unknown_reminder_says_so(store: ReminderStore) -> None:
    conn, chat = _FakeConnector(), []
    rid = _sent(store)
    store.expire(rid, "expired: test")
    poller = _poller(conn, store, chat)
    try:
        _tap(poller, f"/snooze {rid} 1h")
        _tap(poller, "/done 3f9c0d2a-0000-4000-8000-000000000009")
    finally:
        poller.stop()
    assert [m.body for m in conn.sent] == [GONE, GONE]


def test_words_it_cannot_read_are_said(store: ReminderStore) -> None:
    rid = _sent(store)
    handler = ReminderCommands(
        LocalReminderBackend(data_dir=store.db_path.parent), allowed_users=frozenset()
    )
    reply = handler(f"/snooze {rid} whenever", "42")
    assert reply is not None and reply.text == NOT_UNDERSTOOD
    assert store.get(rid).status == "sent"  # type: ignore[union-attr]


def test_an_unlisted_user_cannot_act(store: ReminderStore) -> None:
    rid = _sent(store, message_id="71")
    handler = ReminderCommands(
        LocalReminderBackend(data_dir=store.db_path.parent), allowed_users=frozenset({"42"})
    )
    for reply in (
        handler(f"/done {rid}", "evil"),
        handler.on_reply("done", "evil", CHAT, "71"),
    ):
        assert reply is not None and "permission" in reply.text
    assert store.get(rid).status == "sent"  # type: ignore[union-attr]


def test_a_failing_backend_is_said_not_swallowed() -> None:
    class _Broken:
        def done(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("api down")

    handler = ReminderCommands(_Broken(), allowed_users=frozenset())  # type: ignore[arg-type]
    reply = handler("/done 3f9c0d2a-0000-4000-8000-000000000001", "42")
    assert reply is not None and reply.text.startswith("Couldn't update that reminder")


def test_the_wording() -> None:
    now = datetime(2026, 9, 28, 13, 3, tzinfo=UTC)  # Mon 8:03 AM CDT
    at = datetime(2026, 9, 28, 14, 3, tzinfo=UTC)
    assert snooze_text(ReminderOutcome(until=at), CT, now) == (
        "⏰ Snoozed — I'll remind you at 9:03 AM."
    )
    tomorrow = datetime(2026, 9, 29, 14, 0, tzinfo=UTC)
    assert snooze_text(ReminderOutcome(until=tomorrow), CT, now) == (
        "⏰ Snoozed — I'll remind you tomorrow at 9:00 AM."
    )
    later = datetime(2026, 10, 1, 23, 0, tzinfo=UTC)
    assert snooze_text(ReminderOutcome(until=later), CT, now) == (
        "⏰ Snoozed — I'll remind you on Thu Oct 1 at 6:00 PM."
    )
    nxt = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)
    assert done_text(ReminderOutcome(next_at=nxt), CT) == "✅ Marked done. Next: Mon Oct 5 8:00 AM."
    assert done_text(ReminderOutcome(), CT) == "✅ Marked done."


# ── a bill's reminder (loop-proof PR 4) ─────────────────────────────────────


def _bill(store: ReminderStore, step: str, *, message_id: str = "80") -> str:
    """A delivered bill reminder due yesterday (Chicago), step ``step``."""
    now = datetime.now(UTC)
    due = (now.astimezone(CT) - timedelta(days=1)).date()
    r = store.create(
        target_kind="bill",
        target_id="due-1",
        remind_at=now - timedelta(minutes=2),
        note="❓ Did you pay Example Card $35.00?",
        dedupe_key=f"bill:due-1:{step}",
        meta={
            "entity": "Example Card",
            "amount": "$35.00",
            "due": due.isoformat(),
            "at": "09:00",
            "step": step,
        },
    )
    store.mark_sent(
        r.id,
        delivered_channels=["telegram"],
        message_refs=[{"channel": "telegram", "chat_id": CHAT, "message_id": message_id}],
    )
    return r.id


def test_paid_button_closes_the_bill_reminder(store: ReminderStore, bus: EventBus) -> None:
    completed: list[Any] = []
    bus.on(REMINDER_COMPLETED, completed.append)
    conn, chat = _FakeConnector(), []
    rid = _bill(store, "ask1")
    poller = _poller(conn, store, chat, bus)
    try:
        _tap(poller, f"/paid {rid}", message_id=80)
    finally:
        poller.stop()
    assert chat == []
    assert store.get(rid).status == "done"  # type: ignore[union-attr]
    assert conn.sent[-1].body == PAID
    [done] = completed
    assert (done.target_kind, done.target_id, done.source) == ("bill", "due-1", "telegram")


def test_not_yet_button_acknowledges_and_says_when_it_asks_again(store: ReminderStore) -> None:
    conn, chat = _FakeConnector(), []
    rid = _bill(store, "ask1")
    last = _bill(store, "ask3", message_id="81")
    poller = _poller(conn, store, chat)
    try:
        _tap(poller, f"/notyet {rid}", message_id=80)
        _tap(poller, f"/notyet {last}", message_id=81)
    finally:
        poller.stop()
    row = store.get(rid)
    assert row is not None and row.status == "expired" and row.closed_reason == "not yet: telegram"
    first, second = [m.body for m in conn.sent][-2:]
    # ask1 of a bill due yesterday: the next ask is tomorrow 9:00.
    assert first == "👍 Noted — not paid yet. I'll ask again tomorrow at 9:00 AM."
    assert second == "👍 Noted — not paid yet. That was the last ask: it stays in the digest."


def test_not_yet_on_a_reminder_that_is_not_a_question_is_refused(store: ReminderStore) -> None:
    conn, chat = _FakeConnector(), []
    rid = _bill(store, "dayof")
    poller = _poller(conn, store, chat)
    try:
        _tap(poller, f"/notyet {rid}", message_id=80)
    finally:
        poller.stop()
    assert store.get(rid).status == "sent"  # type: ignore[union-attr]
    assert conn.sent[-1].body == NOT_UNDERSTOOD


def test_replies_paid_and_not_yet(store: ReminderStore) -> None:
    conn, chat = _FakeConnector(), []
    paid = _bill(store, "ask1", message_id="90")
    asked = _bill(store, "ask2", message_id="91")
    plain = _sent(store, message_id="92")
    poller = _poller(conn, store, chat)
    try:
        _say(poller, "paid", reply_to=90)
        _say(poller, "not yet", reply_to=91)
        _say(poller, "not yet", reply_to=92)  # not a bill's question: chat
    finally:
        poller.stop()
    assert store.get(paid).status == "done"  # type: ignore[union-attr]
    assert store.get(asked).closed_reason == "not yet: telegram"  # type: ignore[union-attr]
    assert store.get(plain).status == "sent"  # type: ignore[union-attr]
    assert chat == ["not yet"]
    bodies = [m.body for m in conn.sent]
    assert PAID in bodies
    assert any(b.startswith("👍 Noted — not paid yet.") for b in bodies)


def test_not_yet_text_wording() -> None:
    now = datetime(2026, 10, 13, 14, 5, tzinfo=UTC)
    nxt = datetime(2026, 10, 14, 14, 0, tzinfo=UTC)
    assert not_yet_text(ReminderOutcome(next_at=nxt), CT, now) == (
        "👍 Noted — not paid yet. I'll ask again tomorrow at 9:00 AM."
    )


# ── Paid from ANY of the bill's messages (PR 4 demo, 2026-09-25) ──────────────


@pytest.mark.parametrize("ended", ["swept", "not_yet"])
def test_paid_on_a_swept_or_not_yet_message_still_closes_the_bill(
    store: ReminderStore, bus: EventBus, ended: str
) -> None:
    """The demo: Paid on the Oct 15 message after the 07:00 sweep expired it, and Paid on
    the Oct 13 message the owner had answered Not yet, both said "isn't open any more"."""
    completed: list[Any] = []
    bus.on(REMINDER_COMPLETED, completed.append)
    conn, chat = _FakeConnector(), []
    rid = _bill(store, "ask3")
    if ended == "swept":
        store.expire(rid, "expired: delivered, not acknowledged")
    else:
        store.acknowledge(rid, source="telegram")
    poller = _poller(conn, store, chat, bus)
    try:
        _tap(poller, f"/paid {rid}", message_id=80)
    finally:
        poller.stop()
    assert conn.sent[-1].body == PAID
    assert store.get(rid).status == "done"  # type: ignore[union-attr]
    [done] = completed
    assert (done.target_kind, done.target_id) == ("bill", "due-1")


def test_paid_on_a_bill_already_marked_paid_says_so(store: ReminderStore, bus: EventBus) -> None:
    completed: list[Any] = []
    bus.on(REMINDER_COMPLETED, completed.append)
    conn, chat = _FakeConnector(), []
    tapped = _bill(store, "t3d", message_id="80")
    _bill(store, "ask1", message_id="81")
    store.close_for_target("bill", "due-1", reason="closed: paid (chat)")
    poller = _poller(conn, store, chat, bus)
    try:
        _tap(poller, f"/paid {tapped}", message_id=80)
        _say(poller, "paid", reply_to=81)
    finally:
        poller.stop()
    assert [m.body for m in conn.sent][-2:] == ["✅ Example Card is already marked paid."] * 2
    assert GONE not in [m.body for m in conn.sent]
    assert completed == []
