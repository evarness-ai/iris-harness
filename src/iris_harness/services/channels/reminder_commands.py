"""Done and Snooze on a reminder from a chat surface (loop-proof D14, PR 3b).

A reminder's Telegram message carries four buttons whose callback data is a command:
``/done <id>`` and ``/snooze <id> 1h|10m|tomorrow_9am`` (each under Telegram's 64-byte
callback limit). A typed command takes the same path, and so does a *reply* to the
reminder's message — "snooze 1h", "tomorrow at 6pm", "done" — resolved by the message
it answers (:meth:`ReminderCommands.on_reply`). The poller tries these after the
approval commands and before chat; anything else is chat, unchanged.

A bill's reminder (loop-proof PR 4) carries ``/paid <id>`` (a Done, answered "Marked
paid") and, on its "Did you pay?" question, ``/notyet <id>``: the backend's snooze with
the ``not_yet`` choice, which acknowledges the question instead of moving it — the next
morning's question is its own row. Typed replies "paid" and "not yet" do the same.
Paid works on any of the bill's messages while the bill is open — one the digest sweep
aged out, or one answered Not yet — and on a bill already closed as paid the reply is
"✅ Discover is already marked paid.", never "isn't open any more".

The pattern is ``approval_commands``': the parsing, the user check and the wording
live here once; where the answer goes is the backend's business. The channel gateway
is a separate process, so it answers through the API (``ApiReminderBackend`` in
``server/channel_gateway/telegram.py``); the runtime's own poller writes the store in
process (:class:`LocalReminderBackend`).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .approval_commands import CommandHandler, CommandReply, Keyboard, load_allowed_users

if TYPE_CHECKING:
    from iris_harness.foundation.eventbus import EventBus
    from iris_harness.services.notifications.models import Reminder
    from iris_harness.services.notifications.store import ReminderStore

logger = logging.getLogger(__name__)

_ID = r"([0-9a-f-]{36})"
_DONE = re.compile(rf"^/done\s+{_ID}\s*$", re.IGNORECASE)
_SNOOZE = re.compile(rf"^/snooze\s+{_ID}\s+(\S.{{0,63}})$", re.IGNORECASE)
_PAID = re.compile(rf"^/paid\s+{_ID}\s*$", re.IGNORECASE)
_NOT_YET = re.compile(rf"^/notyet\s+{_ID}\s*$", re.IGNORECASE)

#: The snooze choice that means Not yet (``notifications.bills.NOT_YET_CHOICE``).
NOT_YET_CHOICE = "not_yet"

NOT_UNDERSTOOD = "I didn't catch that. Reply “done”, “10m”, “1h” or “tomorrow 9am”."
GONE = "That reminder isn't open any more: it was deleted or already ended."
FAILED = "Couldn't update that reminder just now. Try again in a minute."
PAID = "✅ Marked paid. No more reminders for this bill."


@dataclass(frozen=True)
class ReminderOutcome:
    """What a Done or Snooze did, enough to word the reply."""

    until: datetime | None = None  # a snooze: when it fires again
    next_at: datetime | None = None  # a Done on a repeating reminder: the next one
    #: Paid on a bill already closed as paid (PR 4): nothing changed; ``entity`` names it.
    already_paid: bool = False
    entity: str = ""


class ReminderNotFound(LookupError):
    """No such reminder (or no reminder sent that message)."""


class SnoozeNotUnderstood(ValueError):
    """The snooze words were not understood."""


class ReminderBackend(Protocol):
    """Where a Done / Snooze goes: the API (channel gateway) or the store (in process).

    Each method raises ``ReminderNotFound`` for an unknown reminder or message,
    ``SnoozeNotUnderstood`` for words it cannot read, ``ValueError`` when the reminder
    already ended."""

    def done(self, reminder_id: str, *, source: str) -> ReminderOutcome: ...

    def snooze(self, reminder_id: str, spoken: str, *, source: str) -> ReminderOutcome: ...

    def done_by_message(
        self, channel: str, chat_id: str, message_id: str, *, source: str
    ) -> ReminderOutcome: ...

    def snooze_by_message(
        self, channel: str, chat_id: str, message_id: str, spoken: str, *, source: str
    ) -> ReminderOutcome: ...


def reminder_keyboard(reminder_id: str) -> Keyboard:
    """The buttons on a reminder's Telegram message (the prototype's four)."""
    return [
        [
            {"text": "✅ Done", "callback_data": f"/done {reminder_id}"},
            {"text": "⏰ 1 hour", "callback_data": f"/snooze {reminder_id} 1h"},
        ],
        [
            {"text": "10 min", "callback_data": f"/snooze {reminder_id} 10m"},
            {"text": "Tomorrow 9am", "callback_data": f"/snooze {reminder_id} tomorrow_9am"},
        ],
    ]


# ── wording ────────────────────────────────────────────────────────────────


def _clock(local: datetime) -> str:
    hour = local.hour % 12 or 12
    return f"{hour}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}"


def _day_clock(local: datetime) -> str:
    """ "Mon Oct 5 8:00 AM"."""
    return f"{local:%a %b} {local.day} {_clock(local)}"


def done_text(outcome: ReminderOutcome, tz: tzinfo) -> str:
    """ "✅ Marked done. Next: Mon Oct 5 8:00 AM." (the second sentence when it repeats)."""
    if outcome.next_at is None:
        return "✅ Marked done."
    return f"✅ Marked done. Next: {_day_clock(outcome.next_at.astimezone(tz))}."


def not_yet_text(outcome: ReminderOutcome, tz: tzinfo, now: datetime) -> str:
    """ "👍 Noted — not paid yet. I'll ask again tomorrow at 9:00 AM." (``next_at`` is
    the next question), or, after the last ask, that it stays in the digest."""
    if outcome.next_at is None:
        return "👍 Noted — not paid yet. That was the last ask: it stays in the digest."
    local = outcome.next_at.astimezone(tz)
    today = now.astimezone(tz).date()
    if local.date() == today + timedelta(days=1):
        when = f"tomorrow at {_clock(local)}"
    elif local.date() == today:
        when = f"at {_clock(local)}"
    else:
        when = f"on {local:%a %b} {local.day} at {_clock(local)}"
    return f"👍 Noted — not paid yet. I'll ask again {when}."


def snooze_text(outcome: ReminderOutcome, tz: tzinfo, now: datetime) -> str:
    """ "⏰ Snoozed — I'll remind you at 9:03 AM." / "… tomorrow at 9:00 AM." /
    "… on Tue Sep 29 at 9:00 AM."."""
    if outcome.until is None:
        return "⏰ Snoozed."
    local = outcome.until.astimezone(tz)
    today = now.astimezone(tz).date()
    if local.date() == today:
        when = f"at {_clock(local)}"
    elif local.date() == today + timedelta(days=1):
        when = f"tomorrow at {_clock(local)}"
    else:
        when = f"on {local:%a %b} {local.day} at {_clock(local)}"
    return f"⏰ Snoozed — I'll remind you {when}."


def _paid_text(outcome: ReminderOutcome) -> str:
    """ "✅ Marked paid. …", or "✅ Discover is already marked paid." when it was."""
    if outcome.already_paid:
        from iris_harness.services.notifications.bills import (
            already_paid_text,
        )

        return already_paid_text(outcome.entity)
    return PAID


def _owner_zone() -> tzinfo:
    from iris_harness.services.digest.settings import iris_timezone

    return iris_timezone()


# ── the handler ────────────────────────────────────────────────────────────


class ReminderCommands:
    """``/done`` and ``/snooze`` commands, and replies to a reminder's message.

    Call it as a ``CommandHandler`` (``(text, user_id) -> CommandReply | None``);
    :meth:`on_reply` answers a reply to one of the bot's messages. ``allowed_users``
    is the finer check within the allowed chat (``None`` reads the approvals' file;
    empty means everyone in the chat), as for approvals.
    """

    def __init__(
        self,
        backend: ReminderBackend,
        *,
        channel: str = "telegram",
        allowed_users: frozenset[str] | None = None,
        tz: tzinfo | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._backend = backend
        self._channel = channel
        self._allowed_users = allowed_users
        self._tz = tz
        self._clock = clock or (lambda: datetime.now(UTC))

    def _zone(self) -> tzinfo:
        return self._tz or _owner_zone()

    def _refused(self, user_id: str) -> CommandReply | None:
        users = load_allowed_users() if self._allowed_users is None else self._allowed_users
        if users and user_id not in users:
            logger.warning("%s user %s may not act on reminders", self._channel, user_id)
            return CommandReply("You don't have permission to act on IRIS reminders.")
        return None

    def __call__(self, text: str, user_id: str) -> CommandReply | None:
        stripped = text.strip()
        done = _DONE.match(stripped)
        snooze = None if done else _SNOOZE.match(stripped)
        paid = _PAID.match(stripped)
        not_yet = _NOT_YET.match(stripped)
        if done is None and snooze is None and paid is None and not_yet is None:
            return None
        refused = self._refused(user_id)
        if refused is not None:
            return refused
        now = self._clock()
        try:
            if paid is not None:
                outcome = self._backend.done(paid.group(1).lower(), source=self._channel)
                return CommandReply(_paid_text(outcome))
            if not_yet is not None:
                outcome = self._backend.snooze(
                    not_yet.group(1).lower(), NOT_YET_CHOICE, source=self._channel
                )
                return CommandReply(not_yet_text(outcome, self._zone(), now))
            if done is not None:
                outcome = self._backend.done(done.group(1).lower(), source=self._channel)
                if outcome.already_paid:
                    return CommandReply(_paid_text(outcome))
                return CommandReply(done_text(outcome, self._zone()))
            assert snooze is not None  # narrowed above
            outcome = self._backend.snooze(
                snooze.group(1).lower(), snooze.group(2).strip(), source=self._channel
            )
            return CommandReply(snooze_text(outcome, self._zone(), now))
        except ReminderNotFound:
            return CommandReply(GONE)
        except SnoozeNotUnderstood:
            return CommandReply(NOT_UNDERSTOOD)
        except ValueError:
            return CommandReply(GONE)
        except Exception:  # said to the owner, never dropped
            logger.exception("reminder command failed: %s", stripped[:80])
            return CommandReply(FAILED)

    def on_reply(
        self, text: str, user_id: str, chat_id: str, message_id: str
    ) -> CommandReply | None:
        """A reply to the bot's message ``message_id``. ``None`` — so it goes to chat —
        unless the text is a Done or a snooze AND that message was a reminder."""
        from iris_harness.services.notifications.snooze import parse_reply

        now = self._clock()
        tz = self._zone()
        asked = parse_reply(text, now, tz)
        if asked is None:
            return None
        refused = self._refused(user_id)
        if refused is not None:
            return refused
        if asked.action == "not_yet":
            return self._not_yet_reply(chat_id, message_id, tz, now)
        try:
            if asked.action == "done":
                outcome = self._backend.done_by_message(
                    self._channel, chat_id, message_id, source=self._channel
                )
                if asked.paid or outcome.already_paid:
                    return CommandReply(_paid_text(outcome))
                return CommandReply(done_text(outcome, tz))
            outcome = self._backend.snooze_by_message(
                self._channel, chat_id, message_id, text.strip(), source=self._channel
            )
            return CommandReply(snooze_text(outcome, tz, now))
        except ReminderNotFound:
            return None  # not a reminder's message: an ordinary reply, for chat
        except SnoozeNotUnderstood:
            return CommandReply(NOT_UNDERSTOOD)
        except ValueError:
            return CommandReply(GONE)

    def _not_yet_reply(
        self, chat_id: str, message_id: str, tz: tzinfo, now: datetime
    ) -> CommandReply | None:
        """ "not yet" in reply to a bill's question. Anything else it answers — not a
        reminder, or a reminder that is not a question — is ordinary chat."""
        try:
            outcome = self._backend.snooze_by_message(
                self._channel, chat_id, message_id, NOT_YET_CHOICE, source=self._channel
            )
        except (ReminderNotFound, SnoozeNotUnderstood):
            return None
        except ValueError:
            return CommandReply(GONE)
        return CommandReply(not_yet_text(outcome, tz, now))


ReplyHandler = Callable[[str, str, str, str], CommandReply | None]


class ChainedCommands:
    """Several command handlers tried in order (approvals, then reminders); the first
    answer wins. ``on_reply`` is the first handler's that has one."""

    def __init__(self, *handlers: CommandHandler) -> None:
        self._handlers = handlers

    def __call__(self, text: str, user_id: str) -> CommandReply | None:
        for handler in self._handlers:
            reply = handler(text, user_id)
            if reply is not None:
                return reply
        return None

    def on_reply(
        self, text: str, user_id: str, chat_id: str, message_id: str
    ) -> CommandReply | None:
        for handler in self._handlers:
            answer: ReplyHandler | None = getattr(handler, "on_reply", None)
            if answer is not None:
                reply = answer(text, user_id, chat_id, message_id)
                if reply is not None:
                    return reply
        return None


class LocalReminderBackend:
    """Writes the one reminder store in process, for the runtime's own Telegram poller."""

    def __init__(self, *, data_dir: Path, bus: EventBus | None = None) -> None:
        self._db = Path(data_dir) / "tasks.db"
        self._bus = bus

    def _store(self) -> ReminderStore:
        from iris_harness.services.notifications.store import ReminderStore

        store = ReminderStore(db_path=self._db, bus=self._bus, tz=_owner_zone())
        store.ensure_schema()
        return store

    def _by_message(self, store: ReminderStore, channel: str, chat_id: str, message_id: str) -> str:
        found = store.find_by_message(channel, chat_id, message_id)
        if found is None:
            raise ReminderNotFound(message_id)
        return str(found.id)

    def done(self, reminder_id: str, *, source: str) -> ReminderOutcome:
        from iris_harness.services.notifications.bills import BillAlreadyPaid

        store = self._store()
        if store.get(reminder_id) is None:
            raise ReminderNotFound(reminder_id)
        try:
            closed = store.complete(reminder_id, source=source)
        except BillAlreadyPaid as paid:
            return ReminderOutcome(already_paid=True, entity=paid.entity)
        return ReminderOutcome(next_at=_next_in_series(store, closed))

    def snooze(self, reminder_id: str, spoken: str, *, source: str) -> ReminderOutcome:
        from iris_harness.services.notifications.snooze import parse_snooze

        store = self._store()
        existing = store.get(reminder_id)
        if existing is None:
            raise ReminderNotFound(reminder_id)
        if spoken.strip().lower() == NOT_YET_CHOICE:
            return _not_yet(store, existing, source=source)
        until = parse_snooze(spoken, datetime.now(UTC), _owner_zone())
        if until is None:
            raise SnoozeNotUnderstood(spoken)
        snoozed = store.snooze(reminder_id, until, source=source)
        return ReminderOutcome(until=snoozed.remind_at)

    def done_by_message(
        self, channel: str, chat_id: str, message_id: str, *, source: str
    ) -> ReminderOutcome:
        return self.done(
            self._by_message(self._store(), channel, chat_id, message_id), source=source
        )

    def snooze_by_message(
        self, channel: str, chat_id: str, message_id: str, spoken: str, *, source: str
    ) -> ReminderOutcome:
        reminder_id = self._by_message(self._store(), channel, chat_id, message_id)
        return self.snooze(reminder_id, spoken, source=source)


def _not_yet(store: ReminderStore, reminder: Reminder, *, source: str) -> ReminderOutcome:
    """Not yet on a bill's question: acknowledged; ``next_at`` is the next question."""
    from iris_harness.services.notifications.bills import (
        is_ask,
        next_question_at,
    )

    if not is_ask(reminder):
        raise SnoozeNotUnderstood(NOT_YET_CHOICE)
    store.acknowledge(reminder.id, source=source)
    return ReminderOutcome(next_at=next_question_at(reminder, _owner_zone()))


def _next_in_series(store: ReminderStore, reminder: Reminder) -> datetime | None:
    if not reminder.series_id:
        return None
    later = [
        r.remind_at
        for r in store.list(
            statuses=("pending", "sending"), include_fired=True, include_dismissed=True
        )
        if r.series_id == reminder.series_id
        and r.id != reminder.id
        and r.remind_at > reminder.remind_at
    ]
    return min(later) if later else None


__all__ = [
    "FAILED",
    "GONE",
    "NOT_UNDERSTOOD",
    "NOT_YET_CHOICE",
    "PAID",
    "ChainedCommands",
    "LocalReminderBackend",
    "ReminderBackend",
    "ReminderCommands",
    "ReminderNotFound",
    "ReminderOutcome",
    "ReplyHandler",
    "SnoozeNotUnderstood",
    "done_text",
    "not_yet_text",
    "reminder_keyboard",
    "snooze_text",
]
