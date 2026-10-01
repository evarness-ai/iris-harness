"""Event topic + payload for the reminder subsystem.

Per ADR-0013, subsystem-private topics live with their producer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .models import TargetKind

REMINDER_FIRED = "reminder.fired"


@dataclass(frozen=True)
class ReminderFiredPayload:
    """Emitted after a channel ACCEPTED a reminder (``ReminderStore.mark_sent``,
    loop-proof D14 accept-then-fire), or by the ``iris reminder tick`` CLI's direct
    ``fire``. Delivery itself happens before this event, in
    ``notifications.channels.deliver_due`` — subscribers react to a delivered
    reminder, they do not send it.
    """

    reminder_id: str
    target_kind: TargetKind
    target_id: str
    remind_at: datetime
    fired_at: datetime
    channel: str
    note: str


REMINDER_COMPLETED = "reminder.completed"


@dataclass(frozen=True)
class ReminderCompletedPayload:
    """Emitted on the process bus when a user marks a reminder completed
    (``ReminderStore.complete``, and the legacy reminder API's PATCH until it retires).
    ``task`` is the reminder's title.

    A domain that reacts to completed work subscribes — the finance plugin resolves
    open dues whose label the completed task names. With no subscriber the
    completion simply completes.
    """

    reminder_id: str
    task: str
    #: The surface the owner used (``telegram``, ``push``, ``sheet``, ``chat``, …).
    source: str = ""
    #: What the reminder is about (PR 4): a bill's Paid closes that due by its id
    #: (``target_kind="bill"``, ``target_id=<due id>``) rather than by its words.
    target_kind: str = ""
    target_id: str = ""


REMINDER_SNOOZED = "reminder.snoozed"


@dataclass(frozen=True)
class ReminderSnoozedPayload:
    """Emitted when the owner snoozes a reminder (``ReminderStore.snooze``, PR 3b).

    ``from_at`` is when it was due, ``until`` when it fires again (both UTC);
    ``source`` is the surface the owner used (``telegram``, ``push``, ``sheet``,
    ``chat``, ``action_center``, ``api``). ``series_id`` groups a repeating reminder's
    occurrences, so a learner can see "snoozed to 18:00 twice" (graph §9, V35).
    """

    reminder_id: str
    series_id: str | None
    from_at: datetime
    until: datetime
    source: str


REMINDER_REOPENED = "reminder.reopened"


@dataclass(frozen=True)
class ReminderReopenedPayload:
    """Emitted when the owner undoes a Done — or says a bill closed as paid was not
    ("Not paid — reopen", PR 4) — through ``ReminderStore.reopen``.

    A domain that closed its object on the Done subscribes: finance reopens the due
    of a ``bill`` reminder. ``closed_reason`` is what the row said before (``done:
    sheet``, ``closed: paid (payment_email)``)."""

    reminder_id: str
    target_kind: str
    target_id: str
    source: str = ""
    closed_reason: str = ""


REMINDER_ACKNOWLEDGED = "reminder.acknowledged"


@dataclass(frozen=True)
class ReminderAcknowledgedPayload:
    """Emitted when the owner answers a bill's "Did you pay?" with Not yet, through
    ``ReminderStore.acknowledge`` — from Telegram, the notification or the sheet (PR 4).

    The domain that owns the target subscribes: finance records the Not yet on the due
    (``target_kind="bill"``, ``target_id=<due id>``), so the Action Center's "Did you
    pay?" card waits until tomorrow, exactly as for a Not yet tapped on the card."""

    reminder_id: str
    target_kind: str
    target_id: str
    source: str = ""
