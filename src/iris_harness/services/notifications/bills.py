"""How a bill's reminder reads and what it offers (loop-proof PR 4, plan D6, graph §5).

A bill reminder is a row with ``target_kind="bill"``, ``target_id=<due id>`` and a
``dedupe_key`` ``bill:<due id>:<step>``. The finance plugin creates the rows
(``finance_workflows/bill_reminders.py``) with display-ready strings in ``meta`` —
``entity``, ``amount`` ("$35.00"), ``statement`` ("$1,284.50"), ``due`` (ISO date) —
so this module words them without knowing anything about money. The steps:

======== ======================================================== =================
step     message                                                  buttons
======== ======================================================== =================
t3d      💳 Discover — $35.00 min due Mon Oct 13                   ✅ Paid · ⏰ Tomorrow
         in 3 days · statement $1,284.50
dayof    💳 Discover — $35.00 min due today                        ✅ Paid · ⏰ 1 hour
         Mon Oct 13 · statement $1,284.50
ask1..3  ❓ Did you pay Discover $35.00?                            ✅ Paid · Not yet
         It was due Mon Oct 13 · no payment email seen
         (ask3 adds "· last ask — then it stays in the digest")
changed  💳 Discover — amount changed: $70.00 min due Mon Oct 13   ✅ Paid · ⏰ Tomorrow
         was $35.00 · statement $2,900.00
paid     ✅ Discover marked paid                                    (none)
         Seen: a payment email. No more reminders for this bill.
======== ======================================================== =================

Paid is Done (``/paid <id>`` → the store's ``complete``); Not yet acknowledges the
question without snoozing it (``/notyet <id>`` → ``acknowledge``) — tomorrow's question
is its own row.

Paid works from ANY of the bill's messages, for as long as the bill is open: a row the
digest sweep aged out ("expired: delivered, not acknowledged"), one the owner answered
Not yet, or one whose bill was reopened after it still closes the bill
(:func:`payable_after_end`). A row whose bill was already closed as paid answers
:class:`BillAlreadyPaid` — "✅ Discover is already marked paid." — never "gone". The
"marked paid" confirmation asks nothing, so it ends as soon as it is delivered
(:data:`DELIVERED_INFO_REASON`) instead of waiting to be swept as unanswered.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, tzinfo

from .models import Reminder

#: Steps that ask "Did you pay?" — the ones with a Not yet.
ASK_STEPS: tuple[str, ...] = ("ask1", "ask2", "ask3")
#: The confirmation after a payment email closed the bill — nothing to act on.
PAID_STEP = "paid"
LAST_ASK = "ask3"

#: The words a snooze choice carries for Not yet (a button payload, not vocabulary).
NOT_YET_CHOICE = "not_yet"

#: ``closed_reason`` of a bill's row that ended before its bill was reopened (the
#: store's ``reopen_for_target``): the bill is open again, so Paid on it still counts.
REOPENED_REASON = "expired: bill reopened"
#: ``closed_reason`` of the "marked paid" confirmation once delivered: nothing to answer.
DELIVERED_INFO_REASON = "delivered: nothing to answer"
#: How a row ends when its bill is closed as paid (``close_for_target``'s reason).
PAID_CLOSE_PREFIX = "closed: paid"
#: Reasons a bill's row ended WITHOUT its bill closing — aged out by the digest sweep
#: (``digest.expiry.UNACKNOWLEDGED_REMINDER_REASON``), a Not yet (the store's
#: ``NOT_YET``), or the bill was reopened after it.
_ENDED_WHILE_OPEN = ("expired: delivered", "not yet", REOPENED_REASON)


class BillAlreadyPaid(ValueError):
    """Paid on a bill's row whose bill is already closed as paid. ``entity`` names the
    bill ("Discover") for the reply; a ``ValueError`` so older callers still see an
    ended row."""

    def __init__(self, reminder_id: str, entity: str = "") -> None:
        super().__init__(f"reminder {reminder_id}: its bill is already marked paid")
        self.reminder_id = reminder_id
        self.entity = entity


def already_paid_text(entity: str) -> str:
    """ "✅ Discover Card is already marked paid." (the bill's name when known)."""
    return f"✅ {entity or 'That bill'} is already marked paid."


def bill_key(due_id: str, step: str, snapshot: str | None = None) -> str:
    """``bill:<due id>:<step>`` (``bill:<id>:changed:<snapshot>``)."""
    key = f"bill:{due_id}:{step}"
    return f"{key}:{snapshot}" if snapshot else key


def bill_step(reminder: Reminder) -> str | None:
    """``t3d`` / ``dayof`` / ``ask1``..``ask3`` / ``changed`` / ``paid``; ``None`` when
    the row is not a bill's generated reminder."""
    if reminder.target_kind != "bill":
        return None
    step = reminder.meta.get("step")
    if step:
        return step
    parts = (reminder.dedupe_key or "").split(":")
    return parts[2] if len(parts) >= 3 and parts[0] == "bill" else None


def is_ask(reminder: Reminder) -> bool:
    return bill_step(reminder) in ASK_STEPS


def asks_nothing(reminder: Reminder) -> bool:
    """The "marked paid" confirmation: delivered, it is over (nothing to answer)."""
    return bill_step(reminder) == PAID_STEP


def closed_as_paid(reminder: Reminder) -> bool:
    """This bill row ended because its bill was closed as paid (any surface)."""
    return reminder.target_kind == "bill" and (reminder.closed_reason or "").startswith(
        PAID_CLOSE_PREFIX
    )


def payable_after_end(reminder: Reminder) -> bool:
    """An ended bill row whose bill did NOT close with it — aged out unanswered, a Not
    yet, or the bill reopened since — so Paid on its message still closes the bill."""
    return (
        reminder.target_kind == "bill"
        and reminder.status == "expired"
        and not asks_nothing(reminder)
        and (reminder.closed_reason or "").startswith(_ENDED_WHILE_OPEN)
    )


def _due_day(meta: dict[str, str]) -> str:
    """ "Mon Oct 13" from the ISO due date ("" when missing)."""
    try:
        due = date.fromisoformat(meta.get("due", ""))
    except ValueError:
        return ""
    return f"{due:%a %b} {due.day}"


def _owed(meta: dict[str, str], *, when: str) -> str:
    """ "$35.00 min due Mon Oct 13" — "min" when a statement total stands beside it."""
    amount = meta.get("amount", "")
    if not amount:
        return f"payment due {when}".strip()
    label = "min due" if meta.get("statement") else "due"
    return f"{amount} {label} {when}".strip()


def _statement(meta: dict[str, str]) -> str:
    return f"statement {meta['statement']}" if meta.get("statement") else ""


def _join(*parts: str) -> str:
    return " · ".join(p for p in parts if p)


def bill_wording(reminder: Reminder) -> tuple[str, str]:
    """``(headline, body)`` for a bill reminder, from its ``meta``. A row without the
    strings (hand-made) reads as its note."""
    return wording(reminder.meta, bill_step(reminder), note=reminder.note)


def wording(meta: dict[str, str], step: str | None, *, note: str = "") -> tuple[str, str]:
    """``(headline, body)`` of one step for a bill described by ``meta``."""
    entity = meta.get("entity", "")
    if not entity or step is None:
        return (note or "Bill reminder", "")
    due_day = _due_day(meta)
    if step == "t3d":
        return (f"💳 {entity} — {_owed(meta, when=due_day)}", _join("in 3 days", _statement(meta)))
    if step == "dayof":
        return (f"💳 {entity} — {_owed(meta, when='today')}", _join(due_day, _statement(meta)))
    if step in ASK_STEPS:
        amount = f" {meta['amount']}" if meta.get("amount") else ""
        last = "last ask — then it stays in the digest" if step == LAST_ASK else ""
        return (
            f"❓ Did you pay {entity}{amount}?",
            _join(f"It was due {due_day}" if due_day else "", "no payment email seen", last),
        )
    if step == "changed":
        was = f"was {meta['was']}" if meta.get("was") else ""
        return (
            f"💳 {entity} — amount changed: {_owed(meta, when=due_day)}",
            _join(was, _statement(meta)),
        )
    if step == PAID_STEP:
        seen = meta.get("seen") or "a payment email"
        return (f"✅ {entity} marked paid", f"Seen: {seen}. No more reminders for this bill.")
    return (note or f"💳 {entity}", "")


def digest_wording(meta: dict[str, str], *, today: date) -> str:
    """The digest's Bills line for an open bill, in the pushes' own words (after the
    name): up to its due day "$35.00 min due Mon Oct 13 (in 3 days) · statement
    $1,284.50"; after it, while unpaid, "$35.00 min — due Mon Oct 13, not marked paid"."""
    try:
        due = date.fromisoformat(meta.get("due", ""))
    except ValueError:
        return _join(_owed(meta, when=""), _statement(meta))
    due_day = _due_day(meta)
    days = (due - today).days
    if days < 0:
        amount = meta.get("amount", "")
        head = (f"{amount} min" if meta.get("statement") else amount) if amount else "payment"
        return f"{head} — due {due_day}, not marked paid"
    when = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"
    return _join(f"{_owed(meta, when=due_day)} ({when})", _statement(meta))


# ── the buttons ────────────────────────────────────────────────────────────────

_SNOOZE_BUTTON = {
    "t3d": ("⏰ Tomorrow", "tomorrow_9am"),
    "changed": ("⏰ Tomorrow", "tomorrow_9am"),
    "dayof": ("⏰ 1 hour", "1h"),
}


def bill_keyboard(reminder: Reminder) -> list[list[dict[str, str]]]:
    """The Telegram buttons: ✅ Paid beside ⏰ Tomorrow / ⏰ 1 hour, or Not yet for a
    question; none on the "marked paid" confirmation. Every callback is ≤ 64 bytes."""
    step = bill_step(reminder)
    if step == PAID_STEP:
        return []
    paid = {"text": "✅ Paid", "callback_data": f"/paid {reminder.id}"}
    if step in ASK_STEPS:
        return [[paid, {"text": "Not yet", "callback_data": f"/notyet {reminder.id}"}]]
    label, choice = _SNOOZE_BUTTON.get(step or "", ("⏰ Tomorrow", "tomorrow_9am"))
    return [[paid, {"text": label, "callback_data": f"/snooze {reminder.id} {choice}"}]]


def bill_push_actions(reminder: Reminder) -> list[dict[str, str]]:
    """The notification's two buttons (``public/sw.js`` ``REMINDER_ACTIONS``)."""
    step = bill_step(reminder)
    if step == PAID_STEP:
        return []
    paid = {"action": "paid", "title": "Paid"}
    if step in ASK_STEPS:
        return [paid, {"action": "not_yet", "title": "Not yet"}]
    if step == "dayof":
        return [paid, {"action": "snooze_1h", "title": "1 hour"}]
    return [paid, {"action": "snooze_tomorrow", "title": "Tomorrow"}]


def bill_actions(reminder: Reminder) -> list[str]:
    """The API's ``actions`` for an open bill row: Paid, Not yet on a question, and
    the two snoozes the sheet offers (1 hour, tomorrow 9:00)."""
    step = bill_step(reminder)
    if step == PAID_STEP:
        return []
    if step in ASK_STEPS:
        return ["paid", NOT_YET_CHOICE, "1h", "tomorrow_9am"]
    return ["paid", "1h", "tomorrow_9am"]


def next_question_at(reminder: Reminder, tz: tzinfo) -> datetime | None:
    """When the bill's next "Did you pay?" comes after a Not yet on this one: the
    morning after (``due + n + 1`` at the row's ``at``, 09:00), or ``None`` after the
    last ask — the bill stays in the digest only (graph §10)."""
    step = bill_step(reminder)
    if step not in ASK_STEPS or step == LAST_ASK:
        return None
    try:
        due = date.fromisoformat(reminder.meta.get("due", ""))
        hour, minute = (int(x) for x in (reminder.meta.get("at") or "09:00").split(":"))
    except ValueError:
        return None
    n = ASK_STEPS.index(step) + 1
    return datetime.combine(due + timedelta(days=n + 1), time(hour, minute), tzinfo=tz)


_SURFACES = {
    "telegram": "Telegram",
    "push": "the notification",
    "sheet": "the reminder sheet",
    "chat": "chat",
    "chat_panel": "chat",
    "action_center": "Action Center",
    "payment_email": "a payment email",
    "finance": "the Finance page",
    "owner": "you",
    "api": "IRIS",
}


def surface_label(source: str) -> str:
    """ "a payment email" for ``payment_email``, "Telegram" for ``telegram``."""
    cleaned = source.strip()
    return _SURFACES.get(cleaned, cleaned.replace("_", " ") or "IRIS")


def paid_by(reminder: Reminder) -> str | None:
    """Who closed the bill, when this row says it was paid: the owner's Done
    (``done: telegram``) or the bill's close (``closed: paid (payment_email)``)."""
    reason = reminder.closed_reason or ""
    if reminder.status == "done":
        return surface_label(reason.removeprefix("done:").strip() or "owner")
    if reason.startswith(PAID_CLOSE_PREFIX):
        inner = reason.removeprefix(PAID_CLOSE_PREFIX).strip()
        return surface_label(inner.strip("()").strip() or "owner")
    return None


__all__ = [
    "ASK_STEPS",
    "DELIVERED_INFO_REASON",
    "LAST_ASK",
    "NOT_YET_CHOICE",
    "PAID_CLOSE_PREFIX",
    "PAID_STEP",
    "REOPENED_REASON",
    "BillAlreadyPaid",
    "already_paid_text",
    "asks_nothing",
    "bill_actions",
    "bill_key",
    "bill_keyboard",
    "bill_push_actions",
    "bill_step",
    "bill_wording",
    "closed_as_paid",
    "digest_wording",
    "is_ask",
    "next_question_at",
    "paid_by",
    "payable_after_end",
    "surface_label",
    "wording",
]
