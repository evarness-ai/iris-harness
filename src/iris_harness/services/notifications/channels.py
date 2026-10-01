"""Reminder delivery — accept-then-fire (loop-proof D14).

The heartbeat hands every due reminder to ``deliver_due``: the row is claimed
(``sending``), sent to each reminder channel (Telegram and web push by default,
``config/notifications.yaml`` ``reminder_channels``), and recorded by what the
channels answered:

- at least one ``SENT`` → the row is ``sent``, ``reminder.fired`` is emitted, and a
  repeating reminder's next occurrence is created;
- none → one more attempt used; retried 5 minutes later, and after the fourth
  attempt (T, +5, +10, +15) the row is ``failed`` — the ``push_delivery`` health check
  and the next digest's "missed" line take it from there.

A web push with no subscribed browser answers ``SKIPPED``: not a failure, and not a
delivery either. ``reminder.fired`` is emitted only after a channel accepted, so a
subscriber never hears about a reminder the owner did not get.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, tzinfo
from pathlib import Path

import yaml

from iris_harness.foundation.paths import config_dir as resolve_config_dir
from iris_harness.services.channels import ChannelGateway, ChannelMessage, DeliveryStatus
from iris_harness.services.channels.models import DeliveryReceipt
from iris_harness.services.channels.reminder_commands import reminder_keyboard

from .bills import bill_keyboard, bill_push_actions, bill_step, bill_wording
from .models import Reminder
from .recurrence import describe
from .store import MAX_ATTEMPTS, ReminderStore
from .targets import lookup_target_title, reminder_title

logger = logging.getLogger(__name__)

#: Kept for callers of the pre-D14 name; the lookup lives in ``targets``.
_lookup_target_title = lookup_target_title

#: Where a reminder goes when ``notifications.yaml`` does not say.
DEFAULT_REMINDER_CHANNELS: tuple[str, ...] = ("telegram", "web_push")


def load_reminder_channels(config_dir: Path | None = None) -> tuple[str, ...]:
    """``reminder_channels`` from ``notifications.yaml``; the default on any problem."""
    path = (config_dir or resolve_config_dir()) / "notifications.yaml"
    try:
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            listed = raw.get("reminder_channels") if isinstance(raw, dict) else None
            if isinstance(listed, list):
                names = tuple(str(c).strip() for c in listed if str(c).strip())
                if names:
                    return names
    except Exception:  # config must never stop delivery
        logger.warning("failed to read %s; using default reminder channels", path, exc_info=True)
    return DEFAULT_REMINDER_CHANNELS


#: The two buttons a reminder's push notification carries (D14). ``action`` is what
#: the service worker receives in ``notificationclick``; browsers show at most two.
PUSH_ACTIONS: tuple[dict[str, str], ...] = (
    {"action": "done", "title": "Done"},
    {"action": "snooze_1h", "title": "Snooze 1h"},
)


# ── wording ────────────────────────────────────────────────────────────────


def _clock(local: datetime) -> str:
    hour = local.hour % 12 or 12
    return f"{hour}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}"


def _when(local: datetime) -> str:
    """ "Mon Sep 28 · 8:00 AM"."""
    return f"{local:%a %b} {local.day} · {_clock(local)}"


@dataclass(frozen=True)
class ReminderText:
    """One reminder in words, in the owner's zone."""

    title: str
    when: str  # "Mon Sep 28 · 8:00 AM"
    clock: str  # "8:00 AM"
    repeats: str  # "every Monday" or ""
    note: str

    def details(self, *, short: bool = False) -> str:
        parts = [self.clock if short else self.when]
        if self.repeats:
            parts.append(f"repeats {self.repeats}")
        if self.note and self.note != self.title:
            parts.append(self.note)
        return " · ".join(parts)


def reminder_text(
    reminder: Reminder, tasks_db: Path, tz: tzinfo, *, calendar_db: Path | None = None
) -> ReminderText:
    local = reminder.remind_at.astimezone(tz)
    return ReminderText(
        title=reminder_title(reminder, tasks_db, calendar_db=calendar_db),
        when=_when(local),
        clock=_clock(local),
        repeats=describe(reminder.recurrence, reminder.remind_at, tz),
        note=reminder.note,
    )


def build_message(
    reminder: Reminder,
    channel: str,
    text: ReminderText,
    *,
    now: datetime | None = None,
    tz: tzinfo = UTC,
) -> ChannelMessage:
    """The message for one channel.

    Telegram: "⏰ <b>title</b>" over "Mon Sep 28 · 8:00 AM · repeats every Monday",
    with Done / 1 hour / 10 min / Tomorrow 9am buttons.
    Web push: the title line as the notification title, the details as its body (just
    the time when it is today), its own tag so two reminders never replace each other,
    and a tap that opens the reminder.
    """
    metadata: dict[str, object] = {
        "reminder_id": reminder.id,
        "target_kind": reminder.target_kind,
        "target_id": reminder.target_id,
        "remind_at": reminder.remind_at.isoformat(),
        "tag": f"reminder:{reminder.id}",
        "url": f"/reminders/{reminder.id}",
        # A snoozed reminder comes back under the same tag: it must alert again.
        "renotify": True,
    }
    if bill_step(reminder) is not None and reminder.meta.get("entity"):
        return _bill_message(reminder, channel, metadata)
    headline = f"⏰ {text.title}"
    if channel == "telegram":
        metadata["parse_mode"] = "HTML"
        # Done / Snooze under the message (PR 3b); the poller clears them on a tap.
        metadata["inline_keyboard"] = reminder_keyboard(reminder.id)
        body = f"⏰ <b>{html.escape(text.title)}</b>\n{html.escape(text.details())}"
        return ChannelMessage(recipient="", body=body, subject="IRIS Reminder", metadata=metadata)
    if channel == "web_push":
        today = (now or datetime.now(UTC)).astimezone(tz).date()
        short = reminder.remind_at.astimezone(tz).date() == today
        # Buttons on the notification (Chrome, Android, desktop; iOS ignores them and a
        # tap opens the sheet at ``url``). The worker posts these to the reminder API.
        metadata["actions"] = [dict(a) for a in PUSH_ACTIONS]
        return ChannelMessage(
            recipient="", body=text.details(short=short), subject=headline, metadata=metadata
        )
    return ChannelMessage(
        recipient="",
        body=f"{headline}\n{text.details()}",
        subject="IRIS Reminder",
        metadata=metadata,
    )


def _bill_message(reminder: Reminder, channel: str, metadata: dict[str, object]) -> ChannelMessage:
    """A bill's reminder (PR 4, ``bills``): its own headline and line, ✅ Paid in place
    of Done, and Not yet on a "Did you pay?" question — the same message on every
    day it is sent, because the words come from the row, not the clock."""
    headline, line = bill_wording(reminder)
    metadata["kind"] = "bill"
    metadata["bill_step"] = bill_step(reminder) or ""
    if channel == "telegram":
        metadata["parse_mode"] = "HTML"
        keyboard = bill_keyboard(reminder)
        if keyboard:
            metadata["inline_keyboard"] = keyboard
        body = f"<b>{html.escape(headline)}</b>"
        if line:
            body += f"\n{html.escape(line)}"
        return ChannelMessage(recipient="", body=body, subject="IRIS Bill", metadata=metadata)
    if channel == "web_push":
        actions = bill_push_actions(reminder)
        if actions:
            metadata["actions"] = actions
        return ChannelMessage(recipient="", body=line, subject=headline, metadata=metadata)
    body = f"{headline}\n{line}" if line else headline
    return ChannelMessage(recipient="", body=body, subject="IRIS Bill", metadata=metadata)


# ── delivery ───────────────────────────────────────────────────────────────


@dataclass
class DeliveryReport:
    """What one tick did."""

    sent: list[str] = field(default_factory=list)
    retrying: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    created_next: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.retrying and not self.failed and not self.errors

    def summary(self) -> str:
        line = (
            f"notification_reminder_tick sent={len(self.sent)} retrying={len(self.retrying)} "
            f"failed={len(self.failed)} next={len(self.created_next)}"
        )
        return f"{line}; {'; '.join(self.errors)}" if self.errors else line


def resolve_channels(
    gateway: ChannelGateway, configured: Sequence[str], *, default_channel: str | None = None
) -> list[str]:
    """The configured reminder channels that are registered, in order; the default
    channel alone when none is (a console-only dev stack still sees its reminders)."""
    registered = set(gateway.channels())
    targets = [c for c in configured if c in registered]
    if not targets and default_channel and default_channel in registered:
        targets = [default_channel]
    return targets


def _chat_id(gateway: ChannelGateway, channel: str, message: ChannelMessage) -> str:
    if message.recipient:
        return message.recipient
    try:
        connector = gateway.get(channel)
    except KeyError:
        return ""
    for attr in ("default_chat_id", "_default_chat_id"):
        value = getattr(connector, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def send_reminder(
    reminder: Reminder,
    gateway: ChannelGateway,
    channels: Sequence[str],
    text: ReminderText,
    *,
    tz: tzinfo,
    now: datetime | None = None,
) -> tuple[list[DeliveryReceipt], list[dict[str, str]]]:
    """Send to every channel; returns every receipt and the refs of the accepted ones."""
    receipts: list[DeliveryReceipt] = []
    refs: list[dict[str, str]] = []
    for channel in channels:
        message = build_message(reminder, channel, text, now=now, tz=tz)
        receipt = gateway.send(channel, message)
        receipts.append(receipt)
        if receipt.status is DeliveryStatus.SENT:
            refs.append(
                {
                    "channel": channel,
                    "chat_id": _chat_id(gateway, channel, message),
                    "message_id": receipt.message_id,
                }
            )
    return receipts, refs


def _describe_receipts(receipts: Sequence[DeliveryReceipt], channels: Sequence[str]) -> str:
    if not channels:
        return "no reminder channel is registered"
    parts = []
    for receipt in receipts:
        if receipt.status is DeliveryStatus.SENT:
            continue
        reason = receipt.error or receipt.status.value
        parts.append(f"{receipt.channel}: {receipt.status.value} ({reason})"[:200])
    return "; ".join(parts)


def deliver_due(
    store: ReminderStore,
    gateway: ChannelGateway,
    *,
    channels: Sequence[str],
    tz: tzinfo,
    now: datetime | None = None,
    calendar_db: Path | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> DeliveryReport:
    """Send every due reminder once; see the module doc for the outcomes."""
    when = now or datetime.now(UTC)
    report = DeliveryReport()
    for due in store.list_due(now=when):
        try:
            claimed = store.claim(due.id, now=when)
            if claimed is None:
                continue
            text = reminder_text(claimed, store.db_path, tz, calendar_db=calendar_db)
            receipts, refs = send_reminder(claimed, gateway, channels, text, tz=tz, now=when)
            problems = _describe_receipts(receipts, channels)
            accepted = [r["channel"] for r in refs]
            if accepted:
                _, nxt = store.mark_sent(
                    claimed.id,
                    delivered_channels=accepted,
                    message_refs=refs,
                    errors=problems or None,
                    now=when,
                )
                report.sent.append(claimed.id)
                if nxt is not None:
                    report.created_next.append(nxt.id)
                continue
            updated = store.mark_attempt_failed(
                claimed.id,
                error=problems or "no channel accepted",
                now=when,
                max_attempts=max_attempts,
            )
            if updated.status == "failed":
                report.failed.append(claimed.id)
                logger.warning(
                    "reminder %s undelivered after %d attempts: %s",
                    claimed.id,
                    updated.attempts,
                    updated.last_error,
                )
            else:
                report.retrying.append(claimed.id)
        except Exception as exc:  # one bad row must not stop the others
            logger.exception("reminder delivery failed: id=%s", due.id)
            report.errors.append(f"{due.id}: {type(exc).__name__}: {exc}"[:200])
    return report


def describe_failed(reminders: Sequence[Reminder], tasks_db: Path, tz: tzinfo) -> str:
    """ "2 reminders couldn't be delivered: “A” (due Mon Sep 28 · 8:00 AM); …"."""
    noun = "reminder" if len(reminders) == 1 else "reminders"
    items = []
    for reminder in reminders[:5]:
        text = reminder_text(reminder, tasks_db, tz)
        items.append(f"“{text.title}” (due {text.when})")
    more = f"; and {len(reminders) - 5} more" if len(reminders) > 5 else ""
    return f"{len(reminders)} {noun} couldn't be delivered: {'; '.join(items)}{more}"


__all__ = [
    "DEFAULT_REMINDER_CHANNELS",
    "DeliveryReport",
    "ReminderText",
    "build_message",
    "deliver_due",
    "describe_failed",
    "load_reminder_channels",
    "reminder_text",
    "resolve_channels",
    "send_reminder",
]
