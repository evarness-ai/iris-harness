"""Notifications subsystem — the one reminder store (loop-proof D14).

A reminder points at any addressable user-domain object (task, goal, bill, event) and
carries its own delivery lifecycle: pending → sending → sent / failed, then done,
snoozed, cancelled or expired. Delivery is accept-then-fire (``channels.deliver_due``);
recurrence lives in ``recurrence``.
"""

from __future__ import annotations

from .events import (
    REMINDER_ACKNOWLEDGED,
    REMINDER_COMPLETED,
    REMINDER_FIRED,
    REMINDER_REOPENED,
    REMINDER_SNOOZED,
    ReminderAcknowledgedPayload,
    ReminderCompletedPayload,
    ReminderFiredPayload,
    ReminderReopenedPayload,
    ReminderSnoozedPayload,
)
from .models import TERMINAL_STATUSES, Reminder, ReminderStatus, TargetKind
from .recurrence import RULES, next_occurrence
from .store import ReminderStore

__all__ = [
    "REMINDER_ACKNOWLEDGED",
    "REMINDER_COMPLETED",
    "REMINDER_FIRED",
    "REMINDER_REOPENED",
    "REMINDER_SNOOZED",
    "RULES",
    "TERMINAL_STATUSES",
    "Reminder",
    "ReminderAcknowledgedPayload",
    "ReminderCompletedPayload",
    "ReminderFiredPayload",
    "ReminderReopenedPayload",
    "ReminderSnoozedPayload",
    "ReminderStatus",
    "ReminderStore",
    "TargetKind",
    "next_occurrence",
]
