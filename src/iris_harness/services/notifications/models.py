"""Pydantic models for the reminder primitive.

A ``Reminder`` is a "fire at time T about target X" row. Since loop-proof D14 (one
reminder store) it also carries its own delivery lifecycle — ``status``, send attempts,
the channels that accepted it, recurrence — while the target (Task, Goal, Bill, Event)
still owns what the reminder is about. ``remind_at`` is stored UTC; the owner's zone
(``IRIS_TZ``) is for entry and display only.

Lifecycle (graph §4 / §10)::

    pending -> sending -> sent -> done | pending (snoozed) | expired
                  |  (none of the channels accepted: retry 3x, 5 min apart)
                  +-> failed -> expired (after the digest showed it)
    any open state -> cancelled (owner / event deleted)
    open rows of one target -> cancelled / expired, "closed: …" (its bill was paid)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from iris_harness.foundation.clock import utc_now

TargetKind = Literal["task", "goal", "bill", "event"]
ReminderStatus = Literal["pending", "sending", "sent", "failed", "done", "cancelled", "expired"]

#: States a reminder never leaves.
TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "cancelled", "expired"})


class Reminder(BaseModel):
    """One scheduled notification pointing at a user-domain object.

    Identity is the ``id`` UUID. ``fired_at`` is the last accepted send (cleared by a
    snooze), ``dismissed_at`` the cancel stamp. ``message_refs`` are the sent messages
    (``{channel, chat_id, message_id}``) a reply can be matched against.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(..., min_length=1)
    target_kind: TargetKind
    target_id: str = Field(..., min_length=1)
    remind_at: datetime
    channel: str = Field(default="default", min_length=1, max_length=64)
    note: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    fired_at: datetime | None = None
    dismissed_at: datetime | None = None
    # ── delivery lifecycle (D14) ──
    status: ReminderStatus = "pending"
    attempts: int = 0
    next_attempt_at: datetime | None = None
    last_error: str | None = None
    delivered_channels: tuple[str, ...] = ()
    message_refs: tuple[dict[str, str], ...] = ()
    # ── recurrence (D14) ──
    recurrence: str | None = None
    recur_until: datetime | None = None
    series_id: str | None = None
    # ── ending (D18) ──
    missed_digests: int = 0
    closed_reason: str | None = None
    # ── generated reminders (loop-proof PR 4) ──
    #: Unique when set: a generator (the bill schedule) creates each key once, ever.
    dedupe_key: str | None = None
    #: What a generated reminder says, as display-ready strings (a bill's entity,
    #: amount, due date, step) — the wording is built from it at send time.
    meta: dict[str, str] = Field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status not in TERMINAL_STATUSES
