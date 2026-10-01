"""``approval.requested`` — an approval was raised and somebody was told.

Per ADR-0013 a subsystem-private topic lives with its producer, so it is here
rather than in ``services.notifications``.

Why an event rather than another ``ApprovalChannel``: ``ChannelRouter.select``
picks exactly ONE destination — where the request came from — and that is
right. A push is not a destination the approval lives at; it is a nudge to go
look at the one it does. Competing for the slot would mean a web-originated
approval reaching the phone INSTEAD of the browser that asked.

So the router emits, and anything that wants to nudge subscribes. The kernel
learns nothing about push, Telegram or the web, which is the same reason the
transport itself sits above this layer (OSS plan M6, decision 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

APPROVAL_REQUESTED = "approval.requested"
# A code caller's approval has run its course: the pinned call ran, failed, or was
# denied at execution, or it never ran because the owner rejected it or it lapsed
# (plugin-capabilities decision 1). The caller's code returned long before the owner
# answered, so this is how the answer reaches it: a plugin subscribes with
# ``api.on_approved_call`` and hears only its own calls.
APPROVAL_CALL_COMPLETED = "approval.call_completed"

#: How a code caller's approval ended.
ApprovedCallStatus = Literal["ran", "failed", "denied", "rejected", "expired"]


@dataclass(frozen=True)
class ApprovalRequestedPayload:
    """Emitted by ``ChannelRouter.notify`` once the approval has been routed.

    ``channel`` is where it was actually delivered, so a subscriber can decide
    not to double up — there is little point buzzing a phone about something
    that just arrived on that same phone's Telegram.
    """

    approval_id: str
    run_id: str
    signal: str
    context_summary: str
    timeout_at: str
    channel: str


@dataclass(frozen=True)
class ApprovalCallCompletedPayload:
    """Emitted once per code caller's approval, when nothing more will happen to it.

    ``status``: ``ran`` (the tool ran and did not raise), ``failed`` (it raised),
    ``denied`` (governance refused it at execution — the caller's permission was
    narrowed, or the call no longer matches), ``rejected`` or ``expired`` (never ran).
    ``summary`` is display-masked and truncated; the audit ledger holds the rest.
    """

    approval_id: str
    caller: str
    tool: str
    status: ApprovedCallStatus
    summary: str


__all__ = [
    "APPROVAL_CALL_COMPLETED",
    "APPROVAL_REQUESTED",
    "ApprovalCallCompletedPayload",
    "ApprovalRequestedPayload",
    "ApprovedCallStatus",
]
