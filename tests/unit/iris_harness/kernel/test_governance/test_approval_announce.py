"""``approval.requested`` — the router says something is waiting (track 2b PR 10).

Why an event and not another ``ApprovalChannel``: ``select()`` picks exactly
ONE destination, where the request came from. A push is not a destination the
approval lives at, it is a nudge to go look at the one it does — competing for
the slot would send a web-originated approval to the phone INSTEAD of the
browser that asked.

So the contract worth protecting is narrow: routing happens first, the event
reports where it went, and a broken subscriber cannot take an approval down
with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from iris_harness.foundation.eventbus import get_default_bus
from iris_harness.kernel.governance.approvals.events import (
    APPROVAL_REQUESTED,
    ApprovalRequestedPayload,
)
from iris_harness.kernel.governance.approvals.router import ChannelRouter
from iris_harness.kernel.governance.approvals.store import ApprovalRow


@dataclass
class _Channel:
    """A destination that records what it was asked to deliver."""

    name: str = "telegram"
    delivered: list[str] = field(default_factory=list)

    def is_configured(self) -> bool:
        return True

    def notify(self, approval: ApprovalRow) -> None:
        self.delivered.append(approval.approval_id)

    def notify_timeout(self, approval: ApprovalRow) -> None:
        pass


def _router(channel: Any) -> ChannelRouter:
    """A router whose one destination is the given channel.

    `queue` is required but unused on this path: `notify` routes an already
    stored row, and nothing here re-reads it.
    """
    return ChannelRouter(queue=cast(Any, None), remote=channel)


def _row(**kw: Any) -> ApprovalRow:
    fields: dict[str, Any] = {
        "approval_id": "ap-1",
        "run_id": "run-1",
        "checkpoint_id": None,
        "signal": "send_email",
        "context_summary": "Reply to the landlord about the boiler.",
        "requested_at": "2026-09-21T10:00:00Z",
        # `telegram` so the injected remote channel is the one selected: a
        # `web` row routes to the kernel's own WebChannel, which is right and
        # not what is under test here.
        "channel": "telegram",
        "status": "pending",
        "responded_at": None,
        "response_actor": None,
        "timeout_at": "2026-09-21T11:30:00Z",
        "policy_on_timeout": "reject",
        "session_id": None,
    }
    fields.update(kw)
    return ApprovalRow(**fields)


@pytest.fixture()
def heard() -> list[ApprovalRequestedPayload]:
    received: list[ApprovalRequestedPayload] = []
    get_default_bus().on(APPROVAL_REQUESTED, received.append)
    return received


def test_routing_an_approval_announces_it(heard: list[ApprovalRequestedPayload]) -> None:
    channel = _Channel()
    _router(channel).notify(_row())

    assert channel.delivered == ["ap-1"], "it must still be delivered"
    assert [p.approval_id for p in heard] == ["ap-1"]


def test_the_event_says_where_it_actually_went(heard: list[ApprovalRequestedPayload]) -> None:
    """So a subscriber can decline to double up.

    There is little point buzzing a phone about something that just arrived on
    that same phone's Telegram.
    """
    _router(_Channel(name="telegram")).notify(_row())
    assert heard[0].channel == "telegram"


def test_the_payload_carries_what_a_notification_needs(
    heard: list[ApprovalRequestedPayload],
) -> None:
    _router(_Channel()).notify(_row())
    payload = heard[0]
    assert payload.signal == "send_email"
    assert payload.context_summary.startswith("Reply to the landlord")
    assert payload.run_id == "run-1"
    assert payload.timeout_at


def test_a_broken_subscriber_cannot_break_an_approval() -> None:
    """The nudge is best-effort; the approval is not.

    A subscriber that raises must not propagate into routing — the run is
    already halted and the queue is the source of truth.
    """

    def explode(_payload: object) -> None:
        raise RuntimeError("push service on fire")

    get_default_bus().on(APPROVAL_REQUESTED, explode)
    channel = _Channel()

    _router(channel).notify(_row(approval_id="ap-2"))

    assert channel.delivered == ["ap-2"], "delivery must survive a broken listener"


def test_delivery_happens_before_the_announcement() -> None:
    """Order matters: the event says an approval WAS routed, not that one is
    about to be. A subscriber reading the queue must find it there."""
    order: list[str] = []

    @dataclass
    class Recorder(_Channel):
        def notify(self, approval: ApprovalRow) -> None:  # type: ignore[override]
            order.append("delivered")

    get_default_bus().on(APPROVAL_REQUESTED, lambda _p: order.append("announced"))
    _router(Recorder()).notify(_row(approval_id="ap-3"))

    assert order == ["delivered", "announced"]
