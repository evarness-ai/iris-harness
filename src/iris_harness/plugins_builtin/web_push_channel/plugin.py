"""``setup(api)`` for the Web Push channel (track 2b PR 10).

Two registrations, and the second is the interesting one.

**A channel.** The health watch already calls ``gateway.broadcast(...)`` with
``channels: all``, so the moment ``web_push`` is a registered channel, a
revoked credential reaches the owner's phone with nothing changed at the call
site. That is the whole reason the connector was written to the ordinary
``IChannelConnector`` shape.

**A nudge on approvals.** ``ChannelRouter`` delivers an approval to exactly one
destination — where it was asked from — and that is correct. So push does not
compete for that slot; it subscribes to ``approval.requested`` and taps the
owner on the shoulder. The approval still lives in the queue, and the
notification is a pointer to it.

Mounted always, and inert with nobody subscribed: the connector reports
``healthy() == False`` and ``send`` returns SKIPPED rather than pretending.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.channels import (
    APPROVAL_REQUESTED,
    ApprovalRequestedPayload,
    ChannelMessage,
    PushSubscriptionStore,
    WebPushConnector,
)

logger = logging.getLogger(__name__)

#: Where a tap lands, per kind of notice. The connector owns this mapping
#: because it is the web UI's channel and so the one thing here entitled to
#: know the web UI's routes.
APPROVAL_URL = "/actions"


def setup(api: PluginAPI) -> None:
    try:
        store = PushSubscriptionStore()
    except Exception as exc:  # noqa: BLE001 - an unwritable data dir is not fatal
        logger.warning("web_push_channel not mounted: %s", exc)
        return

    api.register_channel(WebPushConnector(store=store))
    # The router emits on the process bus. Subscribing through the API puts the
    # handler inside the registry's fault boundary, recorded against this plugin.
    api.subscribe(APPROVAL_REQUESTED, _approval_nudge(store), scope="process")
    logger.debug("web_push_channel: registered")


def _approval_nudge(store: PushSubscriptionStore) -> Callable[[ApprovalRequestedPayload], None]:
    """Nudge the phone when a run halts waiting on the owner."""

    def on_approval(payload: ApprovalRequestedPayload) -> None:
        connector = WebPushConnector(store=store)
        if not connector.healthy():
            return  # nobody subscribed; nothing to say and nothing to log about
        receipt = connector.send(
            ChannelMessage(
                recipient="*",
                subject="Waiting on you",
                # The signal, not the summary: a summary can be a paragraph,
                # and a notification is a glance. The detail is one tap away.
                body=payload.signal,
                metadata={"url": APPROVAL_URL, "tag": "approval"},
            )
        )
        if receipt.error:
            logger.warning("approval push: %s", receipt.error)

    return on_approval
