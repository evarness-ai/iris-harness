"""__tmpl_title: a delivery channel on IRIS's gateway.

A channel is an ``IChannelConnector``: a ``name``, ``send(message) -> DeliveryReceipt``
and ``healthy() -> bool``. ``setup(api)`` registers it with ``api.register_channel``; the
gateway then delivers to it (briefs, reminders, approval nudges) behind the plugin fault
boundary, so a connector that raises degrades this plugin instead of the turn.

This connector delivers to a local outbox file (one JSON line per message) under the
IRIS data directory, so it works offline. Replace :meth:`Connector._deliver` with your
transport (an HTTP API, a chat bot, a webhook) and :meth:`Connector.healthy` with a
cheap configuration check -- never a network call.
"""

from __future__ import annotations

import json
from pathlib import Path

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.channels import (
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
    IChannelConnector,
)
from iris_harness.sdk.persistence import data_path

CHANNEL = "__tmpl_tool"


class Connector(IChannelConnector):
    """The channel ``CHANNEL``: appends each message to ``outbox``.

    It subclasses the protocol rather than matching it by shape, so a type checker holds
    ``send`` and ``healthy`` to the signatures the gateway calls.
    """

    name = CHANNEL

    def __init__(self, outbox: Path | None = None) -> None:
        # Resolved when the connector is built, inside the harness's data directory: never
        # a path under the user's home, so tests and demos leave the owner's profile alone.
        self.outbox = outbox if outbox is not None else data_path(f"{CHANNEL}_outbox.jsonl")

    def healthy(self) -> bool:
        return True

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        if not message.body.strip():
            return DeliveryReceipt(
                channel=self.name, status=DeliveryStatus.SKIPPED, error="empty message"
            )
        try:
            message_id = self._deliver(message)
        except OSError as exc:
            return DeliveryReceipt(channel=self.name, status=DeliveryStatus.FAILED, error=str(exc))
        return DeliveryReceipt(channel=self.name, status=DeliveryStatus.SENT, message_id=message_id)

    def _deliver(self, message: ChannelMessage) -> str:
        """Hand the message to the transport; return its id there."""
        self.outbox.parent.mkdir(parents=True, exist_ok=True)
        record = {"recipient": message.recipient, "subject": message.subject, "body": message.body}
        with self.outbox.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        with self.outbox.open(encoding="utf-8") as handle:
            return str(sum(1 for _ in handle))


def setup(api: PluginAPI) -> None:
    api.register_channel(Connector())
