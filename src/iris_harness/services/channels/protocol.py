"""Channel connector protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import ChannelMessage, DeliveryReceipt


@runtime_checkable
class IChannelConnector(Protocol):
    """Abstract interface every channel connector must implement.

    Implementations may be sync or async. The gateway treats `send` as
    synchronous; async connectors should expose a thin sync wrapper.
    """

    name: str

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        """Deliver a message via the channel and return a receipt."""

    def healthy(self) -> bool:
        """Return True when the connector is configured and reachable."""
