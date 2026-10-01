"""Channel gateway — registry of named connectors with broadcast helpers."""

from __future__ import annotations

import logging

from .models import ChannelMessage, DeliveryReceipt, DeliveryStatus
from .protocol import IChannelConnector

logger = logging.getLogger(__name__)


class ChannelNotFoundError(KeyError):
    """Raised when a request targets a channel that is not registered."""


class ChannelGateway:
    """Registry + dispatcher for channel connectors."""

    def __init__(self) -> None:
        self._connectors: dict[str, IChannelConnector] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, connector: IChannelConnector) -> None:
        if not connector.name:
            raise ValueError("connector.name must be a non-empty string")
        self._connectors[connector.name] = connector

    def unregister(self, name: str) -> bool:
        return self._connectors.pop(name, None) is not None

    def get(self, name: str) -> IChannelConnector:
        try:
            return self._connectors[name]
        except KeyError as exc:
            raise ChannelNotFoundError(name) from exc

    def channels(self) -> list[str]:
        return sorted(self._connectors)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def send(self, channel: str, message: ChannelMessage) -> DeliveryReceipt:
        connector = self.get(channel)
        try:
            return connector.send(message)
        except Exception as exc:  # connectors must not crash callers
            logger.exception("channel %s failed to deliver", channel)
            return DeliveryReceipt(
                channel=channel,
                status=DeliveryStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )

    def broadcast(
        self, message: ChannelMessage, *, channels: list[str] | None = None
    ) -> list[DeliveryReceipt]:
        targets = channels if channels is not None else self.channels()
        return [self.send(c, message) for c in targets]
