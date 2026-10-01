"""Console connector — writes messages to a TextIO stream (defaults to stdout)."""

from __future__ import annotations

import sys
import uuid
from typing import TextIO

from ..models import ChannelMessage, DeliveryReceipt, DeliveryStatus


class ConsoleConnector:
    """Simple connector that prints messages to a stream. Useful for dev + tests."""

    def __init__(self, name: str = "console", *, stream: TextIO | None = None) -> None:
        self.name = name
        self._stream: TextIO = stream if stream is not None else sys.stdout

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        prefix = f"[{self.name}] -> {message.recipient}"
        if message.subject:
            prefix += f" :: {message.subject}"
        self._stream.write(f"{prefix}\n{message.body}\n")
        self._stream.flush()
        return DeliveryReceipt(
            channel=self.name,
            status=DeliveryStatus.SENT,
            message_id=uuid.uuid4().hex,
        )

    def healthy(self) -> bool:
        return not self._stream.closed
