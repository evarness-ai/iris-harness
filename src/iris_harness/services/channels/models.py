"""Channel domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class DeliveryStatus(StrEnum):
    SENT = "sent"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ChannelMessage:
    """A message destined for one or more channels.

    ``recipient`` is interpreted by the connector (chat id for Telegram, file
    handle for console, etc.).
    """

    recipient: str
    body: str
    subject: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class DeliveryReceipt:
    channel: str
    status: DeliveryStatus
    message_id: str = ""
    error: str = ""
    delivered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
