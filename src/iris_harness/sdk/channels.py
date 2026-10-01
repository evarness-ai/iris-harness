"""What a chat-surface plugin needs to be a channel.

``IChannelConnector``, the protocol a connector implements (``name``, ``send``,
``healthy``) and ``PluginAPI.register_channel`` takes; the wire models a connector
returns, the one parser for `config/channels.yaml`, the Telegram and Web Push
transports the built-in surfaces drive, and the ``approval.requested`` topic a surface
subscribes to when it nudges the owner about a waiting approval (the kernel emits it;
``PluginAPI.subscribe`` with ``scope="process"`` receives it). A surface plugin is the
only kind that needs these; everything else reaches channels through
`HarnessServices.channels`.
"""

from __future__ import annotations

from iris_harness.kernel.governance.approvals.events import (
    APPROVAL_REQUESTED,
    ApprovalRequestedPayload,
)
from iris_harness.services.channels.channel_config import load_channel_config, telegram_rows
from iris_harness.services.channels.connectors.telegram import TelegramConnector
from iris_harness.services.channels.models import (
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
)
from iris_harness.services.channels.protocol import IChannelConnector
from iris_harness.services.channels.web_push import PushSubscriptionStore, WebPushConnector

__all__ = [
    "APPROVAL_REQUESTED",
    "ApprovalRequestedPayload",
    "ChannelMessage",
    "DeliveryReceipt",
    "DeliveryStatus",
    "IChannelConnector",
    "PushSubscriptionStore",
    "TelegramConnector",
    "WebPushConnector",
    "load_channel_config",
    "telegram_rows",
]
