"""IRIS Channel Gateway — multi-platform delivery via pluggable connectors."""

from .gateway import ChannelGateway, ChannelNotFoundError
from .models import ChannelMessage, DeliveryReceipt, DeliveryStatus
from .protocol import IChannelConnector

__all__ = [
    "ChannelGateway",
    "ChannelMessage",
    "ChannelNotFoundError",
    "DeliveryReceipt",
    "DeliveryStatus",
    "IChannelConnector",
]
