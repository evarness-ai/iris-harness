"""Generic event bus for the IRIS runtime.

This package hosts the topic-agnostic infrastructure (``EventBus``,
handlers, default-bus singleton). Subsystem-specific topic names and
payload models live with their producers — see e.g.
a subsystem's own ``events`` module for its private topics.

See ADR-Q9 (canonical doc §3.4) and ADR-0013.
"""

from .bus import EventBus, EventHandler, get_default_bus, reset_default_bus

__all__ = ["EventBus", "EventHandler", "get_default_bus", "reset_default_bus"]
