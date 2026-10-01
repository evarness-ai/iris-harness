"""The process-wide event bus.

A plugin publishes and subscribes through `api.publish` / `api.subscribe` while it is
mounted. Code that runs outside a mount -- a store emitting "task.completed" from a
CLI command, a subscriber wired at import -- uses `get_default_bus()`, the same bus
the runtime hands its plugins. Topic names and payload models live with the plugin
that produces them, never here.
"""

from __future__ import annotations

from iris_harness.foundation.eventbus import EventBus, EventHandler, get_default_bus

__all__ = ["EventBus", "EventHandler", "get_default_bus"]
