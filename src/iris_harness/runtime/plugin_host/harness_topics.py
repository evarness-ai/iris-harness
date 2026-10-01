"""Topics only the harness publishes and hears, and the one place that says so.

``approval.call_completed`` carries every plugin's approved calls and their summaries
(plugin-capabilities decision 1). A plugin hears its own through
``api.on_approved_call``, which filters by caller; no plugin may subscribe to the raw
topic (it would read other plugins' outcomes) or publish on it (it would forge one).

A plugin reaches the runtime's bus two ways: ``api.subscribe`` / ``api.publish``, and
``services.events`` itself. Both go through :func:`refuse_harness_topic`:
``PluginAPI`` calls it, and ``services.events`` is a :class:`GuardedEventBus` — a view
of the runtime's bus that refuses the same topics on every verb. The harness's own
code (the ``ToolService`` that emits the topic, ``PluginAPI`` delivering the filtered
subscription) holds the raw bus. A new harness-owned topic is one entry in
:data:`HARNESS_TOPICS`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from iris_harness.kernel.governance.approvals.events import APPROVAL_CALL_COMPLETED

if TYPE_CHECKING:
    from collections.abc import Callable

    from iris_harness.foundation.eventbus import EventBus

#: Harness-owned topic -> the sanctioned way a plugin hears it (None: it does not).
HARNESS_TOPICS: dict[str, str | None] = {
    APPROVAL_CALL_COMPLETED: "api.on_approved_call",
}


def refuse_harness_topic(plugin: str | None, topic: str, verb: str) -> None:
    """Raise ``PermissionError`` when a plugin would ``verb`` a harness-owned topic."""
    if topic not in HARNESS_TOPICS:
        return
    who = f"plugin {plugin!r}" if plugin else "a plugin"
    instead = HARNESS_TOPICS[topic]
    hint = f"; use {instead}, which delivers only your own calls" if instead else ""
    raise PermissionError(f"{who} may not {verb} {topic!r}: the harness owns it{hint}")


class GuardedEventBus:
    """The runtime's event bus as plugins see it (``HarnessServices.events``).

    Every verb — ``on``, ``off``, ``emit``, ``emit_sync`` — refuses a harness-owned
    topic and forwards anything else to the runtime's bus unchanged, so the
    ``EventBusService`` contract is the same for every other topic.
    """

    def __init__(self, bus: EventBus) -> None:
        self._harness_bus = bus

    def on(self, event: str, handler: Callable[[Any], Any]) -> Any:
        refuse_harness_topic(None, event, "subscribe to")
        return self._harness_bus.on(event, handler)

    def off(self, event: str, handler: Callable[[Any], Any]) -> None:
        refuse_harness_topic(None, event, "unsubscribe from")
        self._harness_bus.off(event, handler)

    async def emit(self, event: str, payload: Any = None) -> None:
        refuse_harness_topic(None, event, "publish")
        await self._harness_bus.emit(event, payload)

    def emit_sync(self, event: str, payload: Any = None) -> None:
        refuse_harness_topic(None, event, "publish")
        self._harness_bus.emit_sync(event, payload)


def harness_bus(bus: Any) -> Any:
    """The raw bus behind a guarded view (or ``bus`` itself). Harness code only:
    plugins cannot import ``iris_harness.runtime`` (import contract, gate 2)."""
    return bus._harness_bus if isinstance(bus, GuardedEventBus) else bus


__all__ = ["HARNESS_TOPICS", "GuardedEventBus", "harness_bus", "refuse_harness_topic"]
