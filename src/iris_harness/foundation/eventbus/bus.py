"""Topic-agnostic event bus for the IRIS runtime.

Producers emit events identified by topic strings; subscribers register
handlers per topic. The bus is in-process; durability is via state
polling (see canonical doc §3.4) — there is no persistent queue.

This module is intentionally generic: it knows nothing about specific
topics, payload shapes, or timing helpers. Subsystems that want
convenience wrappers (e.g. context managers that emit start/end pairs)
should publish them as module-level functions in their own packages,
not as methods on ``EventBus``. See ``iris_harness.services.activities.events`` for
an example.

See ADR-Q9 (canonical doc §3.4) for the design and ADR-0013 for the
promotion implementation decisions.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from iris_harness.foundation.process_state import register_process_state

_log = logging.getLogger(__name__)

EventHandler = Callable[[Any], None | Awaitable[None]]


@dataclass
class EventBus:
    """Topic-based async/sync event bus.

    Handlers may be plain callables or async coroutine functions.
    Exceptions in handlers are caught and logged — they never abort the caller.
    """

    _handlers: dict[str, list[EventHandler]] = field(default_factory=lambda: defaultdict(list))
    # Handler tasks scheduled by emit_sync. The event loop keeps only a weak reference to
    # a task, so one nobody holds can be garbage-collected before it finishes.
    _pending: set[asyncio.Task[Any]] = field(default_factory=set)

    def on(self, event: str, handler: EventHandler) -> EventHandler:
        """Register handler for event. Returns handler for use as a decorator."""
        self._handlers[event].append(handler)
        return handler

    def off(self, event: str, handler: EventHandler) -> None:
        """Unregister a previously registered handler."""
        handlers = self._handlers.get(event)
        if handlers and handler in handlers:
            handlers.remove(handler)

    async def emit(self, event: str, payload: Any = None) -> None:
        """Emit event asynchronously — awaits async handlers in place."""
        for handler in list(self._handlers.get(event, [])):
            try:
                result = handler(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 — one handler must not stop the others; logged
                _log.exception("async handler %r raised for event %r", handler, event)

    def emit_sync(self, event: str, payload: Any = None) -> None:
        """Emit event from synchronous code.

        Async handlers are scheduled on the running loop (fire-and-forget)
        or run synchronously when no loop is active.
        """
        for handler in list(self._handlers.get(event, [])):
            try:
                result = handler(payload)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:  # no loop running in this thread: run it here
                        asyncio.run(result)
                    else:
                        task = loop.create_task(result)
                        self._pending.add(task)
                        task.add_done_callback(self._settle)
            except Exception:  # noqa: BLE001 — one handler must not stop the others; logged
                _log.exception("sync handler %r raised for event %r", handler, event)

    def _settle(self, task: asyncio.Task[Any]) -> None:
        """Release a finished handler task, logging what it raised (as emit does)."""
        self._pending.discard(task)
        if not task.cancelled() and task.exception() is not None:
            _log.error("async handler task raised", exc_info=task.exception())


# ---------------------------------------------------------------------------
# Module-level shared bus (default; tests can inject their own)
# ---------------------------------------------------------------------------

_default_bus: EventBus | None = None


def get_default_bus() -> EventBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def reset_default_bus() -> None:
    """Replace the shared bus — useful in tests to prevent handler bleed."""
    global _default_bus
    _default_bus = EventBus()


_Subscriptions = dict[str, list[EventHandler]]


def _save_default_bus() -> tuple[EventBus | None, _Subscriptions]:
    bus = _default_bus
    handlers = {t: list(h) for t, h in bus._handlers.items()} if bus is not None else {}
    return bus, handlers


def _restore_default_bus(saved: tuple[EventBus | None, _Subscriptions]) -> None:
    """The shared bus the process had, with exactly the subscribers it had: a plugin
    that subscribed at ``scope="process"`` during a harness run is dropped."""
    global _default_bus
    bus, handlers = saved
    _default_bus = bus
    if bus is not None:
        bus._handlers.clear()
        for topic, subscribed in handlers.items():
            bus._handlers.setdefault(topic, []).extend(subscribed)


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
register_process_state(f"{__name__}._default_bus", _save_default_bus, _restore_default_bus)
