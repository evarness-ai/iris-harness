"""Activity notices — the harness's notify leg, carved out of ``IrisRuntime`` (OSS plan
M5.7 track C, slice 8).

Two things put a notice in front of the user without a turn asking for it: a
background Activity finishing or failing (the async spine, see
docs/architecture/async-activities-and-notifications.md), and an approval window
lapsing (ADR-0108). Both land the same two ways — an assistant turn in the session
that started the work, and a best-effort message on an outbound channel. This is the
"notify" half of the owner's track A rule: core keeps the mechanism that invokes
notification messaging, and everything it says is composed by whoever asked.

:class:`ActivityNoticeHost` declares the six runtime members the notices read, so mypy
checks the runtime still supplies them — the enforcement ``TurnHost`` and
``EscalationHost`` gave their seams. The host is read **at call time**, not captured.

The lazily-built :class:`~iris_harness.services.activities.ActivityRunner` moved here with the
code that builds it. That makes the collaborator's identity load-bearing: the runner
subscribes this object's completion handlers to the runtime's bus once, so the runtime
must build **one** of these per runtime — a second one would build a second runner and
a second pair of subscriptions, and every completion would be announced twice.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Protocol

from iris_harness.memory.compactor import ConversationTurn
from iris_harness.services.channels import ChannelMessage, DeliveryStatus

if TYPE_CHECKING:
    from pathlib import Path

    from iris_harness.foundation.eventbus import EventBus
    from iris_harness.memory.store import MemoryStore
    from iris_harness.runtime.session_memory import SessionMemory
    from iris_harness.services.channels import ChannelGateway

logger = logging.getLogger(__name__)


class ActivityNoticeHost(Protocol):
    """The six runtime members the notices reach.

    ``_event_bus`` is declared as-is, underscore included, for the reason ``TurnHost``
    gives: renaming a member is its own change. ``sessions`` is session memory; a notice
    is appended to its conversation window (OSS plan M5.7 track C slice 16).
    """

    data_dir: Path
    memory_store: MemoryStore
    channels: ChannelGateway
    default_channel: str
    _event_bus: EventBus
    sessions: SessionMemory


class ActivityNotices:
    """Builds the Activity runner and delivers notices for one runtime. See the module
    docstring. Satisfies ``governance.approvals.service.LapseNotifier``."""

    def __init__(self, host: ActivityNoticeHost) -> None:
        self._host = host
        self._runner: Any = None

    def activities(self) -> Any:
        """Lazily build the background ActivityRunner + wire completion notify.

        Uses this runtime's private ``_event_bus`` so the completion subscribers
        fire only for THIS runtime's activities (the process-global default bus
        would cross-talk between runtimes in tests) and so plugin subscribers
        registered at build time are on the same bus. The read path
        (`GET /activities`) reads the store directly, so it doesn't need the bus.
        """
        runner = self._runner
        if runner is not None:
            return runner
        from iris_harness.services.activities import (
            ACTIVITY_COMPLETED,
            ACTIVITY_FAILED,
            ActivityRunner,
            ActivityStore,
        )

        bus = self._host._event_bus
        store = ActivityStore(db_path=self._host.data_dir / "activities.db", bus=bus)
        store.ensure_schema()
        try:
            workers = max(1, int(os.getenv("IRIS_ACTIVITY_WORKERS", "1")))
        except ValueError:
            workers = 1
        runner = ActivityRunner(store=store, bus=bus, max_workers=workers)
        bus.on(ACTIVITY_COMPLETED, self._on_activity_completed)
        bus.on(ACTIVITY_FAILED, self._on_activity_failed)
        self._runner = runner
        return runner

    def inject_system_notice(self, session_id: str, text: str) -> None:
        """Append a system/assistant notice to a session's history + durable log.

        The in-chat leg of a completion notification: it lands in the session so
        it shows in web chat history, and — because it's an assistant turn — the
        follow-up ("file 1 and 3") reads the parked analysis the job stored.
        True live push to an idle REPL is a later (SSE) upgrade.
        """
        history = self._host.sessions.conversations.setdefault(session_id, [])
        history.append(ConversationTurn(role="assistant", content=text))
        try:
            self._host.memory_store.save_conversation_turns(session_id, [("assistant", text)])
        except Exception:
            logger.exception("activity notice: failed to persist turn for %s", session_id)

    def deliver_lapse_notice(self, *, session_id: str | None, channel: str, text: str) -> None:
        """Satisfies ``governance.approvals.service.LapseNotifier``.

        Reuses the Activity spine's two legs rather than inventing a third: the in-chat
        notice is the one that matters, because the conversation that was told "paused
        for approval" is where the silence afterwards is conspicuous. ``session_id`` is
        None for an approval written before the column existed, or raised outside any
        conversation; those still get the channel leg.
        """
        if session_id:
            self.inject_system_notice(session_id, text)
        self._deliver_activity_notification(channel, subject="IRIS approval expired", body=text)

    def _on_activity_completed(self, payload: Any) -> None:
        """Turn a finished Activity into an in-chat notice + a channel message."""
        from iris_harness.services.activities import ActivityCompletedPayload

        if not isinstance(payload, ActivityCompletedPayload):
            return
        summary = payload.result_summary or f"{payload.title} finished."
        origin = payload.origin or ""
        if origin.startswith("chat:"):
            session_id = origin.split(":", 1)[1]
            self.inject_system_notice(session_id, summary)
        channel = str(payload.metadata.get("channel") or self._host.default_channel)
        self._deliver_activity_notification(channel, subject="IRIS Activity", body=summary)

    def _on_activity_failed(self, payload: Any) -> None:
        from iris_harness.services.activities import ActivityFailedPayload

        if not isinstance(payload, ActivityFailedPayload):
            return
        body = f"I couldn't finish '{payload.title}': {payload.error}"
        origin = payload.origin or ""
        if origin.startswith("chat:"):
            self.inject_system_notice(origin.split(":", 1)[1], body)
        self._deliver_activity_notification("console", subject="IRIS Activity failed", body=body)

    def _deliver_activity_notification(self, channel: str, *, subject: str, body: str) -> None:
        """Best-effort outbound delivery of a completion notice.

        Mirrors the reminder-channels resolution: use the requested channel when
        it's a registered non-default connector, else the runtime default; skip
        silently when nothing outbound is registered (web/console-only setups
        still get the in-chat notice + the Activity feed)."""
        try:
            registered = set(self._host.channels.channels())
        except Exception:  # noqa: BLE001
            return
        name = (
            channel
            if channel in registered and channel != "default"
            else self._host.default_channel
        )
        if name not in registered:
            return
        try:
            receipt = self._host.channels.send(
                name,
                ChannelMessage(
                    recipient="", body=body, subject=subject, metadata={"activity": True}
                ),
            )
        except Exception:
            logger.exception("activity notice: channel send failed on %s", name)
            return
        if receipt.status is not DeliveryStatus.SENT:
            logger.warning(
                "activity notice: delivery not SENT on %s: %s",
                name,
                receipt.error or "<no error>",
            )
