"""In-process background runner for Activities.

``submit(kind, title, work, origin=...)`` creates a ``queued`` row, hands
``work`` to a small ``ThreadPoolExecutor``, and drives it through
``running -> completed/failed`` while persisting status + progress via the
``ActivityStore``. ``work`` receives a ``progress(frac, msg)`` callback.

This is deliberately in-process (runs inside ``iris_api``): the goal is to
unblock the chat turn, not to add a service. Extraction to the planned
Background Worker (:8005) is a later step if isolation/backpressure demand
it. Default single worker (``IRIS_ACTIVITY_WORKERS``) serializes jobs so a
vision batch + a cleanup embed pass don't contend on CPU/Apple Vision.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from iris_harness.foundation.eventbus import EventBus

from .models import ActivityOutcome
from .store import ActivityStore

logger = logging.getLogger(__name__)

# work(progress) -> outcome; progress(frac, message) updates the row.
ProgressFn = Callable[[float, str], None]
WorkFn = Callable[[ProgressFn], ActivityOutcome]


@dataclass
class ActivityRunner:
    """Submits background work and tracks it as an Activity."""

    store: ActivityStore
    bus: EventBus | None = None
    max_workers: int = 1
    _executor: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, self.max_workers), thread_name_prefix="iris-activity"
        )

    def submit(
        self,
        *,
        kind: str,
        title: str,
        work: WorkFn,
        origin: str = "",
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Create a queued Activity, run ``work`` in the background, return the id."""
        activity = self.store.create(kind=kind, title=title, origin=origin, metadata=metadata or {})
        self._executor.submit(self._run, activity.id, work)
        return activity.id

    def run_now(
        self,
        *,
        kind: str,
        title: str,
        work: WorkFn,
        origin: str = "",
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Synchronous variant — run ``work`` inline (tests, tiny jobs)."""
        activity = self.store.create(kind=kind, title=title, origin=origin, metadata=metadata or {})
        self._run(activity.id, work)
        return activity.id

    def _run(self, activity_id: str, work: WorkFn) -> None:
        self.store.mark_running(activity_id)

        def progress(frac: float, message: str = "") -> None:
            try:
                self.store.mark_progress(activity_id, frac, message)
            except Exception:
                logger.exception("activity %s: progress update failed", activity_id)

        try:
            outcome = work(progress)
        except Exception as exc:
            logger.exception("activity %s failed", activity_id)
            self.store.mark_failed(activity_id, str(exc) or exc.__class__.__name__)
            return
        try:
            self.store.mark_completed(
                activity_id,
                result_summary=outcome.result_summary,
                undo_ref=outcome.undo_ref,
                metadata=outcome.metadata,
            )
        except Exception:
            # Completion bookkeeping (incl. the notification subscriber) failed after
            # the work itself succeeded — record it, don't crash the worker thread.
            logger.exception("activity %s: completion handling failed", activity_id)

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait)
