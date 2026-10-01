"""Post-response hook that records learning signals into the metrics store."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import Any

from iris_harness.foundation.process_state import track_globals
from iris_harness.services.learning.store import LearningMetricsStore

logger = logging.getLogger(__name__)

# Process-level signal-stream health (learning-observability.md §4.1, D2).
# A dropped signal is NOT random — recording is most likely to be cut short on
# the same failure/slow paths we want to learn from, so the loss correlates with
# the outcome being measured. We therefore make completeness observable instead
# of silently swallowing: count what we wrote and what we lost. These are an
# always-available in-process floor; the durable counter on the store
# (``signals_dropped_total``) survives restarts.
_HEALTH_LOCK = threading.Lock()
_RECORDED = 0
_DROPPED = 0


def signal_health() -> dict[str, Any]:
    """Return the in-process signal-stream completeness counters."""
    with _HEALTH_LOCK:
        recorded, dropped = _RECORDED, _DROPPED
    total = recorded + dropped
    return {
        "signals_recorded": recorded,
        "signals_dropped": dropped,
        "drop_rate": (dropped / total) if total else 0.0,
    }


def reset_signal_health() -> None:
    """Reset the process counters (tests only)."""
    global _RECORDED, _DROPPED
    with _HEALTH_LOCK:
        _RECORDED = 0
        _DROPPED = 0


def _bump_recorded(n: int) -> None:
    global _RECORDED
    with _HEALTH_LOCK:
        _RECORDED += n


def _bump_dropped(n: int) -> None:
    global _DROPPED
    with _HEALTH_LOCK:
        _DROPPED += n


class LearningSignalCollector:
    """Capture per-response signals and persist them via :class:`LearningMetricsStore`.

    The collector is intentionally permissive: failures are logged but never
    propagated, because learning telemetry must not break the chat path. Unlike
    the original silent-swallow, a failure now increments an observable
    dropped-signal counter (in-process + durable) so completeness is measurable.
    """

    def __init__(self, store: LearningMetricsStore, *, source: str = "chat") -> None:
        self._store = store
        self._source = source

    def record_response(
        self,
        *,
        intent: str,
        agent_type: str,
        intent_confidence: float | None,
        has_errors: bool,
        latency_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        resolved_tier: str | None = None,
        resolved_agent: str | None = None,
    ) -> None:
        """Record one chat response as several derived metrics.

        Correlation fields (``session_id``/``turn_id``/``trace_id``/``span_id``)
        and the resolved tier/agent are stored as first-class columns so each
        metric is auditable back to the turn and span that produced it
        (learning-observability.md §4.1, D3/D4). ``resolved_agent`` defaults to
        ``agent_type`` when the caller does not distinguish requested vs. resolved.
        """
        meta = {"intent": intent, "agent_type": agent_type, **dict(metadata or {})}
        correlation: dict[str, Any] = {
            "session_id": session_id,
            "turn_id": turn_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "resolved_tier": resolved_tier,
            "resolved_agent": resolved_agent if resolved_agent is not None else agent_type,
        }
        written = 0
        try:
            if intent_confidence is not None:
                self._store.record_signal(
                    source=self._source,
                    metric_name="intent_confidence",
                    value=float(intent_confidence),
                    success=not has_errors,
                    latency_ms=latency_ms,
                    metadata=meta,
                    **correlation,
                )
                written += 1
            self._store.record_signal(
                source=self._source,
                metric_name="response_has_errors",
                value=1.0 if has_errors else 0.0,
                success=not has_errors,
                latency_ms=latency_ms,
                metadata=meta,
                **correlation,
            )
            written += 1
            _bump_recorded(written)
        except Exception:  # signals are best-effort, but never silent
            logger.exception("failed to record learning signal")
            _bump_dropped(1)
            # Persist the drop on an independently-guarded path: the signal
            # write failed, but a single-integer counter bump usually won't.
            try:
                self._store.increment_counter("signals_dropped_total")
            except Exception:  # noqa: BLE001, S110 — last-resort, nothing more we can do
                pass

    def record_escalation_shadow(
        self,
        *,
        verdict: Mapping[str, Any],
        has_errors: bool,
        intent: str | None = None,
        latency_ms: float | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        resolved_tier: str | None = None,
        resolved_agent: str | None = None,
    ) -> None:
        """Record the shadow escalation judge's would-be decision (ADR-0068 L2).

        The ``escalation_shadow`` metric pairs the would-be action with the
        turn's available outcome proxy (``has_errors``) — the §4.3 counterfactual
        seed: did the turns the judge would have acted on actually fare worse?
        ``value`` is 1.0 when the judge would act (anything but ``accept``), so a
        simple average is the would-act rate. Best-effort, drop-counted like the
        primary signals.
        """
        action = str(verdict.get("action") or "accept")
        meta = {**dict(verdict), "has_errors": has_errors, "intent": intent}
        try:
            self._store.record_signal(
                source=self._source,
                metric_name="escalation_shadow",
                value=1.0 if action != "accept" else 0.0,
                success=not has_errors,
                latency_ms=latency_ms,
                metadata=meta,
                session_id=session_id,
                turn_id=turn_id,
                trace_id=trace_id,
                span_id=span_id,
                resolved_tier=resolved_tier,
                resolved_agent=resolved_agent,
            )
            _bump_recorded(1)
        except Exception:  # best-effort, never silent
            logger.exception("failed to record escalation shadow signal")
            _bump_dropped(1)
            try:
                self._store.increment_counter("signals_dropped_total")
            except Exception:  # noqa: BLE001, S110 — last-resort
                pass

    def record_feedback(
        self,
        *,
        sentiment: str,
        session_id: str | None = None,
        turn_id: str | None = None,
        trace_id: str | None = None,
        rating: int | None = None,
        note: str | None = None,
        intent: str | None = None,
        agent_type: str | None = None,
    ) -> None:
        """Record one EXPLICIT user-feedback signal (ADR-0072 slice 1).

        A thumbs ``"up"``/``"down"`` on a single answer, with an optional 1-5
        ``rating`` and a short ``note``. Stored as the ``user_feedback`` metric so it
        sits beside measured outcomes and feeds the learning loop, correlated to the
        turn via session/turn/trace ids. Best-effort + drop-counted like the other
        recorders — capturing feedback must never break the request path.
        """
        positive = sentiment.strip().lower() == "up"
        metadata: dict[str, Any] = {"sentiment": "up" if positive else "down"}
        if rating is not None:
            metadata["rating"] = int(rating)
        if note and note.strip():
            metadata["note"] = note.strip()[:500]
        if intent:
            metadata["intent"] = intent
        if agent_type:
            metadata["agent_type"] = agent_type
        try:
            self._store.record_signal(
                source=self._source,
                metric_name="user_feedback",
                value=1.0 if positive else -1.0,
                success=positive,
                metadata=metadata,
                session_id=session_id,
                turn_id=turn_id,
                trace_id=trace_id,
            )
            _bump_recorded(1)
        except Exception:  # best-effort, never silent
            logger.exception("failed to record user feedback")
            _bump_dropped(1)
            try:
                self._store.increment_counter("signals_dropped_total")
            except Exception:  # noqa: BLE001, S110 — last-resort
                pass

    def record_metric(
        self,
        *,
        metric_name: str,
        value: float,
        success: bool,
        latency_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        resolved_tier: str | None = None,
        resolved_agent: str | None = None,
    ) -> None:
        """Record one arbitrary correlated outcome metric (§4.2).

        The general entry point for measured outcome signals (``task_completed``,
        ``turn_tokens``, ``user_correction``). Best-effort + drop-counted like the
        primary signals so completeness stays observable.
        """
        try:
            self._store.record_signal(
                source=self._source,
                metric_name=metric_name,
                value=float(value),
                success=success,
                latency_ms=latency_ms,
                metadata=dict(metadata or {}),
                session_id=session_id,
                turn_id=turn_id,
                trace_id=trace_id,
                span_id=span_id,
                resolved_tier=resolved_tier,
                resolved_agent=resolved_agent,
            )
            _bump_recorded(1)
        except Exception:  # best-effort, never silent
            logger.exception("failed to record outcome metric %s", metric_name)
            _bump_dropped(1)
            try:
                self._store.increment_counter("signals_dropped_total")
            except Exception:  # noqa: BLE001, S110 — last-resort
                pass


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_RECORDED", "_DROPPED")
