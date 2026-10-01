"""Compatibility helpers around the observability bootstrap modules."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from .otlp_setup import TracingState, initialize_tracing

logger = logging.getLogger(__name__)


def setup_tracing() -> Any:
    """Return a configured tracer, or ``None`` when no OTLP endpoint is configured."""
    return setup_tracing_state().tracer


def setup_tracing_state() -> TracingState:
    """Return the full tracing bootstrap result for callers that need metadata."""
    return initialize_tracing()


@contextmanager
def maybe_current_span(tracer: Any, name: str, *, kind: Any = None) -> Generator[Any, None, None]:
    """Create a span and make it current — for plain (non-generator) paths.

    Unlike ``maybe_span``, this attaches the span to the OpenTelemetry
    context so nested spans (pipeline stages, LLM client spans created via
    the global tracer) parent correctly and the active trace id is visible
    to the governance audit log. Safe only where enter and exit happen in
    the same frame — i.e. NOT inside streaming generators; those must keep
    using ``maybe_span``.
    """
    if tracer is None:
        yield None
        return
    kwargs = {"kind": kind} if kind is not None else {}
    try:
        cm = tracer.start_as_current_span(name, **kwargs)
    except AttributeError:
        with maybe_span(tracer, name, kind=kind) as span:
            yield span
        return
    with cm as span:
        try:
            yield span
        except Exception as exc:
            _mark_span_error(span, exc)
            raise


def current_trace_ids() -> tuple[str | None, str | None]:
    """Return ``(trace_id, span_id)`` for the active OTel span as hex strings.

    Lets learning signals correlate to the exact span that produced them
    (learning-observability.md §4.1, D3). Returns ``(None, None)`` when tracing
    is off or no span is recording — learning must never depend on OTel trace
    export being enabled (D1), so this is purely additive context. Never raises.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if not getattr(ctx, "is_valid", False):
            return (None, None)
        return (trace.format_trace_id(ctx.trace_id), trace.format_span_id(ctx.span_id))
    except Exception:  # noqa: BLE001 — correlation is best-effort, never fatal
        return (None, None)


def set_span_attributes(span: Any, attributes: dict[str, Any]) -> None:
    """Best-effort attribute setter that skips None values and never raises."""
    if span is None:
        return
    for key, value in attributes.items():
        if value is None:
            continue
        try:
            span.set_attribute(key, value)
        except Exception:  # noqa: BLE001, S112 - tracing must never break the pipeline
            continue


@contextmanager
def maybe_span(tracer: Any, name: str, *, kind: Any = None) -> Generator[Any, None, None]:
    """Create a span when tracing is live, without binding it to ContextVars.

    ``start_as_current_span`` attaches an OpenTelemetry token to the current
    ``contextvars`` context and detaches it on exit. IRIS uses this helper inside
    streaming generators, which ASGI servers may resume or close from a sibling
    context; token-based detach then raises ``ValueError: token was created in a
    different Context``. Starting a span without making it current still exports
    the runtime span and is safe across generator boundaries.
    """
    if tracer is None:
        yield None
        return
    kwargs = {"kind": kind} if kind is not None else {}
    span = tracer.start_span(name, **kwargs)
    try:
        yield span
    except Exception as exc:
        _mark_span_error(span, exc)
        raise
    finally:
        span.end()


def _mark_span_error(span: Any, exc: Exception) -> None:
    try:
        span.record_exception(exc)
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, str(exc)))
    except Exception:  # noqa: BLE001, S110
        pass
