"""Tests for OpenTelemetry helper behavior."""

from __future__ import annotations

import pytest

from iris_harness.foundation.observability.tracer import current_trace_ids, maybe_span


class _FakeSpan:
    def __init__(self) -> None:
        self.ended = False
        self.exceptions: list[Exception] = []

    def end(self) -> None:
        self.ended = True

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)


class _FakeTracer:
    def __init__(self) -> None:
        self.span = _FakeSpan()
        self.started: list[tuple[str, dict[str, object]]] = []
        self.started_as_current = False

    def start_span(self, name: str, **kwargs: object) -> _FakeSpan:
        self.started.append((name, kwargs))
        return self.span

    def start_as_current_span(self, name: str, **kwargs: object) -> _FakeSpan:
        self.started_as_current = True
        raise AssertionError("maybe_span must not attach a current-span ContextVar token")


def test_maybe_span_does_not_attach_current_context() -> None:
    tracer = _FakeTracer()

    with maybe_span(tracer, "iris.chat_stream") as span:
        assert span is tracer.span

    assert tracer.started == [("iris.chat_stream", {})]
    assert tracer.started_as_current is False
    assert tracer.span.ended is True


def test_maybe_span_records_exception_and_ends_span() -> None:
    tracer = _FakeTracer()
    error = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        with maybe_span(tracer, "iris.chat_stream"):
            raise error

    assert tracer.span.exceptions == [error]
    assert tracer.span.ended is True


def test_current_trace_ids_returns_none_without_active_span() -> None:
    # Learning must never depend on tracing being enabled (D1): no span → (None, None).
    assert current_trace_ids() == (None, None)


def test_current_trace_ids_reads_active_span() -> None:
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("turn"):
        trace_id, span_id = current_trace_ids()
    assert trace_id is not None and len(trace_id) == 32
    assert span_id is not None and len(span_id) == 16
    # Outside the span context the ids clear again.
    assert current_trace_ids() == (None, None)
