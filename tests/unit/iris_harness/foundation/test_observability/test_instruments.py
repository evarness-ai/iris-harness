"""Tests for OTEL auto-instrumentation helpers."""

from __future__ import annotations

import sys
import types

from iris_harness.foundation.observability import instruments


def _install_module(monkeypatch, name: str, module: types.ModuleType) -> None:
    parts = name.split(".")
    for idx in range(1, len(parts)):
        parent_name = ".".join(parts[:idx])
        if parent_name not in sys.modules:
            monkeypatch.setitem(sys.modules, parent_name, types.ModuleType(parent_name))
    monkeypatch.setitem(sys.modules, name, module)


def test_instrument_runtime_disabled_is_noop(monkeypatch) -> None:
    instruments._reset_instrumentation_state()
    monkeypatch.setenv("IRIS_OTEL_ENABLED", "0")

    state = instruments.instrument_runtime(tracer_provider=object())

    assert state.enabled is False
    assert state.instrumented_targets == ()


def test_instrument_runtime_is_idempotent(monkeypatch) -> None:
    langchain_calls: list[object] = []
    httpx_calls: list[object] = []

    class _LangChainInstrumentor:
        def instrument(self, *, tracer_provider) -> None:
            langchain_calls.append(tracer_provider)

    class _HTTPXInstrumentor:
        def instrument(self, *, tracer_provider) -> None:
            httpx_calls.append(tracer_provider)

    langchain_module = types.ModuleType("openinference.instrumentation.langchain")
    langchain_module.LangChainInstrumentor = _LangChainInstrumentor
    httpx_module = types.ModuleType("opentelemetry.instrumentation.httpx")
    httpx_module.HTTPXClientInstrumentor = _HTTPXInstrumentor
    _install_module(monkeypatch, "openinference.instrumentation.langchain", langchain_module)
    _install_module(monkeypatch, "opentelemetry.instrumentation.httpx", httpx_module)

    instruments._reset_instrumentation_state()
    provider = object()
    first = instruments.instrument_runtime(provider, enabled=True)
    second = instruments.instrument_runtime(provider, enabled=True)

    assert first.enabled is True
    assert first.initialized_now is True
    assert set(first.instrumented_targets) == {"langchain", "httpx"}
    assert second.initialized_now is False
    assert langchain_calls == [provider]
    assert httpx_calls == [provider]


def test_instrument_runtime_records_missing_dependency(monkeypatch) -> None:
    instruments._reset_instrumentation_state()
    monkeypatch.setenv("IRIS_OTEL_ENABLED", "1")
    broken_httpx = types.ModuleType("opentelemetry.instrumentation.httpx")
    _install_module(monkeypatch, "opentelemetry.instrumentation.httpx", broken_httpx)

    state = instruments.instrument_runtime(
        tracer_provider=object(),
        enabled=True,
        enable_langchain=False,
        enable_httpx=True,
    )

    assert state.enabled is True
    assert state.instrumented_targets == ()
    assert state.error is not None
