"""Trace export over standard OTLP (ADR-0128): no endpoint, no exporter; an endpoint,
a batched OTLP exporter under the service name; httpx client spans when enabled."""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http import trace_exporter as http_trace_exporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from iris_harness.foundation.auth import auth_headers
from iris_harness.foundation.observability import instruments, otlp_setup
from iris_harness.foundation.observability.otlp_setup import (
    TracingConfig,
    initialize_tracing,
    load_tracing_config,
    redact_endpoint,
)
from iris_harness.server.iris_api.main import create_app

_OTEL_VARS = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_SERVICE_NAME",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_SDK_DISABLED",
    "OTEL_TRACES_EXPORTER",
)


@pytest.fixture(autouse=True)
def clean_otel_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """No OTEL_* from the developer's shell; the global provider is recorded, not set
    (the real one is set-once per process); instrumentation starts fresh."""
    for name in _OTEL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IRIS_OTEL_LANGCHAIN_ENABLED", "0")
    monkeypatch.setenv("IRIS_OTEL_HTTPX_ENABLED", "0")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(trace, "set_tracer_provider", lambda p: seen.__setitem__("global", p))
    instruments._reset_instrumentation_state()
    yield seen
    provider = seen.get("global")
    if provider is not None:
        provider.shutdown()
    instruments._reset_instrumentation_state()


def _batch_processor(provider: TracerProvider) -> BatchSpanProcessor:
    processors = provider._active_span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], BatchSpanProcessor)
    return processors[0]


def _exporter(provider: TracerProvider) -> Any:
    wrapper = _batch_processor(provider).span_exporter
    assert isinstance(wrapper, SpanExporter)
    assert type(wrapper).__name__ == "EgressLoggingSpanExporter"
    return wrapper.inner


# ── no endpoint: nothing ────────────────────────────────────────────────────────


def test_no_endpoint_configures_no_exporter(
    clean_otel_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[Any] = []
    monkeypatch.setattr(http_trace_exporter, "OTLPSpanExporter", lambda **kw: built.append(kw))

    assert load_tracing_config().endpoint is None
    state = initialize_tracing()

    assert state.enabled is False
    assert state.tracer is None
    assert state.tracer_provider is None
    assert state.error is None
    assert built == []
    assert "global" not in clean_otel_env


@pytest.mark.parametrize("env", [{"OTEL_SDK_DISABLED": "true"}, {"OTEL_TRACES_EXPORTER": "none"}])
def test_the_standard_off_switches_win_over_an_endpoint(env: dict[str, str]) -> None:
    cfg = load_tracing_config({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318", **env})
    assert cfg.endpoint is None
    assert initialize_tracing(cfg).enabled is False


def test_healthz_without_an_endpoint_reports_no_observability_error() -> None:
    with TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    ) as client:
        health = client.get("/healthz").json()

    assert health["tracing_enabled"] is False
    assert health["otlp_endpoint"] is None
    assert health["otel_targets"] == []
    assert health["observability_error"] is None
    assert not any("phoenix" in key for key in health)


# ── endpoint resolution: the SDK's own rules ──────────────────────────────────


def test_endpoint_resolution_follows_the_sdk() -> None:
    base = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318/"}
    assert load_tracing_config(base).endpoint == "http://collector:4318/v1/traces"
    assert (
        load_tracing_config(
            {**base, "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://tempo:4318/otlp/v1/traces"}
        ).endpoint
        == "https://tempo:4318/otlp/v1/traces"
    )
    grpc = load_tracing_config(
        {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://jaeger:4317", "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc"}
    )
    assert (grpc.endpoint, grpc.protocol) == ("http://jaeger:4317", "grpc")


def test_redacted_endpoint_drops_credentials_and_query() -> None:
    assert (
        redact_endpoint("https://user:pw@api.example.test:443/v1/traces?key=s3cret")
        == "https://api.example.test:443/v1/traces"
    )


# ── an endpoint: a batched OTLP exporter under the service name ─────────────────


def test_endpoint_builds_a_batched_otlp_exporter_with_the_service_name(
    clean_otel_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=abc123")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "iris-test")

    state = initialize_tracing()

    assert state.enabled is True
    assert state.error is None
    assert state.endpoint == "http://127.0.0.1:4318/v1/traces"
    assert state.service_name == "iris-test"
    provider = state.tracer_provider
    assert isinstance(provider, TracerProvider)
    assert clean_otel_env["global"] is provider
    assert provider.resource.attributes["service.name"] == "iris-test"
    # A plain resource attribute Phoenix files projects under; harmless elsewhere.
    assert provider.resource.attributes["openinference.project.name"] == "iris-test"
    exporter = _exporter(provider)
    assert isinstance(exporter, OTLPSpanExporter)
    # The exporter read the standard variables itself.
    assert exporter._endpoint == "http://127.0.0.1:4318/v1/traces"
    assert exporter._headers["x-api-key"] == "abc123"


def test_service_name_defaults_to_iris(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    state = initialize_tracing()
    assert state.service_name == "iris"
    assert state.tracer_provider.resource.attributes["service.name"] == "iris"


def test_resource_attributes_name_the_service_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=from-attrs,deployment=lab")
    state = initialize_tracing()
    attrs = state.tracer_provider.resource.attributes
    assert (attrs["service.name"], attrs["deployment"]) == ("from-attrs", "lab")


def test_grpc_protocol_builds_the_grpc_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    state = initialize_tracing()
    assert state.error is None
    assert isinstance(_exporter(state.tracer_provider), GrpcSpanExporter)


def test_an_unknown_protocol_is_reported_not_raised() -> None:
    state = initialize_tracing(
        TracingConfig(endpoint="http://127.0.0.1:4318/v1/traces", protocol="http/json")
    )
    assert state.enabled is True
    assert state.tracer is None
    assert state.error is not None and "http/json" in state.error


def test_spans_export_in_batches_and_each_batch_is_an_egress_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    memory = InMemorySpanExporter()
    monkeypatch.setattr(otlp_setup, "_build_exporter", lambda cfg: memory)
    state = initialize_tracing(TracingConfig(endpoint="http://u:p@127.0.0.1:4318/v1/traces"))

    with caplog.at_level(logging.INFO, logger="iris.egress"):
        with state.tracer.start_as_current_span("turn"):
            pass
        state.tracer_provider.force_flush()

    assert [s.name for s in memory.get_finished_spans()] == ["turn"]
    lines = [r.getMessage() for r in caplog.records if r.name == "iris.egress"]
    assert any("-> http://127.0.0.1:4318/v1/traces" in line for line in lines)
    assert not any("u:p@" in line for line in lines)


def test_healthz_with_an_endpoint_reports_it_and_no_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    with TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    ) as client:
        health = client.get("/healthz").json()

    assert health["tracing_enabled"] is True
    assert health["otlp_endpoint"] == "http://127.0.0.1:4318/v1/traces"
    assert health["observability_error"] is None


# ── httpx instrumentation ───────────────────────────────────────────────────────


def test_httpx_instrumentation_is_wired_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wiring, independent of the package: the instrumentor gets our provider."""
    calls: list[Any] = []

    class _HTTPXClientInstrumentor:
        def instrument(self, *, tracer_provider: Any) -> None:
            calls.append(tracer_provider)

    module = types.ModuleType("opentelemetry.instrumentation.httpx")
    module.HTTPXClientInstrumentor = _HTTPXClientInstrumentor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation.httpx", module)
    monkeypatch.setenv("IRIS_OTEL_HTTPX_ENABLED", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")

    state = initialize_tracing()

    assert state.instrumented_targets == ("httpx",)
    assert calls == [state.tracer_provider]
    assert state.error is None


def test_httpx_instrumentation_off_leaves_httpx_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    state = initialize_tracing()  # the fixture sets IRIS_OTEL_HTTPX_ENABLED=0
    assert "httpx" not in state.instrumented_targets


def test_the_real_httpx_instrumentor_is_installed_and_instruments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    httpx_instrumentation = pytest.importorskip(
        "opentelemetry.instrumentation.httpx",
        reason=(
            "opentelemetry-instrumentation-httpx is a core dependency in pyproject.toml but "
            "not installed until the owner re-locks and installs (`poetry lock && poetry "
            "install`)"
        ),
    )
    monkeypatch.setenv("IRIS_OTEL_HTTPX_ENABLED", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    instrumentor = httpx_instrumentation.HTTPXClientInstrumentor()
    try:
        state = initialize_tracing()
        assert "httpx" in state.instrumented_targets
        assert state.error is None
        assert instrumentor.is_instrumented_by_opentelemetry
    finally:
        instrumentor.uninstrument()
