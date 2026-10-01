"""Trace export over standard OTLP to the backend the user configures.

IRIS bundles no trace UI (ADR-0128). OpenTelemetry is core; spans go over OTLP to
whatever backend the operator points the standard OpenTelemetry variables at: Phoenix,
Jaeger, Grafana Tempo, Honeycomb, an OpenTelemetry Collector, anything that speaks OTLP.

Configuration is the OpenTelemetry SDK's own, read the way the SDK reads it:

- ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` (full URL) or ``OTEL_EXPORTER_OTLP_ENDPOINT``
  (base URL; ``/v1/traces`` is appended for OTLP/HTTP). With neither set nothing is
  exported: no provider, no exporter, no error.
- ``OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`` / ``OTEL_EXPORTER_OTLP_PROTOCOL``:
  ``http/protobuf`` (default) or ``grpc``.
- ``OTEL_EXPORTER_OTLP_HEADERS`` / ``..._TRACES_HEADERS``, timeouts, compression and TLS
  files: read by the exporter itself, so every standard knob works unchanged.
- ``OTEL_SERVICE_NAME`` / ``OTEL_RESOURCE_ATTRIBUTES``: read by ``Resource.create``.
  IRIS fills ``service.name=iris`` only when neither names a service.
- ``OTEL_SDK_DISABLED=true`` or ``OTEL_TRACES_EXPORTER=none`` turns export off even
  with an endpoint set.

IRIS adds two switches of its own, because the standard variables have no equivalent
outside the zero-code agent: ``IRIS_OTEL_LANGCHAIN_ENABLED`` and ``IRIS_OTEL_HTTPX_ENABLED``
(both default on) choose which libraries are auto-instrumented.

Export is batched (``BatchSpanProcessor``), and every batch sent is one ``iris.egress``
line (host only), like any other outbound call.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from iris_harness.foundation.env import env_flag

from .instruments import instrument_runtime
from .logging_setup import log_egress

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "iris"
HTTP_PROTOBUF = "http/protobuf"
GRPC = "grpc"
# OpenInference's resource key for the project. A plain OTel resource attribute: Phoenix
# files traces under it, every other backend ignores it. Set to the service name.
_PROJECT_RESOURCE_KEY = "openinference.project.name"
_TRACER_NAME = "iris.runtime"


@dataclass(frozen=True)
class TracingConfig:
    """What the standard OpenTelemetry variables ask for, resolved once."""

    # The traces endpoint spans go to; ``None`` means nothing is exported.
    endpoint: str | None = None
    protocol: str = HTTP_PROTOBUF
    enable_langchain: bool = True
    enable_httpx: bool = True


@dataclass(frozen=True)
class TracingState:
    """Outcome of the tracing bootstrap (what ``/healthz`` and the metrics route show)."""

    enabled: bool
    tracer: Any = None
    tracer_provider: Any = None
    # Where spans go, credentials stripped (safe to show on /healthz).
    endpoint: str | None = None
    protocol: str | None = None
    service_name: str | None = None
    instrumented_targets: tuple[str, ...] = ()
    error: str | None = None


def _get(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


def _append_trace_path(base: str) -> str:
    return base if base.endswith("/v1/traces") else f"{base.rstrip('/')}/v1/traces"


def load_tracing_config(env: Mapping[str, str] | None = None) -> TracingConfig:
    """Resolve the standard OTLP variables (and IRIS's two instrumentation switches)."""
    env = os.environ if env is None else env
    protocol = (
        _get(env, "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
        or _get(env, "OTEL_EXPORTER_OTLP_PROTOCOL")
        or HTTP_PROTOBUF
    ).lower()
    enable_langchain = env_flag("IRIS_OTEL_LANGCHAIN_ENABLED", default=True)
    enable_httpx = env_flag("IRIS_OTEL_HTTPX_ENABLED", default=True)

    sdk_disabled = _get(env, "OTEL_SDK_DISABLED").lower() == "true"
    exporter_none = _get(env, "OTEL_TRACES_EXPORTER").lower() == "none"
    traces_endpoint = _get(env, "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    base_endpoint = _get(env, "OTEL_EXPORTER_OTLP_ENDPOINT")

    endpoint: str | None = None
    if not (sdk_disabled or exporter_none):
        if traces_endpoint:
            endpoint = traces_endpoint
        elif base_endpoint:
            # The OTLP/HTTP exporter appends the signal path to the base; gRPC does not.
            endpoint = base_endpoint if protocol == GRPC else _append_trace_path(base_endpoint)
    return TracingConfig(
        endpoint=endpoint,
        protocol=protocol,
        enable_langchain=enable_langchain,
        enable_httpx=enable_httpx,
    )


def redact_endpoint(endpoint: str | None) -> str | None:
    """The endpoint without userinfo, query or fragment: safe to log and to show."""
    if not endpoint:
        return None
    try:
        parts = urlsplit(endpoint)
        netloc = parts.hostname or ""
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except ValueError:
        return "<unparseable endpoint>"


def initialize_tracing(config: TracingConfig | None = None) -> TracingState:
    """Build the OTLP tracer provider when an endpoint is configured. Never raises."""
    cfg = config or load_tracing_config()
    if not cfg.endpoint:
        logger.info("no OTLP endpoint configured (OTEL_EXPORTER_OTLP_ENDPOINT); traces stay local")
        return TracingState(enabled=False)

    shown = redact_endpoint(cfg.endpoint)
    try:
        exporter = _build_exporter(cfg)
    except Exception as exc:  # noqa: BLE001 - a bad exporter config must not stop the server
        logger.warning("OTLP span exporter unavailable: %s", exc)
        return TracingState(enabled=True, endpoint=shown, protocol=cfg.protocol, error=str(exc))

    try:
        provider = _build_tracer_provider(exporter, destination=shown or "")
    except Exception as exc:
        logger.exception("OpenTelemetry tracer provider setup failed")
        return TracingState(enabled=True, endpoint=shown, protocol=cfg.protocol, error=str(exc))

    instrumentation = instrument_runtime(
        provider,
        enable_langchain=cfg.enable_langchain,
        enable_httpx=cfg.enable_httpx,
    )
    service_name = str(provider.resource.attributes.get("service.name", DEFAULT_SERVICE_NAME))
    logger.info(
        "exporting traces over OTLP (%s) to %s as service %r", cfg.protocol, shown, service_name
    )
    return TracingState(
        enabled=True,
        tracer=provider.get_tracer(_TRACER_NAME),
        tracer_provider=provider,
        endpoint=shown,
        protocol=cfg.protocol,
        service_name=service_name,
        instrumented_targets=instrumentation.instrumented_targets,
        error=instrumentation.error,
    )


def _build_exporter(cfg: TracingConfig) -> Any:
    """The OTLP span exporter for the configured protocol.

    Built with no arguments, so the exporter reads the endpoint, headers, timeout,
    compression and TLS settings from the standard variables itself.
    """
    if cfg.protocol == GRPC:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcSpanExporter,
        )

        return GrpcSpanExporter()
    if cfg.protocol != HTTP_PROTOBUF:
        raise ValueError(
            f"unsupported OTLP protocol {cfg.protocol!r}: use {HTTP_PROTOBUF!r} or {GRPC!r}"
        )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter()


def _build_tracer_provider(exporter: Any, *, destination: str) -> Any:
    """A batched, globally registered SDK tracer provider around *exporter*.

    Registered globally because the LLM-invoke spans (``llm/client.py``) and the audit
    trace ids (``kernel/governance/kernel.py``) read it through ``trace.get_tracer``.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create()
    service_name = str(resource.attributes.get(SERVICE_NAME, ""))
    if not service_name or service_name.startswith("unknown_service"):
        service_name = DEFAULT_SERVICE_NAME
        resource = resource.merge(Resource({SERVICE_NAME: service_name}))
    if _PROJECT_RESOURCE_KEY not in resource.attributes:
        resource = resource.merge(Resource({_PROJECT_RESOURCE_KEY: service_name}))

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(egress_logging_exporter(exporter, destination=destination))
    )
    trace.set_tracer_provider(provider)
    return provider


def egress_logging_exporter(inner: Any, *, destination: str) -> Any:
    """Wrap *inner* so every batch it sends is one ``iris.egress`` line (host only).

    A real ``SpanExporter`` subclass, defined on first use so importing this module
    (which every observability import does) does not pull in the SDK.
    """
    return _egress_exporter_class()(inner, destination=destination)


_EGRESS_EXPORTER_CLASS: Any = None


def _egress_exporter_class() -> Any:
    global _EGRESS_EXPORTER_CLASS
    if _EGRESS_EXPORTER_CLASS is not None:
        return _EGRESS_EXPORTER_CLASS
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class EgressLoggingSpanExporter(SpanExporter):
        def __init__(self, inner: Any, *, destination: str) -> None:
            self.inner = inner
            self._destination = destination

        def export(self, spans: Sequence[Any]) -> SpanExportResult:
            result = self.inner.export(spans)
            log_egress(
                kind="telemetry",
                destination=self._destination,
                method="POST",
                purpose="otlp-traces",
                status=getattr(result, "name", result),
                spans=len(spans),
            )
            return result  # type: ignore[no-any-return]

        def shutdown(self) -> None:
            self.inner.shutdown()

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return bool(self.inner.force_flush(timeout_millis))

    _EGRESS_EXPORTER_CLASS = EgressLoggingSpanExporter
    return _EGRESS_EXPORTER_CLASS
