"""Phoenix bootstrap helpers for local/dev observability.

OpenTelemetry is core; Arize Phoenix is the optional ``phoenix`` extra (Elastic-2.0,
so it stays out of the OSI-only core install, OSS plan R19). Every Phoenix import
here is lazy:

- external mode never needs Phoenix: spans export over OTLP to ``IRIS_PHOENIX_ENDPOINT``,
  which can be a Phoenix server or any other OTLP backend;
- embedded mode launches the Phoenix app in-process, so without the extra it returns an
  error result that names the extra instead of raising;
- ``phoenix.otel.register`` is used when present, else the plain OpenTelemetry SDK builds
  an equivalent provider (project resource, global registration).
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from iris_harness.foundation.env import env_flag

from .instruments import instrument_runtime

logger = logging.getLogger(__name__)

# The one remedy every Phoenix-only feature names when the extra is absent.
PHOENIX_EXTRA_HINT = "install iris-harness[phoenix] (`poetry install -E phoenix`)"
# OpenInference's resource key for the project; Phoenix groups traces by it. Spelled
# out so the no-Phoenix path needs no openinference import.
_PROJECT_RESOURCE_KEY = "openinference.project.name"


def _env_flag(name: str, *, default: bool) -> bool:
    """Thin alias for the shared reader, keeping this module's semantics.

    One of six copies M6.3 found in three disagreeing variants; see
    ``iris_harness.foundation.env`` for what they disagreed about.
    """
    return env_flag(name, default=default)


def _env_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid integer for %s=%r; using %s", name, raw, default)
        return default


@dataclass(frozen=True)
class PhoenixSetupConfig:
    """Runtime Phoenix/OTEL configuration."""

    enabled: bool = False
    mode: str = "embedded"
    project_name: str = "iris"
    port: int = 6006
    # Full OTLP traces endpoint (…/v1/traces). When None it is derived from the
    # port. Used by the span exporter in every mode.
    endpoint: str | None = None
    # Base URL of an external Phoenix UI (external mode). When None it is derived
    # from the port. Purely informational (shown in the console / health).
    endpoint_base: str | None = None
    launch_sleep_seconds: float = 1.0
    enable_langchain: bool = True
    enable_httpx: bool = True
    # When set, embedded Phoenix persists its trace store here (SQLite) instead
    # of keeping spans in memory, so traces survive a restart and can be
    # extracted later for debugging. ``None`` keeps Phoenix's own default.
    working_dir: str | None = None


@dataclass(frozen=True)
class PhoenixSetupResult:
    """Outcome of a Phoenix bootstrap attempt."""

    enabled: bool
    tracer: Any = None
    tracer_provider: Any = None
    phoenix_url: str | None = None
    mode: str = "disabled"
    instrumented_targets: tuple[str, ...] = ()
    error: str | None = None
    working_dir: str | None = None


def load_phoenix_setup_config() -> PhoenixSetupConfig:
    """Load Phoenix settings from environment variables."""
    enabled = _env_flag("IRIS_PHOENIX_ENABLED", default=False)
    mode = os.getenv("IRIS_PHOENIX_MODE", "embedded").strip().lower() or "embedded"
    port = _env_int("IRIS_PHOENIX_PORT", default=6006)
    # IRIS_PHOENIX_ENDPOINT may be a UI base (http://host:6006) or a full traces
    # endpoint (…/v1/traces). Normalize: derive the OTLP endpoint from the base
    # and keep the base for display.
    endpoint_base = os.getenv("IRIS_PHOENIX_ENDPOINT", "").strip() or None
    endpoint: str | None = None
    if endpoint_base:
        endpoint = (
            endpoint_base
            if endpoint_base.endswith("/v1/traces")
            else f"{endpoint_base.rstrip('/')}/v1/traces"
        )
    project_name = os.getenv("IRIS_PHOENIX_PROJECT", "iris").strip() or "iris"
    launch_sleep_ms = _env_int("IRIS_PHOENIX_LAUNCH_SLEEP_MS", default=1000)
    enable_langchain = _env_flag("IRIS_OTEL_LANGCHAIN_ENABLED", default=True)
    enable_httpx = _env_flag("IRIS_OTEL_HTTPX_ENABLED", default=True)
    working_dir_raw = os.getenv("IRIS_PHOENIX_WORKING_DIR")
    working_dir: str | None
    if working_dir_raw is None:
        # Persist by default so an enabled Phoenix keeps traces across restarts
        # (the whole point of "log it so we can debug later"). Opt out with "".
        working_dir = str(Path.home() / ".iris" / "phoenix")
    else:
        working_dir = working_dir_raw.strip() or None
    return PhoenixSetupConfig(
        enabled=enabled,
        mode=mode,
        project_name=project_name,
        port=port,
        endpoint=endpoint,
        endpoint_base=endpoint_base,
        launch_sleep_seconds=max(0.0, launch_sleep_ms / 1000.0),
        enable_langchain=enable_langchain,
        enable_httpx=enable_httpx,
        working_dir=working_dir,
    )


def initialize_phoenix(
    config: PhoenixSetupConfig | None = None,
) -> PhoenixSetupResult:
    """Best-effort Phoenix bootstrap that never raises to callers."""
    cfg = config or load_phoenix_setup_config()
    if not cfg.enabled:
        logger.info("Phoenix tracing disabled via IRIS_PHOENIX_ENABLED")
        return PhoenixSetupResult(enabled=False)
    if cfg.mode not in {"embedded", "external"}:
        message = f"unsupported Phoenix mode: {cfg.mode}"
        logger.warning(message)
        return PhoenixSetupResult(enabled=True, mode=cfg.mode, error=message)

    phoenix_url = cfg.endpoint_base or f"http://127.0.0.1:{cfg.port}"

    # External mode: Phoenix runs as its own process (managed by start_iris.sh);
    # this process only exports spans to it over OTLP. This keeps the Phoenix
    # server (its own web app + growing SQLite trace store) OUT of the iris_api
    # process — the single biggest driver of the harness's runtime footprint.
    if cfg.mode == "external":
        logger.info("Phoenix tracing in external mode; exporting spans to %s", phoenix_url)
        return _wire_tracing(cfg, phoenix_url)

    # Embedded mode (legacy default): launch the full Phoenix app in-process. The
    # app is the `phoenix` extra; `from phoenix import launch_app` (not `import
    # phoenix`) because arize-phoenix-otel alone leaves a bare `phoenix` namespace.
    try:
        from phoenix import launch_app
    except ImportError:
        message = (
            "embedded Phoenix mode needs the Phoenix app: "
            f"{PHOENIX_EXTRA_HINT}, or set IRIS_PHOENIX_MODE=external and "
            "IRIS_PHOENIX_ENDPOINT to any OTLP backend"
        )
        logger.warning(message)
        return PhoenixSetupResult(enabled=True, mode=cfg.mode, error=message)

    os.environ.setdefault("PHOENIX_PORT", str(cfg.port))
    # Bind the embedded UI to loopback (single-user, local-first) unless the
    # operator has overridden it. Phoenix reads PHOENIX_HOST at launch.
    os.environ.setdefault("PHOENIX_HOST", "127.0.0.1")
    if cfg.working_dir:
        try:
            Path(cfg.working_dir).expanduser().mkdir(parents=True, exist_ok=True)
            # Phoenix reads PHOENIX_WORKING_DIR for its on-disk (SQLite) trace
            # store; setting it before launch makes traces durable + extractable.
            # SQLite persistence also caps in-RAM span growth, so this must win
            # over any inherited value — set it, don't setdefault it.
            os.environ["PHOENIX_WORKING_DIR"] = str(Path(cfg.working_dir).expanduser())
        except OSError as exc:
            logger.warning("Phoenix working dir %s unusable: %s", cfg.working_dir, exc)

    _silence_phoenix_noise()

    try:
        session = launch_app()
        time.sleep(cfg.launch_sleep_seconds)
        phoenix_url = str(getattr(session, "url", phoenix_url))
        logger.info("Arize Phoenix started at %s", phoenix_url)
    except Exception as exc:
        logger.exception("Phoenix failed to start; continuing without tracing")
        return PhoenixSetupResult(
            enabled=True,
            mode=cfg.mode,
            phoenix_url=phoenix_url,
            error=str(exc),
        )

    return _wire_tracing(cfg, phoenix_url)


def _wire_tracing(cfg: PhoenixSetupConfig, phoenix_url: str) -> PhoenixSetupResult:
    """Build the OTLP tracer provider + instrument the runtime (mode-agnostic)."""
    tracer_provider = _build_tracer_provider(cfg)
    if tracer_provider is None:
        return PhoenixSetupResult(
            enabled=True,
            mode=cfg.mode,
            phoenix_url=phoenix_url,
            error="unable to configure tracer provider",
        )

    instrumentation = instrument_runtime(
        tracer_provider,
        enabled=True,
        enable_langchain=cfg.enable_langchain,
        enable_httpx=cfg.enable_httpx,
    )
    tracer = tracer_provider.get_tracer(f"{cfg.project_name}.runtime")
    return PhoenixSetupResult(
        enabled=True,
        tracer=tracer,
        tracer_provider=tracer_provider,
        phoenix_url=phoenix_url,
        mode=cfg.mode,
        instrumented_targets=instrumentation.instrumented_targets,
        error=instrumentation.error,
        working_dir=os.environ.get("PHOENIX_WORKING_DIR") or cfg.working_dir,
    )


def _silence_phoenix_noise() -> None:
    """Quiet Phoenix's harmless launch-time warnings so startup logs stay clean.

    Three known-benign lines come from arize-phoenix and its deps when the
    embedded app launches: authlib's deprecation of ``authlib.jose``, SQLAlchemy
    reflecting Phoenix's own expression-based indexes (SAWarning), and Phoenix's
    "install aioboto3 for Bedrock" notice. None affect IRIS; matched narrowly so
    we don't hide anything else."""
    warnings.filterwarnings("ignore", message=r".*authlib\.jose module is deprecated.*")
    warnings.filterwarnings(
        "ignore", message=r".*Skipped unsupported reflection of expression-based index.*"
    )
    # The aioboto3 notice is a logger.warning from phoenix.server.app, not a
    # Python warning — raise that logger's floor instead.
    logging.getLogger("phoenix.server.app").setLevel(logging.ERROR)


def config_with_port(config: PhoenixSetupConfig, port: int) -> PhoenixSetupConfig:
    """Return a copy of *config* with a different Phoenix port."""
    return replace(config, port=port)


def _build_tracer_provider(config: PhoenixSetupConfig) -> Any | None:
    endpoint = config.endpoint or f"http://localhost:{config.port}/v1/traces"
    try:
        from phoenix.otel import register
    except ImportError:
        logger.info(
            "Phoenix extra not installed; exporting spans to %s with the OpenTelemetry SDK",
            endpoint,
        )
        return _manual_otel_setup(endpoint, config.project_name)
    try:
        return register(
            project_name=config.project_name,
            endpoint=endpoint,
            verbose=False,
        )
    except Exception:  # noqa: BLE001
        logger.warning("phoenix.otel.register failed; falling back to manual OTel setup")
        return _manual_otel_setup(endpoint, config.project_name)


def _manual_otel_setup(endpoint: str, project_name: str) -> Any | None:
    """An OTLP/HTTP tracer provider built with the OpenTelemetry SDK alone.

    It does what ``phoenix.otel.register`` does for us: tags the resource with the
    project (so a Phoenix backend still files the traces under it) and registers the
    provider globally, which the LLM-invoke spans (``llm/client.py``) and the audit
    trace ids (``kernel/governance/kernel.py``) read through ``trace.get_tracer``.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {SERVICE_NAME: project_name, _PROJECT_RESOURCE_KEY: project_name}
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        return provider
    except Exception:
        logger.exception("manual OTel setup failed")
        return None
