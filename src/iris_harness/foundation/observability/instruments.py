"""Idempotent OpenTelemetry auto-instrumentation helpers."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from iris_harness.foundation.env import env_flag

logger = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()
_INITIALIZED = False
_INITIALIZED_TARGETS: tuple[str, ...] = ()
_LAST_ERROR: str | None = None


def _env_flag(name: str, *, default: bool) -> bool:
    """Thin alias for the shared reader, keeping this module's semantics.

    One of six copies M6.3 found in three disagreeing variants; see
    ``iris_harness.foundation.env`` for what they disagreed about.
    """
    return env_flag(name, default=default)


@dataclass(frozen=True)
class InstrumentationState:
    """Summary of the currently configured auto-instrumentation state."""

    enabled: bool
    instrumented_targets: tuple[str, ...] = ()
    error: str | None = None
    initialized_now: bool = False


def instrument_runtime(
    tracer_provider: Any | None,
    *,
    enabled: bool | None = None,
    enable_langchain: bool | None = None,
    enable_httpx: bool | None = None,
) -> InstrumentationState:
    """Best-effort OTEL instrumentation for LangChain + httpx.

    The function is intentionally idempotent because multiple FastAPI lifespans
    may bootstrap tracing within the same Python process during tests.
    """
    enabled_flag = _env_flag("IRIS_OTEL_ENABLED", default=True) if enabled is None else enabled
    if not enabled_flag or tracer_provider is None:
        return InstrumentationState(enabled=False)

    langchain_flag = (
        _env_flag("IRIS_OTEL_LANGCHAIN_ENABLED", default=True)
        if enable_langchain is None
        else enable_langchain
    )
    httpx_flag = (
        _env_flag("IRIS_OTEL_HTTPX_ENABLED", default=True) if enable_httpx is None else enable_httpx
    )

    global _INITIALIZED, _INITIALIZED_TARGETS, _LAST_ERROR
    with _INIT_LOCK:
        if _INITIALIZED:
            return InstrumentationState(
                enabled=True,
                instrumented_targets=_INITIALIZED_TARGETS,
                error=_LAST_ERROR,
                initialized_now=False,
            )

        targets: list[str] = []
        errors: list[str] = []

        if langchain_flag:
            try:
                from openinference.instrumentation.langchain import LangChainInstrumentor

                LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
                targets.append("langchain")
            except Exception as exc:  # noqa: BLE001
                logger.warning("LangChain auto-instrumentation unavailable: %s", exc)
                errors.append(f"langchain: {exc}")

        if httpx_flag:
            try:
                from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

                HTTPXClientInstrumentor().instrument(tracer_provider=tracer_provider)
                targets.append("httpx")
            except Exception as exc:  # noqa: BLE001
                logger.warning("httpx auto-instrumentation unavailable: %s", exc)
                errors.append(f"httpx: {exc}")

        _INITIALIZED = True
        _INITIALIZED_TARGETS = tuple(targets)
        _LAST_ERROR = "; ".join(errors) or None
        return InstrumentationState(
            enabled=True,
            instrumented_targets=_INITIALIZED_TARGETS,
            error=_LAST_ERROR,
            initialized_now=True,
        )


def _reset_instrumentation_state() -> None:
    """Reset module globals for tests."""
    global _INITIALIZED, _INITIALIZED_TARGETS, _LAST_ERROR
    with _INIT_LOCK:
        _INITIALIZED = False
        _INITIALIZED_TARGETS = ()
        _LAST_ERROR = None
