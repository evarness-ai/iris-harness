"""The harness without the `phoenix` extra (OSS plan R19).

arize-phoenix is Elastic-2.0, so the core install leaves it out. These tests make the
`phoenix` import root unimportable, the way a core-only install sees it, and assert
that tracing still exports over OTLP, embedded mode names the extra instead of
raising, and a runtime builds and answers a governed turn with tracing on.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http import trace_exporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from iris_harness.foundation.auth import auth_headers
from iris_harness.foundation.observability.instruments import _reset_instrumentation_state
from iris_harness.foundation.observability.phoenix_setup import (
    PhoenixSetupConfig,
    initialize_phoenix,
)
from iris_harness.runtime import build_runtime
from iris_harness.server.iris_api.main import create_app

OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"


class _NoPhoenixFinder(MetaPathFinder):
    """Raise ModuleNotFoundError for `phoenix` and every submodule of it."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> ModuleSpec | None:
        if fullname.split(".", 1)[0] == "phoenix":
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@pytest.fixture
def no_phoenix(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name.split(".", 1)[0] == "phoenix":
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_NoPhoenixFinder(), *sys.meta_path])
    with pytest.raises(ImportError):
        import phoenix.otel  # noqa: F401
    _reset_instrumentation_state()


@pytest.fixture
def otlp_capture(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Swap the OTLP exporter for an in-memory one; record the endpoint and the
    provider registered globally (the real global is set-once per process)."""
    seen: dict[str, Any] = {"exporter": InMemorySpanExporter()}

    def _exporter(*, endpoint: str) -> InMemorySpanExporter:
        seen["endpoint"] = endpoint
        return seen["exporter"]

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", _exporter)
    monkeypatch.setattr(trace, "set_tracer_provider", lambda p: seen.__setitem__("global", p))
    yield seen
    provider = seen.get("global")
    if provider is not None:
        provider.shutdown()


def test_external_mode_exports_over_plain_otel(no_phoenix: None, otlp_capture: dict) -> None:
    result = initialize_phoenix(
        PhoenixSetupConfig(
            enabled=True,
            mode="external",
            project_name="iris",
            endpoint=OTLP_ENDPOINT,
            endpoint_base="http://127.0.0.1:4318",
            enable_langchain=False,
            enable_httpx=False,
        )
    )

    assert result.error is None
    assert result.tracer is not None
    provider = result.tracer_provider
    assert isinstance(provider, TracerProvider)
    # What phoenix.otel.register did for us: the global provider and the project.
    assert otlp_capture["global"] is provider
    assert otlp_capture["endpoint"] == OTLP_ENDPOINT
    assert provider.resource.attributes["openinference.project.name"] == "iris"
    assert provider.resource.attributes["service.name"] == "iris"

    with result.tracer.start_as_current_span("turn"):
        pass
    provider.force_flush()
    assert [s.name for s in otlp_capture["exporter"].get_finished_spans()] == ["turn"]


def test_embedded_mode_names_the_extra_instead_of_raising(no_phoenix: None) -> None:
    result = initialize_phoenix(
        PhoenixSetupConfig(enabled=True, mode="embedded", launch_sleep_seconds=0.0)
    )

    assert result.enabled is True
    assert result.tracer is None
    assert result.error is not None
    assert "iris-harness[phoenix]" in result.error
    assert "IRIS_PHOENIX_MODE=external" in result.error


@pytest.mark.usefixtures("offline_llm")
def test_runtime_builds_and_answers_a_governed_turn_without_phoenix(
    no_phoenix: None,
    otlp_capture: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_PHOENIX_ENABLED", "1")
    monkeypatch.setenv("IRIS_PHOENIX_MODE", "external")
    monkeypatch.setenv("IRIS_PHOENIX_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.setenv("IRIS_OTEL_LANGCHAIN_ENABLED", "0")
    monkeypatch.setenv("IRIS_OTEL_HTTPX_ENABLED", "0")
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()

    runtime = build_runtime(
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        use_background_scheduler=False,
    )
    with TestClient(create_app(runtime=runtime), headers=auth_headers()) as client:
        health = client.get("/healthz").json()
        resp = client.post("/chat", json={"message": "what time is it?"})

    assert health["runtime_ready"] is True
    assert health["phoenix_enabled"] is True
    assert health["observability_error"] is None
    assert otlp_capture["endpoint"] == OTLP_ENDPOINT
    assert isinstance(otlp_capture["global"], TracerProvider)
    assert "phoenix" not in sys.modules

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_errors"] is False
    assert "Current local time:" in body["response"]
