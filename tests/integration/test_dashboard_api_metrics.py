"""Integration tests for the observability metrics API surface."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.foundation.observability import session_log
from iris_harness.server.iris_api.main import create_app

pytestmark = pytest.mark.integration


def _write_log(path, *events: dict[str, object]) -> None:
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def test_llm_metrics_returns_summary_and_backend_status(tmp_path, monkeypatch) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_log(
        log_dir / "session-s1.jsonl",
        {
            "kind": "llm_call",
            "provider": "openrouter",
            "model": "gpt-test",
            "tokens": {"total_tokens": 42},
            "duration_ms": 12.5,
        },
        {
            "kind": "error",
            "phase": "llm_call",
            "message": "timeout",
        },
    )

    monkeypatch.setenv("IRIS_OBSERVABILITY_METRICS_ENABLED", "1")
    monkeypatch.setattr(session_log, "LOG_DIR", log_dir)
    monkeypatch.setattr(
        "iris_harness.foundation.observability.tracer.setup_tracing_state",
        lambda **_: SimpleNamespace(
            enabled=True,
            tracer=object(),
            endpoint="http://localhost:4318/v1/traces",
            instrumented_targets=("langchain", "httpx"),
            error=None,
        ),
    )

    with TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    ) as client:
        resp = client.get("/observability/llm-metrics")

    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"]["healthy"] is True
    assert body["backend"]["endpoint"] == "http://localhost:4318/v1/traces"
    assert body["backend"]["kind"] == "otlp"
    assert body["summary"]["llm_call_count"] == 1
    assert body["summary"]["llm_error_count"] == 1
    assert body["summary"]["total_tokens"] == 42
    assert body["summary"]["provider_counts"] == {"openrouter": 1}


def test_llm_metrics_handles_unavailable_backend_gracefully(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IRIS_OBSERVABILITY_METRICS_ENABLED", "1")
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path / "missing-logs")
    monkeypatch.setattr(
        "iris_harness.foundation.observability.tracer.setup_tracing_state",
        lambda **_: SimpleNamespace(
            enabled=True,
            tracer=None,
            endpoint=None,
            instrumented_targets=(),
            error="otlp exporter unavailable",
        ),
    )

    with TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    ) as client:
        resp = client.get("/observability/llm-metrics")

    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"]["healthy"] is False
    assert body["backend"]["error"] == "otlp exporter unavailable"
    assert body["summary"]["llm_call_count"] == 0


def test_llm_metrics_respects_env_gate(monkeypatch) -> None:
    monkeypatch.delenv("IRIS_OBSERVABILITY_METRICS_ENABLED", raising=False)
    with TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    ) as client:
        resp = client.get("/observability/llm-metrics")

    assert resp.status_code == 404
