"""GET /runtime/inventory (#5) and GET /learning/experiments (#4) — read-only
surfaces. A dummy runtime avoids building the real one."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.learning.store import LearningMetricsStore


def _client(runtime: object) -> TestClient:
    return TestClient(create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers())


@pytest.mark.usefixtures("offline_ollama_inventory")  # model list would ask the live Ollama
def test_runtime_inventory_endpoint_reports_versions() -> None:
    with _client(SimpleNamespace()) as c:
        body = c.get("/runtime/inventory").json()
    assert body["iris_version"]
    assert "fastapi" in body["packages"]
    assert isinstance(body["ollama_models"], list)
    assert "python_version" in body


def test_learning_experiments_empty_without_store() -> None:
    with _client(SimpleNamespace()) as c:  # no learning_store attr
        body = c.get("/learning/experiments").json()
    assert body == {"experiments": [], "status_counts": {}}


def test_learning_experiments_serializes_records() -> None:
    exp = SimpleNamespace(
        id="e1",
        domain="routing",
        hypothesis="tier-2 start for code",
        variant_description="start one tier higher for code intents",
        baseline_metric=0.42,
        current_metric=0.55,
        status="kept",
        created_at=datetime(2026, 6, 20, tzinfo=timezone.utc),
        started_at=None,
        evaluated_at=None,
        evaluation_window_hours=24,
    )
    store = SimpleNamespace(
        list_experiments=lambda: [exp],
        experiment_status_counts=lambda: {"kept": 1},
    )
    with _client(SimpleNamespace(learning_store=store)) as c:
        body = c.get("/learning/experiments").json()

    assert body["status_counts"] == {"kept": 1}
    row = body["experiments"][0]
    assert row["hypothesis"] == "tier-2 start for code"
    assert row["baseline_metric"] == 0.42
    assert row["current_metric"] == 0.55
    assert row["created_at"].startswith("2026-06-20")


def test_learning_intelligence_unavailable_without_store() -> None:
    with _client(SimpleNamespace()) as c:  # no learning_store attr
        body = c.get("/learning/intelligence").json()
    assert body == {"available": False}


def test_learning_intelligence_reports_measured_matrix(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    store.record_signal(
        source="chat",
        metric_name="task_completed",
        value=1.0,
        success=True,
        metadata={"intent": "email"},
        resolved_tier="tier1",
    )
    with _client(SimpleNamespace(learning_store=store)) as c:
        body = c.get("/learning/intelligence").json()

    assert body["available"] is True
    assert "accuracy" in body and "escalation_precision" in body["accuracy"]
    assert body["matrix"][0]["intent"] == "email"
    assert body["matrix"][0]["tier"] == "tier1"


def test_learning_analysis_unavailable_until_analyst_runs(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    with _client(SimpleNamespace(learning_store=store)) as c:
        body = c.get("/learning/analysis").json()
    assert body == {"available": False}


def test_learning_analysis_returns_persisted_run(tmp_path: Path) -> None:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    store.save_analysis(
        {
            "generated_at": "2026-06-20T12:00:00+00:00",
            "model": "local-x",
            "summary": "email on tier1 is weak.",
            "recommendations": [
                {
                    "title": "Route email up",
                    "finding": "60% done",
                    "action": "tier2",
                    "evidence": ["email@tier1"],
                    "confidence": "high",
                }
            ],
        }
    )
    with _client(SimpleNamespace(learning_store=store)) as c:
        body = c.get("/learning/analysis").json()
    assert body["available"] is True
    assert body["summary"].startswith("email")
    assert body["recommendations"][0]["title"] == "Route email up"


def _seed_analysis_with_target(store: LearningMetricsStore) -> None:
    store.record_signal(
        source="chat",
        metric_name="task_completed",
        value=1.0,
        success=True,
        metadata={"intent": "email"},
        resolved_tier="tier1",
    )
    store.save_analysis(
        {
            "generated_at": "2026-06-20T12:00:00+00:00",
            "model": "m",
            "summary": "s",
            "recommendations": [
                {
                    "title": "Route email up",
                    "finding": "weak tier1",
                    "action": "tier2",
                    "evidence": ["email@tier1"],
                    "confidence": "high",
                    "target": {"metric": "completion_rate", "intent": "email", "tier": "tier1"},
                }
            ],
        }
    )


def test_promote_endpoint_creates_experiment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")  # write-gated route
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    _seed_analysis_with_target(store)
    with _client(SimpleNamespace(learning_store=store)) as c:
        resp = c.post("/learning/recommendations/1/promote")
    assert resp.status_code == 200
    body = resp.json()
    assert body["measurable"] is True
    assert body["experiment_id"]
    # The experiment is now in the ledger.
    assert len(store.list_experiments()) == 1


def test_promote_endpoint_gated_when_writes_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    _seed_analysis_with_target(store)
    with _client(SimpleNamespace(learning_store=store)) as c:
        resp = c.post("/learning/recommendations/1/promote")
    assert resp.status_code == 403
    assert store.list_experiments() == []


def test_promote_endpoint_404_for_missing_recommendation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    with _client(SimpleNamespace(learning_store=store)) as c:
        resp = c.post("/learning/recommendations/9/promote")
    assert resp.status_code == 404
