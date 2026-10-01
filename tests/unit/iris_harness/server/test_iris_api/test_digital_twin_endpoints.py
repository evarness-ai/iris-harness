"""Tests for the digital-twin review endpoints (behaviors / signals / intentions).

Three layers, all propose-only: GET lists are open; approve/reject/dismiss are gated
behind IRIS_WEBUI_ALLOW_WRITES. Approving closes the loop into durable identity layers.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.learning.store import LearningMetricsStore


def _store(tmp_path: Path) -> LearningMetricsStore:
    s = LearningMetricsStore(db_path=tmp_path / "learning.db")
    s.ensure_schema()
    return s


def _client(store: LearningMetricsStore | None) -> TestClient:
    rt = SimpleNamespace(learning_store=store) if store is not None else SimpleNamespace()
    return TestClient(create_app(runtime=rt, auto_start_runtime=False), headers=auth_headers())


# --- reads degrade gracefully without a store ---


def test_reads_safe_without_store() -> None:
    with _client(None) as c:
        assert c.get("/learning/behaviors").json() == {
            "status": "pending",
            "count": 0,
            "behaviors": [],
        }
        assert c.get("/learning/signals").json() == {"count": 0, "signals": [], "summary": {}}
        assert c.get("/learning/intentions").json()["intentions"] == []


# --- layer 1: behaviors ---


def test_behaviors_list_approve_reject(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.propose_behavior_pattern("p1", "Checks inbox every morning", "high", ["mon", "tue"])
    s.propose_behavior_pattern("p2", "Asks about weather often", "medium", [])
    with _client(s) as c:
        body = c.get("/learning/behaviors").json()
        assert body["count"] == 2
        assert body["behaviors"][0]["evidence"] == ["mon", "tue"] or body["behaviors"][1][
            "evidence"
        ] == ["mon", "tue"]

        r = c.post("/learning/behaviors/p1/approve")
        assert r.status_code == 200 and r.json()["status"] == "approved"
        r = c.post("/learning/behaviors/p2/reject")
        assert r.status_code == 200 and r.json()["status"] == "rejected"

    assert s.list_behavior_proposals(status="pending") == []
    assert s.get_behavior_proposal("p1").status == "approved"
    # approve/reject both record a steering signal (layer-2 feedback)
    kinds = {x.kind for x in s.list_user_behavior_signals()}
    assert {"pattern_confirmed", "pattern_dismissed"} <= kinds


def test_behaviors_approve_missing_404(tmp_path: Path) -> None:
    with _client(_store(tmp_path)) as c:
        assert c.post("/learning/behaviors/nope/approve").status_code == 404


# --- layer 2: signals (read-only) ---


def test_signals_list_and_summary(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.record_user_behavior_signal("fact_corrected", subject="location", detail="NYC")
    s.record_user_behavior_signal("fact_corrected", subject="blog")
    s.record_user_behavior_signal("pattern_dismissed", subject="weather")
    with _client(s) as c:
        body = c.get("/learning/signals").json()
        assert body["count"] == 3
        assert body["summary"] == {"fact_corrected": 2, "pattern_dismissed": 1}
        only = c.get("/learning/signals?kind=fact_corrected").json()
        assert only["count"] == 2 and {x["subject"] for x in only["signals"]} == {
            "location",
            "blog",
        }


# --- layer 3: intentions ---


def test_intentions_list_approve_dismiss(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.propose_intention("i1", "Establish a morning routine", "groups brief", ["brief"])
    s.propose_intention("i2", "Learn Rust", "", [])
    with _client(s) as c:
        body = c.get("/learning/intentions").json()
        assert body["count"] == 2
        r = c.post("/learning/intentions/i1/approve")
        assert r.status_code == 200 and r.json()["status"] == "active"
        r = c.post("/learning/intentions/i2/dismiss")
        assert r.status_code == 200 and r.json()["status"] == "dismissed"

    assert s.list_intentions(status="proposed") == []
    assert s.get_intention("i1").status == "active"
    kinds = {x.kind for x in s.list_user_behavior_signals()}
    assert {"intention_approved", "intention_dismissed"} <= kinds


def test_intentions_approve_missing_404(tmp_path: Path) -> None:
    with _client(_store(tmp_path)) as c:
        assert c.post("/learning/intentions/nope/approve").status_code == 404


# --- write gate ---


def test_twin_writes_are_gated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)  # override conftest
    s = _store(tmp_path)
    s.propose_behavior_pattern("p1", "x", "low", [])
    s.propose_intention("i1", "y", "", [])
    with _client(s) as c:
        assert c.get("/learning/behaviors").status_code == 200  # reads open
        assert c.get("/learning/signals").status_code == 200
        assert c.get("/learning/intentions").status_code == 200
        assert c.post("/learning/behaviors/p1/approve").status_code == 403  # writes gated
        assert c.post("/learning/behaviors/p1/reject").status_code == 403
        assert c.post("/learning/intentions/i1/approve").status_code == 403
        assert c.post("/learning/intentions/i1/dismiss").status_code == 403
