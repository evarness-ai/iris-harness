"""Tests for POST /surface-feedback (issue 0028)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.learning.suppression import SurfaceFeedbackStore, encode_ref


def _client(tmp_path: Path) -> TestClient:
    runtime = SimpleNamespace(data_dir=tmp_path)
    return TestClient(create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers())


def test_surface_feedback_by_ref_suppresses(tmp_path: Path) -> None:
    client = _client(tmp_path)
    ref = encode_ref("system", "health_alert", {"kind": "credential", "target": "sfbtest-probe"})
    resp = client.post("/surface-feedback", json={"ref": ref, "verdict": "not_useful"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True

    store = SurfaceFeedbackStore()
    store.ensure_schema()
    assert store.should_suppress(
        "system", "health_alert", {"kind": "credential", "target": "sfbtest-probe"}
    )


def test_surface_feedback_by_explicit_dims(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/surface-feedback",
        json={
            "subsystem": "finance",
            "surface_kind": "bill_due",
            "dims": {"label": "netflix", "currency": "inr"},
            "verdict": "not_useful",
        },
    )
    assert resp.status_code == 200, resp.text

    store = SurfaceFeedbackStore()
    store.ensure_schema()
    assert store.should_suppress("finance", "bill_due", {"label": "netflix", "currency": "inr"})


def test_surface_feedback_rejects_bad_verdict(tmp_path: Path) -> None:
    client = _client(tmp_path)
    ref = encode_ref("email", "followup", {"from_domain": "x.com"})
    resp = client.post("/surface-feedback", json={"ref": ref, "verdict": "meh"})
    assert resp.status_code == 422


def test_surface_feedback_requires_ref_or_dims(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post("/surface-feedback", json={"verdict": "not_useful"})
    assert resp.status_code == 422
