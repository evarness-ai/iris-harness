"""Tests for the web-UI write gate (IRIS_WEBUI_ALLOW_WRITES)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api.main import create_app


def _client() -> Iterator[TestClient]:
    with TestClient(create_app(auto_start_runtime=False), headers=auth_headers()) as c:
        yield c


@pytest.fixture
def writes_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    monkeypatch.delenv("IRIS_FEEDBACK_CAPTURE", raising=False)
    monkeypatch.delenv("IRIS_DEPLOYMENT_LABEL", raising=False)
    yield from _client()


@pytest.fixture
def writes_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    monkeypatch.delenv("IRIS_FEEDBACK_CAPTURE", raising=False)
    monkeypatch.delenv("IRIS_DEPLOYMENT_LABEL", raising=False)
    yield from _client()


def test_capabilities_reflects_flag(writes_off: TestClient) -> None:
    assert writes_off.get("/capabilities").json() == {
        "writes_enabled": False,
        "feedback_capture": False,
        "deployment_label": "",
    }


def test_capabilities_on(writes_on: TestClient) -> None:
    assert writes_on.get("/capabilities").json() == {
        "writes_enabled": True,
        "feedback_capture": False,
        "deployment_label": "",
    }


def test_capabilities_feedback_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    monkeypatch.setenv("IRIS_FEEDBACK_CAPTURE", "1")
    monkeypatch.delenv("IRIS_DEPLOYMENT_LABEL", raising=False)
    for client in _client():
        assert client.get("/capabilities").json() == {
            "writes_enabled": False,
            "feedback_capture": True,
            "deployment_label": "",
        }


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("iris-vm (cloud trial)", "iris-vm (cloud trial)"),
        ("  iris-vm  ", "iris-vm"),  # an env file's stray spaces are not part of the name
        ("   ", ""),  # blank is unset: the badge falls back to its build-time label
    ],
)
def test_capabilities_names_the_deployment(
    monkeypatch: pytest.MonkeyPatch, raw: str, label: str
) -> None:
    """The console in the server image is built once for every deployment, so the
    "which harness is this" badge gets the name from the server at runtime."""
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    monkeypatch.delenv("IRIS_FEEDBACK_CAPTURE", raising=False)
    monkeypatch.setenv("IRIS_DEPLOYMENT_LABEL", raw)
    for client in _client():
        assert client.get("/capabilities").json() == {
            "writes_enabled": False,
            "feedback_capture": False,
            "deployment_label": label,
        }


def test_the_deployment_label_needs_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """The label names a private machine; it is data, not part of the public shell."""
    monkeypatch.setenv("IRIS_DEPLOYMENT_LABEL", "iris-vm")
    with TestClient(create_app(auto_start_runtime=False)) as anonymous:
        assert anonymous.get("/capabilities").status_code == 401


def test_control_write_blocked_when_off(writes_off: TestClient) -> None:
    # A gated control mutation → 403 before it ever reaches the handler.
    resp = writes_off.patch("/routines/abc", json={"approval_status": "approved"})
    assert resp.status_code == 403
    assert "IRIS_WEBUI_ALLOW_WRITES" in resp.json()["detail"]


def test_rag_delete_blocked_when_off(writes_off: TestClient) -> None:
    assert writes_off.delete("/rag/documents/rag_x").status_code == 403


def test_reads_never_gated(writes_off: TestClient) -> None:
    # GETs pass through the guard even with writes off (503/runtime errors are
    # fine — the point is they are NOT 403).
    assert writes_off.get("/capabilities").status_code == 200
    assert writes_off.get("/healthz").status_code == 200


def test_control_write_allowed_when_on(writes_on: TestClient) -> None:
    # With the flag on the guard lets it through to the handler (which 503s here
    # because no runtime is wired — crucially NOT 403).
    resp = writes_on.patch("/routines/abc", json={"approval_status": "approved"})
    assert resp.status_code != 403


# ── Deny-by-default (security floor, phase 0) ───────────────────────────────


def test_portfolio_import_gated(writes_off: TestClient) -> None:
    # Previously slipped through the gated-path allowlist.
    assert writes_off.post("/portfolio/import-holdings").status_code == 403


def test_unknown_mutating_route_gated_by_default(writes_off: TestClient) -> None:
    # Deny by default: a mutating route nobody thought about is gated (403),
    # not silently open (404 would mean it reached the router).
    assert writes_off.post("/some/future/mutation").status_code == 403


def test_chat_and_product_surface_stay_open(writes_off: TestClient) -> None:
    # The declared product surface bypasses the gate (handlers may 4xx/5xx for
    # other reasons without a runtime — the point is they are NOT 403).
    for path in ("/chat", "/chat/stream", "/warmup", "/rag/search", "/rag/upload"):
        assert writes_off.post(path).status_code != 403, path


def test_feedback_endpoints_stay_open(writes_off: TestClient) -> None:
    # Local learning telemetry stays open for read-only consoles (ADR-0072 /
    # issue 0028); they enforce their own IRIS_FEEDBACK_CAPTURE flag.
    assert writes_off.post("/api/feedback").status_code != 403
    assert writes_off.post("/surface-feedback").status_code != 403
