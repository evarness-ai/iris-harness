"""API tests for the /playground endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api.main import create_app


def _client(runtime: object | None = None) -> Iterator[TestClient]:
    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as c:
        yield c


@pytest.fixture
def writes_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)


@pytest.fixture
def writes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")


def _fake_runtime() -> SimpleNamespace:
    registry = SimpleNamespace(list_packages=lambda: ())
    router = SimpleNamespace(_tiers={"tier1": object()})
    gateway = SimpleNamespace(channels=lambda: ["console"])
    return SimpleNamespace(skill_registry=registry, tier_router=router, channels=gateway)


def test_suites_lists_committed_suite() -> None:
    for client in _client():
        body = client.get("/playground/suites").json()
    names = {s["name"] for s in body["suites"]}
    assert "core-deterministic" in names


def test_drift_reports_surfaces() -> None:
    for client in _client(_fake_runtime()):
        body = client.get("/playground/drift").json()
    assert {s["surface"] for s in body["surfaces"]} == {
        "skills",
        "llm_tiers",
        "channels",
        "intercepts",
        "plugin_tools",  # ADR-0110: manifest-declared tools vs registered ones
        "plugin_uses",  # plugin-capabilities §4: a grant naming no registered tool
        "plugin_capabilities",  # §2: a declared capability never provided
        "capability_uses",  # §2: a capability used that nothing provides
    }


def test_run_is_write_gated_by_default(writes_off: None) -> None:
    # POST /playground/run drives chat, so the deny-by-default gate blocks it
    # unless writes are enabled.
    for client in _client(_fake_runtime()):
        resp = client.post("/playground/run", json={"suite": "core-deterministic"})
    assert resp.status_code == 403


def test_run_unknown_suite_404(writes_on: None) -> None:
    for client in _client(_fake_runtime()):
        resp = client.post("/playground/run", json={"suite": "does-not-exist"})
    assert resp.status_code == 404
