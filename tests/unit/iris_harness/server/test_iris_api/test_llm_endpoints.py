"""Tests for the /llm/* governor endpoints exposed by the IRIS API."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.llm.arbiter import Mode, OllamaArbiter, PressureSnapshot, ResourceGovernor
from iris_harness.server.iris_api.main import create_app


def _stub_http():
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"models": []}

    class _Http:
        def get(self, url: str):
            return _Resp()

        def post(self, url: str, json):
            return _Resp()

        def close(self) -> None:
            return None

    return _Http()


def _governor_with_snapshot(
    snapshot: PressureSnapshot, *, adaptive: bool = False
) -> ResourceGovernor:
    arb = OllamaArbiter(base_url="http://localhost:11434", http=_stub_http())
    queue = [snapshot]

    def _sampler() -> PressureSnapshot:
        # Always return the seeded snapshot; tests don't care about hysteresis here.
        return queue[0]

    return ResourceGovernor(arbiter=arb, adaptive=adaptive, sampler=_sampler)


def _runtime_with_governor(governor: ResourceGovernor):
    tier_router = SimpleNamespace(governor=governor)
    return SimpleNamespace(tier_router=tier_router)


def _snapshot() -> PressureSnapshot:
    return PressureSnapshot(
        ram_free_gb=18.4,
        cpu_percent=12.0,
        cpu_speed_limit=100,
        thermal_throttled=False,
        sampled_at=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
    )


def test_llm_mode_returns_state() -> None:
    governor = _governor_with_snapshot(_snapshot(), adaptive=True)
    governor.poll()  # seed _last_snapshot
    runtime = _runtime_with_governor(governor)

    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )
    resp = client.get("/llm/mode")

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "active"
    assert body["pin"] is None
    assert body["adaptive"] is True
    assert body["snapshot"]["ram_free_gb"] == 18.4


def test_llm_mode_pin_and_unpin_round_trip() -> None:
    governor = _governor_with_snapshot(_snapshot())
    runtime = _runtime_with_governor(governor)

    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    pinned = client.post("/llm/mode/pin", json={"mode": "thermal"})
    assert pinned.status_code == 200
    assert pinned.json()["pin"] == "thermal"
    assert pinned.json()["mode"] == "thermal"
    assert governor.pin == Mode.THERMAL

    released = client.delete("/llm/mode/pin")
    assert released.status_code == 200
    assert released.json()["pin"] is None
    assert governor.pin is None


def test_llm_pressure_polls_and_returns_snapshot() -> None:
    governor = _governor_with_snapshot(_snapshot())
    runtime = _runtime_with_governor(governor)

    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )
    resp = client.get("/llm/pressure")

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "active"
    assert body["snapshot"]["cpu_speed_limit"] == 100


def test_llm_endpoints_503_when_governor_missing() -> None:
    runtime = SimpleNamespace(tier_router=SimpleNamespace(governor=None))
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    assert client.get("/llm/mode").status_code == 503
    assert client.get("/llm/pressure").status_code == 503
    assert client.post("/llm/mode/pin", json={"mode": "active"}).status_code == 503
    assert client.delete("/llm/mode/pin").status_code == 503


def test_llm_pin_rejects_invalid_mode() -> None:
    governor = _governor_with_snapshot(_snapshot())
    runtime = _runtime_with_governor(governor)
    client = TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    )

    resp = client.post("/llm/mode/pin", json={"mode": "bogus"})
    assert resp.status_code == 422
