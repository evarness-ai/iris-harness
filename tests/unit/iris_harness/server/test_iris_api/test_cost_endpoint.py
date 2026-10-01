"""GET /cost (Track 2 PR 6) — LLM spend from the governance cost ledger.

A thin renderer over ``kernel.governance.cost.summary``, so the rollup's own
behaviour is covered next to it. What is asserted here is the wiring: the route
is authenticated, it is a read, and it carries ``recording`` so the Pulse card
can tell "nobody is counting" apart from "nothing was spent".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance.cost import CostStore
from iris_harness.kernel.governance.cost.summary import COST_LIMITER_ENV
from iris_harness.server.iris_api.main import create_app


@pytest.fixture()
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CostStore:
    """Point the endpoint at a throwaway ledger, never the developer's own."""
    db = tmp_path / "cost-ledger.db"
    monkeypatch.setenv("IRIS_GOVERNANCE_COST_LEDGER_DB_PATH", str(db))
    return CostStore(db_path=db)


def _client() -> TestClient:
    return TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    )


def test_cost_endpoint_reports_spend(ledger: CostStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(COST_LIMITER_ENV, "1")
    ledger.record(
        run_id="r1",
        agent_type="chat",
        tier="tier_3",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.25,
        user_id="local",
    )
    with _client() as client:
        resp = client.get("/cost")

    assert resp.status_code == 200
    body = resp.json()
    assert body["recording"] is True
    assert body["enable_hint"] is None
    assert body["today_usd"] == pytest.approx(0.25)
    assert body["by_tier_usd"]["tier_3"] == pytest.approx(0.25)


def test_cost_endpoint_says_when_nothing_is_recording(
    ledger: CostStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case the whole payload shape exists for.

    With ``CostLimiter`` off the sums are a truthful 0.00 that would render as
    "IRIS is free". The endpoint has to hand the caller enough to say otherwise.
    """
    monkeypatch.delenv(COST_LIMITER_ENV, raising=False)
    with _client() as client:
        body = client.get("/cost").json()

    assert body["recording"] is False
    assert body["enable_hint"] == f"{COST_LIMITER_ENV}=1"
    assert body["today_usd"] == 0.0


def test_the_hint_is_pasteable(ledger: CostStore, monkeypatch: pytest.MonkeyPatch) -> None:
    # It is offered as a copy button, so it must be a line you can paste into an
    # env file — not a sentence about one.
    monkeypatch.delenv(COST_LIMITER_ENV, raising=False)
    with _client() as client:
        hint = client.get("/cost").json()["enable_hint"]
    assert hint is not None
    assert " " not in hint
    assert hint.count("=") == 1


def test_cost_endpoint_follows_iris_user_id(
    ledger: CostStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(COST_LIMITER_ENV, "1")
    monkeypatch.setenv("IRIS_USER_ID", "owner")
    ledger.record(
        run_id="r1",
        agent_type="chat",
        tier="tier_3",
        prompt_tokens=10,
        completion_tokens=10,
        cost_usd=1.50,
        user_id="owner",
    )
    with _client() as client:
        body = client.get("/cost").json()
    assert body["user_id"] == "owner"
    assert body["today_usd"] == pytest.approx(1.50)


def test_cost_endpoint_needs_a_credential(ledger: CostStore) -> None:
    # Spend is not public: it is served behind the same bearer check as the
    # rest of the API, with no header at all.
    with TestClient(create_app(runtime=SimpleNamespace(), auto_start_runtime=False)) as client:
        assert client.get("/cost").status_code == 401


def test_cost_endpoint_is_a_read_not_a_gated_write(
    ledger: CostStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A GET must not be caught by the write guard: the card has to render on a
    # read-scoped device and with IRIS_WEBUI_ALLOW_WRITES unset.
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    with _client() as client:
        assert client.get("/cost").status_code == 200
