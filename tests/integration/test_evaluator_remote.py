"""Integration test: agent ↔ remote evaluator HTTP round-trip (story 12.gov-3.9).

Spins up the FastAPI ``src/iris_harness/server/evaluator`` app in-process and proxies
sync httpx requests through the FastAPI ``TestClient`` so the agent
side talks to the real route handlers (no socket needed). This is the
AC-2/AC-6 proof: same ``SignalResult`` shape between local and remote;
existing in-process mode continues to work.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance.evaluator import EvaluatorHook, StepRecord
from iris_harness.kernel.governance.evaluator.client import RemoteEvaluatorClient
from iris_harness.kernel.governance.evaluator.signals import StepCapSignal
from iris_harness.kernel.governance.hooks.types import HookContext, HookPoint
from iris_harness.server.evaluator.main import create_app


def _proxy_transport(fastapi_app: TestClient) -> httpx.MockTransport:
    """Bridge sync httpx → FastAPI TestClient (which wraps the ASGI app)."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = request.content if request.content else None
        response = fastapi_app.request(
            method=request.method,
            url=str(request.url.path),
            content=body,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        )
        return httpx.Response(
            status_code=response.status_code,
            content=response.content,
            headers=response.headers,
        )

    return httpx.MockTransport(_handler)


@pytest.fixture()
def remote_client() -> RemoteEvaluatorClient:
    test_client = TestClient(create_app(), headers=auth_headers())  # 4 cheap signals
    return RemoteEvaluatorClient(
        base_url="http://evaluator",
        transport=_proxy_transport(test_client),
        timeout_s=2.0,
    )


def test_service_round_trip_with_test_client() -> None:
    """The FastAPI app responds 200 to a well-formed POST /evaluate."""
    client = TestClient(create_app(), headers=auth_headers())

    step_payload = StepRecord(run_id="r", step_id=0, agent_type="chat", thought="hi").model_dump(
        mode="json"
    )

    response = client.post("/evaluate", json=step_payload)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["signals"], list)
    assert len(body["signals"]) > 0  # the 4 cheap signals all fire each step


def test_healthz_reports_signal_count() -> None:
    client = TestClient(create_app(), headers=auth_headers())
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["signal_count"] == 4
    assert data["locked"] is True


def test_hook_routes_through_remote_client(remote_client: RemoteEvaluatorClient) -> None:
    """AC-1 + AC-2: EvaluatorHook with backend= produces the same shape decisions."""
    hook = EvaluatorHook(backend=remote_client)
    ctx = HookContext(
        hook_point=HookPoint.POST_STEP,
        run_id="r1",
        agent_type="chat",
        step_id=0,
        payload={"thought": "exploring", "tool_name": "search", "tool_args_hash": "x"},
    )
    decision = asyncio.run(hook(ctx))
    # All four cheap signals return ok on step 0 → hook allows.
    assert decision.outcome == "allow"
    assert "evaluator:" in decision.reason


def test_remote_step_cap_halts_at_threshold(remote_client: RemoteEvaluatorClient) -> None:
    """Drive the remote registry past the step cap; ensure halt propagates."""
    hook = EvaluatorHook(backend=remote_client)
    final_decision = None
    # Default step_cap for "chat" is 20; cross it with room to spare.
    for step_id in range(0, 25):
        ctx = HookContext(
            hook_point=HookPoint.POST_STEP,
            run_id="rcap",
            agent_type="chat",
            step_id=step_id,
            payload={"thought": f"step {step_id}"},
        )
        final_decision = asyncio.run(hook(ctx))
        if final_decision.outcome == "deny":
            break
    assert final_decision is not None
    assert final_decision.outcome == "deny"
    assert "step_cap" in final_decision.reason


def test_in_process_mode_still_default() -> None:
    """AC-6: existing in-process mode continues to work."""
    from iris_harness.kernel.governance.evaluator import EvaluatorRegistry

    reg = EvaluatorRegistry()
    reg.register(StepCapSignal(default_threshold=20))
    reg.init_lock()
    hook = EvaluatorHook(registry=reg)

    ctx = HookContext(
        hook_point=HookPoint.POST_STEP,
        run_id="r",
        agent_type="chat",
        step_id=0,
        payload={"thought": "hi"},
    )
    decision = asyncio.run(hook(ctx))
    assert decision.outcome == "allow"


def test_reset_endpoint_clears_run_state() -> None:
    """POST /reset/<run_id> drops per-run signal state."""
    test_client = TestClient(create_app(), headers=auth_headers())
    client = RemoteEvaluatorClient(base_url="http://x", transport=_proxy_transport(test_client))
    # Walk the run past most of the step cap, reset, then walk again.
    for step_id in range(0, 15):
        client.evaluate(StepRecord(run_id="r", step_id=step_id, agent_type="chat", thought="t"))
    client.reset_run_state("r")
    # After reset, step_cap state is fresh — step 15 in a new "view" of the run
    # should still be ok (step_cap default 20; reset clears its counter).
    results = client.evaluate(StepRecord(run_id="r", step_id=15, agent_type="chat", thought="t"))
    step_cap = next(r for r in results if r.name == "step_cap")
    assert step_cap.verdict == "ok"
