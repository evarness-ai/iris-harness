"""Tests for the remote evaluator client (story 12.gov-3.9)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from iris_harness.kernel.governance.evaluator import StepRecord
from iris_harness.kernel.governance.evaluator.client import (
    RemoteEvaluatorClient,
    worst,
)


def _step(*, tier: str | None = "tier_3") -> StepRecord:
    return StepRecord(
        run_id="r", step_id=0, agent_type="chat", thought="hi", tier=tier  # type: ignore[arg-type]
    )


def _mock_transport(handler: Any) -> httpx.BaseTransport:
    return httpx.MockTransport(handler)


def test_evaluate_round_trip() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "signals": [
                    {
                        "name": "step_cap",
                        "verdict": "ok",
                        "reason": "step_cap: 0/20",
                    }
                ]
            },
        )

    client = RemoteEvaluatorClient(
        base_url="http://evaluator.test",
        transport=_mock_transport(handler),
    )
    results = client.evaluate(_step(tier="tier_1"))
    assert captured["url"] == "http://evaluator.test/evaluate"
    assert captured["body"]["run_id"] == "r"
    assert len(results) == 1
    assert results[0].name == "step_cap"
    assert results[0].verdict == "ok"


def test_cloud_route_fails_closed_on_error() -> None:
    """AC-3: tier_3 route gets a synthetic halt when the service is down."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "service unavailable"})

    client = RemoteEvaluatorClient(
        base_url="http://evaluator.test",
        transport=_mock_transport(handler),
    )
    results = client.evaluate(_step(tier="tier_3"))
    assert len(results) == 1
    assert results[0].verdict == "halt"
    assert results[0].severity == "critical"
    assert "fail-closed" in results[0].reason
    assert results[0].name == "evaluator_unavailable"


def test_local_route_fails_open_on_error() -> None:
    """AC-3: tier_1/tier_2 routes get a synthetic warn (allow with audit)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = RemoteEvaluatorClient(
        base_url="http://evaluator.test",
        transport=_mock_transport(handler),
    )
    for tier in ("tier_1", "tier_2"):
        results = client.evaluate(_step(tier=tier))
        assert results[0].verdict == "warn"
        assert results[0].severity == "warn"
        assert "fail-open" in results[0].reason


def test_malformed_response_treated_as_failure() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"not_signals": "oops"})

    client = RemoteEvaluatorClient(
        base_url="http://evaluator.test",
        transport=_mock_transport(handler),
    )
    results = client.evaluate(_step(tier="tier_3"))
    assert results[0].verdict == "halt"


def test_parse_error_in_signal_treated_as_failure() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"signals": [{"bogus": True}]},
        )

    client = RemoteEvaluatorClient(
        base_url="http://evaluator.test",
        transport=_mock_transport(handler),
    )
    results = client.evaluate(_step(tier="tier_3"))
    assert results[0].verdict == "halt"


def test_reset_calls_remote_endpoint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(200, json={"ok": True})

    client = RemoteEvaluatorClient(
        base_url="http://evaluator.test",
        transport=_mock_transport(handler),
    )
    client.reset_run_state("abc")
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/reset/abc")


def test_signal_count_via_healthz() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "signal_count": 4})

    client = RemoteEvaluatorClient(
        base_url="http://evaluator.test",
        transport=_mock_transport(handler),
    )
    assert client.signal_count() == 4


def test_signal_count_on_error_returns_zero() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = RemoteEvaluatorClient(
        base_url="http://evaluator.test",
        transport=_mock_transport(handler),
    )
    assert client.signal_count() == 0


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValueError):
        RemoteEvaluatorClient(base_url="http://x", timeout_s=0.0)


def test_worst_helper_matches_registry_semantics() -> None:
    from iris_harness.kernel.governance.evaluator.types import SignalResult

    rs = (
        SignalResult(name="a", verdict="ok", reason="ok"),
        SignalResult(name="b", verdict="warn", reason="w"),
        SignalResult(name="c", verdict="halt", reason="h"),
        SignalResult(name="d", verdict="require_approval", reason="r"),
    )
    assert worst(rs).verdict == "halt"  # type: ignore[union-attr]
    assert worst(()) is None
