"""Shadow tier-escalation judge wired into ResponseCurator (ADR-0068 L2).

The judge must be SHADOW-ONLY: it diagnoses + records, but its signal is always
non-acting (``skipped``) so it can never halt, retry, or banner the response.
"""

from __future__ import annotations

from iris_harness.agent.agent_executor import AgentResult
from iris_harness.agent.escalation import EscalationConfig
from iris_harness.agent.response_curator import ResponseCurator


class _Judge:
    """Mock EscalationJudgeClient: returns a fixed payload, or raises."""

    def __init__(self, payload: str | None = None, *, exc: Exception | None = None) -> None:
        self._payload = payload
        self._exc = exc
        self.calls = 0

    async def judge(self, *, query: str, response: str, intent: str, context: str) -> str:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        assert self._payload is not None
        return self._payload


def _curator(judge: _Judge | None, *, enabled: bool = True) -> ResponseCurator:
    return ResponseCurator(
        escalation_judge=judge,
        escalation_config=EscalationConfig(enabled=enabled, mode="shadow", sample_rate=1.0),
    )


def _curate(curator: ResponseCurator):
    return curator.curate(
        [AgentResult(agent_type="chat", output="a short answer", success=True)],
        query="explain quantum error correction in depth",
        intent="general",
    )


def _escalation_signal(curated):
    bundle = curated.metadata["judge_bundle"]
    sigs = [s for s in bundle["signals"] if s["name"] == "escalation"]
    assert sigs, "escalation signal missing from judge bundle"
    return sigs[0]


def test_shadow_verdict_is_non_acting_but_recorded() -> None:
    judge = _Judge(
        '{"action": "escalate", "diagnosis": "capability_gap",'
        ' "confidence": 0.8, "reason": "needs a bigger model"}'
    )
    curated = _curate(_curator(judge))

    assert judge.calls == 1
    assert curated.has_errors is False
    # Shadow signal never acts: verdict is skipped, no governance banner.
    sig = _escalation_signal(curated)
    assert sig["verdict"] == "skipped"
    assert "warning_banner" not in curated.metadata
    # ...but the would-be decision is captured for learning.
    verdict = sig["metadata"]["escalation"]
    assert verdict["action"] == "escalate"
    assert verdict["diagnosis"] == "capability_gap"
    assert sig["metadata"]["shadow"] is True


def test_disabled_config_skips_without_calling_judge() -> None:
    judge = _Judge('{"action": "escalate", "diagnosis": "capability_gap", "confidence": 1.0}')
    curated = _curate(_curator(judge, enabled=False))
    assert judge.calls == 0
    sig = _escalation_signal(curated)
    assert sig["verdict"] == "skipped"
    assert "not configured" in sig["reason"]


def test_no_judge_is_skipped() -> None:
    curated = _curate(_curator(None))
    sig = _escalation_signal(curated)
    assert sig["verdict"] == "skipped"


def test_judge_failure_is_inconclusive_not_fatal() -> None:
    curated = _curate(_curator(_Judge(exc=RuntimeError("judge down"))))
    assert curated.has_errors is False  # curation never breaks on a judge failure
    sig = _escalation_signal(curated)
    assert sig["verdict"] == "skipped"
    assert "inconclusive" in sig["reason"]


def test_unparseable_verdict_is_inconclusive() -> None:
    curated = _curate(_curator(_Judge("the model rambled without json")))
    sig = _escalation_signal(curated)
    assert sig["verdict"] == "skipped"
    assert "inconclusive" in sig["reason"]


def test_accept_verdict_records_not_would_act() -> None:
    judge = _Judge('{"action": "accept", "diagnosis": "acceptable", "confidence": 0.9}')
    curated = _curate(_curator(judge))
    verdict = _escalation_signal(curated)["metadata"]["escalation"]
    assert verdict["action"] == "accept"
