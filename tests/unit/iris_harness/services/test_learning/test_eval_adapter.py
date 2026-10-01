"""Tests for the eval-harness live adapter (ADR-0070, slice 1).

The runtime build + real chat turns need a live model backend, so they aren't
unit-tested. What IS tested: the pure ChatResult->RunOutcome mapping and that the
adapter applies the arm's priors and maps the result — driven by a fake runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iris_harness.services.learning.eval_adapter import (
    chat_result_to_outcome,
    make_runtime_run_query,
    run_preflight,
)
from iris_harness.services.learning.eval_harness import EvalQuery, VariantConfig


@dataclass
class _FakeResult:
    has_errors: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def test_outcome_completed_from_has_errors() -> None:
    assert chat_result_to_outcome(_FakeResult(has_errors=False)).completed is True
    assert chat_result_to_outcome(_FakeResult(has_errors=True)).completed is False


def test_outcome_reads_tools_from_metadata() -> None:
    out = chat_result_to_outcome(_FakeResult(metadata={"tools": ["read_email", "search_inbox"]}))
    assert out.tools_called == ("read_email", "search_inbox")
    # alt key + string coercion
    out2 = chat_result_to_outcome(_FakeResult(metadata={"tools_called": "read_email"}))
    assert out2.tools_called == ("read_email",)


def test_outcome_no_tools_when_metadata_absent() -> None:
    assert chat_result_to_outcome(_FakeResult()).tools_called == ()


class _FakeRouter:
    def __init__(self) -> None:
        self.priors: dict[str, str] = {}

    def set_intent_tier_priors(self, priors: dict[str, str]) -> None:
        self.priors = dict(priors)


class _FakeRuntime:
    def __init__(self, result: _FakeResult) -> None:
        self.tier_router = _FakeRouter()
        self._result = result
        self.calls: list[tuple[str, str]] = []

    def chat(self, message: str, *, session_id: str = "default", **_: Any) -> _FakeResult:
        self.calls.append((message, session_id))
        return self._result


def test_run_query_applies_variant_priors_and_maps_result() -> None:
    rt = _FakeRuntime(_FakeResult(has_errors=False, metadata={"tools": ["read_email"]}))
    run = make_runtime_run_query(rt)

    cfg = VariantConfig(label="email@tier2", intent_tier_priors={"email": "tier2"})
    outcome = run(EvalQuery(query="summarize that email", intent="email"), cfg)

    assert rt.tier_router.priors == {"email": "tier2"}  # variant applied
    assert outcome.completed is True
    assert outcome.tools_called == ("read_email",)
    assert rt.calls[0][0] == "summarize that email"


def test_run_query_baseline_clears_priors() -> None:
    rt = _FakeRuntime(_FakeResult())
    run = make_runtime_run_query(rt)
    run(EvalQuery(query="q", intent="email"), VariantConfig(label="baseline"))
    assert rt.tier_router.priors == {}  # baseline arm = no overrides


class _ArmAwareRuntime:
    """Completes only under the variant arm — so the pre-flight should say 'better'."""

    def __init__(self) -> None:
        self.tier_router = _FakeRouter()

    def chat(self, message: str, *, session_id: str = "default", **_: Any) -> _FakeResult:
        variant = bool(self.tier_router.priors)
        return _FakeResult(has_errors=not variant)


def test_run_preflight_returns_comparison() -> None:
    rt = _ArmAwareRuntime()
    workload = [EvalQuery(query="q1", intent="email"), EvalQuery(query="q2", intent="email")]
    cmp = run_preflight(
        rt, workload, VariantConfig(label="email@tier2", intent_tier_priors={"email": "tier2"})
    )
    assert cmp.verdict == "better"
    assert cmp.variant.completion_rate == 1.0
    assert cmp.baseline.completion_rate == 0.0


def test_run_preflight_empty_workload_inconclusive() -> None:
    cmp = run_preflight(_FakeRuntime(_FakeResult()), [], VariantConfig(label="v"))
    assert cmp.verdict == "inconclusive"
