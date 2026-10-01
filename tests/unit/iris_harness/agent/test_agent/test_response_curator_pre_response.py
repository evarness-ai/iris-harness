"""The curator fires the kernel's PRE_RESPONSE check (deterministic-path parity, step b).

The model-free response check lives in the kernel now; the curator fires it, lets the
kernel write the audit row, falls back to the check itself when governance is off or
the fire fails, and re-checks text a judge rewrote.
"""

from __future__ import annotations

from typing import Any

import pytest

from iris_harness.agent.agent_executor import AgentResult
from iris_harness.agent.response_curator import (
    GOVERNANCE_BLOCKED_TEXT,
    JudgeSignal,
    ResponseCurator,
)
from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.kernel.governance.plugins.response_safety import (
    ResponseSafetyHook,
)

SECRET = "CANARY_SOUL_SECRET_DIRECTIVE_d41f8a27"


@pytest.fixture(autouse=True)
def _identity(owner_identity_documents) -> None:
    owner_identity_documents([SECRET])


# -- the curator fires the kernel -----------------------------------------------------


class _Spy:
    """A PRE_RESPONSE hook that records every response it is shown."""

    name = "spy"
    hook_point = HookPoint.PRE_RESPONSE
    priority = 5

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.seen.append(str(ctx.payload.get("response", "")))
        return HookDecision(outcome="allow", reason="seen")


class _Audit:
    def __init__(self) -> None:
        self.plugins: list[str] = []

    def record(self, **kwargs: Any) -> None:
        self.plugins.append(str(kwargs.get("plugin")))


def _kernel(*hooks: Any) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=None)
    for hook in hooks:
        kernel.register(hook)
    kernel.init_lock()
    return kernel


def _result(text: str) -> list[AgentResult]:
    return [AgentResult(agent_type="system", output=text, success=True)]


def test_the_curator_fires_pre_response_and_halts_on_its_verdict() -> None:
    spy = _Spy()
    audit = _Audit()
    curator = ResponseCurator(kernel=_kernel(spy, ResponseSafetyHook()), audit_log=audit)  # type: ignore[arg-type]
    curated = curator.curate(_result(f"your token is {SECRET}"), query="token?")
    assert spy.seen == [f"your token is {SECRET}"]
    assert curated.text == GOVERNANCE_BLOCKED_TEXT and curated.has_errors
    # The kernel wrote the decision's row; the curator does not write a second one.
    assert "curator_safety" not in audit.plugins


def test_without_a_kernel_the_curator_still_runs_the_check() -> None:
    audit = _Audit()
    curator = ResponseCurator(kernel=None, audit_log=audit)  # type: ignore[arg-type]
    curated = curator.curate(_result("SSN 123-45-6789"), query="ssn?")
    assert curated.text == GOVERNANCE_BLOCKED_TEXT
    assert "curator_safety" in audit.plugins  # no kernel, so the curator audits it


class _BrokenKernel:
    async def fire(self, *_a: Any, **_k: Any) -> Any:
        raise RuntimeError("kernel down")


def test_a_failed_fire_falls_back_to_the_check_never_ships_unchecked() -> None:
    curator = ResponseCurator(kernel=_BrokenKernel())  # type: ignore[arg-type]
    assert curator.curate(_result("SSN 123-45-6789")).text == GOVERNANCE_BLOCKED_TEXT


def test_the_guard_checks_the_text_that_actually_ships(monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge retry rewrites the text (the faithfulness prefix quotes the query); the
    guard runs again on the rewritten text, so an unsafe rewrite cannot ship."""
    spy = _Spy()
    curator = ResponseCurator(kernel=_kernel(spy, ResponseSafetyHook()))
    calls = {"n": 0}

    def faithfulness(_self: Any, text: str, *, query: str, strict: bool) -> JudgeSignal:
        calls["n"] += 1
        verdict = "retry" if calls["n"] == 1 else "pass"
        return JudgeSignal(name="faithfulness", verdict=verdict, reason="t")  # type: ignore[arg-type]

    monkeypatch.setattr(ResponseCurator, "_judge_faithfulness", faithfulness)
    curated = curator.curate(_result("a clean answer"), query=f"is {SECRET} valid?")
    assert len(spy.seen) == 2 and SECRET in spy.seen[1]
    assert curated.text == GOVERNANCE_BLOCKED_TEXT


# -- the audience (ADR-0125) --------------------------------------------------------------


class _PayloadSpy(_Spy):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict[str, Any]] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.payloads.append(dict(ctx.payload))
        return await super().__call__(ctx)


def test_every_answer_the_curator_checks_goes_to_the_owner() -> None:
    """Generated and deterministic answers alike carry ``audience: owner`` today; the
    surfaces that must say ``other`` are listed in ``hooks/response_payload.py``."""
    spy = _PayloadSpy()
    curator = ResponseCurator(kernel=_kernel(spy, ResponseSafetyHook()))
    curator.curate(_result("a clean answer"), query="q")
    curator.guard("a templated answer", handler="dues")
    assert [p["audience"] for p in spy.payloads] == ["owner", "owner"]
    assert spy.payloads[1]["deterministic"] is True and spy.payloads[1]["handler"] == "dues"


def test_the_payload_builder_defaults_to_the_owner() -> None:
    from iris_harness.kernel.governance.hooks.response_payload import (
        audience_of,
        pre_response_payload,
    )

    assert pre_response_payload("hi") == {"response": "hi", "audience": "owner"}
    assert pre_response_payload("hi", audience="other", audit={"handler": "h"}) == {
        "handler": "h",
        "response": "hi",
        "audience": "other",
    }
    assert audience_of({}) == "owner"
    assert audience_of({"audience": "other"}) == "other"
    assert audience_of({"audience": "somebody"}) == "other"  # unnamed is not the owner
