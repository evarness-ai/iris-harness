"""Kernel basics: registration, dispatch, init-lock CI gate.

The ``test_runtime_registration_forbidden`` test is the CI gate
described in design §5.3 (hook plugin trust boundary): hook plugins
cannot be added after kernel init, because hooks observe every prompt
and every tool call.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from iris_harness.foundation.observability.session_log import session_scope
from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
    HookRegistrationLockedError,
    KernelNotInitializedError,
)
from iris_harness.kernel.governance.audit.log import AuditLog


class _AllowHook:
    name: str = "allow"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 100

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="allowed")


class _DenyHook:
    name: str = "deny"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 50

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="deny", reason="denied for test")


class _TransformHook:
    name: str = "transform"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 10

    async def __call__(self, ctx: HookContext) -> HookDecision:
        new_payload: dict[str, Any] = {**ctx.payload, "transformed": True}
        return HookDecision(
            outcome="transform",
            reason="redacted",
            transformed_payload=new_payload,
        )


class _AssertTransformedHook:
    name: str = "assert_transformed"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 20

    async def __call__(self, ctx: HookContext) -> HookDecision:
        if ctx.payload.get("transformed") is not True:
            return HookDecision(outcome="deny", reason="expected transformed payload")
        return HookDecision(outcome="allow", reason="ok")


class _SetClassificationHook:
    name: str = "set_class"
    hook_point: HookPoint = HookPoint.PRE_CLASSIFY
    priority: int = 10

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(
            outcome="allow",
            reason="classified as personal",
            set_classification="personal",
        )


class _DenyAndAnnotateHook:
    """Demonstrates annotations applied even when outcome is deny."""

    name: str = "deny_annotate"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 10

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(
            outcome="deny",
            reason="blocked but annotated",
            set_classification="secret",
            set_tier="tier_1",
        )


class _RaisingHook:
    name: str = "raising"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 5

    async def __call__(self, ctx: HookContext) -> HookDecision:
        raise RuntimeError("simulated plugin crash")


def _ctx(point: HookPoint = HookPoint.PRE_LLM_CALL) -> HookContext:
    return HookContext(
        hook_point=point,
        run_id="run-1",
        agent_type="chat",
        payload={"prompt": "hello"},
    )


async def test_no_hooks_allows_by_default() -> None:
    kernel = GovernanceKernel()
    kernel.init_lock()
    decision, ctx = await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())
    assert decision.outcome == "allow"
    assert ctx.classification is None


async def test_single_allow_hook_dispatches() -> None:
    kernel = GovernanceKernel()
    kernel.register(_AllowHook())
    kernel.init_lock()
    decision, _ = await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())
    assert decision.outcome == "allow"
    assert decision.reason == "allowed"


async def test_deny_short_circuits_before_lower_priority_allow() -> None:
    kernel = GovernanceKernel()
    kernel.register(_AllowHook())
    kernel.register(_DenyHook())
    kernel.init_lock()
    decision, _ = await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())
    assert decision.outcome == "deny"
    assert "denied" in decision.reason


async def test_transform_rebinds_payload_for_next_hook() -> None:
    kernel = GovernanceKernel()
    kernel.register(_TransformHook())
    kernel.register(_AssertTransformedHook())
    kernel.init_lock()
    decision, ctx = await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())
    assert decision.outcome == "allow"
    assert ctx.payload.get("transformed") is True


async def test_set_classification_propagates_to_final_context() -> None:
    kernel = GovernanceKernel()
    kernel.register(_SetClassificationHook())
    kernel.init_lock()
    decision, ctx = await kernel.fire(HookPoint.PRE_CLASSIFY, _ctx(HookPoint.PRE_CLASSIFY))
    assert decision.outcome == "allow"
    assert ctx.classification == "personal"


async def test_annotations_apply_even_when_decision_is_deny() -> None:
    """A hook can deny *and* still record the classification it inferred."""
    kernel = GovernanceKernel()
    kernel.register(_DenyAndAnnotateHook())
    kernel.init_lock()
    decision, ctx = await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())
    assert decision.outcome == "deny"
    assert ctx.classification == "secret"
    assert ctx.tier == "tier_1"


async def test_hook_exception_is_fail_closed_deny() -> None:
    kernel = GovernanceKernel()
    kernel.register(_RaisingHook())
    kernel.init_lock()
    decision, _ = await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())
    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert "raising" in decision.reason


async def test_runtime_registration_forbidden() -> None:
    """CI gate for design §5.3 hook plugin trust boundary."""
    kernel = GovernanceKernel()
    kernel.init_lock()
    with pytest.raises(HookRegistrationLockedError):
        kernel.register(_AllowHook())


async def test_fire_before_init_lock_raises() -> None:
    kernel = GovernanceKernel()
    with pytest.raises(KernelNotInitializedError):
        await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())


async def test_other_hook_points_unaffected_by_registration() -> None:
    kernel = GovernanceKernel()
    kernel.register(_DenyHook())  # registers on PRE_LLM_CALL
    kernel.init_lock()
    decision, _ = await kernel.fire(HookPoint.PRE_TOOL_USE, _ctx(HookPoint.PRE_TOOL_USE))
    assert decision.outcome == "allow"


def test_hook_count_diagnostic() -> None:
    kernel = GovernanceKernel()
    kernel.register(_AllowHook())
    kernel.register(_DenyHook())
    kernel.init_lock()
    assert kernel.hook_count(HookPoint.PRE_LLM_CALL) == 2
    assert kernel.hook_count(HookPoint.PRE_TOOL_USE) == 0


def test_is_locked_property() -> None:
    kernel = GovernanceKernel()
    assert kernel.is_locked is False
    kernel.init_lock()
    assert kernel.is_locked is True


async def test_audit_stamps_active_session_id(tmp_path) -> None:
    """Audit rows carry the active session id so traces can correlate hooks.

    Every pipeline stage stamps its own run_id, so the kernel records the
    session_id from the session_scope contextvar (see GovernanceKernel._audit).
    """
    audit = AuditLog(db_path=tmp_path / "audit.db")
    kernel = GovernanceKernel(audit_log=audit)
    kernel.register(_AllowHook())
    kernel.init_lock()

    with session_scope("sess-stamp"):
        await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())

    rows = audit.query()
    assert rows, "expected an audit row"
    payload = json.loads(rows[-1].payload_json)
    assert payload.get("session_id") == "sess-stamp"
    # run_id is still the stage's own id, not the session — that's why we stamp.
    assert rows[-1].run_id == "run-1"


async def test_audit_stamps_the_caller_from_metadata(tmp_path) -> None:
    """A tool call's caller rides in metadata; every row of the call records it.

    Before, only the hooks that copied it into ``audit_metadata`` (caller_policy,
    mcp_client_egress) recorded who made the call, so the other rows of an MCP client's
    call said nothing about it and a caller filter missed them.
    """
    audit = AuditLog(db_path=tmp_path / "audit.db")
    kernel = GovernanceKernel(audit_log=audit)
    kernel.register(_AllowHook())
    kernel.init_lock()

    ctx = _ctx().model_copy(update={"metadata": {"caller": "mcp:desktop"}})
    await kernel.fire(HookPoint.PRE_LLM_CALL, ctx)
    await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())

    first, second = audit.query()
    assert json.loads(first.payload_json)["caller"] == "mcp:desktop"
    assert "caller" not in json.loads(second.payload_json)
    assert [r.id for r in audit.query(caller="mcp:")] == [first.id]
    assert [r.id for r in audit.query(caller="mcp:desktop")] == [first.id]
    assert audit.query(caller="mcp:desk") == ()
    assert audit.callers() == ("mcp:desktop",)
