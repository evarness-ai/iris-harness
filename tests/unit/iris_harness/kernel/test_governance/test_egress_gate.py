"""EgressGate tests — full coverage of the §6.3 policy matrix.

The matrix is (classification × target_tier) × outcome, on the shipped
``config/governance/egress.yaml``. tier_1/tier_2 run on the owner's machines,
tier_3 leaves them; secret is local only (owner, 2026-09-30):

- secret    @ tier_1  → allow
- secret    @ tier_2  → allow
- secret    @ tier_3  → deny (critical)
- personal  @ tier_1  → allow
- personal  @ tier_2  → allow
- personal  @ tier_3  → require_approval (or allow if trusted_cloud_for)
- internal  @ tier_1  → allow
- internal  @ tier_2  → allow
- internal  @ tier_3  → allow (warn — redaction recommended)
- public    @ {all}   → allow
- None      @ tier_3  → deny (fail-closed)
- None      @ tier_1/2 → allow (warn)
- tier None → deny (caller bug)
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance import (
    DataClassification,
    GovernanceKernel,
    HookContext,
    HookPoint,
    LLMTier,
)
from iris_harness.kernel.governance.plugins import EgressGate


def _ctx(
    classification: DataClassification | None,
    tier: LLMTier | None,
) -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="run-egress",
        agent_type="chat",
        classification=classification,
        tier=tier,
        payload={"prompt": "hello"},
    )


@pytest.fixture()
def gate() -> EgressGate:
    return EgressGate()


@pytest.mark.parametrize(
    "tier",
    ["tier_1", "tier_2", "tier_3"],
)
async def test_public_always_allowed(gate: EgressGate, tier: LLMTier) -> None:
    decision = await gate(_ctx("public", tier))
    assert decision.outcome == "allow"


@pytest.mark.parametrize("tier", ["tier_1", "tier_2"])
async def test_secret_to_any_local_tier_allowed(gate: EgressGate, tier: LLMTier) -> None:
    decision = await gate(_ctx("secret", tier))
    assert decision.outcome == "allow"
    assert decision.severity == "info"


async def test_secret_to_tier_3_denied_critical(gate: EgressGate) -> None:
    decision = await gate(_ctx("secret", "tier_3"))
    assert decision.outcome == "deny"
    assert decision.severity == "critical"
    assert decision.reason == (
        "egress_gate: secret data never leaves the owner's machines (target=tier_3)"
    )
    assert decision.audit_metadata["max_tier"] == "tier_2"


@pytest.mark.parametrize("tier", ["tier_1", "tier_2"])
async def test_personal_to_local_allowed(gate: EgressGate, tier: LLMTier) -> None:
    decision = await gate(_ctx("personal", tier))
    assert decision.outcome == "allow"


async def test_personal_to_tier_3_requires_approval(gate: EgressGate) -> None:
    decision = await gate(_ctx("personal", "tier_3"))
    assert decision.outcome == "require_approval"
    assert decision.severity == "warn"
    assert "personal" in decision.reason
    assert decision.audit_metadata["classification"] == "personal"
    assert decision.audit_metadata["target_tier"] == "tier_3"


async def test_personal_to_tier_3_with_trusted_cloud_bypass() -> None:
    gate = EgressGate(trusted_cloud_for=frozenset({"personal"}))
    decision = await gate(_ctx("personal", "tier_3"))
    assert decision.outcome == "allow"
    assert decision.severity == "warn"
    assert decision.audit_metadata["bypass"] == "trusted_cloud_for"


def test_trusted_cloud_for_rejects_secret() -> None:
    """Defense-in-depth: secret can never be in trusted_cloud_for."""
    with pytest.raises(ValueError, match="secret"):
        EgressGate(trusted_cloud_for=frozenset({"secret"}))


@pytest.mark.parametrize("tier", ["tier_1", "tier_2"])
async def test_internal_to_local_allowed_no_warn(gate: EgressGate, tier: LLMTier) -> None:
    decision = await gate(_ctx("internal", tier))
    assert decision.outcome == "allow"
    assert decision.severity == "info"


async def test_internal_to_tier_3_allowed_with_warn_and_redact_hint(
    gate: EgressGate,
) -> None:
    decision = await gate(_ctx("internal", "tier_3"))
    assert decision.outcome == "allow"
    assert decision.severity == "warn"
    assert decision.audit_metadata.get("would_redact") is True


async def test_unclassified_to_cloud_fails_closed(gate: EgressGate) -> None:
    decision = await gate(_ctx(None, "tier_3"))
    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert "unclassified" in decision.reason


@pytest.mark.parametrize("tier", ["tier_1", "tier_2"])
async def test_unclassified_local_allowed_with_warn(gate: EgressGate, tier: LLMTier) -> None:
    decision = await gate(_ctx(None, tier))
    assert decision.outcome == "allow"
    assert decision.severity == "warn"


async def test_missing_tier_fails_closed(gate: EgressGate) -> None:
    decision = await gate(_ctx("public", None))
    assert decision.outcome == "deny"
    assert decision.severity == "error"
    assert "tier" in decision.reason


# --- Integration: classifier + egress gate through the kernel -----------------


async def test_classifier_then_egress_gate_pipeline_blocks_secret_to_cloud() -> None:
    """End-to-end: a prompt with an API key, targeted at cloud, gets denied."""
    from iris_harness.kernel.governance.plugins import DataClassifierHook

    kernel = GovernanceKernel()
    kernel.register(DataClassifierHook())
    kernel.register(EgressGate())
    kernel.init_lock()

    # PreClassify: classify the inbound prompt
    classify_ctx = HookContext(
        hook_point=HookPoint.PRE_CLASSIFY,
        run_id="run-e2e",
        agent_type="chat",
        payload={"prompt": "here is my key sk-ABC1234567890abcdef1234"},
    )
    _, classified_ctx = await kernel.fire(HookPoint.PRE_CLASSIFY, classify_ctx)
    assert classified_ctx.classification == "secret"

    # PreLLMCall: caller targets tier_3 (cloud)
    llm_ctx = HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="run-e2e",
        agent_type="chat",
        classification=classified_ctx.classification,
        tier="tier_3",
        payload=classify_ctx.payload,
    )
    decision, _ = await kernel.fire(HookPoint.PRE_LLM_CALL, llm_ctx)
    assert decision.outcome == "deny"
    assert decision.severity == "critical"


async def test_classifier_then_egress_gate_pipeline_asks_personal_to_cloud() -> None:
    from iris_harness.kernel.governance.plugins import DataClassifierHook

    kernel = GovernanceKernel()
    kernel.register(DataClassifierHook())
    kernel.register(EgressGate())
    kernel.init_lock()

    classify_ctx = HookContext(
        hook_point=HookPoint.PRE_CLASSIFY,
        run_id="run-e2e-personal",
        agent_type="chat",
        payload={"prompt": "reach me at alice@example.com please"},
    )
    _, classified_ctx = await kernel.fire(HookPoint.PRE_CLASSIFY, classify_ctx)
    assert classified_ctx.classification == "personal"

    llm_ctx = HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="run-e2e-personal",
        agent_type="chat",
        classification=classified_ctx.classification,
        tier="tier_3",
        payload=classify_ctx.payload,
    )
    decision, _ = await kernel.fire(HookPoint.PRE_LLM_CALL, llm_ctx)
    assert decision.outcome == "require_approval"


async def test_classifier_then_egress_gate_pipeline_allows_public_to_cloud() -> None:
    from iris_harness.kernel.governance.plugins import DataClassifierHook

    kernel = GovernanceKernel()
    kernel.register(DataClassifierHook())
    kernel.register(EgressGate())
    kernel.init_lock()

    classify_ctx = HookContext(
        hook_point=HookPoint.PRE_CLASSIFY,
        run_id="run-e2e-public",
        agent_type="chat",
        payload={"prompt": "explain how transformers work"},
    )
    _, classified_ctx = await kernel.fire(HookPoint.PRE_CLASSIFY, classify_ctx)
    assert classified_ctx.classification == "public"

    llm_ctx = HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="run-e2e-public",
        agent_type="chat",
        classification=classified_ctx.classification,
        tier="tier_3",
        payload=classify_ctx.payload,
    )
    decision, _ = await kernel.fire(HookPoint.PRE_LLM_CALL, llm_ctx)
    assert decision.outcome == "allow"
