"""Governance adversarial scenarios — deliberate policy violations.

Fires crafted contexts through the real ``kernel_from_env()`` kernel
(requires ``IRIS_GOVERNANCE_ENABLED=1``; the runner sets it) and checks the
data-classification × tier policy matrix from the egress gate:

    secret   → max tier_1 (deny above, severity critical)
    personal → max tier_2 (tier_3 requires approval)
    internal → tier_3 allowed with warn
    unclassified at tier_3 → deny (fail-closed)

Plus one end-to-end case: a real ``CodingLLMClient.invoke`` against the
tier-3 config with a secret-bearing prompt must raise before any model call.

NOTE: these runs write real rows into the governance audit log — that is
intentional; the audit rows ARE the proof artifact.
"""

from __future__ import annotations

import uuid
from typing import Any

from scenarios.common import REPO_ROOT, ScenarioRecord, Timer

# Deliberately fake but pattern-valid markers from the classifier regex packs.
FAKE_OPENAI_KEY = "sk-FAKE000000000000000000000000abcd"  # noqa: S105 - synthetic
FAKE_SSN = "123-45-6789"
VAULT_HANDLE = "vault://gmail/scenario-account"

CASES: list[dict[str, Any]] = [
    {
        "id": "gov-secret-t3-deny",
        "prompt": f"please summarize this config, api key is {FAKE_OPENAI_KEY}",
        "tier": "tier_3",
        "expect_outcome": "deny",
        "expect_classification": "secret",
    },
    {
        "id": "gov-secret-t2-deny",
        "prompt": f"store this for me: {FAKE_OPENAI_KEY}",
        "tier": "tier_2",
        "expect_outcome": "deny",
        "expect_classification": "secret",
    },
    {
        "id": "gov-personal-t3-approval",
        "prompt": f"my ssn is {FAKE_SSN}, fill in the form draft",
        "tier": "tier_3",
        "expect_outcome": "require_approval",
        "expect_classification": "personal",
    },
    {
        "id": "gov-personal-t2-allow",
        "prompt": f"my ssn is {FAKE_SSN}, fill in the form draft",
        "tier": "tier_2",
        "expect_outcome": "allow",
        "expect_classification": "personal",
    },
    {
        "id": "gov-internal-t3-allow",
        "prompt": f"use {VAULT_HANDLE} to authenticate the sync",
        "tier": "tier_3",
        "expect_outcome": "allow",
        "expect_classification": "internal",
    },
    {
        "id": "gov-public-t3-allow",
        "prompt": "write a short poem about mountains",
        "tier": "tier_3",
        "expect_outcome": "allow",
        "expect_classification": "public",
    },
]


def _fire_matrix_case(kernel: Any, case: dict[str, Any]) -> tuple[str, str | None]:
    """PRE_CLASSIFY then PRE_LLM_CALL, mirroring the production client flow."""
    from iris_harness.governance import HookContext, HookPoint

    run_id = f"scenario-{uuid.uuid4()}"
    classify_ctx = HookContext(
        hook_point=HookPoint.PRE_CLASSIFY,
        run_id=run_id,
        agent_type="scenario",
        payload={"prompt": case["prompt"]},
    )
    _, classified_ctx = kernel.fire_sync(HookPoint.PRE_CLASSIFY, classify_ctx)

    llm_ctx = HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id=run_id,
        agent_type="scenario",
        classification=classified_ctx.classification,
        tier=case["tier"],
        payload={"prompt": case["prompt"]},
    )
    decision, _ = kernel.fire_sync(HookPoint.PRE_LLM_CALL, llm_ctx)
    return decision.outcome, classified_ctx.classification


def _fail_closed_case(kernel: Any) -> str:
    """PRE_LLM_CALL at tier_3 with NO classification — must deny."""
    from iris_harness.governance import HookContext, HookPoint

    ctx = HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id=f"scenario-{uuid.uuid4()}",
        agent_type="scenario",
        classification=None,
        tier="tier_3",
        payload={"prompt": "unclassified text straight to cloud tier"},
    )
    decision, _ = kernel.fire_sync(HookPoint.PRE_LLM_CALL, ctx)
    return decision.outcome


def _end_to_end_client_block() -> tuple[bool, str]:
    """Secret prompt through a real tier-3 client: must raise pre-model."""
    from iris_harness.llm.client import CodingLLMClient, CodingLLMInvocationError
    from iris_harness.llm.tier_router import TierRouter

    router = TierRouter.load_from_yaml(REPO_ROOT / "config" / "llm_tiers.yaml")
    client = CodingLLMClient(router.get_llm_config("skill_writing"))  # tier3
    try:
        client.invoke(
            system_prompt="",
            user_prompt=f"summarize: {FAKE_OPENAI_KEY}",
        )
    except CodingLLMInvocationError as exc:
        return True, str(exc)
    return False, "invoke completed without governance block"


def run() -> list[ScenarioRecord]:
    from iris_harness.governance import kernel_from_env

    kernel = kernel_from_env()
    if kernel is None:
        return [
            ScenarioRecord(
                scenario="governance",
                case_id="kernel-missing",
                verdict="fail",
                expected="kernel from IRIS_GOVERNANCE_ENABLED=1",
                actual="kernel_from_env() returned None",
                latency_ms=0.0,
            )
        ]

    records: list[ScenarioRecord] = []
    for case in CASES:
        with Timer() as timer:
            outcome, classification = _fire_matrix_case(kernel, case)
        ok = outcome == case["expect_outcome"] and classification == case["expect_classification"]
        records.append(
            ScenarioRecord(
                scenario="governance",
                case_id=case["id"],
                verdict="pass" if ok else "fail",
                expected=f"{case['expect_classification']}→{case['tier']}={case['expect_outcome']}",
                actual=f"{classification}→{case['tier']}={outcome}",
                latency_ms=round(timer.elapsed_ms, 2),
                detail={"tier": case["tier"]},
            )
        )

    with Timer() as timer:
        outcome = _fail_closed_case(kernel)
    records.append(
        ScenarioRecord(
            scenario="governance",
            case_id="gov-unclassified-t3-failclosed",
            verdict="pass" if outcome == "deny" else "fail",
            expected="None→tier_3=deny (fail-closed)",
            actual=f"None→tier_3={outcome}",
            latency_ms=round(timer.elapsed_ms, 2),
        )
    )

    with Timer() as timer:
        blocked, message = _end_to_end_client_block()
    records.append(
        ScenarioRecord(
            scenario="governance",
            case_id="gov-e2e-client-secret-block",
            verdict="pass" if blocked else "fail",
            expected="CodingLLMInvocationError before model call",
            actual=message[:160],
            latency_ms=round(timer.elapsed_ms, 2),
        )
    )
    return records
