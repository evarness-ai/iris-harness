"""Tests for the CostLimiter PreLLMCall plugin (story 12.gov-3.7 / AC-2, AC-5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.kernel.governance.cost import CostStore
from iris_harness.kernel.governance.hooks.types import HookContext, HookPoint
from iris_harness.kernel.governance.plugins import CostLimiter


@pytest.fixture()
def store(tmp_path: Path) -> CostStore:
    return CostStore(db_path=tmp_path / "cost.db")


def _ctx(*, tier: str | None = "tier_3", prompt: str = "x" * 400) -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="r1",
        agent_type="chat",
        tier=tier,
        payload={"prompt": prompt, "max_tokens": 100},
    )


async def test_under_budget_allows_and_records(store: CostStore) -> None:
    limiter = CostLimiter(store=store, daily_cap_usd=100.00)
    decision = await limiter(_ctx())
    assert decision.outcome == "allow"
    assert decision.audit_metadata["estimated_cost_usd"] > 0
    assert decision.audit_metadata["today_spend_usd"] == 0.0
    assert store.count() == 1
    # The recorded row matches the audit-metadata estimate.
    recorded = store.sum_today(user_id="local")
    assert recorded == pytest.approx(decision.audit_metadata["estimated_cost_usd"])


async def test_over_budget_denies_critical_and_does_not_record(store: CostStore) -> None:
    """AC-2: at >= 100% of cap, deny PreLLMCall with severity critical."""
    # Pre-populate ledger at the cap. Any non-zero estimate (every
    # tier_3 call) on top will push us over.
    store.record(
        run_id="prior",
        agent_type="chat",
        tier="tier_3",
        prompt_tokens=10_000,
        completion_tokens=10_000,
        cost_usd=5.00,
        user_id="local",
    )
    limiter = CostLimiter(store=store, daily_cap_usd=5.00)

    decision = await limiter(_ctx())
    assert decision.outcome == "deny"
    assert decision.severity == "critical"
    assert "exceeds daily cap" in decision.reason
    # No new row written on deny — count stays at the pre-populated one.
    assert store.count() == 1


async def test_local_tier_1_records_zero_cost(store: CostStore) -> None:
    """AC-5: tier_1 is free."""
    limiter = CostLimiter(store=store, daily_cap_usd=5.00)
    decision = await limiter(_ctx(tier="tier_1"))
    assert decision.outcome == "allow"
    assert decision.audit_metadata["estimated_cost_usd"] == 0.0
    assert store.sum_today(user_id="local") == 0.0


async def test_local_tier_2_records_zero_cost(store: CostStore) -> None:
    limiter = CostLimiter(store=store, daily_cap_usd=5.00)
    decision = await limiter(_ctx(tier="tier_2"))
    assert decision.outcome == "allow"
    assert decision.audit_metadata["estimated_cost_usd"] == 0.0


async def test_local_tier_bypasses_cap_even_when_exhausted(store: CostStore) -> None:
    """A user past their cap on tier_3 can still use local tier_1/2."""
    store.record(
        run_id="prior",
        agent_type="chat",
        tier="tier_3",
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=999.0,
        user_id="local",
    )
    limiter = CostLimiter(store=store, daily_cap_usd=5.00)

    decision = await limiter(_ctx(tier="tier_1"))
    assert decision.outcome == "allow"


# AC-4 (unknown tier doesn't silently treat as free) is covered at the
# pricing layer in test_cost_pricing.py — HookContext.tier is a Literal,
# so the limiter never actually sees an unknown string at this boundary.


async def test_zero_cap_blocks_any_paid_call(store: CostStore) -> None:
    """A cap of $0 means cloud is effectively disabled."""
    limiter = CostLimiter(store=store, daily_cap_usd=0.0)
    decision = await limiter(_ctx(tier="tier_3"))
    assert decision.outcome == "deny"
    assert decision.severity == "critical"


async def test_per_user_isolation(store: CostStore) -> None:
    """Another user's spend must NOT count against this user's cap."""
    store.record(
        run_id="other",
        agent_type="chat",
        tier="tier_3",
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=999.0,
        user_id="someone_else",
    )
    limiter = CostLimiter(store=store, daily_cap_usd=5.00, user_id="local")
    decision = await limiter(_ctx())
    assert decision.outcome == "allow"


async def test_payload_max_tokens_drives_completion_estimate(store: CostStore) -> None:
    """A bigger max_tokens → larger estimate, even with the same prompt."""
    small = HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="r",
        agent_type="chat",
        tier="tier_3",
        payload={"prompt": "x", "max_tokens": 10},
    )
    big = HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="r",
        agent_type="chat",
        tier="tier_3",
        payload={"prompt": "x", "max_tokens": 10_000},
    )

    limiter = CostLimiter(store=store, daily_cap_usd=100.00)
    a = await limiter(small)
    b = await limiter(big)
    assert b.audit_metadata["estimated_cost_usd"] > a.audit_metadata["estimated_cost_usd"]


def test_invalid_cap_rejected(store: CostStore) -> None:
    with pytest.raises(ValueError):
        CostLimiter(store=store, daily_cap_usd=-1.0)


# ── enforce=False: record the spend, refuse nothing ────────────────────────
#
# The owner's call (2026-09-20): keep enforcement, but put it behind its own
# flag, so the cloud harness can start counting before anyone picks a cap.
# The ledger is a mechanism; the cap is a policy.


def _at_the_cap(store: CostStore) -> None:
    store.record(
        run_id="prior",
        agent_type="chat",
        tier="tier_3",
        prompt_tokens=10_000,
        completion_tokens=10_000,
        cost_usd=5.00,
        user_id="local",
    )


async def test_not_enforcing_allows_the_call_that_would_have_been_denied(
    store: CostStore,
) -> None:
    _at_the_cap(store)
    limiter = CostLimiter(store=store, daily_cap_usd=5.00, enforce=False)

    decision = await limiter(_ctx())
    assert decision.outcome == "allow"


async def test_not_enforcing_still_records_the_over_cap_spend(store: CostStore) -> None:
    """The point of the flag. Denying does not record, because the call never
    happens; allowing must, or the ledger under-reports exactly the spend the
    owner turned recording on to see."""
    _at_the_cap(store)
    limiter = CostLimiter(store=store, daily_cap_usd=5.00, enforce=False)

    await limiter(_ctx())
    assert store.count() == 2
    assert store.sum_today(user_id="local") > 5.00


async def test_an_unenforced_breach_is_visible_in_the_audit_trail(store: CostStore) -> None:
    # Allowed, but it must not read like a normal day: this is the line to
    # grep for after a surprising bill.
    _at_the_cap(store)
    limiter = CostLimiter(store=store, daily_cap_usd=5.00, enforce=False)

    decision = await limiter(_ctx())
    assert "OVER CAP, not enforced" in decision.reason
    assert decision.audit_metadata["over_cap"] is True
    assert decision.audit_metadata["enforced"] is False


async def test_enforcing_is_the_default(store: CostStore) -> None:
    # Constructed without the argument, behaviour is exactly as before.
    _at_the_cap(store)
    limiter = CostLimiter(store=store, daily_cap_usd=5.00)

    decision = await limiter(_ctx())
    assert decision.outcome == "deny"
    assert decision.audit_metadata["enforced"] is True


async def test_under_cap_is_unremarkable_either_way(store: CostStore) -> None:
    for enforce in (True, False):
        fresh = CostStore(db_path=store.db_path.parent / f"c-{enforce}.db")
        decision = await CostLimiter(store=fresh, daily_cap_usd=5.00, enforce=enforce)(_ctx())
        assert decision.outcome == "allow"
        assert decision.audit_metadata["over_cap"] is False
        assert "OVER CAP" not in decision.reason
