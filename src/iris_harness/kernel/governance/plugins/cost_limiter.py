"""CostLimiter — ``PreLLMCall`` plugin that enforces a per-user daily cap.

Story 12.gov-3.7 / design §9.1 ``cost_budget``. The plugin runs at
``PreLLMCall`` priority 40 (after the egress gate at 30) so it never
denies a call the egress gate would have denied anyway — keeps the
audit trail aligned with the privacy promise's primary control.

Algorithm:

1. Look at ``ctx.tier`` (set by the caller) and ``ctx.payload["prompt"]``.
2. Estimate prompt tokens (``len(prompt) // 4``) and completion tokens
   (caller-supplied ``max_tokens`` or a sane fallback).
3. Look up the tier's pricing; estimate cost.
4. Sum the user's today-spend from the ledger.
5. If ``today + estimate > cap`` → ``deny`` with severity ``critical``.
6. Otherwise → ``allow`` AND write the estimated cost to the ledger so
   subsequent calls in the same day see it. Local tiers (tier_1/2)
   contribute zero by design.

If a follow-up wires actual usage reconciliation, it can simply
``UPDATE`` the row by ``id`` post-call. For Phase 3 v1 we accept the
estimate as ground truth — it's strictly conservative (uses
``max_tokens`` for completion) so the cap can't be silently breached.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from iris_harness.kernel.governance.cost.pricing import (
    TierPricing,
    estimate_cost_usd,
    estimate_tokens,
)
from iris_harness.kernel.governance.cost.store import CostStore
from iris_harness.kernel.governance.hooks.types import (
    HookContext,
    HookDecision,
    HookPoint,
)

logger = logging.getLogger(__name__)

# Conservative default for completion tokens when the caller hasn't
# supplied one. Tier configs typically cap completions at 4096 — the
# CostLimiter assumes the worst-case completion to keep the cap strict.
_DEFAULT_MAX_COMPLETION_TOKENS = 4096


class CostLimiter:
    """``PreLLMCall`` hook enforcing a per-user daily USD cap."""

    name: str = "cost_limiter"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 40  # after egress_gate (30) and redaction (20)

    def __init__(
        self,
        *,
        store: CostStore,
        daily_cap_usd: float,
        user_id: str = "local",
        enforce: bool = True,
        pricing: Mapping[str, TierPricing] | None = None,
        default_max_completion_tokens: int = _DEFAULT_MAX_COMPLETION_TOKENS,
    ) -> None:
        if daily_cap_usd < 0:
            raise ValueError("daily_cap_usd must be >= 0")
        if default_max_completion_tokens < 0:
            raise ValueError("default_max_completion_tokens must be >= 0")
        self._store = store
        self._cap = float(daily_cap_usd)
        # Recording and enforcing are separate concerns: the ledger is a
        # mechanism, the cap is a policy. With ``enforce=False`` every call is
        # still priced and written, and none is refused — which is what lets an
        # operator watch spend for a while before deciding what cap to set.
        self._enforce = enforce
        self._user_id = user_id
        self._pricing = pricing
        self._default_max_completion_tokens = default_max_completion_tokens

    async def __call__(self, ctx: HookContext) -> HookDecision:
        prompt = _str_payload(ctx.payload.get("prompt"))
        tier = ctx.tier
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = _completion_token_estimate(
            ctx.payload, default=self._default_max_completion_tokens
        )
        estimated = estimate_cost_usd(
            tier=tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            pricing=self._pricing,
        )

        # No-cost calls (local tiers) bypass the budget check entirely
        # but still record a zero-cost ledger row for AC-5 visibility.
        if estimated <= 0.0:
            self._record(
                ctx,
                tier=tier,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=0.0,
            )
            return HookDecision(
                outcome="allow",
                reason=f"cost_limiter: tier={tier} estimated cost $0.00 (local)",
                audit_metadata=_audit(
                    tier=tier,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated=0.0,
                    today_spend=self._store.sum_today(user_id=self._user_id),
                    cap=self._cap,
                    over_cap=False,
                    enforced=self._enforce,
                ),
            )

        today_spend = self._store.sum_today(user_id=self._user_id)
        projected = today_spend + estimated
        # cap == 0 means "no paid calls allowed" — only the zero-cost
        # tiers (handled above) get through. Any non-zero estimate
        # trips the cap. Operators who want the limiter fully disabled
        # should leave it unwired rather than passing cap=0.
        over_cap = projected > self._cap
        if over_cap and self._enforce:
            return HookDecision(
                outcome="deny",
                reason=(
                    f"cost_limiter: projected ${projected:.4f} exceeds daily cap "
                    f"${self._cap:.2f} (today=${today_spend:.4f} + "
                    f"estimate=${estimated:.4f})"
                ),
                severity="critical",
                audit_metadata=_audit(
                    tier=tier,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated=estimated,
                    today_spend=today_spend,
                    cap=self._cap,
                    over_cap=True,
                    enforced=True,
                ),
            )

        # Allow + record. The estimate is a conservative upper bound
        # (uses ``max_tokens`` for completion); see module docstring.
        self._record(
            ctx,
            tier=tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=estimated,
        )
        reason = (
            f"cost_limiter: tier={tier} estimate=${estimated:.4f} "
            f"today=${today_spend:.4f} cap=${self._cap:.2f}"
        )
        if over_cap:
            # Allowed, but the ledger must not look like a normal day: this is
            # the line an operator greps for after a surprising bill.
            reason = f"{reason} — OVER CAP, not enforced"
        return HookDecision(
            outcome="allow",
            reason=reason,
            audit_metadata=_audit(
                tier=tier,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated=estimated,
                today_spend=today_spend,
                cap=self._cap,
                over_cap=over_cap,
                enforced=self._enforce,
            ),
        )

    def _record(
        self,
        ctx: HookContext,
        *,
        tier: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        try:
            self._store.record(
                run_id=ctx.run_id,
                agent_type=ctx.agent_type,
                tier=tier or "unknown",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                user_id=self._user_id,
            )
        except Exception as exc:  # noqa: BLE001 - ledger write must not break enforcement
            logger.warning("cost_limiter: ledger write failed (%s); allowing call", exc)


def _str_payload(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _completion_token_estimate(payload: dict[str, object], *, default: int) -> int:
    """Pull ``max_tokens`` or ``completion_tokens`` out of the payload."""
    for key in ("max_tokens", "completion_tokens", "max_completion_tokens"):
        value = payload.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return default


def _audit(
    *,
    tier: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    estimated: float,
    today_spend: float,
    cap: float,
    over_cap: bool = False,
    enforced: bool = True,
) -> dict[str, object]:
    return {
        "tier": tier,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": estimated,
        "today_spend_usd": today_spend,
        "daily_cap_usd": cap,
        "over_cap": over_cap,
        "enforced": enforced,
    }
