"""EgressGate — enforces class→tier policy at ``PreLLMCall``.

This is the runtime enforcement of the privacy promise. The classifier
annotates the context with a ``classification``; the caller sets
``ctx.tier`` to the LLM tier it's about to call; this hook decides.

The tiers mean where the model runs (``llm/locality.py``): ``tier_1`` and
``tier_2`` are models on the owner's machines, ``tier_3`` leaves them. The
class→tier table is config, ``config/governance/egress.yaml``
(``kernel/governance/egress_policy.py``; design §6.3). As shipped:

==========  =========  =================================
class       max tier   on violation (target > max)
==========  =========  =================================
secret      tier_2     hard deny (critical) -- local only
personal    tier_2     require_approval (warn)
internal    tier_3     n/a — never exceeds its max
public      tier_3     n/a — never exceeds its max
==========  =========  =================================

``secret`` never reaching ``tier_3``, and always as a hard deny, is enforced
by the policy loader: a table that loosens it does not load.

Additional rules:

- ``internal`` at ``tier_3``: allow with ``warn`` severity. The
  RedactionFilter plugin (separate commit) replaces this with a
  ``transform`` once it lands.
- ``classification is None``: fail-closed for cloud (``tier_3``),
  allow for local. A correctly-wired IntentRouter sets classification
  before this fires; ``None`` indicates a wiring bug.
- ``tier is None``: hard deny — caller bug.
- ``trusted_cloud_for`` (§6.4): bypass approval for listed classes on
  cloud egress; ``secret`` is permanently rejected from this list.
"""

from __future__ import annotations

import logging

from iris_harness.kernel.governance.egress_policy import EgressPolicy, load_egress_policy
from iris_harness.kernel.governance.hooks.types import (
    DataClassification,
    HookContext,
    HookDecision,
    HookPoint,
    LLMTier,
)

logger = logging.getLogger(__name__)


class EgressGate:
    """``PreLLMCall`` hook enforcing class→tier policy."""

    name: str = "egress_gate"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 30  # after classifier (10) and future redaction (20)

    def __init__(
        self,
        *,
        trusted_cloud_for: frozenset[DataClassification] = frozenset(),
        policy: EgressPolicy | None = None,
    ) -> None:
        if "secret" in trusted_cloud_for:
            raise ValueError(
                "'secret' classification cannot be in trusted_cloud_for; "
                "secret data must never leave the owner's machines."
            )
        self._trusted_cloud_for = trusted_cloud_for
        self._policy = policy if policy is not None else load_egress_policy()

    async def __call__(self, ctx: HookContext) -> HookDecision:
        if ctx.tier is None:
            return HookDecision(
                outcome="deny",
                reason="egress_gate: target tier not set on PreLLMCall context",
                severity="error",
                audit_metadata={"missing": "tier"},
            )

        target_tier: LLMTier = ctx.tier

        if ctx.classification is None:
            if target_tier == "tier_3":
                return HookDecision(
                    outcome="deny",
                    reason="egress_gate: unclassified prompt to cloud (fail-closed)",
                    severity="error",
                    audit_metadata={"target_tier": target_tier},
                )
            return HookDecision(
                outcome="allow",
                reason=f"egress_gate: unclassified prompt to {target_tier} permitted (local)",
                severity="warn",
                audit_metadata={"target_tier": target_tier},
            )

        classification: DataClassification = ctx.classification

        if not self._policy.allows(classification, target_tier):
            return self._handle_violation(classification, target_tier)

        if classification == "internal" and target_tier == "tier_3":
            return HookDecision(
                outcome="allow",
                reason="egress_gate: internal data to cloud — redaction recommended",
                severity="warn",
                audit_metadata={
                    "classification": classification,
                    "target_tier": target_tier,
                    "would_redact": True,
                },
            )

        return HookDecision(
            outcome="allow",
            reason=f"egress_gate: {classification} -> {target_tier} permitted",
            audit_metadata={
                "classification": classification,
                "target_tier": target_tier,
            },
        )

    def _handle_violation(
        self,
        classification: DataClassification,
        target_tier: LLMTier,
    ) -> HookDecision:
        rule = self._policy.rule(classification)
        metadata = {
            "classification": classification,
            "target_tier": target_tier,
            "max_tier": rule.max_tier,
        }

        if rule.on_violation == "require_approval":
            if target_tier == "tier_3" and classification in self._trusted_cloud_for:
                logger.warning(
                    "egress_gate: %s->cloud bypass via trusted_cloud_for", classification
                )
                return HookDecision(
                    outcome="allow",
                    reason=(
                        f"egress_gate: {classification} data to trusted cloud (approval bypassed)"
                    ),
                    severity="warn",
                    audit_metadata={**metadata, "bypass": "trusted_cloud_for"},
                )
            return HookDecision(
                outcome="require_approval",
                reason=(
                    f"egress_gate: {classification} data to {target_tier} requires user approval"
                ),
                severity="warn",
                audit_metadata=metadata,
            )

        if target_tier == "tier_3":
            reason = (
                f"egress_gate: {classification} data never leaves the owner's machines "
                f"(target={target_tier})"
            )
        else:
            reason = (
                f"egress_gate: {classification} data may reach at most {rule.max_tier} "
                f"(target={target_tier})"
            )
        return HookDecision(
            outcome="deny",
            reason=reason,
            severity="critical",
            audit_metadata=metadata,
        )
