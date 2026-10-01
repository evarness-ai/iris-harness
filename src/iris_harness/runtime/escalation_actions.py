"""The escalation action loop, carved out of ``IrisRuntime`` (OSS plan M5.7 track C).

ADR-0068 L3: when the escalation judge runs in enforce mode, act on its verdict —
**escalate** to a stronger tier, **reroute** at the same tier with a grounding
directive, or **clarify** by handing the judge's question back to the user. Bounded
by ``max_escalations`` (D3), the local-only egress guard (D4) and the resource
governor's veto (D6).

This is the first slice of the carve, and it was chosen because the block already
had a seam: its test drove the methods against a stand-in ``self`` carrying exactly
four runtime members. :class:`EscalationHost` declares those four, so mypy checks
that the runtime still supplies them — the same enforcement ``TurnHost`` gave the
turn pipeline.

The host is read **at call time**, not captured at construction, so a collaborator
swapped on the runtime after this object exists is still the one used — the same
resolution the methods had while they lived on the class. The egress classifier
cache moved here with the code that owns it; nothing else on the runtime read it.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from iris_harness.agent.escalation import (
    EscalationConfig,
    egress_eligible,
    more_restrictive,
)
from iris_harness.foundation.observability.session_log import current_session_id, current_turn_id
from iris_harness.foundation.observability.tracer import current_trace_ids
from iris_harness.llm.locality import provider_locality
from iris_harness.runtime.turn_capture import (
    _coerce_total_tokens,
    _extract_escalation_verdict,
)

if TYPE_CHECKING:
    from iris_harness.agent.agent_executor import AgentResult
    from iris_harness.agent.intent_router import IntentResult
    from iris_harness.agent.response_curator import CuratedResponse, ResponseCurator
    from iris_harness.agent.task_planner import TaskPlan
    from iris_harness.llm.tier_router import TierRouter
    from iris_harness.memory.retriever import MemoryContext
    from iris_harness.services.learning.signals import LearningSignalCollector

logger = logging.getLogger(__name__)


def _why_not_rerunnable(results: list[AgentResult]) -> str | None:
    """Why acting on a verdict could repeat or hide a change, or None when it cannot.

    ADR-0118 decision 5. Every action replaces the answer: escalate and reroute re-run
    the plan, clarify swaps the answer for a question. After a write, a re-run can
    repeat it and a question hides what was done. Fail-closed: a result that does
    not REPORT ``effects_executed`` is treated as unknown, because a plugin handler
    can write outside the loop, and a loop that wrote and then fell back to a
    deterministic answer loses its report.
    """
    for result in results:
        meta = result.metadata
        if meta.get("resumed"):
            return f"{result.agent_type} continued a paused run"
        if meta.get("pending_approval_id"):
            # Re-running would raise a second approval for the same request (ADR-0118).
            return f"{result.agent_type} is waiting on approval {meta['pending_approval_id']}"
        effects = meta.get("effects_executed")
        if not isinstance(effects, list | tuple):
            return f"{result.agent_type} did not report what it changed"
        if effects:
            return f"{result.agent_type} executed {', '.join(sorted(set(map(str, effects))))}"
    return None


class EscalationHost(Protocol):
    """The four runtime members the action loop reaches.

    ``_execute_plan`` is declared as-is, underscore included, for the reason
    ``TurnHost`` gives: renaming a member is its own change, not a rider on the one
    that makes the rename checkable.
    """

    tier_router: TierRouter
    response_curator: ResponseCurator
    signal_collector: LearningSignalCollector

    def _execute_plan(
        self,
        plan: TaskPlan,
        intent: IntentResult,
        memory_ctx: MemoryContext,
        session_id: str,
        *,
        preferred_model: str | None = None,
        provider_profile: str | None = None,
    ) -> list[AgentResult]: ...


class EscalationActions:
    """Acts on an escalation verdict for one runtime. See the module docstring."""

    def __init__(self, host: EscalationHost) -> None:
        self._host = host
        # Lazily-built data classifier for the cloud-escalation egress gate (D4).
        self._egress_classifier_cache: Any = None

    def _resolve_escalation_target(
        self,
        verdict: dict[str, Any],
        cfg: EscalationConfig,
        *,
        current_model: str,
        egress_eligible: bool = False,
    ) -> tuple[str, str] | None:
        """Pick a safe target (tier_name, model) for an escalate verdict, or None.

        Escalation targets LOCAL tiers freely. A CLOUD tier is allowed only when
        ``cfg.allow_cloud`` is on AND ``egress_eligible`` (the content classified
        as cloud-safe; ``secret``/``personal`` never) — the D4 egress invariant.
        Honors the resource governor's downshift as a veto (D6) and never
        escalates to the tier already in use.
        """
        candidate = str(verdict.get("target_tier") or "").strip() or cfg.judge_tier
        tier = self._host.tier_router.get_tier_by_name(candidate)
        if tier is None and candidate != cfg.judge_tier:
            candidate = cfg.judge_tier
            tier = self._host.tier_router.get_tier_by_name(candidate)
        if tier is None:
            return None
        # ADR-0068 D4: a tier whose provider runs in the cloud (llm/locality.py; an
        # undeclared provider counts as cloud) is a target only when cloud escalation is on
        # and the content is egress-eligible, so personal/secret never leaves via escalation.
        localities = getattr(self._host.tier_router, "provider_localities", None)
        declared = localities() if callable(localities) else None
        if provider_locality(tier.provider, declared) == "cloud" and not (
            cfg.allow_cloud and egress_eligible
        ):
            logger.debug(
                "escalation skipped: target tier %s is cloud and not egress-eligible", candidate
            )
            return None
        if tier.model == current_model:
            return None  # already at this tier — nothing to escalate to
        governor = getattr(self._host.tier_router, "governor", None)
        if governor is not None:
            try:
                if governor.recommend_tier_name(candidate) != candidate:
                    logger.debug("escalation vetoed by resource governor for %s", candidate)
                    return None
            except Exception:  # noqa: BLE001, S110 — veto check is best-effort
                pass
        return candidate, tier.model

    def _escalation_egress_ok(
        self,
        cfg: EscalationConfig,
        *,
        message: str,
        curated: CuratedResponse,
        memory_ctx: MemoryContext | None,
    ) -> bool:
        """Is the content cloud-safe enough to escalate off-box? (D4, fail-closed).

        Classifies everything that would be SENT to a cloud tier — the prompt, the
        current answer, and the recalled memory context — and folds it to the
        worst classification. Cloud escalation is permitted only when that is in
        ``cfg.cloud_classifications`` (``secret`` never). Any error -> not eligible.
        The governance kernel's egress gate is still the final backstop.
        """
        if not cfg.allow_cloud:
            return False
        try:
            if self._egress_classifier_cache is None:
                from iris_harness.kernel.governance.plugins.classifier import (
                    DataClassifier,
                )

                self._egress_classifier_cache = DataClassifier()
            classifier = self._egress_classifier_cache
            parts = [message or "", curated.text or ""]
            if memory_ctx is not None:
                parts.extend(memory_ctx.recent_turns)
            worst = "public"
            for part in parts:
                if part:
                    worst = more_restrictive(worst, classifier.classify(part).classification)
            return egress_eligible(worst, cfg.cloud_classifications)
        except Exception:  # fail closed: never escalate off-box on error
            logger.debug(
                "egress eligibility check failed; blocking cloud escalation", exc_info=True
            )
            return False

    def maybe_escalate(
        self,
        *,
        results: list[AgentResult],
        curated: CuratedResponse,
        plan: TaskPlan,
        intent_result: IntentResult,
        memory_ctx: MemoryContext | None,
        session_id: str,
        message: str,
        preferred_model: str | None,
        provider_profile: str | None,
        strict: bool,
        history_text: tuple[str, ...],
    ) -> tuple[list[AgentResult], CuratedResponse, str | None, str | None]:
        """Act on the escalation verdict — escalate / reroute / clarify (ADR-0068).

        - **escalate** (capability_gap): re-run at a stronger LOCAL tier.
        - **reroute** (grounding_gap): re-run at the SAME tier with a grounding /
          tool directive so the agent retrieves before answering.
        - **clarify** (ambiguity): stop and return the judge's question to the
          user (confidence-gated); no re-execution — no model can invent a
          missing constraint.

        escalate and reroute act only at ``action_confidence_floor`` or above, as
        clarify does at ``clarify_confidence_floor``. Bounded by
        ``max_escalations`` (D3) + local-only egress guard (D4) +
        resource-governor veto (D6). Returns ``(results, curated, escalated_tier,
        clarify_question)`` — ``escalated_tier`` is set only when the tier
        actually changed (escalate); ``clarify_question`` replaces the response.
        """
        cfg = self._host.response_curator.escalation_config
        if not cfg.acts or memory_ctx is None:
            return results, curated, None, None
        current_model = preferred_model or self._host.tier_router.model_for_intent(
            intent_result.intent
        )
        from_tier = self._host.tier_router.tier_name_for_model(current_model)
        # Egress decision for cloud escalation (D4) — computed once per turn, only
        # when cloud is allowed; fail-closed. Local escalation never needs it.
        egress_ok = self._escalation_egress_ok(
            cfg, message=message, curated=curated, memory_ctx=memory_ctx
        )
        escalated_tier: str | None = None
        hops = 0
        while hops < max(0, cfg.max_escalations):
            # Checked every hop: a re-run can itself write before the next one.
            held = _why_not_rerunnable(results)
            if held is not None:
                logger.info("escalation not acted on: %s", held)
                break
            verdict = _extract_escalation_verdict(curated)
            if verdict is None:
                break
            action = verdict.get("action")

            if action == "clarify":
                question = str(verdict.get("question") or "").strip()
                confidence = float(verdict.get("confidence") or 0.0)
                # D8: asking too often is worse than a mediocre answer — gate it.
                if question and confidence >= cfg.clarify_confidence_floor:
                    self._record_action_signal(
                        "clarify",
                        verdict,
                        from_tier=from_tier,
                        to_tier=None,
                        new_curated=curated,
                        intent=intent_result.intent,
                    )
                    return results, curated, escalated_tier, question
                break

            # Re-running replaces an answer the user would otherwise get, so a verdict
            # the judge is not confident in must not act — the same gate clarify has.
            if float(verdict.get("confidence") or 0.0) < cfg.action_confidence_floor:
                break

            run_plan = plan
            if action == "escalate":
                target = self._resolve_escalation_target(
                    verdict, cfg, current_model=current_model, egress_eligible=egress_ok
                )
                if target is None:
                    break
                target_name, run_model = target
            elif action == "reroute":
                # Same tier; nudge the agent to ground via a tool before answering.
                target_name = from_tier or "same"
                run_model = current_model
                hint = verdict.get("tool_hint") or "available retrieval/search"
                directive = (
                    "\n\n[Grounding required: the previous answer lacked support. Use the "
                    f"{hint} tool to look up facts before answering.]"
                )
                run_plan = replace(plan, query=f"{plan.query}{directive}")
            else:
                break  # accept / unknown — nothing to do

            try:
                new_results = self._host._execute_plan(
                    run_plan,
                    intent_result,
                    memory_ctx,
                    session_id,
                    preferred_model=run_model if action == "escalate" else preferred_model,
                    provider_profile=provider_profile,
                )
                new_curated = self._host.response_curator.curate(
                    new_results,
                    query=message,
                    strict=strict,
                    session_id=session_id,
                    conversation_history=history_text,
                    intent=intent_result.intent,
                )
            except Exception:  # an action never breaks the turn
                logger.exception("escalation action re-execution failed; keeping prior answer")
                break

            hops += 1
            self._record_action_signal(
                action,
                verdict,
                from_tier=from_tier,
                to_tier=target_name,
                new_curated=new_curated,
                intent=intent_result.intent,
            )
            results, curated = new_results, new_curated
            if action == "escalate":
                current_model, from_tier, escalated_tier = run_model, target_name, target_name
        return results, curated, escalated_tier, None

    def _record_action_signal(
        self,
        action: str,
        verdict: dict[str, Any],
        *,
        from_tier: str | None,
        to_tier: str | None,
        new_curated: CuratedResponse,
        intent: str | None = None,
    ) -> None:
        """Record escalation/reroute/clarify outcome signals (decision + cost)."""
        trace_id, span_id = current_trace_ids()
        corr: dict[str, Any] = {
            "session_id": current_session_id(),
            "turn_id": current_turn_id(),
            "trace_id": trace_id,
            "span_id": span_id,
            "resolved_tier": to_tier or from_tier,
        }
        self._host.signal_collector.record_metric(
            metric_name="escalation_decision",
            value=1.0,
            success=not new_curated.has_errors,
            metadata={
                "acted": True,
                "action": action,
                "diagnosis": verdict.get("diagnosis"),
                "from_tier": from_tier,
                "to_tier": to_tier,
                "confidence": verdict.get("confidence"),
                "intent": intent,
            },
            **corr,
        )
        # Re-executing actions (escalate/reroute) carry an extra-hop cost; clarify
        # spends no extra generation.
        if action in {"escalate", "reroute"}:
            tokens = _coerce_total_tokens(new_curated.metadata.get("total_tokens")) or 0
            self._host.signal_collector.record_metric(
                metric_name="escalation_cost",
                value=float(tokens),
                success=not new_curated.has_errors,
                metadata={"action": action, "to_tier": to_tier, "hops": 1},
                **corr,
            )
