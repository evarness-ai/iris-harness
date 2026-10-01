"""The governance judge, wired: a post-run safety review of every loop run that used tools.

``LLMJudge`` (``kernel/governance/evaluator/judge.py``, design §9.2 of
``docs/architecture/unified-governance-layer.md``) shipped without a caller, so
``IRIS_GOVERNANCE_JUDGE_ENABLED`` did nothing. This module is that caller. The owner's
rules (2026-09-23):

1. **The owner picks the judge's model** (``IRIS_GOVERNANCE_JUDGE_TIER``): ``run`` (the
   default) judges each run on the model route the run itself used, so a Mac-only run is
   judged Mac-only; any tier named in ``llm_tiers.yaml`` — local or cloud — judges every
   run on that tier instead. Nothing here locks the judge to local or to cloud: the
   owner decides. The judge's client is governed (``CodingLLMClient`` fires the pre-LLM
   hooks), so the egress gate judges the trace like any other prompt — personal data to
   a cloud tier still needs what any cloud call needs.
2. **Only runs that used tools.** A plain answer has no tool misuse or dangerous action
   to review.
3. **Alert on halt, record warn.** ``halt_next`` is broadcast to the owner's channels the
   way the health watch alerts; ``warn`` and ``allow`` go to the audit log only, where
   ``iris run inspect`` shows them.

It never delays or breaks a run: the run hands its trace over and returns, one worker
judges in the background, and a full queue drops the review with a log line rather
than piling work onto the Mac's models.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from iris_harness.agent.run_review import CompletedRun
from iris_harness.foundation.env import env_flag
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.evaluator.judge import JudgeVerdict, LLMJudge
from iris_harness.kernel.governance.hooks.types import LLMTier

logger = logging.getLogger(__name__)

ENABLED_ENV = "IRIS_GOVERNANCE_JUDGE_ENABLED"
TIMEOUT_ENV = "IRIS_GOVERNANCE_JUDGE_TIMEOUT_S"
TIER_ENV = "IRIS_GOVERNANCE_JUDGE_TIER"
SAME_AS_RUN = "run"
_DEFAULT_TIMEOUT_S = 30.0
# Reviews waiting or running at once. One worker judges; more than this queued means the
# Mac is behind, and a skipped review is better than a backlog that slows real turns.
_MAX_PENDING = 4
_EXCERPT = 120

Invoke = Callable[[str, str], str]
"""``(system_prompt, user_prompt) -> text`` on one model route."""

Target = tuple[str, str]
"""What the judge runs on: ``("intent", <routing intent>)`` — the run's own route — or
``("tier", <llm_tiers.yaml tier name>)``, the owner's pick."""


def judge_enabled() -> bool:
    """Read per run, so turning the judge on or off applies to the next run."""
    return env_flag(ENABLED_ENV, default=False)


def judge_tier_setting() -> str:
    """``run`` or a tier name. Read per review, so a change applies to the next run."""
    import os

    return os.environ.get(TIER_ENV, "").strip() or SAME_AS_RUN


def judge_timeout_s() -> float:
    import os

    raw = os.environ.get(TIMEOUT_ENV, "").strip()
    try:
        value = float(raw) if raw else _DEFAULT_TIMEOUT_S
    except ValueError:
        return _DEFAULT_TIMEOUT_S
    return value if value > 0 else _DEFAULT_TIMEOUT_S


class _RouteClient:
    """``JudgeClient`` over one model route. The route, not ``model_tier``, picks the model;
    the tier only labels the audit row."""

    def __init__(self, invoke: Invoke) -> None:
        self._invoke = invoke

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_tier: LLMTier,
        timeout_s: float,
    ) -> str:
        return await asyncio.wait_for(
            asyncio.to_thread(self._invoke, system_prompt, user_prompt), timeout=timeout_s
        )


def judge_steps(run: CompletedRun) -> list[dict[str, str | None]]:
    """The run as the judge's template reads it: string fields, and the final answer
    last, because the judge scores the trace AND the response the owner got."""
    steps: list[dict[str, str | None]] = []
    for step in run.steps:
        action_input = step.get("action_input")
        steps.append(
            {
                "thought": _text(step.get("thought")),
                "action": _text(step.get("action")),
                "action_input": (
                    json.dumps(action_input, ensure_ascii=False, default=str)
                    if isinstance(action_input, (dict, list))
                    else _text(action_input)
                ),
                "observation": _text(step.get("observation")),
            }
        )
    steps.append(
        {
            "thought": None,
            "action": None,
            "action_input": None,
            "observation": f"Final answer given to the user: {run.final_answer}",
        }
    )
    return steps


def _text(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


class GovernanceJudge:
    """Takes finished runs and judges the ones that used tools, off the run's thread."""

    def __init__(
        self,
        *,
        invoke_for: Callable[[Target], Invoke],
        tier_for: Callable[[Target], LLMTier],
        known_tier: Callable[[str], bool] = lambda _name: False,
        audit_log: Callable[[], AuditLog] | None,
        notify: Callable[[str, str], None] | None,
        executor: ThreadPoolExecutor | None = None,
        max_pending: int = _MAX_PENDING,
    ) -> None:
        self._invoke_for = invoke_for
        self._tier_for = tier_for
        self._known_tier = known_tier
        self._audit_log = audit_log
        self._notify = notify
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="iris-governance-judge"
        )
        self._max_pending = max_pending
        self._pending = 0
        self._lock = threading.Lock()

    def review(self, run: CompletedRun, route: str, agent_type: str) -> bool:
        """The process's run reviewer (``agent.run_review``): same as :meth:`submit`."""
        return self.submit(run, route=route, agent_type=agent_type)

    def submit(self, run: CompletedRun, *, route: str, agent_type: str) -> bool:
        """Queue ``run`` for review on ``route`` (the routing intent the run used).

        Returns whether it was queued: not when the judge is off, the run used no tools,
        or too many reviews are already waiting.
        """
        if not judge_enabled() or not run.used_tools:
            return False
        with self._lock:
            if self._pending >= self._max_pending:
                logger.warning(
                    "governance judge: %d reviews pending; skipping run %s",
                    self._pending,
                    run.run_id,
                )
                return False
            self._pending += 1
        self._executor.submit(self._review, run, route, agent_type)
        return True

    def target_for(self, route: str) -> Target:
        """The owner's tier if it names a real one, else the run's own route."""
        choice = judge_tier_setting()
        if choice == SAME_AS_RUN:
            return ("intent", route)
        if self._known_tier(choice):
            return ("tier", choice)
        logger.warning(
            "governance judge: %s=%r is not a tier in llm_tiers.yaml; judging on the run's route",
            TIER_ENV,
            choice,
        )
        return ("intent", route)

    def _review(self, run: CompletedRun, route: str, agent_type: str) -> JudgeVerdict | None:
        try:
            target = self.target_for(route)
            judge = LLMJudge(
                client=_RouteClient(self._invoke_for(target)),
                audit_log=self._audit_log() if self._audit_log is not None else None,
                enabled=True,
                model_tier=self._tier_for(target),
                timeout_s=judge_timeout_s(),
            )
            verdict = asyncio.run(
                judge.judge_trace(
                    run_id=run.run_id,
                    agent_type=agent_type,
                    original_task=run.query,
                    steps=judge_steps(run),
                )
            )
            if verdict is not None and verdict.recommend == "halt_next":
                self._alert(run, verdict, agent_type)
            return verdict
        except Exception:  # the judge never escalates its own failure
            logger.exception("governance judge: review of run %s failed", run.run_id)
            return None
        finally:
            with self._lock:
                self._pending -= 1

    def _alert(self, run: CompletedRun, verdict: JudgeVerdict, agent_type: str) -> None:
        if self._notify is None:
            return
        question = run.query.strip().replace("\n", " ")
        if len(question) > _EXCERPT:
            question = question[: _EXCERPT - 1] + "…"
        body = (
            f"The governance judge recommends stopping before the next run like this.\n"
            f"Request: {question}\n"
            f"Why: {verdict.rationale[:300]}\n"
            f"Scores: danger {verdict.danger:.2f}, tool misuse {verdict.tool_misuse:.2f}, "
            f"hallucination {verdict.hallucination:.2f}\n"
            f"Agent: {agent_type}. Details: iris run inspect {run.run_id}"
        )
        try:
            self._notify("Governance judge: halt recommended", body)
        except Exception:  # an undelivered alert is logged, not raised
            logger.exception("governance judge: alert for run %s not delivered", run.run_id)


def build_governance_judge(*, tier_router: Any, channels: Any) -> GovernanceJudge:
    """The judge over the runtime's tier router and channel gateway.

    ``invoke_for(route)`` builds a governed client on the model ``route`` resolves to —
    the same one the run used — at temperature 0. The audit log is the kernel's own
    ledger, opened on first review so a harness with the judge off creates nothing.
    """
    from iris_harness.foundation.paths import audit_db_path
    from iris_harness.llm.tier_router import governance_tier_for_intent
    from iris_harness.services.health.watch import broadcast_notifier

    def invoke_for(target: Target) -> Invoke:
        from iris_harness.llm.client import CodingLLMClient

        kind, name = target
        cfg = (
            tier_router.get_llm_config_for_tier(name)
            if kind == "tier"
            else tier_router.get_llm_config(name)
        )
        cfg = cfg.model_copy(update={"temperature": 0.0, "max_tokens": 400})
        client = CodingLLMClient(cfg, governance_agent_type="chat")

        def _invoke(system: str, user: str) -> str:
            return str(client.invoke(system_prompt=system, user_prompt=user))

        return _invoke

    notifier = broadcast_notifier(channels) if channels is not None else None
    ledger: list[AuditLog] = []

    def audit_log() -> AuditLog:
        if not ledger:
            ledger.append(AuditLog(db_path=audit_db_path()))
        return ledger[0]

    return GovernanceJudge(
        invoke_for=invoke_for,
        tier_for=lambda target: (
            _tier_label(tier_router, target[1])
            if target[0] == "tier"
            else governance_tier_for_intent(tier_router, target[1])
        ),
        known_tier=lambda name: tier_router.get_tier_by_name(name) is not None,
        audit_log=audit_log,
        notify=(
            (lambda subject, body: notifier(subject, body, None)) if notifier is not None else None
        ),
    )


def _tier_label(tier_router: Any, name: str) -> LLMTier:
    """Governance's tier label for a tier named in llm_tiers.yaml (it labels the audit row
    and decides the kernel's tier rules; the tier itself picks the model). Where the tier's
    provider runs decides cloud or local, never the name (``governance_tier_for``)."""
    from iris_harness.llm.tier_router import TierConfig, TierRouter, governance_tier_for

    if isinstance(tier_router, TierRouter):
        return tier_router.governance_tier_for_tier(name)
    tier = tier_router.get_tier_by_name(name)
    return governance_tier_for(name, tier if isinstance(tier, TierConfig) else None)
