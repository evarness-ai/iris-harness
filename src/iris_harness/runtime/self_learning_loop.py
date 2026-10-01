"""The self-learning loop's periodic jobs — escalation priors, skill crystallization,
sandbox pre-flight, routine reflection and experiment re-measure (ADR-0068 L5, ADR-0069,
ADR-0070; ``docs/architecture/learning-subsystem.md``).

Extracted from ``IrisRuntime`` as ``HeartbeatRunnersMixin`` in Phase 2 (byte-identical
bodies, inherited). OSS plan M5.7 track C turned it into a collaborator in two steps:
slice 9 took learning analysis, behavior mining and intention rollup to
``runtime/learning_controls.py`` with the console controls that toggle them, and slice 11
made the rest ``SelfLearningLoop(host)``, held as ``runtime.learning_loop``.

It holds no state. :class:`SelfLearningHost` declares the seven runtime members the jobs
read, so mypy checks the runtime still supplies them; the host is read **at call time**,
not captured. The heartbeat adapters and ``refresh_escalation_priors`` are public because
the runtime registers the former and calls the latter at startup.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from iris_harness.agent.response_curator import ResponseCurator
from iris_harness.llm.tier_router import TierRouter
from iris_harness.services.heartbeat import HeartbeatDefinition, HeartbeatRun, HeartbeatStatus
from iris_harness.services.learning.crystallizer import SkillCrystallizer
from iris_harness.services.learning.escalation_priors import recommend_start_tiers
from iris_harness.services.learning.signals import LearningSignalCollector
from iris_harness.services.learning.store import LearningMetricsStore
from iris_harness.services.routines import RoutineApprovalRequestStatus, RoutineStore
from iris_harness.services.routines.reflection import reflect_on_routines

logger = logging.getLogger(__name__)


_SANDBOX_MAX_ITEMS = 10


_SANDBOX_REPEATS = 3


class SelfLearningHost(Protocol):
    """The seven runtime members the self-learning jobs reach."""

    config_dir: Path
    learning_store: LearningMetricsStore
    response_curator: ResponseCurator
    routine_store: RoutineStore
    signal_collector: LearningSignalCollector
    skill_crystallizer: SkillCrystallizer
    tier_router: TierRouter


class SelfLearningLoop:
    """The self-learning loop's periodic jobs for one runtime. See the module docstring."""

    def __init__(self, host: SelfLearningHost) -> None:
        self._host = host

    def refresh_escalation_priors(self) -> int:
        """Mine escalation history into predictive start-tier priors (L5).

        Computes per-intent recommendations from ``escalation_shadow`` history and,
        when ``priors.enabled``, applies them to the router so those intents start
        higher (predictive routing). When disabled the recommendations are recorded
        but NOT applied — shadow -> validate -> enforce. Best-effort; returns the
        number of recommendations.
        """
        try:
            cfg = self._host.response_curator.escalation_config
            recs = recommend_start_tiers(
                self._host.learning_store,
                current_tiers=self._host.tier_router.intent_tier_map(),
                window=timedelta(hours=cfg.priors_window_hours),
                min_samples=cfg.priors_min_samples,
                escalate_rate_threshold=cfg.priors_escalate_rate,
            )
            for intent, tier in recs.items():
                self._host.signal_collector.record_metric(
                    metric_name="escalation_prior_recommendation",
                    value=1.0,
                    success=True,
                    metadata={"intent": intent, "to_tier": tier, "applied": cfg.priors_enabled},
                    resolved_tier=tier,
                )
            if cfg.priors_enabled and recs:
                self._host.tier_router.set_intent_tier_priors(recs)
                logger.info("applied %d learned escalation start-tier priors", len(recs))
            elif recs:
                logger.info(
                    "computed %d escalation start-tier priors (shadow; not applied)", len(recs)
                )
            return len(recs)
        except Exception:  # priors are advisory, never fatal
            logger.debug("escalation priors refresh failed", exc_info=True)
            return 0

    def escalation_priors_heartbeat(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """Heartbeat adapter that periodically refreshes escalation priors (L5)."""
        count = self.refresh_escalation_priors()
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            finished_at=datetime.now(UTC),
            output=json.dumps({"recommendations": count}, sort_keys=True),
        )

    def routine_reflection_heartbeat(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """Heartbeat adapter for the routines learning loop (propose-only)."""
        count = self._reflect_on_routines()
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            finished_at=datetime.now(UTC),
            output=json.dumps({"proposals": count}, sort_keys=True),
        )

    def experiment_remeasure_heartbeat(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """Heartbeat adapter that re-measures promoted recommendation-experiments."""
        count = self._remeasure_promoted_experiments()
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            finished_at=datetime.now(UTC),
            output=json.dumps({"remeasured": count}, sort_keys=True),
        )

    def _run_sandbox_preflights(self) -> int:
        """Sandbox replay-eval pre-flight for promotable experiments (ADR-0070 s2).

        Opt-in (``IRIS_SANDBOX_PREFLIGHT``, off by default — it builds an isolated
        runtime and runs local LLMs). For each promoted experiment that carries a
        structured applicable change and hasn't been pre-flighted, replay a frozen,
        bounded workload through an ISOLATED eval runtime under baseline vs variant
        and stamp the verdict. Never applies the change. Best-effort; returns the
        number pre-flighted.
        """
        if os.getenv("IRIS_SANDBOX_PREFLIGHT", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return 0
        import shutil

        from iris_harness.runtime.eval_runtime import build_eval_runtime
        from iris_harness.services.learning.eval_adapter import run_preflight
        from iris_harness.services.learning.eval_harness import COMPLETION_RATE
        from iris_harness.services.learning.eval_preflight import (
            needs_preflight,
            run_experiment_preflight,
        )
        from iris_harness.services.learning.eval_workload import (
            build_workload_from_traces,
        )

        try:
            pending = [
                e for e in self._host.learning_store.list_experiments() if needs_preflight(e)
            ]
        except Exception:
            logger.debug("sandbox preflight: listing experiments failed", exc_info=True)
            return 0
        if not pending:
            return 0

        eval_runtime: Any = None
        scratch = None
        done = 0
        try:
            eval_runtime, scratch = build_eval_runtime(config_dir=self._host.config_dir)

            def _evaluate(workload: Any, variant: Any) -> Any:
                return run_preflight(
                    eval_runtime,
                    workload,
                    variant,
                    repeats=_SANDBOX_REPEATS,
                    metric=COMPLETION_RATE,
                )

            def _load(intent: str) -> Any:
                return build_workload_from_traces(intent=intent, max_items=_SANDBOX_MAX_ITEMS)

            for exp in pending:
                try:
                    if (
                        run_experiment_preflight(
                            self._host.learning_store,
                            exp,
                            evaluate_fn=_evaluate,
                            load_workload=_load,
                        )
                        is not None
                    ):
                        done += 1
                except Exception:  # one experiment failing must not stop the batch
                    logger.debug("sandbox preflight failed for %s", exp.id, exc_info=True)
        except Exception:  # the whole batch is best-effort
            logger.debug("sandbox preflight batch failed", exc_info=True)
        finally:
            if eval_runtime is not None:
                try:
                    eval_runtime.shutdown()
                except Exception:  # noqa: BLE001, S110
                    pass
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
        return done

    def sandbox_preflight_heartbeat(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """Heartbeat adapter for the opt-in sandbox pre-flight (ADR-0070 s2)."""
        count = self._run_sandbox_preflights()
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            finished_at=datetime.now(UTC),
            output=json.dumps({"preflighted": count}, sort_keys=True),
        )

    def crystallize_heartbeat(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """Heartbeat adapter that mines clean successes into skill proposals."""
        try:
            report = self._host.skill_crystallizer.crystallize()
        except Exception as exc:  # must not crash the scheduler
            logger.exception("crystallize_tick failed")
            return HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.FAILED,
                finished_at=datetime.now(UTC),
                error=f"{type(exc).__name__}: {exc}",
            )
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            finished_at=datetime.now(UTC),
            output=json.dumps(
                {
                    "proposals": len(report.proposals),
                    "skipped": len(report.skipped_existing),
                    "rejected": len(report.rejected),
                },
                sort_keys=True,
            ),
        )

    def _reflect_on_routines(self) -> int:
        """Propose routine changes from the outcome ledger via the HITL queue.

        Opt-in (``IRIS_ROUTINE_REFLECTION``). For each failing/stale routine it
        posts a routine approval request (deduped against pending reflection
        requests) and records a ``routine_reflection`` signal — it NEVER mutates a
        routine; the user reviews + acts. Best-effort; returns proposals created.
        """
        if os.getenv("IRIS_ROUTINE_REFLECTION", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return 0
        try:
            specs = self._host.routine_store.list_all()
            reflections = reflect_on_routines(self._host.learning_store, specs)
            if not reflections:
                return 0
            pending = self._host.routine_store.list_approval_requests(
                status=RoutineApprovalRequestStatus.PENDING
            )
            already = {
                req.routine_id for req in pending if req.metadata.get("source") == "reflection"
            }
            created = 0
            for ref in reflections:
                if ref.routine_id in already:
                    continue
                self._host.routine_store.create_approval_request(
                    routine_id=ref.routine_id,
                    session_id="routine-reflection",
                    prompt=f"{ref.reason} — recommend: {ref.recommendation}.",
                    metadata={
                        "source": "reflection",
                        "kind": ref.kind,
                        "recommendation": ref.recommendation,
                    },
                )
                self._host.signal_collector.record_metric(
                    metric_name="routine_reflection",
                    value=1.0,
                    success=True,
                    metadata={
                        "routine_id": ref.routine_id,
                        "kind": ref.kind,
                        "recommendation": ref.recommendation,
                    },
                )
                created += 1
            return created
        except Exception:  # reflection is advisory, never fatal
            logger.debug("routine reflection failed", exc_info=True)
            return 0

    def _remeasure_promoted_experiments(self) -> int:
        """Re-measure promoted recommendation-experiments (ADR-0069 #4 s3).

        Always-on (no flag): it only touches experiments a human explicitly
        promoted, capturing their target metric's current value and keeping/
        discarding past the window. Best-effort; returns how many were updated.
        """
        from iris_harness.services.learning.promote import (
            remeasure_promoted_experiments,
        )

        try:
            return remeasure_promoted_experiments(self._host.learning_store)
        except Exception:  # measurement is advisory, never fatal
            logger.debug("promoted-experiment re-measure failed", exc_info=True)
            return 0
