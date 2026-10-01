"""LearningEngine — ties strategies, signals, and the experiment loop together.

Driven by the ``learning_tick`` heartbeat. On each tick the engine:

1. For every strategy with no active experiment, evaluates the current
   aggregate of its metric. If the trigger fires and we have enough
   samples, a new experiment is created + started.
2. For every active experiment past its evaluation window, the latest
   metric is recorded and the experiment is evaluated (kept / discarded).
3. Stale experiments are auto-discarded.

Every state change is persisted via :class:`LearningMetricsStore.upsert_experiment`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from iris_harness.services.heartbeat.models import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatStatus,
)
from iris_harness.services.learning.experiment_loop import ExperimentLoop
from iris_harness.services.learning.models import Experiment
from iris_harness.services.learning.store import LearningMetricsStore
from iris_harness.services.learning.strategies import StrategyDefinition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LearningTickReport:
    """Summary of what one ``learning_tick`` invocation did."""

    started: tuple[str, ...]
    evaluated: tuple[str, ...]
    discarded: tuple[str, ...]
    skipped: tuple[str, ...]


class LearningEngine:
    """Coordinate strategies + ExperimentLoop + LearningMetricsStore."""

    def __init__(
        self,
        *,
        loop: ExperimentLoop,
        store: LearningMetricsStore,
        strategies: list[StrategyDefinition],
    ) -> None:
        self._loop = loop
        self._store = store
        self._strategies = list(strategies)
        self._strategy_to_experiment: dict[str, str] = {}
        self._hydrate_from_store()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self, *, now: datetime | None = None) -> LearningTickReport:
        """Run one learning iteration. Safe to call from a heartbeat handler."""
        moment = now or datetime.now(UTC)
        started: list[str] = []
        evaluated: list[str] = []
        discarded_names: list[str] = []
        skipped: list[str] = []

        # 1) Auto-discard stale active experiments.
        for stale in self._loop.discard_stale_experiments(now=moment):
            self._persist(stale)
            discarded_names.append(stale.id)

        # 2) Spawn new experiments where strategies fire.
        for strategy in self._strategies:
            if self._strategy_to_experiment.get(strategy.name):
                continue
            aggregate = self._store.aggregate_metric(metric_name=strategy.metric, now=moment)
            if aggregate.sample_count < strategy.baseline_min_samples:
                skipped.append(strategy.name)
                continue
            if not strategy.trigger.fires(aggregate.average):
                continue
            try:
                experiment = self._loop.create_experiment(
                    domain=strategy.domain,
                    hypothesis=f"strategy:{strategy.name}",
                    variant_description=strategy.variant.description,
                    config_changes=strategy.variant.config_changes,
                    rollback_config=strategy.variant.rollback,
                    baseline_metric=aggregate.average,
                    evaluation_window_hours=strategy.evaluation.window_hours,
                )
            except ValueError as exc:
                logger.info("strategy %s could not start experiment: %s", strategy.name, exc)
                continue
            self._loop.start_experiment(experiment.id, started_at=moment)
            self._strategy_to_experiment[strategy.name] = experiment.id
            self._persist(experiment, strategy_name=strategy.name)
            started.append(experiment.id)

        # 3) Evaluate experiments whose window has elapsed.
        for strategy_name, experiment_id in list(self._strategy_to_experiment.items()):
            try:
                experiment = self._loop.get_experiment(experiment_id)
            except KeyError:
                self._strategy_to_experiment.pop(strategy_name, None)
                continue
            if experiment.status not in {"running", "evaluating"}:
                continue
            if experiment.started_at is None:
                continue
            due_at = experiment.started_at + timedelta(hours=experiment.evaluation_window_hours)
            if moment < due_at:
                continue
            strategy_def = self._lookup_strategy(strategy_name)
            metric_name = strategy_def.metric if strategy_def is not None else "intent_confidence"
            aggregate = self._store.aggregate_metric(
                metric_name=metric_name,
                window=timedelta(hours=experiment.evaluation_window_hours),
                now=moment,
            )
            self._loop.record_measurement(experiment_id, aggregate.average)
            outcome = self._loop.evaluate_experiment(experiment_id, evaluated_at=moment)
            self._persist(outcome, strategy_name=strategy_name)
            self._strategy_to_experiment.pop(strategy_name, None)
            evaluated.append(experiment_id)

        # Meta-observability: let the loop report on itself so "is self-learning
        # working?" is answerable from health_summary() without reading the DB
        # (learning-observability.md §4.4). Best-effort; never break the tick.
        self._bump_meta_counters(started=started, evaluated=evaluated, discarded=discarded_names)

        return LearningTickReport(
            started=tuple(started),
            evaluated=tuple(evaluated),
            discarded=tuple(discarded_names),
            skipped=tuple(skipped),
        )

    def _bump_meta_counters(
        self,
        *,
        started: list[str],
        evaluated: list[str],
        discarded: list[str],
    ) -> None:
        counts = {
            "experiments_started_total": len(started),
            "experiments_evaluated_total": len(evaluated),
            "experiments_discarded_total": len(discarded),
            "learning_ticks_total": 1,
        }
        for name, value in counts.items():
            if value:
                try:
                    self._store.increment_counter(name, by=value)
                except Exception:  # noqa: BLE001, S110 — meta-telemetry is best-effort
                    pass

    def heartbeat_handler(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """Adapter so the engine can be registered as a heartbeat handler."""
        try:
            report = self.tick()
        except Exception as exc:  # must not crash scheduler
            logger.exception("learning_tick failed")
            return HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.FAILED,
                finished_at=datetime.now(UTC),
                error=f"{type(exc).__name__}: {exc}",
            )
        output: dict[str, list[str]] = {
            "started": list(report.started),
            "evaluated": list(report.evaluated),
            "discarded": list(report.discarded),
            "skipped": list(report.skipped),
        }
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            finished_at=datetime.now(UTC),
            output=json.dumps(output, sort_keys=True),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hydrate_from_store(self) -> None:
        """Re-attach in-flight experiments from the store after a restart."""
        for status in ("pending", "running", "evaluating"):
            for stored in self._store.list_experiments(status=status):
                if stored.id in self._loop._experiments:  # internal hydration
                    continue
                self._loop._experiments[stored.id] = stored
                strategy_name = stored.hypothesis.removeprefix("strategy:")
                if strategy_name and strategy_name != stored.hypothesis:
                    self._strategy_to_experiment.setdefault(strategy_name, stored.id)

    def _lookup_strategy(self, name: str) -> StrategyDefinition | None:
        for strategy in self._strategies:
            if strategy.name == name:
                return strategy
        return None

    def _persist(self, experiment: Experiment, *, strategy_name: str | None = None) -> None:
        try:
            self._store.upsert_experiment(experiment, strategy_name=strategy_name)
        except Exception:
            logger.exception("failed to persist experiment %s", experiment.id)
