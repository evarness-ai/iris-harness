from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from iris_harness.services.learning.models import Experiment

_ACTIVE_STATUSES = frozenset({"pending", "running", "evaluating"})
_TERMINAL_STATUSES = frozenset({"kept", "discarded", "failed"})


class ExperimentLoop:
    """Run a small, guarded experiment lifecycle for autonomous tuning."""

    def __init__(
        self,
        *,
        max_concurrent: int = 1,
        measurement_window_hours: int = 24,
        auto_discard_hours: int = 48,
        min_improvement_pct: float = 5.0,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.measurement_window_hours = measurement_window_hours
        self.auto_discard_hours = auto_discard_hours
        self.min_improvement_pct = min_improvement_pct
        self._experiments: dict[str, Experiment] = {}
        self._history: list[Experiment] = []

    def create_experiment(
        self,
        *,
        domain: str,
        hypothesis: str,
        variant_description: str,
        config_changes: Mapping[str, object],
        baseline_metric: float,
        rollback_config: Mapping[str, object] | None = None,
        experiment_id: str | None = None,
        created_at: datetime | None = None,
        evaluation_window_hours: int | None = None,
    ) -> Experiment:
        """Create a new experiment if the active-experiment guard allows it."""
        if self.active_experiment_count() >= self.max_concurrent:
            raise ValueError("max active experiments reached")

        experiment = Experiment(
            id=experiment_id or f"exp-{uuid4().hex[:12]}",
            domain=domain,
            hypothesis=hypothesis,
            variant_description=variant_description,
            config_changes=dict(config_changes),
            baseline_metric=baseline_metric,
            created_at=created_at or datetime.now(UTC),
            evaluation_window_hours=evaluation_window_hours or self.measurement_window_hours,
            rollback_config=dict(rollback_config or {}),
        )
        self._experiments[experiment.id] = experiment
        return experiment

    def start_experiment(
        self, experiment_id: str, *, started_at: datetime | None = None
    ) -> Experiment:
        """Mark an experiment as running."""
        experiment = self.get_experiment(experiment_id)
        if experiment.status != "pending":
            raise ValueError(f"experiment '{experiment_id}' is not pending")
        experiment.status = "running"
        experiment.started_at = started_at or datetime.now(UTC)
        return experiment

    def record_measurement(self, experiment_id: str, metric_value: float) -> Experiment:
        """Record the latest measured metric value for an active experiment."""
        experiment = self.get_experiment(experiment_id)
        if experiment.status not in {"running", "evaluating"}:
            raise ValueError(f"experiment '{experiment_id}' is not running")
        experiment.current_metric = metric_value
        experiment.status = "evaluating"
        return experiment

    def evaluate_experiment(
        self, experiment_id: str, *, evaluated_at: datetime | None = None
    ) -> Experiment:
        """Decide whether to keep or discard the experiment based on improvement."""
        experiment = self.get_experiment(experiment_id)
        if experiment.current_metric is None:
            raise ValueError(f"experiment '{experiment_id}' has no recorded metric")

        improvement = self.calculate_improvement_pct(
            baseline_metric=experiment.baseline_metric,
            current_metric=experiment.current_metric,
        )
        experiment.status = "kept" if improvement >= self.min_improvement_pct else "discarded"
        experiment.evaluated_at = evaluated_at or datetime.now(UTC)
        self._archive_terminal_experiment(experiment)
        return experiment

    def discard_stale_experiments(self, *, now: datetime | None = None) -> list[Experiment]:
        """Auto-discard active experiments that exceed the allowed measurement window."""
        current_time = now or datetime.now(UTC)
        discarded: list[Experiment] = []
        for experiment in self._experiments.values():
            if experiment.status not in _ACTIVE_STATUSES or experiment.started_at is None:
                continue
            expiry_time = experiment.started_at + timedelta(hours=self.auto_discard_hours)
            if current_time < expiry_time:
                continue
            experiment.status = "discarded"
            experiment.evaluated_at = current_time
            self._archive_terminal_experiment(experiment)
            discarded.append(experiment)
        return discarded

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Return a tracked experiment by id."""
        try:
            return self._experiments[experiment_id]
        except KeyError as exc:  # pragma: no cover - exercised via callers
            raise KeyError(f"unknown experiment '{experiment_id}'") from exc

    def get_history(self) -> list[Experiment]:
        """Return immutable snapshots of completed experiments."""
        return [replace(experiment) for experiment in self._history]

    def active_experiment_count(self) -> int:
        """Return the number of currently active experiments."""
        return sum(
            1 for experiment in self._experiments.values() if experiment.status in _ACTIVE_STATUSES
        )

    @staticmethod
    def calculate_improvement_pct(*, baseline_metric: float, current_metric: float) -> float:
        """Calculate relative improvement as a percentage."""
        if baseline_metric == 0:
            return 100.0 if current_metric > 0 else 0.0
        return ((current_metric - baseline_metric) / abs(baseline_metric)) * 100.0

    def _archive_terminal_experiment(self, experiment: Experiment) -> None:
        if experiment.status not in _TERMINAL_STATUSES:
            return
        snapshot = replace(experiment)
        for index, existing in enumerate(self._history):
            if existing.id == snapshot.id:
                self._history[index] = snapshot
                return
        self._history.append(snapshot)
