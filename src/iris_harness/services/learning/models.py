from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

ExperimentStatus = Literal["pending", "running", "evaluating", "kept", "discarded", "failed"]


@dataclass
class IterationRecord:
    """
    Represents a single iteration's data in the learning process.
    """

    iteration_id: str
    metrics: list[str]
    timestamp: float


@dataclass
class ExperimentResult:
    """
    Represents the result of an experiment.
    """

    experiment_id: str
    success: bool
    details: str
    timestamp: float


@dataclass
class MetricSnapshot:
    """
    Represents a snapshot of metrics at a specific point in time.
    """

    snapshot_id: str
    metrics: list[str]
    timestamp: float


@dataclass
class Experiment:
    """Represents an autonomous self-improvement experiment."""

    id: str
    domain: str
    hypothesis: str
    variant_description: str
    config_changes: dict[str, object]
    baseline_metric: float
    current_metric: float | None = None
    status: ExperimentStatus = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    evaluated_at: datetime | None = None
    evaluation_window_hours: int = 24
    rollback_config: dict[str, object] = field(default_factory=dict)
