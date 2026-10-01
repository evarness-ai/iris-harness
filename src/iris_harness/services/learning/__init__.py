"""Self-learning subsystem: experiments, signals, and autonomous tuning."""

from iris_harness.services.learning.engine import LearningEngine, LearningTickReport
from iris_harness.services.learning.experiment_loop import ExperimentLoop
from iris_harness.services.learning.models import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    IterationRecord,
    MetricSnapshot,
)
from iris_harness.services.learning.signals import LearningSignalCollector
from iris_harness.services.learning.store import LearningMetricsStore, MetricAggregate, SignalRecord
from iris_harness.services.learning.strategies import (
    StrategyDefinition,
    StrategyEvaluation,
    StrategyTrigger,
    StrategyVariant,
    load_strategies,
)

__all__ = [
    "Experiment",
    "ExperimentLoop",
    "ExperimentResult",
    "ExperimentStatus",
    "IterationRecord",
    "LearningEngine",
    "LearningMetricsStore",
    "LearningSignalCollector",
    "LearningTickReport",
    "MetricAggregate",
    "MetricSnapshot",
    "SignalRecord",
    "StrategyDefinition",
    "StrategyEvaluation",
    "StrategyTrigger",
    "StrategyVariant",
    "load_strategies",
]
