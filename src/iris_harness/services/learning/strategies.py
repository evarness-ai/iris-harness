"""Strategy definitions for the IRIS self-learning loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StrategyTrigger:
    """Threshold that decides when to spawn an experiment."""

    below: float | None = None
    above: float | None = None

    def fires(self, value: float) -> bool:
        if self.below is not None and value < self.below:
            return True
        if self.above is not None and value > self.above:
            return True
        return False


@dataclass(frozen=True)
class StrategyVariant:
    """Variant configuration applied when an experiment is started."""

    description: str
    config_changes: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyEvaluation:
    """Evaluation policy for a strategy's experiments."""

    window_hours: int = 24
    min_improvement_pct: float = 5.0


@dataclass(frozen=True)
class StrategyDefinition:
    """A single autonomous-tuning strategy."""

    name: str
    domain: str
    metric: str
    baseline_min_samples: int
    trigger: StrategyTrigger
    variant: StrategyVariant
    evaluation: StrategyEvaluation


def load_strategies(path: Path) -> list[StrategyDefinition]:
    """Load strategy definitions from a YAML file.

    Returns an empty list when the file does not exist (the learning loop is
    optional and degrades gracefully when no strategies are configured).
    """
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_strategies = payload.get("strategies") or []
    if not isinstance(raw_strategies, list):
        raise ValueError(f"strategies file must contain a list under 'strategies': {path}")
    strategies: list[StrategyDefinition] = []
    for entry in raw_strategies:
        strategies.append(_parse_strategy(entry, path=path))
    return strategies


def _parse_strategy(entry: dict[str, Any], *, path: Path) -> StrategyDefinition:
    try:
        trigger_raw = entry.get("trigger") or {}
        variant_raw = entry.get("variant") or {}
        evaluation_raw = entry.get("evaluation") or {}
        return StrategyDefinition(
            name=str(entry["name"]),
            domain=str(entry["domain"]),
            metric=str(entry["metric"]),
            baseline_min_samples=int(entry.get("baseline_min_samples", 10)),
            trigger=StrategyTrigger(
                below=_optional_float(trigger_raw.get("below")),
                above=_optional_float(trigger_raw.get("above")),
            ),
            variant=StrategyVariant(
                description=str(variant_raw.get("description", "")),
                config_changes=dict(variant_raw.get("config_changes") or {}),
                rollback=dict(variant_raw.get("rollback") or {}),
            ),
            evaluation=StrategyEvaluation(
                window_hours=int(evaluation_raw.get("window_hours", 24)),
                min_improvement_pct=float(evaluation_raw.get("min_improvement_pct", 5.0)),
            ),
        )
    except KeyError as exc:
        raise ValueError(f"strategy is missing required field {exc}: {path}") from exc


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
