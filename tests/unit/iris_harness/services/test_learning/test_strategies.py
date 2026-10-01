"""Unit tests for :mod:`iris_harness.services.learning.strategies`."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.learning.strategies import StrategyTrigger, load_strategies


def test_load_strategies_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_strategies(tmp_path / "missing.yaml") == []


def test_load_strategies_parses_full_definition(tmp_path: Path) -> None:
    path = tmp_path / "strategies.yaml"
    path.write_text(
        """
strategies:
  - name: demo
    domain: intent_routing
    metric: intent_confidence
    baseline_min_samples: 5
    trigger:
      below: 0.7
    variant:
      description: bump
      config_changes:
        threshold: 0.8
      rollback:
        threshold: 0.6
    evaluation:
      window_hours: 6
      min_improvement_pct: 7.5
""",
        encoding="utf-8",
    )
    [strategy] = load_strategies(path)
    assert strategy.name == "demo"
    assert strategy.metric == "intent_confidence"
    assert strategy.trigger.below == 0.7
    assert strategy.variant.config_changes == {"threshold": 0.8}
    assert strategy.evaluation.window_hours == 6


def test_strategy_trigger_fires_only_when_outside_thresholds() -> None:
    trigger = StrategyTrigger(below=0.5, above=0.9)
    assert trigger.fires(0.4) is True
    assert trigger.fires(0.95) is True
    assert trigger.fires(0.7) is False


def test_load_strategies_rejects_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "strategies.yaml"
    path.write_text("strategies:\n  - name: only_name\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_strategies(path)
