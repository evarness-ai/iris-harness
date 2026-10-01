"""Integration test: chat → signal capture → learning_tick heartbeat."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.runtime import build_runtime

pytestmark = pytest.mark.integration


def test_chat_records_signals_and_heartbeat_runs(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()

    # Provide a strategies file so the engine has something to evaluate.
    learning_dir = config_dir / "learning"
    learning_dir.mkdir()
    (learning_dir / "strategies.yaml").write_text(
        """
strategies:
  - name: bump_threshold
    domain: intent_routing
    metric: intent_confidence
    baseline_min_samples: 1
    trigger:
      below: 1.5
    variant:
      description: bump
      config_changes:
        threshold: 0.7
      rollback:
        threshold: 0.6
    evaluation:
      window_hours: 1
      min_improvement_pct: 5.0
""",
        encoding="utf-8",
    )

    runtime = build_runtime(
        config_dir=config_dir,
        data_dir=data_dir,
        use_background_scheduler=False,
    )
    runtime.startup()
    try:
        runtime.chat("what time is it?")

        confidence_rows = runtime.learning_store.recent_signals(metric_name="intent_confidence")
        assert len(confidence_rows) >= 1

        report = runtime.learning_engine.tick()
        assert len(report.started) == 1
        persisted = runtime.learning_store.list_experiments(strategy_name="bump_threshold")
        assert persisted[0].status == "running"
    finally:
        runtime.shutdown()
