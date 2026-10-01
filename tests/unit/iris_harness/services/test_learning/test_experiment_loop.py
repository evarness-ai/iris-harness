from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from iris_harness.services.learning.experiment_loop import ExperimentLoop


def test_create_start_measure_and_keep_experiment() -> None:
    loop = ExperimentLoop(min_improvement_pct=5.0)

    experiment = loop.create_experiment(
        domain="intent_routing",
        hypothesis="raising confidence gating improves routing accuracy",
        variant_description="increase routing threshold to 0.8",
        config_changes={"threshold": 0.8},
        rollback_config={"threshold": 0.7},
        baseline_metric=0.80,
        experiment_id="exp_keep",
    )
    loop.start_experiment(experiment.id, started_at=datetime.now(UTC))
    loop.record_measurement(experiment.id, 0.85)
    evaluated = loop.evaluate_experiment(experiment.id)

    assert evaluated.status == "kept"
    assert evaluated.current_metric == 0.85
    assert loop.get_history()[-1].id == experiment.id
    assert loop.get_history()[-1].status == "kept"


def test_discard_experiment_when_improvement_is_below_threshold() -> None:
    loop = ExperimentLoop(min_improvement_pct=10.0)

    experiment = loop.create_experiment(
        domain="email_classification",
        hypothesis="prompt tweak improves accuracy",
        variant_description="add more category examples",
        config_changes={"prompt": "variant-a"},
        baseline_metric=0.80,
        experiment_id="exp_discard",
    )
    loop.start_experiment(experiment.id)
    loop.record_measurement(experiment.id, 0.84)
    evaluated = loop.evaluate_experiment(experiment.id)

    assert evaluated.status == "discarded"
    assert loop.get_history()[-1].status == "discarded"


def test_prevents_overlapping_active_experiments() -> None:
    loop = ExperimentLoop(max_concurrent=1)
    loop.create_experiment(
        domain="finance_categorization",
        hypothesis="variant improves correction rate",
        variant_description="lower temperature",
        config_changes={"temperature": 0.1},
        baseline_metric=0.73,
        experiment_id="exp_one",
    )

    with pytest.raises(ValueError, match="max active experiments reached"):
        loop.create_experiment(
            domain="finance_categorization",
            hypothesis="second variant overlaps",
            variant_description="raise threshold",
            config_changes={"threshold": 0.9},
            baseline_metric=0.73,
            experiment_id="exp_two",
        )


def test_auto_discards_stale_experiments() -> None:
    loop = ExperimentLoop(auto_discard_hours=48)
    started_at = datetime.now(UTC) - timedelta(hours=49)

    experiment = loop.create_experiment(
        domain="response_quality",
        hypothesis="prompt tune lowers correction rate",
        variant_description="shorter answer instructions",
        config_changes={"prompt_style": "concise"},
        baseline_metric=0.60,
        experiment_id="exp_stale",
    )
    loop.start_experiment(experiment.id, started_at=started_at)
    discarded = loop.discard_stale_experiments(now=datetime.now(UTC))

    assert [item.id for item in discarded] == [experiment.id]
    assert discarded[0].status == "discarded"
    assert loop.get_history()[-1].id == experiment.id
