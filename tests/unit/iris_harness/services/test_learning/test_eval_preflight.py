"""Tests for the experiment sandbox pre-flight wiring (ADR-0070, slice 2).

Covers the analyst's structured applicable change, the change->variant mapping,
promotion stamping the change onto the experiment, and the orchestration that
replay-evaluates a promoted experiment (with an injected fake eval).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from iris_harness.services.learning.analyst import AppliedChange, parse_learning_analysis
from iris_harness.services.learning.eval_harness import (
    ArmScore,
    EvalComparison,
    EvalQuery,
    VariantConfig,
)
from iris_harness.services.learning.eval_preflight import (
    SANDBOX_KEY,
    needs_preflight,
    run_experiment_preflight,
    variant_from_change,
    variant_from_experiment,
)
from iris_harness.services.learning.promote import CHANGE_KEY, promote_recommendation
from iris_harness.services.learning.store import LearningMetricsStore

_NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


# ---- analyst applicable change --------------------------------------------


def test_analyst_parses_route_change() -> None:
    raw = json.dumps(
        {
            "summary": "s",
            "recommendations": [
                {
                    "title": "t",
                    "finding": "f",
                    "action": "a",
                    "confidence": "high",
                    "change": {"kind": "route_intent_tier", "intent": "email", "to_tier": "tier2"},
                }
            ],
        }
    )
    analysis = parse_learning_analysis(raw, generated_at=_NOW, window_hours=168.0, model="m")
    assert analysis is not None
    change = analysis.recommendations[0].change
    assert change is not None
    assert change.kind == "route_intent_tier"
    assert change.intent == "email"
    assert change.to_tier == "tier2"


def test_route_change_requires_intent_and_tier() -> None:
    assert AppliedChange.from_dict({"kind": "route_intent_tier", "intent": "email"}) is None
    assert AppliedChange.from_dict({"kind": "unknown_kind", "intent": "e", "to_tier": "t"}) is None


# ---- change -> variant -----------------------------------------------------


def test_variant_from_change() -> None:
    v = variant_from_change(
        AppliedChange(kind="route_intent_tier", intent="email", to_tier="tier2")
    )
    assert v is not None
    assert v.intent_tier_priors == {"email": "tier2"}
    assert not v.is_baseline


# ---- promotion stamps the change ------------------------------------------


def _store(tmp_path: Path) -> LearningMetricsStore:
    store = LearningMetricsStore(db_path=tmp_path / "learning.db")
    store.ensure_schema()
    return store


def _save_analysis_with_change(store: LearningMetricsStore) -> None:
    store.record_signal(
        source="chat",
        metric_name="task_completed",
        value=1.0,
        success=True,
        metadata={"intent": "email"},
        resolved_tier="tier1",
    )
    store.save_analysis(
        {
            "generated_at": _NOW.isoformat(),
            "model": "m",
            "summary": "s",
            "recommendations": [
                {
                    "title": "Route email up",
                    "finding": "weak tier1",
                    "action": "tier2",
                    "confidence": "high",
                    "target": {"metric": "completion_rate", "intent": "email", "tier": "tier1"},
                    "change": {"kind": "route_intent_tier", "intent": "email", "to_tier": "tier2"},
                }
            ],
        }
    )


def test_promotion_stamps_change_and_variant_recoverable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save_analysis_with_change(store)
    result = promote_recommendation(store, index=1, experiment_id="e1", now=_NOW)
    assert result is not None
    exp = result.experiment
    assert exp.config_changes[CHANGE_KEY]["to_tier"] == "tier2"
    variant = variant_from_experiment(exp)
    assert variant is not None and variant.intent_tier_priors == {"email": "tier2"}
    assert needs_preflight(exp) is True


# ---- orchestration ---------------------------------------------------------


def _comparison(verdict: str) -> EvalComparison:
    arm = ArmScore(label="x", runs=4, completion_rate=0.5, tool_correctness=None, labelled_runs=0)
    return EvalComparison(
        metric="completion_rate",
        baseline=arm,
        variant=arm,
        improvement_pct=10.0,
        verdict=verdict,
        note="n",
    )


def test_run_experiment_preflight_stamps_verdict(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save_analysis_with_change(store)
    exp = promote_recommendation(store, index=1, experiment_id="e1", now=_NOW).experiment  # type: ignore[union-attr]

    seen: dict[str, object] = {}

    def fake_eval(workload, variant: VariantConfig) -> EvalComparison:
        seen["intent_priors"] = variant.intent_tier_priors
        seen["n"] = len(list(workload))
        return _comparison("better")

    def load_workload(intent: str):
        seen["intent"] = intent
        return [EvalQuery(query="q1", intent=intent), EvalQuery(query="q2", intent=intent)]

    cmp = run_experiment_preflight(
        store, exp, evaluate_fn=fake_eval, load_workload=load_workload, now=_NOW
    )
    assert cmp is not None and cmp.verdict == "better"
    assert seen["intent"] == "email"
    assert seen["intent_priors"] == {"email": "tier2"}
    # Verdict persisted on the experiment in the ledger.
    [stored] = store.list_experiments()
    sandbox = stored.config_changes[SANDBOX_KEY]
    assert sandbox["verdict"] == "better"
    assert sandbox["workload_size"] == 2
    assert needs_preflight(stored) is False  # already done


def test_preflight_skips_experiment_without_change(tmp_path: Path) -> None:
    from iris_harness.services.learning.models import Experiment

    store = _store(tmp_path)
    exp = Experiment(
        id="e2",
        domain="email",
        hypothesis="h",
        variant_description="v",
        config_changes={},
        baseline_metric=0.5,
        created_at=_NOW,
    )
    called = {"n": 0}

    def fake_eval(workload, variant):  # pragma: no cover - must not be called
        called["n"] += 1
        return _comparison("better")

    out = run_experiment_preflight(
        store, exp, evaluate_fn=fake_eval, load_workload=lambda i: [EvalQuery(query="q", intent=i)]
    )
    assert out is None
    assert called["n"] == 0


def test_preflight_skips_empty_workload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save_analysis_with_change(store)
    exp = promote_recommendation(store, index=1, experiment_id="e1", now=_NOW).experiment  # type: ignore[union-attr]
    out = run_experiment_preflight(
        store, exp, evaluate_fn=lambda w, v: _comparison("better"), load_workload=lambda i: []
    )
    assert out is None
    assert SANDBOX_KEY not in store.list_experiments()[0].config_changes
