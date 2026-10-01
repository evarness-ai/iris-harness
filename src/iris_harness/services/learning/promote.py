"""Close the learning loop (ADR-0069 #4, slice 3).

Slice 1 measures, slice 2 interprets — this slice *tests*. It turns an accepted
:class:`~iris_harness.services.learning.analyst.Recommendation` into a tracked
:class:`~iris_harness.services.learning.models.Experiment` so a proposal stops being advice and
becomes a hypothesis with a baseline and an outcome.

Two pieces, both deterministic and side-effect-light (they read/write the
learning store only):

* :func:`promote_recommendation` — the HITL action. Given the latest analysis and
  a 1-based index, it captures the baseline value of the recommendation's measured
  target and persists a ``pending`` experiment. It does NOT apply the change —
  applying the recommended config/routing edit stays a human step (the program's
  "never auto-apply a large change" rule). The experiment's *measurement* is
  automated; its *application* is not.
* :func:`remeasure_promoted_experiments` — re-reads each promoted experiment's
  target from a fresh measured report, records the current value, and (past the
  evaluation window) keeps or discards it on the same improvement rule the
  autonomous experiment loop uses. Lower-is-better metrics are inverted.

The measurement target rides in ``Experiment.config_changes`` under
:data:`TARGET_KEY` — no schema change, and it doubles as the marker that an
experiment came from a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from iris_harness.services.learning.analyst import LearningAnalysis, MetricTarget
from iris_harness.services.learning.experiment_loop import ExperimentLoop
from iris_harness.services.learning.intelligence import IntelligenceReport, build_intelligence
from iris_harness.services.learning.models import Experiment
from iris_harness.services.learning.store import LearningMetricsStore

# Reserved key under Experiment.config_changes carrying the measurement target.
# Its presence also marks an experiment as recommendation-derived.
TARGET_KEY = "__measure_target__"
SOURCE_KEY = "__source__"
# The structured applicable change (ADR-0070), when the recommendation carried one.
# Present => the experiment can be sandbox pre-flighted.
CHANGE_KEY = "__applied_change__"

# Metrics where a LOWER value is the improvement (so % improvement is inverted).
_LOWER_IS_BETTER = frozenset({"correction_rate", "drop_rate", "avg_tokens"})

_TERMINAL_STATUSES = frozenset({"kept", "discarded", "failed"})
_DEFAULT_WINDOW_HOURS = 48
_MIN_IMPROVEMENT_PCT = 5.0


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of promoting a recommendation."""

    experiment: Experiment
    measurable: bool  # False when no usable target → baseline couldn't be captured
    note: str


def resolve_target_value(report: IntelligenceReport, target: MetricTarget) -> float | None:
    """Read the current value of a target metric from a measured report.

    Returns ``None`` when the cell/metric isn't present (e.g. no traffic for that
    intent+tier yet), so callers can treat it as "not measurable right now".
    """
    if target.metric == "escalation_precision":
        return report.accuracy.escalation_precision
    if target.metric == "drop_rate":
        return report.accuracy.drop_rate
    for cell in report.matrix:
        if cell.intent == target.intent and cell.tier == target.tier:
            value = getattr(cell, target.metric, None)
            return None if value is None else float(value)
    return None


def improvement_pct(*, baseline: float, current: float, metric: str) -> float:
    """Percent improvement, with lower-is-better metrics inverted."""
    raw = ExperimentLoop.calculate_improvement_pct(baseline_metric=baseline, current_metric=current)
    return -raw if metric in _LOWER_IS_BETTER else raw


def promote_recommendation(
    store: LearningMetricsStore,
    *,
    index: int,
    experiment_id: str | None = None,
    now: datetime | None = None,
    evaluation_window_hours: int = _DEFAULT_WINDOW_HOURS,
) -> PromotionResult | None:
    """Promote the recommendation at ``index`` (1-based) to a tracked experiment.

    Returns ``None`` when there is no stored analysis or the index is out of
    range. Persists a ``pending`` experiment carrying the measurement target; the
    re-measure tick takes it from there. Never applies the recommended change.
    """
    payload = store.latest_analysis()
    if not payload:
        return None
    analysis = LearningAnalysis.from_dict(payload)
    if analysis is None or not analysis.recommendations:
        return None
    if index < 1 or index > len(analysis.recommendations):
        return None
    rec = analysis.recommendations[index - 1]
    moment = now or datetime.now(UTC)

    target = rec.target
    baseline: float | None = None
    if target is not None:
        report = build_intelligence(store, now=moment)
        baseline = resolve_target_value(report, target)

    if target is not None and baseline is not None:
        measurable = True
        scope = f" for {target.intent}@{target.tier}" if target.intent else " (global)"
        note = f"Tracking {target.metric}{scope}"
    elif target is not None:
        measurable = False
        note = (
            f"Target {target.metric} has no measured value yet — the experiment will "
            "start measuring once that traffic appears."
        )
    else:
        measurable = False
        note = "No measurable target on this recommendation — tracked for the record only."

    config: dict[str, Any] = {SOURCE_KEY: "recommendation", "rec_title": rec.title}
    if target is not None:
        config[TARGET_KEY] = target.as_dict()
    if rec.change is not None:
        config[CHANGE_KEY] = rec.change.as_dict()

    experiment = Experiment(
        id=experiment_id or f"rec-exp-{uuid4().hex[:12]}",
        domain=(target.intent if target and target.intent else "learning"),
        hypothesis=f"{rec.title}: {rec.finding}".strip().rstrip(":"),
        variant_description=rec.action or rec.title,
        config_changes=config,
        baseline_metric=baseline if baseline is not None else 0.0,
        created_at=moment,
        evaluation_window_hours=evaluation_window_hours,
    )
    store.upsert_experiment(experiment)
    return PromotionResult(experiment=experiment, measurable=measurable, note=note)


def remeasure_promoted_experiments(
    store: LearningMetricsStore,
    *,
    now: datetime | None = None,
    min_improvement_pct: float = _MIN_IMPROVEMENT_PCT,
) -> int:
    """Re-measure every non-terminal recommendation-derived experiment.

    Reads each one's target from a fresh report, records the current value, and
    (once past its evaluation window) keeps it if it improved by
    ``min_improvement_pct`` or discards it. Returns how many were updated.
    """
    moment = now or datetime.now(UTC)
    report = build_intelligence(store, now=moment)
    updated = 0
    for exp in store.list_experiments():
        if exp.status in _TERMINAL_STATUSES:
            continue
        raw_target = exp.config_changes.get(TARGET_KEY)
        if not isinstance(raw_target, dict):
            continue  # not a recommendation experiment (or has no target)
        target = MetricTarget.from_dict(raw_target)
        if target is None:
            continue
        current = resolve_target_value(report, target)
        if current is None:
            continue  # still no measured value — leave it pending
        exp.current_metric = current
        if exp.started_at is None:
            exp.started_at = moment
        exp.status = "evaluating"
        if moment >= exp.created_at + timedelta(hours=exp.evaluation_window_hours):
            imp = improvement_pct(
                baseline=exp.baseline_metric, current=current, metric=target.metric
            )
            exp.status = "kept" if imp >= min_improvement_pct else "discarded"
            exp.evaluated_at = moment
        store.upsert_experiment(exp)
        updated += 1
    return updated


__all__ = [
    "SOURCE_KEY",
    "TARGET_KEY",
    "PromotionResult",
    "improvement_pct",
    "promote_recommendation",
    "remeasure_promoted_experiments",
    "resolve_target_value",
]
