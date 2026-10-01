"""Wire a promoted experiment to a sandbox pre-flight (ADR-0070, slice 2).

Slice 1 built the harness; this connects it to the loop. A promoted experiment
that carries a structured applicable change (``__applied_change__``, stamped by
``promote`` from the analyst's ``Recommendation.change``) can be turned into a
sandbox :class:`VariantConfig` and replay-evaluated *before* anyone applies the
change. The verdict is stamped back onto the experiment so the ledger shows it.

The orchestration is pure given its injected pieces — ``evaluate_fn`` (runs the
replay; the live impl wraps ``eval_adapter.run_preflight`` over an isolated eval
runtime) and ``load_workload`` (yields the frozen queries for an intent). That
keeps the wiring unit-testable without a model; only the thin runtime glue that
supplies those callables needs a live host.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from iris_harness.services.learning.analyst import AppliedChange
from iris_harness.services.learning.eval_harness import EvalComparison, EvalQuery, VariantConfig
from iris_harness.services.learning.models import Experiment
from iris_harness.services.learning.promote import CHANGE_KEY
from iris_harness.services.learning.store import LearningMetricsStore

# Key under config_changes where the sandbox pre-flight verdict is stamped.
SANDBOX_KEY = "__sandbox__"

_TERMINAL_STATUSES = frozenset({"kept", "discarded", "failed"})

EvaluateFn = Callable[[Sequence[EvalQuery], VariantConfig], EvalComparison]
WorkloadLoader = Callable[[str], Sequence[EvalQuery]]


def variant_from_change(change: AppliedChange) -> VariantConfig | None:
    """Map a structured applicable change to a sandbox variant. None if unmappable."""
    if change.kind == "route_intent_tier" and change.intent and change.to_tier:
        return VariantConfig(
            label=f"{change.intent}@{change.to_tier}",
            intent_tier_priors={change.intent: change.to_tier},
        )
    return None


def variant_from_experiment(experiment: Experiment) -> VariantConfig | None:
    """Read a promoted experiment's stamped change and map it to a variant."""
    raw = experiment.config_changes.get(CHANGE_KEY)
    if not isinstance(raw, dict):
        return None
    change = AppliedChange.from_dict(raw)
    return variant_from_change(change) if change else None


def needs_preflight(experiment: Experiment) -> bool:
    """True when an experiment can be sandbox-evaluated and hasn't been yet."""
    if experiment.status in _TERMINAL_STATUSES:
        return False
    if SANDBOX_KEY in experiment.config_changes:
        return False
    return variant_from_experiment(experiment) is not None


def run_experiment_preflight(
    store: LearningMetricsStore,
    experiment: Experiment,
    *,
    evaluate_fn: EvaluateFn,
    load_workload: WorkloadLoader,
    now: datetime | None = None,
) -> EvalComparison | None:
    """Replay-evaluate one promoted experiment and stamp the verdict onto it.

    Returns the comparison, or ``None`` when the experiment isn't evaluable (no
    mappable change) or has no workload. The verdict is persisted under
    ``config_changes[SANDBOX_KEY]`` — advisory, exactly like the rest of slice 3:
    it never applies the change.
    """
    raw = experiment.config_changes.get(CHANGE_KEY)
    change = AppliedChange.from_dict(raw) if isinstance(raw, dict) else None
    if change is None:
        return None
    variant = variant_from_change(change)
    if variant is None:
        return None
    intent = change.intent or experiment.domain
    workload = list(load_workload(intent))
    if not workload:
        return None
    comparison = evaluate_fn(workload, variant)
    moment = now or datetime.now(UTC)
    experiment.config_changes[SANDBOX_KEY] = {
        **comparison.as_dict(),
        "evaluated_at": moment.isoformat(),
        "workload_size": len(workload),
    }
    store.upsert_experiment(experiment)
    return comparison


__all__ = [
    "SANDBOX_KEY",
    "needs_preflight",
    "run_experiment_preflight",
    "variant_from_change",
    "variant_from_experiment",
]
