"""Offline replay-eval harness for learning experiments (ADR-0070, slice 1).

The *test* that ADR-0069 #4 slice 3 couldn't do quickly: instead of waiting for
live traffic and measuring a confounded before/after, replay a **frozen** set of
queries through an isolated runtime under **baseline vs variant** config and
compare the outcomes directly. Same queries, only the config differs — a clean
counterfactual (the "only the routed model changes" rigor from the instruct-models
post).

This module is the harness *core* — pure and deterministic given its inputs. It
does no LLM IO itself: the caller injects a ``RunQuery`` that actually executes a
query under a config and reports what happened (which tools it called, whether it
completed). That keeps the scoring/comparison logic unit-testable without a model,
and lets the heavy, governed execution adapter live separately (and stay opt-in).

Scope (ADR-0070): only **offline-reproducible** metrics —

* ``completion_rate`` — share of runs that completed without errors. Needs no
  labels; the honest default.
* ``tool_correctness`` — share of *labelled* runs whose expected tool was called.
  Only computed for queries carrying an ``expected_tool``.

``user_correction`` / ``downstream_reuse`` are deliberately absent: they depend on
real future user behaviour and cannot be replayed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

# Metrics this harness can evaluate offline. Higher is better for both.
Metric = str
COMPLETION_RATE: Metric = "completion_rate"
TOOL_CORRECTNESS: Metric = "tool_correctness"
_METRICS = frozenset({COMPLETION_RATE, TOOL_CORRECTNESS})

_DEFAULT_MIN_IMPROVEMENT_PCT = 5.0


@dataclass(frozen=True)
class EvalQuery:
    """One item of the frozen replay workload."""

    query: str
    intent: str
    # Optional ground-truth label: the tool this query *should* lead to. Present
    # only on curated/labelled items; enables the tool_correctness metric.
    expected_tool: str | None = None


@dataclass(frozen=True)
class VariantConfig:
    """A config arm to evaluate. The empty arm (no overrides) is the baseline.

    v1 carries the one applicable knob this harness supports — an intent→tier
    routing override (``TierRouter.set_intent_tier_priors``). The structure is
    deliberately open so future arms (flags, thresholds) slot in without changing
    the core.
    """

    label: str
    intent_tier_priors: dict[str, str] = field(default_factory=dict)

    @property
    def is_baseline(self) -> bool:
        return not self.intent_tier_priors


@dataclass(frozen=True)
class RunOutcome:
    """What one execution of one query produced (reported by the injected RunQuery)."""

    completed: bool
    tools_called: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArmScore:
    """Aggregate outcome of one config arm over the workload × repeats."""

    label: str
    runs: int
    completion_rate: float
    tool_correctness: float | None
    labelled_runs: int

    def value(self, metric: Metric) -> float | None:
        if metric == COMPLETION_RATE:
            return self.completion_rate
        if metric == TOOL_CORRECTNESS:
            return self.tool_correctness
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "runs": self.runs,
            "completion_rate": round(self.completion_rate, 4),
            "tool_correctness": (
                None if self.tool_correctness is None else round(self.tool_correctness, 4)
            ),
            "labelled_runs": self.labelled_runs,
        }


@dataclass(frozen=True)
class EvalComparison:
    """Baseline-vs-variant verdict on one metric."""

    metric: Metric
    baseline: ArmScore
    variant: ArmScore
    improvement_pct: float | None
    verdict: str  # "better" | "worse" | "inconclusive"
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline.as_dict(),
            "variant": self.variant.as_dict(),
            "improvement_pct": (
                None if self.improvement_pct is None else round(self.improvement_pct, 2)
            ),
            "verdict": self.verdict,
            "note": self.note,
        }


# (query, config) -> outcome. Injected so the core never touches an LLM.
RunQuery = Callable[[EvalQuery, VariantConfig], RunOutcome]


def _improvement_pct(*, baseline: float, variant: float) -> float:
    """Relative improvement of variant over baseline, as a percent."""
    if baseline == 0:
        return 100.0 if variant > 0 else 0.0
    return ((variant - baseline) / abs(baseline)) * 100.0


def score_arm(
    workload: Sequence[EvalQuery],
    *,
    run_query: RunQuery,
    config: VariantConfig,
    repeats: int = 1,
) -> ArmScore:
    """Run every query ``repeats`` times under ``config`` and aggregate outcomes.

    ``repeats`` > 1 averages out local-model run-to-run variance (a real effect —
    same prompt, different answers). A query that raises is scored as a
    non-completion rather than aborting the arm.
    """
    reps = max(1, repeats)
    runs = 0
    completed = 0
    labelled_runs = 0
    tool_hits = 0
    for item in workload:
        for _ in range(reps):
            try:
                outcome = run_query(item, config)
            except Exception:  # noqa: BLE001 — a failed run is a non-completion, not a crash
                outcome = RunOutcome(completed=False)
            runs += 1
            if outcome.completed:
                completed += 1
            if item.expected_tool is not None:
                labelled_runs += 1
                if item.expected_tool in outcome.tools_called:
                    tool_hits += 1
    return ArmScore(
        label=config.label,
        runs=runs,
        completion_rate=(completed / runs) if runs else 0.0,
        tool_correctness=(tool_hits / labelled_runs) if labelled_runs else None,
        labelled_runs=labelled_runs,
    )


def compare(
    *,
    baseline: ArmScore,
    variant: ArmScore,
    metric: Metric = COMPLETION_RATE,
    min_improvement_pct: float = _DEFAULT_MIN_IMPROVEMENT_PCT,
    min_runs: int = 1,
) -> EvalComparison:
    """Decide whether the variant beats the baseline on ``metric``.

    Returns ``inconclusive`` (never a false win) when the metric is unavailable on
    either arm or there aren't enough runs to trust the comparison.
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    b_val = baseline.value(metric)
    v_val = variant.value(metric)
    if b_val is None or v_val is None:
        reason = "no labelled runs" if metric == TOOL_CORRECTNESS else "metric unavailable"
        return EvalComparison(
            metric=metric,
            baseline=baseline,
            variant=variant,
            improvement_pct=None,
            verdict="inconclusive",
            note=f"{metric} not measurable ({reason}).",
        )
    if baseline.runs < min_runs or variant.runs < min_runs:
        return EvalComparison(
            metric=metric,
            baseline=baseline,
            variant=variant,
            improvement_pct=None,
            verdict="inconclusive",
            note=f"too few runs (need >= {min_runs} per arm).",
        )
    imp = _improvement_pct(baseline=b_val, variant=v_val)
    if imp >= min_improvement_pct:
        verdict = "better"
    elif imp <= -min_improvement_pct:
        verdict = "worse"
    else:
        verdict = "inconclusive"
    return EvalComparison(
        metric=metric,
        baseline=baseline,
        variant=variant,
        improvement_pct=imp,
        verdict=verdict,
        note=f"{metric}: {b_val:.3f} -> {v_val:.3f} ({imp:+.1f}%).",
    )


def evaluate(
    workload: Sequence[EvalQuery],
    *,
    run_query: RunQuery,
    variant: VariantConfig,
    baseline: VariantConfig | None = None,
    repeats: int = 1,
    metric: Metric = COMPLETION_RATE,
    min_improvement_pct: float = _DEFAULT_MIN_IMPROVEMENT_PCT,
    min_runs: int = 1,
) -> EvalComparison:
    """Score baseline + variant arms over the workload and compare them.

    ``baseline`` defaults to the empty (no-override) arm — the current config.
    """
    base_cfg = baseline or VariantConfig(label="baseline")
    base = score_arm(workload, run_query=run_query, config=base_cfg, repeats=repeats)
    var = score_arm(workload, run_query=run_query, config=variant, repeats=repeats)
    return compare(
        baseline=base,
        variant=var,
        metric=metric,
        min_improvement_pct=min_improvement_pct,
        min_runs=min_runs,
    )


__all__ = [
    "COMPLETION_RATE",
    "TOOL_CORRECTNESS",
    "ArmScore",
    "EvalComparison",
    "EvalQuery",
    "RunOutcome",
    "RunQuery",
    "VariantConfig",
    "compare",
    "evaluate",
    "score_arm",
]
