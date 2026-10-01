"""Learning intelligence — the deterministic measurement layer (ADR-0069 #4).

This is the *measure* half of "how is the system learning, and how accurate are
its signals?". It computes nothing speculative: every number here is a plain
aggregate over the correlated signals the self-learning program already records
(``learning.db``, ADR-0068). Interpreting these numbers — naming patterns,
flagging issues, recommending changes — is deliberately NOT done here; that is
the job of the agentic analyst (slice 2), which reads this report. Keeping the
substrate free of hand-coded thresholds keeps it trustworthy and keeps the
judgment with a model (the house rule: measure don't guess; agentic over
heuristic).

Two measured blocks:

* :class:`SignalAccuracy` — are the signals themselves trustworthy? The shadow
  escalation judge's precision against its outcome proxy (of the turns it would
  have escalated, how many actually erred), plus signal volume and drop rate.
* :class:`OutcomeCell` matrix — per ``(intent, resolved_tier)``: task completion
  rate, user-correction rate (when that judge is enabled), downstream-reuse
  count, and average token cost. The raw shape a reader (human or model) needs
  to spot "intent X on tier1 underperforms".

The existing experiments ledger (``experiment_status_counts``) is referenced,
not recomputed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iris_harness.services.learning.store import LearningMetricsStore, SignalRecord

# Metrics the matrix + accuracy figures are built from. All correlated and
# (except user_correction) always-on per learning-observability.md §4.2.
_MATRIX_METRICS = ("task_completed", "turn_tokens", "user_correction", "downstream_reuse")
_ACCURACY_METRICS = ("escalation_shadow",)
_FEEDBACK_METRIC = "user_feedback"

_DEFAULT_WINDOW = timedelta(days=7)
_UNKNOWN = "unknown"


@dataclass(frozen=True)
class SignalAccuracy:
    """Trustworthiness of the learning signals themselves.

    ``escalation_precision`` is the fraction of would-escalate turns that
    actually carried an error — the §4.3 counterfactual read: when the shadow
    judge says "a bigger model would help", was the turn genuinely worse? It is
    ``None`` until there is at least one would-act sample (no false precision
    from an empty set).
    """

    escalation_shadow_total: int
    escalation_would_act: int
    escalation_would_act_rate: float
    escalation_precision: float | None
    signals_recorded_total: int
    signals_dropped_total: int
    drop_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "escalation_shadow_total": self.escalation_shadow_total,
            "escalation_would_act": self.escalation_would_act,
            "escalation_would_act_rate": round(self.escalation_would_act_rate, 4),
            "escalation_precision": (
                None if self.escalation_precision is None else round(self.escalation_precision, 4)
            ),
            "signals_recorded_total": self.signals_recorded_total,
            "signals_dropped_total": self.signals_dropped_total,
            "drop_rate": round(self.drop_rate, 4),
        }


@dataclass(frozen=True)
class OutcomeCell:
    """Measured outcomes for one ``(intent, tier)`` slice of traffic."""

    intent: str
    tier: str
    samples: int
    completion_rate: float
    correction_rate: float | None
    correction_samples: int
    reuse_count: int
    avg_tokens: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "tier": self.tier,
            "samples": self.samples,
            "completion_rate": round(self.completion_rate, 4),
            "correction_rate": (
                None if self.correction_rate is None else round(self.correction_rate, 4)
            ),
            "correction_samples": self.correction_samples,
            "reuse_count": self.reuse_count,
            "avg_tokens": (None if self.avg_tokens is None else round(self.avg_tokens, 1)),
        }


@dataclass(frozen=True)
class FeedbackIntent:
    """Explicit user feedback (ADR-0072) for one intent over the window."""

    intent: str
    positive: int
    negative: int

    @property
    def total(self) -> int:
        return self.positive + self.negative

    @property
    def satisfaction(self) -> float:
        return self.positive / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "positive": self.positive,
            "negative": self.negative,
            "satisfaction": round(self.satisfaction, 4),
        }


@dataclass(frozen=True)
class FeedbackSummary:
    """Aggregate of the ``user_feedback`` signal (ADR-0072 slice 4) over the window.

    The first EXPLICIT signal in the loop: a thumbs up/down the user volunteered,
    correlated to a turn. Surfaced in the report so the analyst (slice 2) can weigh
    self-reported dissatisfaction alongside the measured outcome matrix.
    """

    total: int
    positive: int
    negative: int
    by_intent: tuple[FeedbackIntent, ...] = ()

    @property
    def satisfaction(self) -> float:
        return self.positive / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "positive": self.positive,
            "negative": self.negative,
            "satisfaction": round(self.satisfaction, 4) if self.total else None,
            "by_intent": [b.as_dict() for b in self.by_intent],
        }


@dataclass(frozen=True)
class IntelligenceReport:
    """The full measured snapshot the agentic analyst (slice 2) consumes."""

    sampled_at: datetime
    window_hours: float
    accuracy: SignalAccuracy
    matrix: tuple[OutcomeCell, ...]
    experiments: dict[str, int]
    feedback: FeedbackSummary = field(default_factory=lambda: FeedbackSummary(0, 0, 0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "sampled_at": self.sampled_at.isoformat(),
            "window_hours": round(self.window_hours, 2),
            "accuracy": self.accuracy.as_dict(),
            "matrix": [cell.as_dict() for cell in self.matrix],
            "experiments": dict(self.experiments),
            "feedback": self.feedback.as_dict(),
        }


def _intent_of(sig: SignalRecord) -> str:
    value = sig.metadata.get("intent")
    return str(value) if value else _UNKNOWN


def _tier_of(sig: SignalRecord) -> str:
    return sig.resolved_tier or _UNKNOWN


def _build_accuracy(shadow: list[SignalRecord], *, health: dict[str, Any]) -> SignalAccuracy:
    total = len(shadow)
    would_act = [s for s in shadow if s.value >= 1.0]
    n_act = len(would_act)
    # Precision against the has_errors proxy carried on each shadow row.
    precision: float | None
    if n_act:
        errored = sum(1 for s in would_act if bool(s.metadata.get("has_errors")))
        precision = errored / n_act
    else:
        precision = None
    return SignalAccuracy(
        escalation_shadow_total=total,
        escalation_would_act=n_act,
        escalation_would_act_rate=(n_act / total) if total else 0.0,
        escalation_precision=precision,
        signals_recorded_total=int(health.get("signals_recorded_total", 0)),
        signals_dropped_total=int(health.get("signals_dropped_total", 0)),
        drop_rate=float(health.get("drop_rate", 0.0)),
    )


def _build_matrix(rows: list[SignalRecord]) -> tuple[OutcomeCell, ...]:
    # Accumulate per (intent, tier) without holding every row: each metric folds
    # into running sums so the matrix cost is O(signals), not O(cells x signals).
    completion: dict[tuple[str, str], list[float]] = defaultdict(list)
    correction: dict[tuple[str, str], list[float]] = defaultdict(list)
    reuse: dict[tuple[str, str], int] = defaultdict(int)
    tokens: dict[tuple[str, str], list[float]] = defaultdict(list)

    for sig in rows:
        key = (_intent_of(sig), _tier_of(sig))
        if sig.metric_name == "task_completed":
            completion[key].append(sig.value)
        elif sig.metric_name == "user_correction":
            correction[key].append(sig.value)
        elif sig.metric_name == "downstream_reuse":
            reuse[key] += 1
        elif sig.metric_name == "turn_tokens":
            tokens[key].append(sig.value)

    # The matrix is keyed by traffic that actually completed turns; reuse/tokens
    # without a task_completed row are folded in via the union of keys so an
    # all-reuse cell still shows. completion drives the primary sample count.
    keys = set(completion) | set(correction) | set(reuse) | set(tokens)
    cells: list[OutcomeCell] = []
    for key in keys:
        comp = completion.get(key, [])
        corr = correction.get(key, [])
        tok = tokens.get(key, [])
        cells.append(
            OutcomeCell(
                intent=key[0],
                tier=key[1],
                samples=len(comp),
                completion_rate=(sum(comp) / len(comp)) if comp else 0.0,
                correction_rate=(sum(corr) / len(corr)) if corr else None,
                correction_samples=len(corr),
                reuse_count=reuse.get(key, 0),
                avg_tokens=(sum(tok) / len(tok)) if tok else None,
            )
        )
    # Stable, useful order: most-trafficked first, then by name for determinism.
    cells.sort(key=lambda c: (-c.samples, c.intent, c.tier))
    return tuple(cells)


def _build_feedback(rows: list[SignalRecord]) -> FeedbackSummary:
    """Fold ``user_feedback`` rows into totals + a per-intent breakdown.

    Value carries the sign (+1 up / -1 down); intent comes from the metadata the
    feedback POST stamped on the signal.
    """
    positive = 0
    negative = 0
    per_intent: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [positive, negative]
    for sig in rows:
        intent = _intent_of(sig)
        if sig.value > 0:
            positive += 1
            per_intent[intent][0] += 1
        else:
            negative += 1
            per_intent[intent][1] += 1
    by_intent = tuple(
        sorted(
            (FeedbackIntent(intent=i, positive=p, negative=n) for i, (p, n) in per_intent.items()),
            key=lambda f: (-f.total, f.intent),
        )
    )
    return FeedbackSummary(
        total=positive + negative, positive=positive, negative=negative, by_intent=by_intent
    )


def build_intelligence(
    store: LearningMetricsStore,
    *,
    window: timedelta | None = None,
    now: datetime | None = None,
) -> IntelligenceReport:
    """Build the measured learning-intelligence snapshot over a recent window.

    Pure read: no LLM, no thresholds, no side effects. ``window`` defaults to 7
    days; ``now`` is injectable for deterministic tests.
    """
    win = window if window is not None else _DEFAULT_WINDOW
    moment = now or datetime.now(UTC)
    matrix_rows = store.signals_in_window(metric_names=_MATRIX_METRICS, window=win, now=moment)
    shadow_rows = store.signals_in_window(metric_names=_ACCURACY_METRICS, window=win, now=moment)
    feedback_rows = store.signals_in_window(
        metric_names=(_FEEDBACK_METRIC,), window=win, now=moment
    )
    health = store.health_summary(window=win, now=moment)
    return IntelligenceReport(
        sampled_at=moment,
        window_hours=win.total_seconds() / 3600.0,
        accuracy=_build_accuracy(shadow_rows, health=health),
        matrix=_build_matrix(matrix_rows),
        experiments=store.experiment_status_counts(),
        feedback=_build_feedback(feedback_rows),
    )


def render_text(report: IntelligenceReport) -> str:
    """Agent-readable rendering for the ``learning_intelligence`` tool.

    Plain text the ReAct loop can quote: the accuracy line, then a compact
    matrix table. No interpretation — the model does that from the numbers.
    """
    acc = report.accuracy
    lines: list[str] = [
        f"Learning intelligence (last {report.window_hours:.0f}h):",
        "",
        "Signal accuracy / integrity:",
        f"  escalation judge: {acc.escalation_would_act}/{acc.escalation_shadow_total} "
        f"would-escalate"
        + (
            f", precision {acc.escalation_precision:.0%} (of those, share that actually erred)"
            if acc.escalation_precision is not None
            else " (no would-act samples yet)"
        ),
        f"  signals recorded: {acc.signals_recorded_total}, "
        f"dropped: {acc.signals_dropped_total} ({acc.drop_rate:.1%})",
        "",
    ]
    if report.matrix:
        lines.append("Outcomes by intent x tier (completion / correction / reuse / avg tokens):")
        for cell in report.matrix:
            corr = "n/a" if cell.correction_rate is None else f"{cell.correction_rate:.0%}"
            toks = "n/a" if cell.avg_tokens is None else f"{cell.avg_tokens:.0f}"
            lines.append(
                f"  {cell.intent} @ {cell.tier}: "
                f"{cell.completion_rate:.0%} done (n={cell.samples}), "
                f"corr {corr}, reuse {cell.reuse_count}, ~{toks} tok"
            )
    else:
        lines.append("Outcomes by intent x tier: no measured turns in window yet.")
    fb = report.feedback
    if fb.total:
        worst = min(
            (f for f in fb.by_intent if f.total), key=lambda f: f.satisfaction, default=None
        )
        worst_note = (
            f"; lowest: {worst.intent} {worst.satisfaction:.0%} (n={worst.total})" if worst else ""
        )
        lines.extend(
            [
                "",
                f"User feedback: {fb.satisfaction:.0%} satisfied "
                f"({fb.positive} up / {fb.negative} down, n={fb.total}){worst_note}",
            ]
        )
    if report.experiments:
        ledger = ", ".join(f"{k}={v}" for k, v in sorted(report.experiments.items()))
        lines.extend(["", f"Experiments ledger: {ledger}"])
    return "\n".join(lines)


__all__ = [
    "IntelligenceReport",
    "OutcomeCell",
    "SignalAccuracy",
    "FeedbackSummary",
    "FeedbackIntent",
    "build_intelligence",
    "render_text",
]
