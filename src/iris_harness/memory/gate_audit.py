"""Measure the memory garbage-in / garbage-out guarantees on labelled data.

Two defences keep junk out of the user profile (see `docs/architecture/memory-subsystem.md`):

  - **capture gates** (`fact_validation.py`: plausibility + durability + grounding) decide
    what enters the store — *garbage in*;
  - **recall filter** (`retriever._filter_facts`: `min_fact_confidence` drop,
    `uncertain_below` mark) decides what reaches the prompt — *garbage out*.

Both ship with thresholds, but nothing *measured* whether those thresholds are right. This
module is the missing meter: given **labelled** fact candidates (`keep` = a real durable fact
that should be admitted + recalled; `junk` = pollution that should be blocked), it scores the
gates' and the filter's precision/recall, attributes each block to the gate that made it, and
sweeps the recall threshold to suggest a tuned value. Pure + deterministic — no LLM, no I/O —
so it runs on a built-in synthetic corpus out of the box and is fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from iris_harness.memory.fact_keys import canonical_key, is_self_statement
from iris_harness.memory.fact_validation import is_durable_fact, is_fact_grounded, is_plausible_fact

Label = Literal["keep", "junk"]


@dataclass(frozen=True)
class FactCandidate:
    """A labelled candidate. ``message`` is the source utterance (for the grounding gate);
    ``confidence`` is the extractor's score (for the recall filter)."""

    key: str
    value: str
    message: str
    confidence: float
    label: Label


# ---------------------------------------------------------------------------
# Capture-gate audit (garbage IN)
# ---------------------------------------------------------------------------


def _gate_block(c: FactCandidate) -> str | None:
    """Return the name of the first gate that blocks ``c``, or None if all admit.

    Mirrors the live capture order, which now starts with two filters ahead of the
    validation gates: the message has to be the user talking about themselves, and the
    key has to be an allowed profile key (a fact mapping in config/memory/mappings.yaml).
    """
    if not is_self_statement(c.message):
        return "first_person"
    if canonical_key(c.key) is None:
        return "key_allowlist"
    ok, _ = is_plausible_fact(c.key, c.value)
    if not ok:
        return "plausibility"
    ok, _ = is_durable_fact(c.key, c.value)
    if not ok:
        return "durability"
    if not is_fact_grounded(c.value, c.message):
        return "grounding"
    return None


@dataclass(frozen=True)
class GateReport:
    """Precision/recall of the capture gates, admit = positive."""

    total: int
    keep: int
    junk: int
    true_admit: int  # keep admitted (correct)
    false_admit: int  # junk admitted (garbage-IN leak)
    true_reject: int  # junk blocked (correct)
    false_reject: int  # keep blocked (over-blocking, lost a real fact)
    by_gate: dict[str, int]  # which gate blocked how many
    false_reject_by_gate: dict[str, int]  # over-blocks, per gate (the actionable signal)

    @property
    def precision(self) -> float:
        """Of what was admitted, how much is genuinely keep."""
        admitted = self.true_admit + self.false_admit
        return self.true_admit / admitted if admitted else 1.0

    @property
    def recall(self) -> float:
        """Of the real facts, how many the gates let through."""
        return self.true_admit / self.keep if self.keep else 1.0

    @property
    def false_admit_rate(self) -> float:
        return self.false_admit / self.junk if self.junk else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "keep": self.keep,
            "junk": self.junk,
            "true_admit": self.true_admit,
            "false_admit": self.false_admit,
            "true_reject": self.true_reject,
            "false_reject": self.false_reject,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "false_admit_rate": round(self.false_admit_rate, 3),
            "by_gate": dict(self.by_gate),
            "false_reject_by_gate": dict(self.false_reject_by_gate),
        }


def audit_capture_gates(candidates: list[FactCandidate]) -> GateReport:
    """Run each candidate through the three gates and tally the confusion matrix."""
    by_gate: dict[str, int] = {}
    fr_by_gate: dict[str, int] = {}
    ta = fa = tr = fr = 0
    keep = sum(1 for c in candidates if c.label == "keep")
    for c in candidates:
        blocked_by = _gate_block(c)
        admitted = blocked_by is None
        if blocked_by:
            by_gate[blocked_by] = by_gate.get(blocked_by, 0) + 1
        if c.label == "keep":
            if blocked_by is None:
                ta += 1
            else:
                fr += 1
                fr_by_gate[blocked_by] = fr_by_gate.get(blocked_by, 0) + 1
        else:  # junk
            if admitted:
                fa += 1
            else:
                tr += 1
    return GateReport(
        total=len(candidates),
        keep=keep,
        junk=len(candidates) - keep,
        true_admit=ta,
        false_admit=fa,
        true_reject=tr,
        false_reject=fr,
        by_gate=by_gate,
        false_reject_by_gate=fr_by_gate,
    )


# ---------------------------------------------------------------------------
# Recall-filter audit (garbage OUT) + threshold sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecallReport:
    """Precision of what the recall filter lets reach the prompt, at given thresholds."""

    min_fact_confidence: float
    uncertain_below: float
    reaching_prompt: int  # not dropped (uncertain + clean)
    dropped: int
    false_recall: int  # junk that reached the prompt (garbage-OUT leak)
    false_drop: int  # keep that was dropped (lost a real fact)
    uncertain_keep: int  # keep correctly flagged uncertain
    uncertain_junk: int  # junk flagged uncertain (reached prompt, but hedged)

    @property
    def precision(self) -> float:
        return (
            (self.reaching_prompt - self.false_recall) / self.reaching_prompt
            if self.reaching_prompt
            else 1.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_fact_confidence": self.min_fact_confidence,
            "uncertain_below": self.uncertain_below,
            "reaching_prompt": self.reaching_prompt,
            "dropped": self.dropped,
            "false_recall": self.false_recall,
            "false_drop": self.false_drop,
            "uncertain_keep": self.uncertain_keep,
            "uncertain_junk": self.uncertain_junk,
            "precision": round(self.precision, 3),
        }


def audit_recall_filter(
    candidates: list[FactCandidate], *, min_fact_confidence: float, uncertain_below: float
) -> RecallReport:
    """Apply the confidence thresholds and tally what reaches the prompt vs gets dropped."""
    reaching = dropped = fr = fd = unc_keep = unc_junk = 0
    for c in candidates:
        if c.confidence < min_fact_confidence:
            dropped += 1
            if c.label == "keep":
                fd += 1
            continue
        reaching += 1
        uncertain = c.confidence < uncertain_below
        if c.label == "junk":
            fr += 1
            if uncertain:
                unc_junk += 1
        elif uncertain:
            unc_keep += 1
    return RecallReport(
        min_fact_confidence=min_fact_confidence,
        uncertain_below=uncertain_below,
        reaching_prompt=reaching,
        dropped=dropped,
        false_recall=fr,
        false_drop=fd,
        uncertain_keep=unc_keep,
        uncertain_junk=unc_junk,
    )


@dataclass(frozen=True)
class ThresholdSweep:
    rows: list[dict[str, Any]] = field(default_factory=list)
    suggested_min_confidence: float = 0.0
    current_min_confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "suggested_min_confidence": self.suggested_min_confidence,
            "current_min_confidence": self.current_min_confidence,
            "rows": self.rows,
        }


def sweep_recall_threshold(
    candidates: list[FactCandidate],
    *,
    current_min_confidence: float,
    uncertain_below: float,
    step: float = 0.05,
) -> ThresholdSweep:
    """Sweep ``min_fact_confidence`` over [0, 1] and pick the value that best separates
    junk from keep — maximising (true_drop_junk + true_recall_keep) / total. Ties break
    toward the *lower* threshold (keep more real facts)."""
    rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_thr = current_min_confidence
    n = max(1, round(1.0 / step))
    for i in range(n + 1):
        thr = round(i * step, 4)
        rep = audit_recall_filter(
            candidates, min_fact_confidence=thr, uncertain_below=uncertain_below
        )
        keep_total = sum(1 for c in candidates if c.label == "keep")
        junk_total = len(candidates) - keep_total
        kept_keep = keep_total - rep.false_drop
        dropped_junk = junk_total - rep.false_recall
        # Balanced accuracy: average of (keep retained) and (junk blocked).
        ba = 0.5 * (
            (kept_keep / keep_total if keep_total else 1.0)
            + (dropped_junk / junk_total if junk_total else 1.0)
        )
        rows.append(
            {
                "min_confidence": thr,
                "false_recall": rep.false_recall,
                "false_drop": rep.false_drop,
                "balanced_accuracy": round(ba, 3),
            }
        )
        if ba > best_score:
            best_score = ba
            best_thr = thr
    return ThresholdSweep(
        rows=rows,
        suggested_min_confidence=best_thr,
        current_min_confidence=current_min_confidence,
    )


@dataclass(frozen=True)
class FullAudit:
    gates: GateReport
    recall: RecallReport
    sweep: ThresholdSweep

    def as_dict(self) -> dict[str, Any]:
        return {
            "gates": self.gates.as_dict(),
            "recall": self.recall.as_dict(),
            "sweep": self.sweep.as_dict(),
        }


def run_full_audit(
    candidates: list[FactCandidate],
    *,
    min_fact_confidence: float,
    uncertain_below: float,
) -> FullAudit:
    """End-to-end audit modelling the real pipeline: the capture gates run on *all*
    candidates; the recall filter + threshold sweep run only on the ones the gates
    ADMITTED (junk the gates already blocked never reaches the store, so it must not
    inflate the recall-leak count)."""
    gates = audit_capture_gates(candidates)
    admitted = [c for c in candidates if _gate_block(c) is None]
    recall = audit_recall_filter(
        admitted, min_fact_confidence=min_fact_confidence, uncertain_below=uncertain_below
    )
    sweep = sweep_recall_threshold(
        admitted, current_min_confidence=min_fact_confidence, uncertain_below=uncertain_below
    )
    return FullAudit(gates=gates, recall=recall, sweep=sweep)


__all__ = [
    "FactCandidate",
    "FullAudit",
    "GateReport",
    "RecallReport",
    "ThresholdSweep",
    "audit_capture_gates",
    "audit_recall_filter",
    "run_full_audit",
    "sweep_recall_threshold",
]
