"""kNN-gate measurement against the held-out labeled set (Track 1L / ADR-0023).

Loads the per-account ``holdout-labels.jsonl``, embeds each labeled
email, runs the kNN top-1+top-2 lookup, and records (cos_top1, margin,
predicted_root, true_root). Sweeps a (cos_min × margin_min) grid and
recommends a threshold per ADR-0023 §5.

Output is structured (``MeasurementReport`` Pydantic) so the CLI can
render a Rich table AND write a markdown writeup from the same data.

The measurement does NOT call the LLM — pure-kNN only. This is the
ground-truth probe of what the current centroids can do unaided.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from iris_harness.sdk.config import workspace_dir
from iris_harness.sdk.llm import embed_corpus
from iris_harness.sdk.persistence import data_path
from iris_personal.email.category_store import CategoryStore
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.holdout import HoldoutLabel, load_holdout
from iris_personal.plugins.email_workflows.triage import (
    CategoryCentroid,
    build_centroids,
    root_margin,
)

logger = logging.getLogger(__name__)

# Sweep grid per ADR-0023 §4
# 0.35-0.45 added for the root-aware margin re-sweep: the Phase 3
# decomposition showed every abstained holdout message sits at
# cos_top1 0.26-0.498 — the old grid could not even see that region.
DEFAULT_COS_MIN_GRID: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80)
DEFAULT_MARGIN_MIN_GRID: tuple[float, ...] = (0.02, 0.05, 0.08, 0.12, 0.20)

# Threshold-pick constraint per ADR-0023 §5
MIN_GATED_FRACTION = 0.5


# ─── Result types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HoldoutPrediction:
    """One labeled email run through pure-kNN, without any threshold gating.

    The gate is applied later by ``sweep`` — predictions carry the raw
    cosine signals so the sweep can compute (gated_count, accuracy) at
    each (cos_min, margin_min) combination without re-classifying.
    """

    message_id: str
    true_root: str
    predicted_path: str | None  # full path if a centroid existed; None if no categories
    predicted_root: str | None  # path.split('/')[1] for convenience
    cos_top1: float
    margin: float


class SweepCell(BaseModel):
    """One (cos_min, margin_min) row of the sweep grid."""

    model_config = ConfigDict(frozen=True)

    cos_min: float
    margin_min: float
    total: int
    gated_count: int
    gated_correct: int
    gated_accuracy: float = Field(..., ge=0.0, le=1.0)
    queue_rate: float = Field(..., ge=0.0, le=1.0)


class MeasurementReport(BaseModel):
    """Full output of an ``iris email knn-gate`` run."""

    model_config = ConfigDict(frozen=True)

    account_id: str
    measured_at: datetime
    total_labels: int
    label_distribution: dict[str, int]
    predictions: list[dict[str, Any]]  # one per labeled email, serializable
    sweep: list[SweepCell]
    recommended_cos_min: float
    recommended_margin_min: float
    recommended_accuracy: float
    recommended_gated_count: int
    note: str = ""  # e.g. "no combination meets ≥50% gated rate; widening"

    # Root-level confusion: dict[true_root → dict[predicted_root → count]]
    # Built from the sweep's recommended threshold for diagnostics.
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)


# ─── Core ────────────────────────────────────────────────────────────────────


def _label_to_email_message(label: HoldoutLabel) -> EmailMessage:
    """Project the denormalized HoldoutLabel into an EmailMessage shaped
    for classify_pure_knn's embedder."""
    return EmailMessage(
        id=label.message_id,
        provider="gmail",  # provider value doesn't affect kNN; label is account-tagged
        account_id=label.account_id,
        from_address=label.from_address,
        from_domain=label.from_domain,
        subject=label.subject,
        snippet=label.snippet,
        received_at=label.received_at,
    )


def _path_to_root(path: str | None) -> str | None:
    if not path:
        return None
    parts = path.split("/")
    return parts[1] if len(parts) >= 2 else None


def _predict_all(
    labels: list[HoldoutLabel],
    centroids: list[CategoryCentroid],
    *,
    embed_model: str,
    embedder: Any = None,  # injection point for tests
) -> list[HoldoutPrediction]:
    """Embed every labeled email, compute kNN top-1+top-2 against the
    given centroids, return raw predictions. NO threshold gating yet.
    """
    if not centroids:
        return [
            HoldoutPrediction(
                message_id=lbl.message_id,
                true_root=lbl.true_root,
                predicted_path=None,
                predicted_root=None,
                cos_top1=0.0,
                margin=0.0,
            )
            for lbl in labels
        ]

    texts = []
    for lbl in labels:
        # Mirror what classify_pure_knn embeds (sender + subject + snippet)
        msg = _label_to_email_message(lbl)
        domain = msg.from_domain or ""
        texts.append(
            f"From: {msg.from_address} ({domain})\n"
            f"Subject: {msg.subject}\n"
            f"Snippet: {msg.snippet}"
        )

    embed_fn = embedder if embedder is not None else embed_corpus
    email_vecs = embed_fn(texts, embed_model)

    matrix = np.stack([c.centroid for c in centroids])  # [k, D]
    # sims [N, k] — each row is one email's similarity to every centroid
    sims = email_vecs @ matrix.T

    predictions: list[HoldoutPrediction] = []
    for i, lbl in enumerate(labels):
        row = sims[i]
        order = np.argsort(-row)
        top1_idx = int(order[0])
        top1_sim = float(row[top1_idx])
        # Root-aware margin (ADR-0022 amendment, Phase 3): measure
        # cross-root competition only, matching classify_pure_knn.
        ordered = [centroids[int(i)] for i in order]
        ordered_sims = [float(row[int(i)]) for i in order]
        margin = root_margin(ordered_sims, ordered)
        chosen = centroids[top1_idx]
        predictions.append(
            HoldoutPrediction(
                message_id=lbl.message_id,
                true_root=lbl.true_root,
                predicted_path=chosen.path,
                predicted_root=_path_to_root(chosen.path),
                cos_top1=top1_sim,
                margin=margin,
            )
        )
    return predictions


def sweep_grid(
    predictions: list[HoldoutPrediction],
    *,
    cos_min_grid: tuple[float, ...] = DEFAULT_COS_MIN_GRID,
    margin_min_grid: tuple[float, ...] = DEFAULT_MARGIN_MIN_GRID,
) -> list[SweepCell]:
    """For every (cos_min, margin_min) in the grid, compute
    (gated_count, gated_accuracy, queue_rate)."""
    total = len(predictions)
    out: list[SweepCell] = []
    for cos_min in cos_min_grid:
        for margin_min in margin_min_grid:
            gated = [p for p in predictions if p.cos_top1 >= cos_min and p.margin >= margin_min]
            gated_count = len(gated)
            gated_correct = sum(1 for p in gated if p.predicted_root == p.true_root)
            gated_accuracy = (gated_correct / gated_count) if gated_count else 0.0
            queue_rate = ((total - gated_count) / total) if total else 1.0
            out.append(
                SweepCell(
                    cos_min=cos_min,
                    margin_min=margin_min,
                    total=total,
                    gated_count=gated_count,
                    gated_correct=gated_correct,
                    gated_accuracy=gated_accuracy,
                    queue_rate=queue_rate,
                )
            )
    return out


def pick_threshold(
    sweep: list[SweepCell],
    *,
    min_gated_fraction: float = MIN_GATED_FRACTION,
) -> tuple[SweepCell, str]:
    """ADR-0023 §5: max gated_accuracy s.t. gated_count ≥ min_gated_fraction.

    Returns ``(best, note)`` — ``note`` is empty when a regular pick was
    possible, else a string explaining why the constraint had to be
    relaxed (e.g. "no combination meets ≥50% gated rate").
    """
    if not sweep:
        raise ValueError("empty sweep — cannot pick a threshold")
    total = sweep[0].total

    def _key(c: SweepCell) -> tuple[float, int]:
        return (c.gated_accuracy, c.gated_count)

    candidates = [c for c in sweep if total and (c.gated_count / total) >= min_gated_fraction]
    if candidates:
        return max(candidates, key=_key), ""

    # Relax: no combination meets the floor. Recommend the loosest gate
    # so at least *some* triage happens, and warn in the note.
    note = (
        f"no combination meets gated_fraction ≥ {min_gated_fraction:.0%}; "
        "recommending the loosest gate (lowest cos_min + margin_min) as a fallback"
    )
    loosest = min(sweep, key=lambda c: (c.cos_min, c.margin_min))
    return loosest, note


def _confusion_at(
    predictions: list[HoldoutPrediction], cos_min: float, margin_min: float
) -> dict[str, dict[str, int]]:
    """Build a (true_root → predicted_root → count) matrix from the gated
    subset at the given thresholds."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in predictions:
        if p.cos_top1 < cos_min or p.margin < margin_min:
            continue
        true = p.true_root
        pred = p.predicted_root or "(no-centroid)"
        out[true][pred] += 1
    # Convert to plain dicts for serialization
    return {k: dict(v) for k, v in out.items()}


# ─── Corrections → HoldoutLabel projection ──────────────────────────────────


def corrections_as_holdout(
    account_id: str,
    *,
    category_store: CategoryStore,
    email_store: EmailStore,
) -> list[HoldoutLabel]:
    """Project user-classification corrections into synthetic
    ``HoldoutLabel`` rows so they can join the measurement set.

    Per ADR-0024 §3: each correction in ``categories_history`` has a
    ``new_path`` (the user's correct answer) and a ``message_id``
    (the email that was wrong). We synthesize:
      - ``true_root`` = root component of new_path
      - envelope fields come from the live ``emails`` row (best-effort)
      - ``label_source`` tagged so the measurement can distinguish
        spike-imported vs hand-labeled vs correction-derived rows
        when needed later

    Rows where the message is no longer in email.db are dropped
    silently (the correction is still recorded in history; we just
    can't re-classify what we can't read).
    """
    corrections = category_store.list_corrections(account_id=account_id, limit=10_000)
    if not corrections:
        return []
    out: list[HoldoutLabel] = []
    for row in corrections:
        payload = row["payload"]
        message_id = payload.get("message_id")
        new_path = payload.get("new_path") or row["new_path"]
        if not message_id or not new_path:
            continue
        msg = email_store.get(message_id)
        if msg is None:
            continue
        parts = new_path.split("/")
        if len(parts) < 2:
            continue
        true_root = parts[1]
        try:
            out.append(
                HoldoutLabel(
                    message_id=message_id,
                    account_id=account_id,
                    true_root=true_root,
                    label_source="user-classification-correction",
                    from_address=msg.from_address,
                    from_domain=msg.from_domain,
                    subject=msg.subject,
                    snippet=msg.snippet,
                    received_at=msg.received_at,
                )
            )
        except Exception as exc:  # noqa: BLE001 — skip bad rows, keep going
            logger.warning("skipping correction for %s: %s", message_id, exc)
    return out


# ─── Orchestrator ────────────────────────────────────────────────────────────


@dataclass
class KnnGateRunner:
    """Top-level entry point used by ``iris email knn-gate``."""

    workspace_dir: Path = field(default_factory=workspace_dir)
    db_path: Path = field(default_factory=lambda: data_path("iris.db"))
    email_db_path: Path = field(default_factory=lambda: data_path("email.db"))
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder: Any = None  # test injection

    def measure(
        self,
        account_id: str,
        *,
        labels_path: Path | None = None,
        include_corrections: bool = False,
    ) -> MeasurementReport:
        from iris_personal.plugins.email_workflows.holdout import holdout_path

        path = labels_path or holdout_path(self.workspace_dir, account_id)
        labels = load_holdout(path)
        labels = [lbl for lbl in labels if lbl.account_id == account_id]

        # ADR-0024 — augment with user corrections.
        if include_corrections:
            cat_store = CategoryStore(db_path=self.db_path)
            cat_store.ensure_schema()
            email_store = EmailStore(db_path=self.email_db_path)
            email_store.ensure_schema()
            extra = corrections_as_holdout(
                account_id,
                category_store=cat_store,
                email_store=email_store,
            )
            existing_ids = {lbl.message_id for lbl in labels}
            labels = labels + [lbl for lbl in extra if lbl.message_id not in existing_ids]

        if not labels:
            raise ValueError(
                f"no labels for {account_id} at {path}. " "Run `iris email label-holdout` first."
            )

        # Pull centroids — same code path as classify_pure_knn
        store = CategoryStore(db_path=self.db_path)
        store.ensure_schema()
        active = store.list(type="email", account_id=account_id, active_only=True)
        active_paths = {c.path for c in active}
        if not active_paths:
            raise ValueError(
                f"no active categories for {account_id}. "
                "Run `iris email bootstrap-categories` + `iris email "
                "accept-categories` first."
            )

        # Build centroids honoring the embedder injection
        if self.embedder is not None:
            import iris_personal.plugins.email_workflows.triage as triage_mod

            original = triage_mod.embed_corpus
            triage_mod.embed_corpus = self.embedder
            try:
                centroids = build_centroids(
                    self.workspace_dir, account_id, active_paths=active_paths
                )
            finally:
                triage_mod.embed_corpus = original
        else:
            centroids = build_centroids(
                self.workspace_dir,
                account_id,
                active_paths=active_paths,
                embed_model=self.embed_model,
            )

        predictions = _predict_all(
            labels,
            centroids,
            embed_model=self.embed_model,
            embedder=self.embedder,
        )

        sweep = sweep_grid(predictions)
        best, note = pick_threshold(sweep)
        confusion = _confusion_at(predictions, best.cos_min, best.margin_min)

        return MeasurementReport(
            account_id=account_id,
            measured_at=datetime.now(UTC),
            total_labels=len(labels),
            label_distribution=dict(Counter(lbl.true_root for lbl in labels)),
            predictions=[
                {
                    "message_id": p.message_id,
                    "true_root": p.true_root,
                    "predicted_path": p.predicted_path,
                    "predicted_root": p.predicted_root,
                    "cos_top1": p.cos_top1,
                    "margin": p.margin,
                }
                for p in predictions
            ],
            sweep=sweep,
            recommended_cos_min=best.cos_min,
            recommended_margin_min=best.margin_min,
            recommended_accuracy=best.gated_accuracy,
            recommended_gated_count=best.gated_count,
            note=note,
            confusion=confusion,
        )


def render_writeup(report: MeasurementReport) -> str:
    """Markdown writeup per ADR-0023 §7 — pure function, no I/O."""
    lines: list[str] = []
    lines.append(f"# kNN-gate measurement — {report.account_id}")
    lines.append("")
    lines.append(f"Measured: {report.measured_at.isoformat()}")
    lines.append(f"Holdout size: {report.total_labels}")
    lines.append("")
    lines.append("## Label distribution")
    lines.append("")
    lines.append("| Root | Count |")
    lines.append("|---|---:|")
    for root, count in sorted(report.label_distribution.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{root}` | {count} |")
    lines.append("")
    lines.append("## Sweep grid")
    lines.append("")
    lines.append("| cos_min | margin_min | gated | accuracy | queue_rate |")
    lines.append("|---:|---:|---:|---:|---:|")
    for cell in report.sweep:
        lines.append(
            f"| {cell.cos_min:.2f} | {cell.margin_min:.2f} | "
            f"{cell.gated_count}/{cell.total} | "
            f"{cell.gated_accuracy:.1%} | {cell.queue_rate:.1%} |"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"- **cos_min**: `{report.recommended_cos_min:.2f}`")
    lines.append(f"- **margin_min**: `{report.recommended_margin_min:.2f}`")
    lines.append(
        f"- **Gated accuracy**: {report.recommended_accuracy:.1%} on "
        f"{report.recommended_gated_count}/{report.total_labels} labels"
    )
    if report.note:
        lines.append(f"- Note: {report.note}")
    lines.append("")
    if report.confusion:
        lines.append("## Confusion matrix (at recommended thresholds)")
        lines.append("")
        # Header — predicted columns
        all_pred = sorted({p for row in report.confusion.values() for p in row.keys()})
        lines.append("| true \\ pred | " + " | ".join(f"`{p}`" for p in all_pred) + " |")
        lines.append("|---|" + "|".join("---:" for _ in all_pred) + "|")
        for true_root in sorted(report.confusion.keys()):
            row = report.confusion[true_root]
            cells = " | ".join(str(row.get(p, 0)) for p in all_pred)
            lines.append(f"| `{true_root}` | {cells} |")
        lines.append("")
    return "\n".join(lines)
