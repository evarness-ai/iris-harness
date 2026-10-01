"""Lightweight analytics retriever for persona learning signals.

Reads completed task reward summaries from SQLite and exposes per-persona
retry and score aggregations for offline learning feedback loops.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iris_harness.foundation.persistence.sqlite import connect

# Repo-relative path of the coding agent's task database. The retriever only reads
# it (and returns nothing when it is absent); the coding agent imports this constant
# so both sides agree on the location without the core depending on the agent.
CODING_TASK_DB = Path("data/coding_tasks.db")


@dataclass
class PersonaLearningSignal:
    """A single learning signal extracted from a completed task reward summary."""

    task_id: str
    persona_name: str
    metric_name: str
    score: float
    retry_count: int
    iterations_to_pass: int | None
    lesson: str  # evidence string from the reward signal


def load_persona_learning_signals(
    repo_root: Path,
    *,
    persona: str | None = None,
    limit: int = 100,
) -> list[PersonaLearningSignal]:
    """Load persona learning signals from the coding task SQLite database.

    Args:
        repo_root: Root of the project repository.
        persona: Optional persona name to filter by (e.g. ``"developer"``).
        limit: Maximum number of signals to return.

    Returns:
        List of :class:`PersonaLearningSignal` instances ordered by most
        recently updated task first.
    """
    db_path = repo_root / CODING_TASK_DB
    if not db_path.exists():
        return []

    conn = connect(db_path, row_factory=sqlite3.Row)
    try:
        rows = conn.execute(
            "SELECT task_id, reward_summary_json FROM coding_tasks "
            "WHERE reward_summary_json IS NOT NULL "
            "ORDER BY updated_at DESC "
            "LIMIT ?",
            (limit * 10,),  # over-fetch to compensate for per-signal expansion
        ).fetchall()
    finally:
        conn.close()

    signals: list[PersonaLearningSignal] = []
    for row in rows:
        task_id: str = row["task_id"]
        raw = row["reward_summary_json"]
        try:
            summary_data: dict[str, Any] = json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError:
            continue

        iterations_to_pass: int | None = summary_data.get("iterations_to_pass")
        retry_map: dict[str, int] = summary_data.get("persona_retry_map") or {}
        raw_signals = summary_data.get("signals") or []

        for sig in raw_signals:
            sig_persona = sig.get("persona_name") or sig.get("persona_name__value", "")
            # Pydantic model_dump may serialize enum values as str or dict
            if isinstance(sig_persona, dict):
                sig_persona = sig_persona.get("value", "")
            sig_metric = sig.get("metric_name") or sig.get("metric_name__value", "")
            if isinstance(sig_metric, dict):
                sig_metric = sig_metric.get("value", "")

            if persona is not None and sig_persona != persona:
                continue

            retry_count = retry_map.get(sig_persona, 0)
            lesson = sig.get("evidence") or ""

            signals.append(
                PersonaLearningSignal(
                    task_id=task_id,
                    persona_name=str(sig_persona),
                    metric_name=str(sig_metric),
                    score=float(sig.get("score", 0.0)),
                    retry_count=retry_count,
                    iterations_to_pass=(
                        int(iterations_to_pass) if iterations_to_pass is not None else None
                    ),
                    lesson=lesson,
                )
            )
        if len(signals) >= limit:
            break

    return signals[:limit]


def compute_persona_retry_summary(
    signals: list[PersonaLearningSignal],
) -> dict[str, dict[str, float]]:
    """Aggregate retry and score statistics per persona.

    Args:
        signals: List of :class:`PersonaLearningSignal` instances.

    Returns:
        Mapping of persona name → ``{"avg_retry_count": float, "avg_score": float,
        "sample_count": int}``.
    """
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}

    for sig in signals:
        p = sig.persona_name
        if p not in totals:
            totals[p] = {"total_retry": 0.0, "total_score": 0.0}
            counts[p] = 0
        totals[p]["total_retry"] += sig.retry_count
        totals[p]["total_score"] += sig.score
        counts[p] += 1

    result: dict[str, dict[str, float]] = {}
    for p, agg in totals.items():
        n = counts[p]
        result[p] = {
            "avg_retry_count": round(agg["total_retry"] / n, 4),
            "avg_score": round(agg["total_score"] / n, 4),
            "sample_count": float(n),
        }
    return result
