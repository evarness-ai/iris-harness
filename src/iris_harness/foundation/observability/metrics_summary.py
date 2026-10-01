"""Stable summaries of recent LLM activity for operator-facing APIs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import session_log


def summarize_llm_metrics(
    *,
    log_dir: Path | None = None,
    max_sessions: int = 25,
    learning_store: Any | None = None,
    process_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate recent ``session_log`` LLM activity into a stable schema.

    When ``learning_store`` (a ``LearningMetricsStore``) is supplied, a
    ``learning`` section is merged in with signal-stream completeness (recorded
    vs. dropped), per-metric signal volume, and the experiment lifecycle tally,
    so "is self-learning working?" is answerable from one call
    (learning-observability.md §4.4). Decoupled by duck-typing — observability
    never hard-depends on the learning package.

    ``process_signals`` is the in-process signal counters (``learning.signals
    .signal_health()``). It is passed in for the same reason ``learning_store`` is:
    until M6.2 this module imported that function directly, which was the one place the
    "never hard-depends" promise above was not kept — and it is an upward import from
    the bottom layer.
    """
    root = log_dir or session_log.session_log_dir()
    if not root.exists():
        summary = _empty_summary()
        summary["learning"] = _learning_summary(learning_store, process_signals)
        return summary

    call_count = 0
    error_count = 0
    total_tokens = 0
    total_duration_ms = 0.0
    providers: Counter[str] = Counter()
    models: Counter[str] = Counter()

    session_paths = sorted(
        root.glob("session-*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[: max(0, max_sessions)]

    for path in session_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(payload.get("kind") or "")
            if kind == "llm_call":
                call_count += 1
                provider = str(payload.get("provider") or "unknown")
                model = str(payload.get("model") or "unknown")
                providers[provider] += 1
                models[model] += 1
                tokens = payload.get("tokens")
                if isinstance(tokens, dict):
                    total_tokens += _coerce_int(tokens.get("total_tokens"))
                total_duration_ms += _coerce_float(payload.get("duration_ms"))
            elif kind == "error" and str(payload.get("phase") or "") == "llm_call":
                error_count += 1

    return {
        "sessions_scanned": len(session_paths),
        "llm_call_count": call_count,
        "llm_error_count": error_count,
        "total_tokens": total_tokens,
        "total_duration_ms": round(total_duration_ms, 3),
        "provider_counts": dict(providers),
        "model_counts": dict(models),
        "learning": _learning_summary(learning_store, process_signals),
    }


def _learning_summary(
    learning_store: Any | None, process_signals: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Merge durable store health with the in-process signal counters the caller has."""
    summary: dict[str, Any] = {}
    summary["process"] = process_signals if process_signals is not None else {}
    if learning_store is not None:
        try:
            summary.update(learning_store.health_summary())
        except Exception:  # noqa: BLE001 — never let a summary read break callers
            summary["error"] = "health_summary unavailable"
    return summary


def _empty_summary() -> dict[str, Any]:
    return {
        "sessions_scanned": 0,
        "llm_call_count": 0,
        "llm_error_count": 0,
        "total_tokens": 0,
        "total_duration_ms": 0.0,
        "provider_counts": {},
        "model_counts": {},
    }


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
