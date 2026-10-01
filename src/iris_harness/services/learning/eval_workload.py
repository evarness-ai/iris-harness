"""Frozen replay workloads for the eval harness (ADR-0070, slice 1).

A workload is a list of :class:`~iris_harness.services.learning.eval_harness.EvalQuery`. It is
*frozen* on purpose: the counterfactual (baseline vs variant) is only meaningful
if both arms see the exact same queries, and comparisons across runs are only
stable if the workload doesn't drift. So the flow is: snapshot real queries from
the session logs once (:func:`build_workload_from_traces`), persist them
(:func:`save_workload`), and replay that fixed file thereafter
(:func:`load_workload`).

No labels are required — ``completion_rate`` needs none. ``expected_tool`` may be
hand-added to a saved file to unlock the ``tool_correctness`` metric.
"""

from __future__ import annotations

import json
from pathlib import Path

from iris_harness.foundation.observability.session_log import iter_recent_turns
from iris_harness.services.learning.eval_harness import EvalQuery

# Keep a snapshot tight: enough queries to get signal, few enough that a
# baseline+variant × repeats run stays affordable on local models.
_DEFAULT_MAX_ITEMS = 30


def build_workload_from_traces(
    *,
    intent: str | None = None,
    scan_limit: int = 400,
    max_items: int = _DEFAULT_MAX_ITEMS,
    log_dir: Path | None = None,
) -> list[EvalQuery]:
    """Snapshot a frozen workload from recent real session-log turns.

    Newest-first, de-duplicated by query text. ``intent`` filters to one intent
    (the natural unit for a routing experiment); ``None`` takes all intents.

    When ``intent`` is set the filter is pushed *into* the scan (and over-fetched to
    survive de-duplication) so a low-volume intent isn't crowded out of the newest
    ``scan_limit`` turns by high-volume ones — historically this made calendar/files
    workloads come back empty. ``scan_limit`` then bounds how deep the filtered scan
    goes rather than how many raw turns are read.
    """
    if intent is not None:
        # Dedup + require-query while scanning so `limit` counts DISTINCT real user
        # queries — a low-volume intent isn't crowded out, and a heavily repeated
        # query (some recur hundreds of times) doesn't fill the budget. Scan deep
        # (scan_cap) to get past high-volume intents (heartbeats, routine authoring).
        turns = iter_recent_turns(
            limit=max_items,
            log_dir=log_dir,
            intent=intent,
            scan_cap=max(scan_limit, 100_000),
            require_query=True,
            unique_queries=True,
        )
    else:
        turns = iter_recent_turns(limit=scan_limit, log_dir=log_dir)
    seen: set[str] = set()
    items: list[EvalQuery] = []
    for turn in turns:
        query = turn.query.strip()
        if not query or query in seen:
            continue
        seen.add(query)
        items.append(EvalQuery(query=query, intent=turn.intent or (intent or "")))
        if len(items) >= max_items:
            break
    return items


def save_workload(items: list[EvalQuery], path: Path) -> None:
    """Persist a frozen workload as JSON (stable, reviewable, hand-labellable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"query": it.query, "intent": it.intent, "expected_tool": it.expected_tool} for it in items
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_workload(path: Path) -> list[EvalQuery]:
    """Load a frozen workload from JSON. Missing/malformed file -> empty list."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    items: list[EvalQuery] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        query = str(entry.get("query") or "").strip()
        if not query:
            continue
        expected = entry.get("expected_tool")
        items.append(
            EvalQuery(
                query=query,
                intent=str(entry.get("intent") or ""),
                expected_tool=str(expected) if expected else None,
            )
        )
    return items


__all__ = ["build_workload_from_traces", "load_workload", "save_workload"]
