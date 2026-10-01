"""Frozen-workload sourcing from session traces (ADR-0070)."""

from __future__ import annotations

import json
from pathlib import Path

from iris_harness.services.learning.eval_harness import EvalQuery
from iris_harness.services.learning.eval_workload import (
    build_workload_from_traces,
    load_workload,
    save_workload,
)


def _write_session(tmp_path: Path, name: str, events: list[dict]) -> None:
    path = tmp_path / f"session-{name}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def test_intent_workload_not_crowded_out_by_high_volume_intent(tmp_path: Path) -> None:
    # One calendar turn buried under many system turns. The old "newest-N then
    # filter" path returned [] for calendar; the intent-aware scan must find it.
    events = [
        {"kind": "user_message", "text": "any meeting tomorrow?"},
        {"kind": "agent_response", "session_id": "s", "intent": "calendar", "response": "yes"},
    ]
    for i in range(60):
        events += [
            {"kind": "user_message", "text": f"noise{i}"},
            {"kind": "agent_response", "session_id": "s", "intent": "system", "response": "ok"},
        ]
    _write_session(tmp_path, "s", events)

    wl = build_workload_from_traces(intent="calendar", max_items=8, log_dir=tmp_path)
    assert [q.query for q in wl] == ["any meeting tomorrow?"]
    assert wl[0].intent == "calendar"


def test_intent_workload_dedupes_repeated_queries(tmp_path: Path) -> None:
    events = []
    for _ in range(5):
        events += [
            {"kind": "user_message", "text": "approve"},
            {"kind": "agent_response", "session_id": "s", "intent": "calendar", "response": "ok"},
        ]
    events += [
        {"kind": "user_message", "text": "schedule a meeting tomorrow at 3pm"},
        {"kind": "agent_response", "session_id": "s", "intent": "calendar", "response": "done"},
    ]
    _write_session(tmp_path, "s", events)

    wl = build_workload_from_traces(intent="calendar", max_items=8, log_dir=tmp_path)
    queries = [q.query for q in wl]
    assert queries.count("approve") == 1
    assert "schedule a meeting tomorrow at 3pm" in queries


def test_workload_save_load_round_trip(tmp_path: Path) -> None:
    wl = [EvalQuery(query="any meeting tomorrow?", intent="calendar", expected_tool="calendar")]
    path = tmp_path / "wl.json"
    save_workload(wl, path)
    back = load_workload(path)
    assert back[0].query == "any meeting tomorrow?"
    assert back[0].expected_tool == "calendar"
