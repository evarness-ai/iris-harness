"""Tests for the session-log success-trace reader."""

from __future__ import annotations

import json
from pathlib import Path

from iris_harness.foundation.observability.session_log import iter_recent_turns


def _write_session(tmp_path: Path, name: str, events: list[dict]) -> None:
    path = tmp_path / f"session-{name}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def test_pairs_user_message_with_agent_response(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        "a",
        [
            {"kind": "user_message", "text": "what is 2+2?", "turn_id": "t1"},
            {
                "kind": "agent_response",
                "session_id": "a",
                "turn_id": "t1",
                "intent": "general",
                "agent_type": "general",
                "response": "4",
                "has_errors": False,
            },
        ],
    )
    turns = iter_recent_turns(limit=10, log_dir=tmp_path)
    assert len(turns) == 1
    assert turns[0].query == "what is 2+2?"
    assert turns[0].response == "4"
    assert turns[0].intent == "general"
    assert turns[0].turn_id == "t1"
    assert turns[0].has_errors is False


def test_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert iter_recent_turns(log_dir=tmp_path / "nope") == []


def test_log_dir_honors_iris_home_and_override(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Regression: session logs must follow IRIS_HOME (so the test suite never writes
    # into the developer's real ~/.iris/logs) and an explicit override.

    import iris_harness.foundation.observability.session_log as sl

    # Read on every use, not at import: no reload needed to see a moved home.
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("IRIS_SESSION_LOG_DIR", raising=False)
    assert sl.session_log_dir() == tmp_path / "home" / "logs"

    monkeypatch.setenv("IRIS_SESSION_LOG_DIR", str(tmp_path / "explicit"))
    assert sl.session_log_dir() == tmp_path / "explicit"


def test_respects_limit(tmp_path: Path) -> None:
    events = []
    for i in range(5):
        events.append({"kind": "user_message", "text": f"q{i}"})
        events.append(
            {"kind": "agent_response", "session_id": "a", "intent": "x", "response": f"r{i}"}
        )
    _write_session(tmp_path, "a", events)
    assert len(iter_recent_turns(limit=3, log_dir=tmp_path)) == 3


def test_intent_filter_finds_rare_intent_crowded_out(tmp_path: Path) -> None:
    # A high-volume intent dominates; the one calendar turn trails them. Turns are
    # read oldest-first within a file, so a low capped scan stops before reaching it;
    # an intent-filtered scan must keep going and find it.
    events: list[dict] = []
    for i in range(50):
        events.append({"kind": "user_message", "text": f"noise{i}"})
        events.append(
            {"kind": "agent_response", "session_id": "a", "intent": "system", "response": "ok"}
        )
    events.append({"kind": "user_message", "text": "any meeting tomorrow?"})
    events.append(
        {"kind": "agent_response", "session_id": "a", "intent": "calendar", "response": "yes"}
    )
    _write_session(tmp_path, "a", events)

    # Unfiltered, capped low -> the rare calendar turn is crowded out.
    unfiltered = iter_recent_turns(limit=5, log_dir=tmp_path)
    assert not any(t.intent == "calendar" for t in unfiltered)

    # Intent-filtered -> found regardless of how many higher-volume turns precede it.
    cal = iter_recent_turns(limit=5, log_dir=tmp_path, intent="calendar")
    assert [t.query for t in cal] == ["any meeting tomorrow?"]
    assert all(t.intent == "calendar" for t in cal)


def test_require_query_skips_unpaired_turns(tmp_path: Path) -> None:
    # Heartbeat-style agent_response with no preceding user_message -> empty query.
    events = [
        {"kind": "agent_response", "session_id": "a", "intent": "calendar", "response": "auto"},
        {"kind": "user_message", "text": "any meeting tomorrow?"},
        {"kind": "agent_response", "session_id": "a", "intent": "calendar", "response": "yes"},
    ]
    _write_session(tmp_path, "a", events)
    got = iter_recent_turns(limit=10, log_dir=tmp_path, intent="calendar", require_query=True)
    assert [t.query for t in got] == ["any meeting tomorrow?"]


def test_unique_queries_counts_distinct_against_limit(tmp_path: Path) -> None:
    # A heavily-repeated query must not fill `limit` before distinct ones are reached.
    events = []
    for _ in range(20):
        events += [
            {"kind": "user_message", "text": "approve"},
            {"kind": "agent_response", "session_id": "a", "intent": "calendar", "response": "ok"},
        ]
    for q in ("schedule a meeting tomorrow", "any meeting tomorrow?"):
        events += [
            {"kind": "user_message", "text": q},
            {"kind": "agent_response", "session_id": "a", "intent": "calendar", "response": "ok"},
        ]
    _write_session(tmp_path, "a", events)

    got = iter_recent_turns(
        limit=3, log_dir=tmp_path, intent="calendar", require_query=True, unique_queries=True
    )
    assert sorted(t.query for t in got) == [
        "any meeting tomorrow?",
        "approve",
        "schedule a meeting tomorrow",
    ]


def test_scan_cap_bounds_filtered_scan(tmp_path: Path) -> None:
    # Only non-matching turns, more than scan_cap -> returns [] without scanning all.
    events = []
    for i in range(20):
        events.append({"kind": "user_message", "text": f"n{i}"})
        events.append(
            {"kind": "agent_response", "session_id": "a", "intent": "system", "response": "ok"}
        )
    _write_session(tmp_path, "a", events)
    assert iter_recent_turns(limit=5, log_dir=tmp_path, intent="calendar", scan_cap=5) == []
