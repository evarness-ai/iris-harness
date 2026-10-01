"""Tests for CLI session replay formatting."""

from __future__ import annotations

from iris_harness.cli.commands import (
    _parse_replay_args,
    _replay_event_summary,
    _session_turns,
    _trace_event_detail,
)


def test_session_turns_group_events_by_user_message() -> None:
    events = [
        {"kind": "user_message", "text": "one"},
        {"kind": "intent_router.end"},
        {"kind": "agent_response", "response": "done"},
        {"kind": "user_message", "text": "two"},
        {"kind": "agent_response", "response": "done again"},
    ]

    turns = _session_turns(events)

    assert [[event["kind"] for event in turn] for turn in turns] == [
        ["user_message", "intent_router.end", "agent_response"],
        ["user_message", "agent_response"],
    ]


def test_a_turn_the_system_opened_is_a_turn_of_its_own() -> None:
    """ADR-0127: the first-chat welcome starts with ``turn_open``, not a user message."""
    events = [
        {"kind": "turn_open", "opener": "welcome", "label": "First-chat welcome"},
        {"kind": "handler.end"},
        {"kind": "agent_response", "response": "Hi"},
        {"kind": "user_message", "text": "thanks"},
        {"kind": "agent_response", "response": "ok"},
    ]

    turns = _session_turns(events)

    assert [turn[0]["kind"] for turn in turns] == ["turn_open", "user_message"]
    _ts, label, detail = _replay_event_summary(events[0])
    assert label == "open"
    assert detail == "First-chat welcome (opened by IRIS, no user message)"


def test_replay_event_summary_renders_llm_start_payload() -> None:
    event = {
        "kind": "agent.trace",
        "ts": "2026-05-12T03:00:02.419908+00:00",
        "payload": {
            "event": "llm.start",
            "provider": "ollama",
            "model": "llama3.2:3b",
            "num_ctx": 2048,
            "governor_mode": "active",
        },
    }

    ts, label, detail = _replay_event_summary(event)

    assert ts == "03:00:02"
    assert label == "llm"
    assert detail == "start · ollama/llama3.2:3b · ctx 2048 · governor active"


def test_replay_event_summary_renders_memory_context() -> None:
    event = {
        "kind": "memory.context",
        "payload": {
            "has_user_profile": True,
            "has_active_context": True,
            "recent_turns": 3,
            "episodic_patterns": 2,
            "behavior_name": "reminders",
        },
    }

    _ts, label, detail = _replay_event_summary(event)

    assert label == "memory"
    assert detail == "profile=True · active=True · recent=3 · episodic=2 · behavior=reminders"


def test_replay_event_summary_keeps_reasonable_final_response_visible() -> None:
    response = "Glad I could bring some humor to your day! Would you like another joke or is there something else on your mind that you'd like to chat about, Robin?"
    event = {"kind": "agent_response", "response": response}

    _ts, label, detail = _replay_event_summary(event)

    assert label == "final"
    assert detail == response
    assert "…" not in detail


def test_replay_event_summary_full_mode_does_not_truncate_long_response() -> None:
    response = "word " * 200
    event = {"kind": "agent_response", "response": response}

    _ts, _label, detail = _replay_event_summary(event, full=True)

    assert detail == response.strip()


def test_replay_event_summary_renders_agent_result_metadata() -> None:
    event = {
        "kind": "agent.trace",
        "payload": {
            "name": "agent.result",
            "agent_type": "system",
            "provider": "ollama",
            "model": "llama3.2:3b",
            "num_ctx": 2048,
            "total_tokens": 1168,
        },
    }

    _ts, label, detail = _replay_event_summary(event)

    assert label == "agent"
    assert detail == "system · ollama/llama3.2:3b · ctx 2048 · 1168 tok"


def test_parse_replay_args_accepts_full_flag_and_limit() -> None:
    assert _parse_replay_args("3 --full") == (3, True, None)
    assert _parse_replay_args("full") == (1, True, None)


def test_replay_event_summary_renders_tool_invocation() -> None:
    start = {"kind": "tool.invoke.start", "payload": {"tool": "research"}}
    end = {"kind": "tool.invoke.end", "payload": {"tool": "research", "ok": True}}

    assert _replay_event_summary(start)[1:] == ("tool", "start · research")
    assert _replay_event_summary(end)[1:] == ("tool", "end · research · ok=True")


def test_replay_event_summary_renders_response_curator() -> None:
    event = {
        "kind": "response_curator.end",
        "payload": {"response_chars": 42, "has_errors": False},
    }

    _ts, label, detail = _replay_event_summary(event)

    assert label == "curate"
    assert detail == "end · 42 chars · errors=False"


def test_trace_event_detail_includes_llm_input_and_output() -> None:
    event = {
        "kind": "llm_call",
        "provider": "ollama",
        "model": "llama3.2:3b",
        "input_messages": [
            {"role": "system", "content": "You are IRIS."},
            {"role": "user", "content": "show news"},
        ],
        "output": {"text": "answer", "tool_calls": [{"name": "research"}]},
        "tokens": {"total_tokens": 99},
        "duration_ms": 123.4,
    }

    detail = _trace_event_detail(event)

    assert "model: ollama/llama3.2:3b" in detail
    assert "tokens:" in detail
    assert "system: You are IRIS." in detail
    assert "tool_calls:" in detail
    assert "output: answer" in detail
