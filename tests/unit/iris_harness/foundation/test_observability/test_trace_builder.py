"""trace_builder reconstructs the call-trace graph from session JSONL."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from iris_harness.foundation.observability import session_log
from iris_harness.foundation.observability import trace_builder as tb


def _write_session(dir_path: Path, session_id: str, events: list[dict]) -> None:
    path = dir_path / f"session-{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _turn_events(session_id: str) -> list[dict]:
    return [
        {
            "kind": "user_message",
            "ts": "2026-06-16T09:41:12.000000+00:00",
            "session_id": session_id,
            "text": "what time is it today?",
        },
        {
            "kind": "turn.start",
            "ts": "2026-06-16T09:41:12.001000+00:00",
            "session_id": session_id,
            "phase": "turn.start",
        },
        {
            "kind": "pipeline.phase",
            "ts": "2026-06-16T09:41:12.002000+00:00",
            "session_id": session_id,
            "phase": "intent_router.start",
        },
        {
            "kind": "llm_call",
            "ts": "2026-06-16T09:41:12.690000+00:00",
            "session_id": session_id,
            "agent_type": "intent_router",
            "model": "llama3.2:3b",
            "provider": "ollama",
            "input_messages": [{"role": "user", "content": "what time is it today?"}],
            "output": {"text": '{"intent": "system"}'},
            "tokens": {"prompt_tokens": 674, "completion_tokens": 28, "total_tokens": 702},
            "duration_ms": 686.0,
            "resources": {
                "cpu_percent": 78.0,
                "ram_free_gb": 9.4,
                "ram_total_gb": 16.0,
                "gpu_percent": None,
                "thermal_throttled": False,
            },
        },
        {
            "kind": "intent_router.end",
            "ts": "2026-06-16T09:41:12.700000+00:00",
            "session_id": session_id,
            "phase": "intent_router.end",
            "payload": {"intent": "system", "agent_type": "system", "confidence": 0.98},
        },
        {
            "kind": "memory.context",
            "ts": "2026-06-16T09:41:12.720000+00:00",
            "session_id": session_id,
            "phase": "memory.context",
            "payload": {"has_user_profile": True},
        },
        {
            "kind": "planner.end",
            "ts": "2026-06-16T09:41:12.726000+00:00",
            "session_id": session_id,
            "phase": "planner.end",
            "payload": {"plan_size": 1, "first_agent": "system"},
        },
        {
            "kind": "agent.trace",
            "ts": "2026-06-16T09:41:12.730000+00:00",
            "session_id": session_id,
            "phase": "agent.start",
            "text": "agent start: system",
            "payload": {"agent_type": "system", "task_id": "t1"},
        },
        {
            "kind": "llm_call",
            "ts": "2026-06-16T09:41:14.200000+00:00",
            "session_id": session_id,
            "agent_type": "unknown",
            "model": "granite4:latest",
            "provider": "ollama",
            "input_messages": [{"role": "user", "content": "what time is it today?"}],
            "output": {"text": "It's 9:41 AM."},
            "tokens": {"prompt_tokens": 1100, "completion_tokens": 40, "total_tokens": 1140},
            "duration_ms": 1480.0,
        },
        {
            "kind": "agent.trace",
            "ts": "2026-06-16T09:41:14.210000+00:00",
            "session_id": session_id,
            "phase": "agent.result",
            "text": "agent result: system",
            "payload": {"agent_type": "system", "success": True, "latency_ms": 1480.0},
        },
        {
            "kind": "response_curator.start",
            "ts": "2026-06-16T09:41:14.220000+00:00",
            "session_id": session_id,
            "phase": "response_curator.start",
            "payload": {"result_count": 1},
        },
        {
            "kind": "response_curator.end",
            "ts": "2026-06-16T09:41:14.340000+00:00",
            "session_id": session_id,
            "phase": "response_curator.end",
            "payload": {"has_errors": False},
        },
        {
            "kind": "agent_response",
            "ts": "2026-06-16T09:41:14.341000+00:00",
            "session_id": session_id,
            "response": "It's 9:41 AM.",
            "intent": "system",
            "agent_type": "system",
            "has_errors": False,
            "total_tokens": 1842,
            "total_duration_ms": 2341.0,
        },
    ]


def test_list_and_build_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    _write_session(tmp_path, "sess1", _turn_events("sess1"))

    summaries = tb.list_traces()
    assert len(summaries) == 1
    s = summaries[0]
    assert s["trace_id"] == "sess1~0"
    assert s["request"] == "what time is it today?"
    assert s["total_tokens"] == 1842
    assert s["total_duration_ms"] == 2341.0

    trace = tb.get_trace("sess1~0")
    assert trace is not None
    kinds = {n["kind"] for n in trace["nodes"]}
    assert {
        "runtime",
        "intent_router",
        "memory",
        "task_planner",
        "agent",
        "response_curator",
        "llm",
    } <= kinds

    llms = [n for n in trace["nodes"] if n["kind"] == "llm"]
    assert len(llms) == 2
    intent_llm = next(n for n in llms if n["model"] == "llama3.2:3b")
    assert intent_llm["tokens"] == {"prompt": 674, "completion": 28, "total": 702}
    assert intent_llm["resources"]["cpu_percent"] == 78.0
    assert "[user] what time is it today?" in intent_llm["input"]

    # the intent LLM hangs off intent_router; the agent LLM off the agent node.
    edge_pairs = {(e["source"], e["target"]) for e in trace["edges"]}
    assert ("intent_router", intent_llm["id"]) in edge_pairs
    # main chain is connected runtime -> ... -> curator
    assert ("runtime", "intent_router") in edge_pairs
    assert any(src == "agent0" and tgt == "curator" for src, tgt in edge_pairs)


def test_skips_degenerate_turns(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    _write_session(
        tmp_path,
        "sess2",
        [
            {
                "kind": "user_message",
                "ts": "2026-06-16T09:00:00.000000+00:00",
                "session_id": "sess2",
                "text": "approve",
            },
            {
                "kind": "agent_response",
                "ts": "2026-06-16T09:00:00.100000+00:00",
                "session_id": "sess2",
                "response": "done",
                "intent": "system",
                "agent_type": "system",
                "has_errors": False,
            },
        ],
    )
    assert tb.list_traces() == []


def test_unknown_trace_id_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    assert tb.get_trace("nope~0") is None
    assert tb.get_trace("malformed") is None


def _write_audit_db(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, run_id TEXT, "
        "step_id INTEGER, agent_type TEXT, hook_point TEXT, plugin TEXT, decision TEXT, "
        "classification TEXT, tier TEXT, cost_usd REAL, severity TEXT, reason TEXT, payload_json TEXT)"
    )
    for r in rows:
        payload = {"session_id": r["session_id"]} if r.get("session_id") else {}
        conn.execute(
            "INSERT INTO audit_log(ts, run_id, agent_type, hook_point, plugin, decision, severity, reason, payload_json) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                r["ts"],
                r["run_id"],
                "system",
                r["hook_point"],
                r["plugin"],
                r["decision"],
                r.get("severity", "info"),
                r["reason"],
                json.dumps(payload),
            ),
        )
    conn.commit()
    conn.close()


def test_governance_nodes_from_audit_db(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    _write_session(tmp_path, "sess1", _turn_events("sess1"))

    audit_db = tmp_path / "audit.db"
    _write_audit_db(
        audit_db,
        [
            # within the turn window 09:41:12.000 .. 09:41:14.341.
            # Non-curator hooks carry a per-stage UUID run_id and correlate ONLY via
            # the session_id stamped into the audit payload by the kernel.
            {
                "ts": "2026-06-16T09:41:12.100000+00:00",
                "run_id": "uuid-classify",
                "session_id": "sess1",
                "hook_point": "pre_classify",
                "plugin": "data_classifier",
                "decision": "allow",
                "reason": "classified as public",
            },
            {
                "ts": "2026-06-16T09:41:12.150000+00:00",
                "run_id": "uuid-llm",
                "session_id": "sess1",
                "hook_point": "pre_llm_call",
                "plugin": "egress_gate",
                "decision": "allow",
                "reason": "tier_1 permitted",
            },
            # Curator legacy path: run_id == session_id, no payload session_id.
            {
                "ts": "2026-06-16T09:41:14.300000+00:00",
                "run_id": "sess1",
                "hook_point": "pre_response",
                "plugin": "curator_safety",
                "decision": "deny",
                "reason": "architecture_disclosure",
            },
            # a different turn's row (out of window) must NOT appear
            {
                "ts": "2026-06-16T10:00:00.000000+00:00",
                "run_id": "sess1",
                "hook_point": "pre_response",
                "plugin": "curator_safety",
                "decision": "deny",
                "reason": "other turn",
            },
            # another session's row, in-window but different session — must NOT appear
            {
                "ts": "2026-06-16T09:41:12.500000+00:00",
                "run_id": "uuid-other",
                "session_id": "sessZ",
                "hook_point": "pre_llm_call",
                "plugin": "egress_gate",
                "decision": "deny",
                "reason": "other session",
            },
        ],
    )
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(audit_db))

    trace = tb.get_trace("sess1~0")
    assert trace is not None
    gov = [n for n in trace["nodes"] if n["kind"] == "governance"]
    hooks = {n["governance"]["hook"] for n in gov}
    assert hooks == {
        "pre_classify",
        "pre_llm_call",
        "pre_response",
    }  # the out-of-window row excluded

    by_hook = {n["governance"]["hook"]: n for n in gov}
    edge_pairs = {(e["source"], e["target"]) for e in trace["edges"]}
    # pre_classify hosted on intent_router; pre_response (deny) on the curator.
    assert ("intent_router", by_hook["pre_classify"]["id"]) in edge_pairs
    assert ("curator", by_hook["pre_response"]["id"]) in edge_pairs
    assert by_hook["pre_response"]["governance"]["decision"] == "deny"
    assert by_hook["pre_response"]["status"] == "error"
    # the other session's in-window deny must be excluded -> pre_llm_call stays allow
    assert by_hook["pre_llm_call"]["governance"]["decision"] == "allow"
    # pre_llm_call hosted on an llm node.
    llm_ids = {n["id"] for n in trace["nodes"] if n["kind"] == "llm"}
    pre_llm_src = next(
        e["source"] for e in trace["edges"] if e["target"] == by_hook["pre_llm_call"]["id"]
    )
    assert pre_llm_src in llm_ids


def test_missing_audit_db_yields_no_governance(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    _write_session(tmp_path, "sess1", _turn_events("sess1"))
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "does-not-exist.db"))

    trace = tb.get_trace("sess1~0")
    assert trace is not None
    assert [n for n in trace["nodes"] if n["kind"] == "governance"] == []


def _mini_turn(session_id: str, ts: str, text: str) -> list[dict]:
    return [
        {"kind": "user_message", "ts": ts, "session_id": session_id, "text": text},
        {
            "kind": "llm_call",
            "ts": ts,
            "session_id": session_id,
            "agent_type": "intent_router",
            "model": "m",
            "provider": "p",
            "input_messages": [],
            "output": {"text": "x"},
            "tokens": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "duration_ms": 10.0,
        },
        {
            "kind": "agent_response",
            "ts": ts,
            "session_id": session_id,
            "response": "r",
            "intent": "system",
            "agent_type": "system",
            "has_errors": False,
            "total_tokens": 2,
            "total_duration_ms": 10.0,
        },
    ]


def test_list_sessions_groups_turns(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    # session sA: two meaningful turns; sB: one.
    _write_session(
        tmp_path,
        "sA",
        _mini_turn("sA", "2026-06-16T09:00:00.000000+00:00", "first")
        + _mini_turn("sA", "2026-06-16T09:05:00.000000+00:00", "second"),
    )
    _write_session(tmp_path, "sB", _mini_turn("sB", "2026-06-16T08:00:00.000000+00:00", "only"))

    sessions = tb.list_sessions()
    by_id = {s["session_id"]: s for s in sessions}
    assert set(by_id) == {"sA", "sB"}

    a = by_id["sA"]
    assert a["turn_count"] == 2
    assert a["total_tokens"] == 4
    assert a["started_at"] < a["last_at"]
    # turns newest-first, each links to its own trace.
    assert [t["request"] for t in a["turns"]] == ["second", "first"]
    assert {t["trace_id"] for t in a["turns"]} == {"sA~0", "sA~1"}
    # title is the conversation opener (oldest turn), not the newest.
    assert a["title"] == "first"

    # sessions ordered newest-activity first (sA's last turn is after sB's).
    assert sessions[0]["session_id"] == "sA"


def test_session_messages_replays_turns_chronologically(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    _write_session(
        tmp_path,
        "sR",
        _mini_turn("sR", "2026-06-16T09:00:00.000000+00:00", "first")
        + _mini_turn("sR", "2026-06-16T09:05:00.000000+00:00", "second"),
    )

    msgs = tb.session_messages("sR")
    # user/assistant pairs, oldest first, so the chat replays top-to-bottom.
    assert [(m["role"], m["text"]) for m in msgs] == [
        ("user", "first"),
        ("assistant", "r"),
        ("user", "second"),
        ("assistant", "r"),
    ]
    # assistant messages link back to their trace for deep-linking.
    assert msgs[1]["trace_id"] == "sR~0"
    assert msgs[3]["trace_id"] == "sR~1"


def test_session_messages_unknown_session_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    assert tb.session_messages("nope") == []


def test_build_steps_reasoning_timeline(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    ts = "2026-06-16T09:00:00"
    turn = [
        {
            "kind": "user_message",
            "ts": f"{ts}.000000+00:00",
            "session_id": "r1",
            "text": "do a thing",
        },
        {
            "kind": "llm_call",
            "ts": f"{ts}.500000+00:00",
            "session_id": "r1",
            "agent_type": "system",
            "model": "llama3.2:3b",
            "provider": "ollama",
            "iteration": 1,
            "input_messages": [{"role": "user", "content": "do a thing"}],
            "output": {"text": "Thought: I'll search\nAction: research"},
            "tokens": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "duration_ms": 20.0,
        },
        {
            "kind": "intent_router.end",
            "ts": f"{ts}.600000+00:00",
            "session_id": "r1",
            "phase": "intent_router.end",
            "payload": {
                "intent": "system",
                "agent_type": "system",
                "confidence": 0.9,
                "is_multi_step": False,
            },
        },
        {
            "kind": "agent.trace",
            "ts": f"{ts}.900000+00:00",
            "session_id": "r1",
            "phase": "agent.result",
            "payload": {
                "agent_type": "system",
                "success": False,
                "stall_count": 2,
                "iterations": 3,
            },
        },
        {
            "kind": "response_curator.end",
            "ts": f"{ts}.950000+00:00",
            "session_id": "r1",
            "phase": "response_curator.end",
            "payload": {
                "has_errors": False,
                "response_chars": 42,
                "metadata": {
                    "judge_bundle": {
                        "halted": False,
                        "signals": [
                            {"name": "safety", "verdict": "pass", "reason": "ok"},
                            {"name": "faithfulness", "verdict": "skipped", "reason": "no sources"},
                        ],
                    }
                },
            },
        },
        {
            "kind": "agent_response",
            "ts": f"{ts}.960000+00:00",
            "session_id": "r1",
            "response": "Stopped: repeated action without progress.",
            "intent": "system",
            "agent_type": "system",
            "has_errors": False,
            "total_tokens": 15,
            "total_duration_ms": 960.0,
        },
    ]
    _write_session(tmp_path, "r1", turn)

    steps = tb.get_trace("r1~0")["steps"]
    types = [s["type"] for s in steps]
    assert types == ["request", "llm", "intent", "stop", "curator", "response"]

    llm = next(s for s in steps if s["type"] == "llm")
    assert llm["iteration"] == 1 and llm["fields"]["tokens"] == 15
    assert "Thought: I'll search" in llm["detail"]

    stop = next(s for s in steps if s["type"] == "stop")
    assert stop["status"] == "error" and stop["fields"]["stall_count"] == 2

    curator = next(s for s in steps if s["type"] == "curator")
    assert curator["fields"]["safety"] == "pass"
    assert curator["fields"]["faithfulness"] == "skipped"


def test_tool_invoke_events_become_tool_node_and_step(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    ts = "2026-06-16T09:00:00"
    turn = [
        {"kind": "user_message", "ts": f"{ts}.000000+00:00", "session_id": "ti", "text": "aapl?"},
        {
            "kind": "llm_call",
            "ts": f"{ts}.200000+00:00",
            "session_id": "ti",
            "agent_type": "system",
            "model": "qwen2.5:7b",
            "provider": "ollama",
            "input_messages": [],
            "output": {"text": "Action: stock_quote"},
            "tokens": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "duration_ms": 5.0,
        },
        {
            "kind": "tool.invoke.start",
            "ts": f"{ts}.210000+00:00",
            "session_id": "ti",
            "phase": "tool.invoke.start",
            "payload": {
                "tool": "stock_quote",
                "tool_call_id": "c1",
                "arguments": {"symbol": "AAPL"},
            },
        },
        {
            "kind": "tool.invoke.end",
            "ts": f"{ts}.900000+00:00",
            "session_id": "ti",
            "phase": "tool.invoke.end",
            "payload": {
                "tool": "stock_quote",
                "tool_call_id": "c1",
                "ok": True,
                "result_preview": "AAPL: 299.24 USD",
            },
        },
        {
            "kind": "agent_response",
            "ts": f"{ts}.950000+00:00",
            "session_id": "ti",
            "response": "AAPL is 299.24 USD",
            "intent": "search",
            "agent_type": "system",
            "has_errors": False,
            "total_tokens": 2,
            "total_duration_ms": 950.0,
        },
    ]
    _write_session(tmp_path, "ti", turn)

    trace = tb.get_trace("ti~0")
    assert trace is not None

    tool_nodes = [n for n in trace["nodes"] if n["kind"] == "tool"]
    assert len(tool_nodes) == 1
    tnode = tool_nodes[0]
    assert tnode["label"] == "stock_quote"
    assert tnode["status"] == "ok"
    assert "AAPL: 299.24 USD" in tnode["output"]
    assert "AAPL" in tnode["input"]  # args carried from the paired start

    tool_steps = [s for s in trace["steps"] if s["type"] == "tool"]
    assert tool_steps and tool_steps[0]["fields"].get("tool") == "stock_quote"


def _llm(ts: str, agent_type: str, model: str, duration_ms: float, **extra: object) -> dict:
    return {
        "kind": "llm_call",
        "ts": ts,
        "session_id": "seq",
        "agent_type": agent_type,
        "model": model,
        "provider": "ollama",
        "input_messages": [],
        "output": {"text": ""},
        "duration_ms": duration_ms,
        **extra,
    }


def _ordered_email_turn(curator_agent: str) -> list[dict]:
    """The shape of a live email turn: the agent's own calls, a tool that calls a model
    while it runs, and the curator's judge call made after the agent finished."""
    t = "2026-09-16T16:46:"
    return [
        {"kind": "user_message", "ts": f"{t}00.500000+00:00", "session_id": "seq", "text": "hi"},
        {"kind": "pipeline.phase", "ts": f"{t}01.100000+00:00", "phase": "intent_router.start"},
        {"kind": "intent_router.end", "ts": f"{t}01.200000+00:00", "payload": {"intent": "x"}},
        {"kind": "memory.context", "ts": f"{t}01.700000+00:00", "payload": {}},
        {"kind": "planner.end", "ts": f"{t}01.700000+00:00", "payload": {"plan_size": 1}},
        {
            "kind": "agent.trace",
            "ts": f"{t}01.750000+00:00",
            "phase": "agent.start",
            "payload": {"agent_type": "email"},
        },
        _llm(f"{t}07.900000+00:00", "email", "qwen2.5:7b-instruct", 5900.0, tier="tier2"),
        {
            "kind": "tool.invoke.start",
            "ts": f"{t}08.000000+00:00",
            "payload": {"tool": "inbox_digest", "tool_call_id": "c1", "arguments": {}},
        },
        _llm(f"{t}09.500000+00:00", "email", "granite4:latest", 1200.0, tier="tier1"),
        {
            "kind": "tool.invoke.end",
            "ts": f"{t}09.600000+00:00",
            "payload": {"tool": "inbox_digest", "tool_call_id": "c1", "ok": True},
        },
        _llm(f"{t}18.100000+00:00", "email", "qwen2.5:7b-instruct", 6300.0, tier="tier2"),
        {
            "kind": "agent.trace",
            "ts": f"{t}18.190000+00:00",
            "phase": "agent.result",
            "payload": {"agent_type": "email", "success": True, "latency_ms": 16400.0},
        },
        {"kind": "response_curator.start", "ts": f"{t}18.200000+00:00", "payload": {}},
        _llm(f"{t}19.600000+00:00", curator_agent, "granite4:latest", 1250.0, tier="tier1"),
        {"kind": "response_curator.end", "ts": f"{t}28.100000+00:00", "payload": {}},
        _llm(f"{t}28.600000+00:00", "turn_capture", "llama3.2:3b", 400.0, tier="router"),
        {
            "kind": "agent_response",
            "ts": f"{t}28.700000+00:00",
            "response": "done",
            "intent": "communication",
            "agent_type": "email",
            "has_errors": False,
            "total_duration_ms": 28200.0,
        },
    ]


def _edges_by_label(trace: dict) -> list[tuple[int, str, str]]:
    label = {n["id"]: n["label"] for n in trace["nodes"]}
    return [(e["seq"], label[e["source"]], label[e["target"]]) for e in trace["edges"]]


def test_llm_calls_hang_off_the_step_that_made_them(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    _write_session(tmp_path, "seq", _ordered_email_turn(curator_agent="response_curator"))

    trace = tb.get_trace("seq~0")
    assert trace is not None
    llm_edges = [(src, dst) for _seq, src, dst in _edges_by_label(trace) if dst != "email_agent"]

    assert ("inbox_digest", "granite4:latest") in llm_edges  # made while the tool ran
    assert ("response_curator", "granite4:latest") in llm_edges  # the curator's judge
    assert ("IrisRuntime.chat", "llama3.2:3b") in llm_edges  # record stage fact capture
    assert llm_edges.count(("email_agent", "qwen2.5:7b-instruct")) == 2

    by_label = {n["label"]: n for n in trace["nodes"] if n["kind"] == "llm"}
    assert by_label["llama3.2:3b"]["tier"] == "router"
    assert by_label["llama3.2:3b"]["agent"] == "turn_capture"


def test_older_logs_without_agent_names_still_credit_the_curator(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    _write_session(tmp_path, "seq", _ordered_email_turn(curator_agent="unknown"))

    trace = tb.get_trace("seq~0")
    assert trace is not None
    assert any(
        src == "response_curator" and dst == "granite4:latest"
        for _seq, src, dst in _edges_by_label(trace)
    )


def test_edges_are_numbered_in_execution_order(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    _write_session(tmp_path, "seq", _ordered_email_turn(curator_agent="response_curator"))

    trace = tb.get_trace("seq~0")
    assert trace is not None
    edges = _edges_by_label(trace)

    assert [seq for seq, _s, _d in edges] == list(range(1, len(edges) + 1))
    targets = [dst for _seq, _src, dst in edges]
    assert targets == [
        "intent_router",
        "memory_retriever",
        "task_planner",
        "email_agent",
        "qwen2.5:7b-instruct",
        "inbox_digest",
        "granite4:latest",
        "qwen2.5:7b-instruct",
        "response_curator",
        "granite4:latest",
        "llama3.2:3b",
    ]
    offsets = [n["t_offset_ms"] for n in trace["nodes"]]
    assert offsets == sorted(offsets)


def test_list_sessions_skips_before_the_limit(monkeypatch, tmp_path):
    """Hidden sessions never shorten the list: skip is applied before limit."""
    import os

    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    _write_session(tmp_path, "mine", _mini_turn("mine", "2026-06-16T08:00:00.000000+00:00", "hi"))
    _write_session(tmp_path, "run-1", _mini_turn("run-1", "2026-06-16T09:00:00.000000+00:00", "x"))
    os.utime(tmp_path / "session-mine.jsonl", (1, 1))  # the owner's is the OLDER file

    everything = tb.list_sessions(limit=1)
    kept = tb.list_sessions(limit=1, skip=lambda sid: sid.startswith("run-"))

    assert [s["session_id"] for s in everything] == ["run-1"]
    assert [s["session_id"] for s in kept] == ["mine"]


# -- the agent node names the code that answered, read off the registered handler ------


class _FakeAgentHandler:
    def handle(self, task: object) -> str:
        return "ok"


def _make_fake_handler():  # type: ignore[no-untyped-def]
    def handler(task: object) -> str:
        return "ok"

    return handler


def test_agent_node_names_the_registered_handler(monkeypatch, tmp_path):
    """No literal module table: the agent node's module/method/file come from the
    callable the executor registered, so a moved handler is named where it now lives."""
    from iris_harness.agent.agent_executor import AgentExecutor

    monkeypatch.setattr(tb, "_AGENT_SOURCES", {})
    AgentExecutor().register("system", _FakeAgentHandler().handle)
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    _write_session(tmp_path, "sess1", _turn_events("sess1"))

    trace = tb.get_trace("sess1~0")
    assert trace is not None
    agent = next(n for n in trace["nodes"] if n["kind"] == "agent")
    assert agent["module"] == __name__
    assert agent["method"] == "_FakeAgentHandler.handle"
    # From the package root: ``src/iris_harness/…`` for core code in a checkout.
    assert agent["component_path"] == __name__.replace(".", "/") + ".py"
    # The node keeps the shape the Agent Call Trace screen reads.
    assert {"id", "kind", "label", "module", "method", "component_path", "status"} <= set(agent)


def test_callable_meta_sees_through_wrappers_and_closures() -> None:
    """A plugin handler arrives wrapped by the fault boundary, and is often a closure:
    the trace names the plugin's factory, not the wrapper or ``<locals>``."""
    import functools

    from iris_harness.runtime.plugin_host.manifest import RegistrationKind
    from iris_harness.runtime.plugin_host.registry import PluginRegistry

    inner = _make_fake_handler()
    guarded = PluginRegistry().guard("p", RegistrationKind.INTENT_HANDLER, "x", inner, degrade=None)
    assert tb.callable_meta(guarded) == tb.callable_meta(inner)
    assert tb.callable_meta(inner)[:2] == (__name__, "_make_fake_handler")
    assert tb.callable_meta(functools.partial(inner))[1] == "_make_fake_handler"
    assert tb.callable_meta(_FakeAgentHandler().handle)[1] == "_FakeAgentHandler.handle"
    # Core code in a checkout reads as it did in the old literal table.
    assert tb.callable_meta(tb.callable_meta) == (
        "iris_harness.foundation.observability.trace_builder",
        "callable_meta",
        "src/iris_harness/foundation/observability/trace_builder.py",
    )


def test_unregistered_agent_falls_back_to_system_then_blank(monkeypatch) -> None:
    monkeypatch.setattr(tb, "_AGENT_SOURCES", {})
    assert tb._agent_meta("email") == ("", "", "")
    tb.register_agent_source("system", _FakeAgentHandler().handle)
    assert tb._agent_meta("email")[1] == "_FakeAgentHandler.handle"


def test_tool_events_pair_by_call_id_with_the_old_key_as_the_fallback():
    # #134: ``call_id`` is the runner's minted id; a session log written before it (or a
    # general-lane builtin tool, which mints none) has only ``tool_call_id``.
    assert tb._call_id_of({"call_id": "01A", "tool_call_id": "01A"}) == "01A"
    assert tb._call_id_of({"call_id": "01A", "tool_call_id": "other"}) == "01A"
    assert tb._call_id_of({"tool_call_id": "c1"}) == "c1"
    assert tb._call_id_of({}) is None
