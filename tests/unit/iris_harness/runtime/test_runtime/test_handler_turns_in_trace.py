"""A turn a deterministic handler answered shows in Sessions and Call trace.

Such a turn used to log only its request and its answer, and the trace builder lists the
turns that went through the pipeline, so every handler-answered turn (the clock, dues,
the brief…) was missing from both screens although the ledger had its audit rows. The
intercept stage now logs ``handler.end`` and the guard stage ``guard.end``, and Call trace
draws both. Run through the real pipeline on both entry points, with no model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from iris_harness.foundation.observability import trace_builder as tb
from iris_harness.foundation.observability.session_log import session_log_dir


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    from iris_harness.runtime import build_runtime

    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    return build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )


def _run(runtime: Any, entry: str, session_id: str) -> None:
    if entry == "chat":
        runtime.chat("what time is it?", session_id=session_id, channel="web")
    else:
        events = list(runtime.chat_stream("what time is it?", session_id=session_id))
        assert events[-1].kind == "done"


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_handler_answered_turn_is_in_sessions_and_call_trace(runtime: Any, entry: str) -> None:
    session_id = f"handler-trace-{entry}"
    _run(runtime, entry, session_id)

    log = (session_log_dir() / f"session-{session_id}.jsonl").read_text().splitlines()
    kinds = [json.loads(line)["kind"] for line in log]
    assert "handler.end" in kinds and "guard.end" in kinds
    assert not any(k == "llm_call" for k in kinds), "the clock answers with no model"

    traces = [t for t in tb.list_traces() if t["session_id"] == session_id]
    assert [t["request"] for t in traces] == ["what time is it?"]
    sessions = [s for s in tb.list_sessions() if s["session_id"] == session_id]
    assert len(sessions) == 1 and sessions[0]["turn_count"] == 1

    trace = tb.get_trace(traces[0]["trace_id"])
    assert trace is not None
    nodes = {n["id"]: n for n in trace["nodes"]}
    assert nodes["handler"]["kind"] == "handler"
    assert nodes["handler"]["label"] == "time_date"
    assert nodes["guard"]["kind"] == "guard" and nodes["guard"]["status"] == "ok"
    data_edges = {(e["source"], e["target"]) for e in trace["edges"] if e["kind"] == "data"}
    assert {("runtime", "handler"), ("handler", "guard")} <= data_edges

    # The response check's audit row hangs off the guard, the input screen's off the turn.
    hooks = {e["hook_point"] for e in trace["governance"]}
    assert {"pre_turn", "pre_response"} <= hooks
    gov_hosts = {
        nodes[e["target"]]["governance"]["hook"]: e["source"]
        for e in trace["edges"]
        if e["target"].startswith("gov")
    }
    assert gov_hosts["pre_response"] == "guard"

    steps = [s["type"] for s in trace["steps"]]
    assert steps == ["request", "handler", "guard", "response"]
