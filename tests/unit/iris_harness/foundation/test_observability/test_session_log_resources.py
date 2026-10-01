"""Resource metrics are captured on the chat-path session-log events.

The trace UI renders CPU/GPU/RAM per step, so ``llm_call``, ``tool_run`` and
``agent_response`` records must carry a ``resources`` block. GPU is reserved as
``None`` (unavailable via psutil on macOS); CPU/RAM come from the tier governor's
pressure sampler. Sampling is best-effort and must never break logging.
"""

from __future__ import annotations

import json
from pathlib import Path

from iris_harness.foundation.observability import session_log


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_llm_call_and_tool_run_carry_resources(monkeypatch, tmp_path):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)

    with session_log.session_scope("res-test"):
        with session_log.agent_scope("system", iteration=1):
            with session_log.llm_call_scope(
                model="granite4:latest",
                provider="ollama",
                input_messages=[{"role": "user", "content": "what time is it today?"}],
            ) as state:
                state["output_text"] = "It's 9:41 AM."
                state["tokens"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            session_log.log_tool_run(cmd="echo hi", exit_code=0, duration_ms=1.0, stdout="hi")
        session_log.log_agent_response(
            "res-test",
            response="It's 9:41 AM.",
            intent="system",
            agent_type="system",
            has_errors=False,
        )

    events = _read_events(session_log.session_log_path("res-test"))
    by_kind = {e["kind"]: e for e in events}

    for kind in ("llm_call", "tool_run", "agent_response"):
        assert kind in by_kind, f"missing {kind} event"
        res = by_kind[kind]["resources"]
        # best-effort: may be None if psutil/arbiter unavailable in this env
        if res is not None:
            assert set(res) >= {
                "cpu_percent",
                "ram_free_gb",
                "ram_total_gb",
                "gpu_percent",
                "thermal_throttled",
            }
            assert res["gpu_percent"] is None  # reserved on macOS/Apple Silicon


def test_sample_resources_never_raises(monkeypatch):
    # Even if the sampler blows up, the helper returns None rather than propagating.
    def boom() -> object:
        raise RuntimeError("no sampler")

    monkeypatch.setattr("iris_harness.llm.arbiter.sample_pressure", boom, raising=False)
    assert session_log._sample_resources() is None or isinstance(
        session_log._sample_resources(), dict
    )
