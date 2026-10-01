"""Behavioral tests for agent executor and response curator."""

from __future__ import annotations

from dataclasses import dataclass

from iris_harness.agent.agent_executor import AgentExecutor, AgentResult, AgentTask
from iris_harness.agent.response_curator import ResponseCurator


def _make_executor(agent_type: str = "system", response: str = "ok") -> AgentExecutor:
    executor = AgentExecutor()
    executor.register(agent_type, lambda task: response)
    return executor


def test_agent_executor_dispatches_to_registered_handler() -> None:
    executor = _make_executor("email", "inbox has 3 emails")
    task = AgentTask(query="check email", agent_type="email")
    result = executor.execute(task)

    assert result.success
    assert result.output == "inbox has 3 emails"
    assert result.agent_type == "email"


def test_agent_executor_returns_failure_for_unknown_agent() -> None:
    executor = AgentExecutor()
    task = AgentTask(query="test", agent_type="nonexistent")
    result = executor.execute(task)

    assert not result.success
    assert result.error is not None


def test_agent_executor_captures_exception_as_failure() -> None:
    executor = AgentExecutor()
    executor.register("broken", lambda task: (_ for _ in ()).throw(RuntimeError("boom")))
    task = AgentTask(query="test", agent_type="broken")
    result = executor.execute(task)

    assert not result.success
    assert "boom" in (result.error or "")


def test_agent_executor_records_latency() -> None:
    executor = _make_executor()
    result = executor.execute(AgentTask(query="q", agent_type="system"))

    assert result.latency_ms >= 0


def test_execute_stream_falls_back_to_nonstreaming_handler() -> None:
    # planner/coding_agent register execute() but not execute_stream(); the
    # streaming chat path must still run them instead of reporting
    # "unsupported agent_type". Regression for the planner chat error.
    executor = AgentExecutor()
    executor.register("planner", lambda task: "today: 1 followup")
    items = list(executor.execute_stream(AgentTask(query="how is the day", agent_type="planner")))

    chunks = [i for i in items if isinstance(i, str)]
    results = [i for i in items if isinstance(i, AgentResult)]
    assert chunks == ["today: 1 followup"]
    assert len(results) == 1
    assert results[0].success
    assert results[0].output == "today: 1 followup"
    assert results[0].error is None


def test_execute_stream_unknown_agent_still_fails() -> None:
    executor = AgentExecutor()
    items = list(executor.execute_stream(AgentTask(query="q", agent_type="ghost")))
    results = [i for i in items if isinstance(i, AgentResult)]

    assert len(results) == 1
    assert not results[0].success
    assert "ghost" in (results[0].error or "")


def test_response_curator_single_success() -> None:
    results = [AgentResult(agent_type="email", output="3 new emails", success=True)]
    curator = ResponseCurator()
    response = curator.curate(results)

    assert response.text == "3 new emails"
    assert not response.has_errors
    assert "email" in response.sources


def test_response_curator_merges_multiple_results() -> None:
    results = [
        AgentResult(agent_type="email", output="emails fetched", success=True),
        AgentResult(agent_type="finance", output="portfolio up 2%", success=True),
    ]
    curator = ResponseCurator()
    response = curator.curate(results)

    assert "[email]" in response.text
    assert "[finance]" in response.text
    assert len(response.sources) == 2


def test_response_curator_partial_failure_includes_error_summary() -> None:
    results = [
        AgentResult(agent_type="email", output="ok", success=True),
        AgentResult(agent_type="calendar", output="", success=False, error="timeout"),
    ]
    curator = ResponseCurator()
    response = curator.curate(results)

    assert response.text == "ok"
    assert response.has_errors
    assert response.error_summary is not None
    assert "timeout" in response.error_summary


def test_response_curator_all_failed_returns_error_response() -> None:
    results = [
        AgentResult(agent_type="email", output="", success=False, error="connection refused")
    ]
    curator = ResponseCurator()
    response = curator.curate(results)

    assert response.has_errors
    assert not response.text.startswith("[email]")


def test_response_curator_empty_results() -> None:
    curator = ResponseCurator()
    response = curator.curate([])

    assert response.has_errors
    assert response.error_summary == "empty result set"


def test_response_curator_safety_signal_halts_unsafe_output() -> None:
    curator = ResponseCurator()
    response = curator.curate(
        [
            AgentResult(
                agent_type="system", output="Use AKIA1234567890ABCDEF right now", success=True
            )
        ]
    )

    assert response.has_errors
    assert "unsafe" in (response.text.lower() + (response.error_summary or "").lower())
    bundle = response.metadata.get("judge_bundle")
    assert isinstance(bundle, dict)
    signals = bundle.get("signals")
    assert isinstance(signals, list)
    assert any(s.get("name") == "safety" and s.get("verdict") == "halt" for s in signals)


def test_response_curator_schema_retry_uses_json_block_repair() -> None:
    curator = ResponseCurator()
    response = curator.curate(
        [
            AgentResult(
                agent_type="system",
                output='Result:\n```json\n{"status": "ok"}\n```',
                success=True,
                metadata={"expected_schema": {"required": ["status"]}},
            )
        ]
    )

    assert response.has_errors is False
    assert "Governance warning" not in response.text
    bundle = response.metadata.get("judge_bundle")
    assert isinstance(bundle, dict)
    assert bundle.get("retries_used", 0) >= 1


def test_response_curator_strict_faithfulness_warns_on_low_overlap() -> None:
    curator = ResponseCurator()
    response = curator.curate(
        [AgentResult(agent_type="system", output="The stock market closed higher.", success=True)],
        query="How do I bake sourdough bread?",
        strict=True,
    )

    assert response.has_errors is False
    bundle = response.metadata.get("judge_bundle")
    assert isinstance(bundle, dict)
    assert bundle.get("retries_used", 0) >= 1
    signals = bundle.get("signals")
    assert isinstance(signals, list)
    assert any(s.get("name") == "faithfulness" and s.get("verdict") == "pass" for s in signals)


def test_response_curator_consistency_warns_on_known_name_conflict() -> None:
    curator = ResponseCurator()
    response = curator.curate(
        [AgentResult(agent_type="system", output="Your name is Alice.", success=True)],
        query="what is my name?",
        conversation_history=("user: my name is Bob",),
    )

    assert "Governance warning" in response.text
    bundle = response.metadata.get("judge_bundle")
    assert isinstance(bundle, dict)
    signals = bundle.get("signals")
    assert isinstance(signals, list)
    assert any(s.get("name") == "consistency" and s.get("verdict") == "warn" for s in signals)


@dataclass
class _StubFaithfulnessJudge:
    verdicts: list[str]

    async def judge(self, *, query: str, response: str) -> str:
        if self.verdicts:
            return self.verdicts.pop(0)
        return '{"addresses_question": true, "confidence": 0.5, "rationale": "default"}'


def test_response_curator_faithfulness_llm_retry_then_pass() -> None:
    curator = ResponseCurator(
        faithfulness_judge=_StubFaithfulnessJudge(
            verdicts=[
                '{"addresses_question": false, "confidence": 0.1, "rationale": "off-topic"}',
                '{"addresses_question": true, "confidence": 0.8, "rationale": "answers question"}',
            ]
        )
    )
    response = curator.curate(
        [AgentResult(agent_type="system", output="Market summary.", success=True)],
        query="How do I bake sourdough bread?",
        strict=True,
    )

    assert response.has_errors is False
    bundle = response.metadata.get("judge_bundle")
    assert isinstance(bundle, dict)
    assert bundle.get("retries_used", 0) >= 1
    signals = bundle.get("signals")
    assert isinstance(signals, list)
    faithfulness = next((s for s in signals if s.get("name") == "faithfulness"), None)
    assert isinstance(faithfulness, dict)
    assert faithfulness.get("verdict") == "pass"
    assert faithfulness.get("metadata", {}).get("judge_mode") == "llm"


def test_response_curator_faithfulness_llm_invalid_payload_warns_after_retry() -> None:
    curator = ResponseCurator(
        faithfulness_judge=_StubFaithfulnessJudge(verdicts=["not-json", "still-not-json"])
    )
    response = curator.curate(
        [AgentResult(agent_type="system", output="Random answer.", success=True)],
        query="How do I bake sourdough bread?",
        strict=True,
    )

    assert "Governance warning" in response.text
    bundle = response.metadata.get("judge_bundle")
    assert isinstance(bundle, dict)
    signals = bundle.get("signals")
    assert isinstance(signals, list)
    faithfulness = next((s for s in signals if s.get("name") == "faithfulness"), None)
    assert isinstance(faithfulness, dict)
    assert faithfulness.get("verdict") == "warn"


# ─── parallel wave execution (Planner P1) ──────────────────────────────────


def test_execute_wave_runs_concurrently_and_preserves_order() -> None:
    import threading
    import time

    executor = AgentExecutor()
    barrier = threading.Barrier(3, timeout=2.0)

    def _slow(task: AgentTask) -> str:
        # If the three run concurrently they all reach the barrier; if they
        # run sequentially the barrier times out → BrokenBarrierError.
        barrier.wait()
        return task.query

    executor.register("a", _slow)
    tasks = [AgentTask(query=f"q{i}", agent_type="a") for i in range(3)]
    start = time.monotonic()
    results = executor.execute_wave(tasks)
    elapsed = time.monotonic() - start

    assert [r.output for r in results] == ["q0", "q1", "q2"]  # input order preserved
    assert all(r.success for r in results)
    assert elapsed < 1.5  # concurrent, not 3× a serial wait


def test_execute_wave_isolates_failures() -> None:
    executor = AgentExecutor()

    def _maybe_fail(task: AgentTask) -> str:
        if task.query == "boom":
            raise ValueError("kaboom")
        return "ok"

    executor.register("a", _maybe_fail)
    results = executor.execute_wave(
        [AgentTask(query="boom", agent_type="a"), AgentTask(query="fine", agent_type="a")]
    )
    assert results[0].success is False and "kaboom" in (results[0].error or "")
    assert results[1].success is True and results[1].output == "ok"


def test_execute_waves_orders_waves_then_in_wave() -> None:
    executor = AgentExecutor()
    executor.register("a", lambda task: task.query)
    waves = [
        [AgentTask(query="w0a", agent_type="a"), AgentTask(query="w0b", agent_type="a")],
        [AgentTask(query="w1", agent_type="a")],
    ]
    results = executor.execute_waves(waves)
    assert [r.output for r in results] == ["w0a", "w0b", "w1"]


def test_execute_wave_single_task_runs_inline() -> None:
    executor = AgentExecutor()
    executor.register("a", lambda task: "solo")
    results = executor.execute_wave([AgentTask(query="x", agent_type="a")])
    assert [r.output for r in results] == ["solo"]


def test_execute_wave_keeps_the_turn_context_in_each_task() -> None:
    from iris_harness.foundation.observability.session_log import (
        current_turn_id,
        session_scope,
        turn_scope,
    )

    executor = AgentExecutor()
    executor.register("a", lambda task: str(current_turn_id()))
    tasks = [AgentTask(query=f"q{i}", agent_type="a") for i in range(3)]

    with session_scope("s"), turn_scope("turn-9"):
        results = executor.execute_wave(tasks)

    assert [r.output for r in results] == ["turn-9"] * 3


def test_execute_and_execute_stream_run_under_the_task_agent_name() -> None:
    from iris_harness.foundation.observability.session_log import _agent_type_var

    executor = AgentExecutor()
    executor.register("email", lambda task: str(_agent_type_var.get()))

    def _stream(task: AgentTask):  # type: ignore[no-untyped-def]
        yield str(_agent_type_var.get())

    executor.register_stream("email", _stream)
    task = AgentTask(query="q", agent_type="email")

    assert executor.execute(task).output == "email"
    streamed = [c for c in executor.execute_stream(task) if isinstance(c, str)]
    assert streamed == ["email"]
    assert _agent_type_var.get() is None


def test_a_crashing_handler_is_logged_and_still_a_failed_result(caplog) -> None:  # type: ignore[no-untyped-def]
    """No caller logs a failed AgentResult, so the executor does (agent + type only)."""
    import logging

    def boom(task: AgentTask) -> str:
        raise RuntimeError("db locked")

    executor = AgentExecutor()
    executor.register("email", boom)
    with caplog.at_level(logging.WARNING, logger="iris_harness.agent.agent_executor"):
        result = executor.execute(AgentTask(query="my secret question", agent_type="email"))

    assert not result.success and result.error == "db locked"
    [record] = [r for r in caplog.records if r.name == "iris_harness.agent.agent_executor"]
    assert record.getMessage() == "agent email handler failed (RuntimeError)"
    assert "my secret question" not in record.getMessage()
