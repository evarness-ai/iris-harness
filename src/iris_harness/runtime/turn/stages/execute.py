"""Stage 6 — run the plan through the governed AgentExecutor.

The first task streams token-by-token (the REPL / web UI experience); the rest
of its wave, and every later wave, run through ``execute_wave`` so independent
sub-tasks stay concurrent (ADR-0045). One code path serves both ``chat`` and
``chat_stream`` — ``chat`` simply drops the tokens.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from iris_harness.agent.agent_executor import ActivityChunk, AgentResult, AgentTask, TraceChunk
from iris_harness.foundation.observability.session_log import log_timeline_event
from iris_harness.foundation.observability.tracer import set_span_attributes
from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.stages import stage_span
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import StreamEvent


def _start_event(task: AgentTask, task_id: str, session_id: str) -> StreamEvent:
    payload = {
        "name": "agent.start",
        "agent_type": task.agent_type,
        "task_id": task_id,
        "query_chars": len(task.query),
    }
    log_timeline_event(
        "agent.trace",
        phase="agent.start",
        text=f"agent start: {task.agent_type}",
        payload=payload,
        session_id=session_id,
    )
    return StreamEvent(kind="trace", text=f"agent start: {task.agent_type}", payload=payload)


def _result_event(result: AgentResult, session_id: str) -> StreamEvent:
    payload = {
        "name": "agent.result",
        "agent_type": result.agent_type,
        "success": result.success,
        "latency_ms": result.latency_ms,
        "output_chars": len(result.output),
        **result.metadata,
    }
    log_timeline_event(
        "agent.trace",
        phase="agent.result",
        text=f"agent result: {result.agent_type}",
        payload=payload,
        session_id=session_id,
    )
    return StreamEvent(kind="trace", text=f"agent result: {result.agent_type}", payload=payload)


def _stream_task(
    runtime: TurnHost, task: AgentTask, session_id: str, results: list[AgentResult]
) -> Iterator[StreamEvent]:
    for chunk in runtime.agent_executor.execute_stream(task):
        if isinstance(chunk, str):
            yield StreamEvent(kind="token", text=chunk)
        elif isinstance(chunk, ActivityChunk):
            yield StreamEvent(kind="activity", text=chunk.text)
        elif isinstance(chunk, TraceChunk):
            payload: dict[str, Any] | None = None
            stripped = chunk.text.strip()
            if stripped.startswith("{"):
                try:
                    loaded = json.loads(stripped)
                    if isinstance(loaded, dict):
                        payload = loaded
                except json.JSONDecodeError:
                    payload = None
            log_timeline_event(
                "agent.trace",
                phase="agent.trace",
                text=chunk.text,
                payload=payload,
                session_id=session_id,
            )
            yield StreamEvent(kind="trace", text=chunk.text, payload=payload)
        elif isinstance(chunk, AgentResult):
            yield _result_event(chunk, session_id)
            results.append(chunk)


def with_upstream(task: AgentTask, *, upstream: list[tuple[str, str]]) -> AgentTask:
    """ADR-0111: a dependent sub-task sees the request, its own step, and what the
    steps it depends on produced. Returns the task unchanged when nothing is upstream."""
    if not upstream:
        return task
    request = str(task.params.get("request") or "").strip()
    lines = []
    if request and request != task.query:
        lines.append(f"Overall request: {request}")
    lines.append(f"Your step: {task.query}")
    lines.append("Results from the earlier steps this one depends on:")
    for dep_id, output in upstream:
        lines.append(f"[{dep_id}] {output.strip()}")
    return replace(task, query="\n".join(lines))


def run(runtime: TurnHost, state: TurnState) -> Iterator[StreamEvent]:
    session_id = state.session_id
    waves: list[list[AgentTask]] = state.__dict__.pop("_waves", [])
    task_ids: list[str] = state.__dict__.pop("_task_ids", [])
    deps: dict[str, list[str]] = state.__dict__.pop("_deps", {})
    ids = iter(task_ids)
    results: list[AgentResult] = []
    by_id: dict[str, AgentResult] = {}

    def _bind(task: AgentTask, task_id: str) -> AgentTask:
        upstream = [(d, by_id[d].output) for d in deps.get(task_id, []) if d in by_id]
        return with_upstream(task, upstream=upstream)

    with stage_span(runtime, state, "iris.stage.agent_executor") as span:
        for wave_index, wave in enumerate(waves):
            if wave_index == 0:
                first, rest = wave[0], wave[1:]
                first_id = next(ids, "t1")
                yield _start_event(first, first_id, session_id)
                before = len(results)
                yield from _stream_task(runtime, _bind(first, first_id), session_id, results)
                if len(results) > before:
                    by_id[first_id] = results[-1]
            else:
                rest = wave
            if rest:
                rest_ids = [next(ids, task.agent_type) for task in rest]
                for task, task_id in zip(rest, rest_ids, strict=True):
                    yield _start_event(task, task_id, session_id)
                bound = [_bind(task, task_id) for task, task_id in zip(rest, rest_ids, strict=True)]
                for task_id, result in zip(
                    rest_ids, runtime.agent_executor.execute_wave(bound), strict=True
                ):
                    yield _result_event(result, session_id)
                    results.append(result)
                    by_id[task_id] = result
        state.results = results
        set_span_attributes(
            span,
            {
                "iris.result_count": len(results),
                "iris.agents": ",".join(r.agent_type for r in results),
                "iris.has_errors": any(not r.success for r in results),
            },
        )
    if not results:
        raise RuntimeError("streaming pipeline produced no results")
