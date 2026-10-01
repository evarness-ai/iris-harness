"""Agent dispatch with memory-context injection and result collection."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from iris_harness.foundation.observability.session_log import agent_scope, bind_context
from iris_harness.foundation.observability.trace_builder import register_agent_source
from iris_harness.memory.retriever import MemoryContext

logger = logging.getLogger(__name__)

# Default cap on concurrent agents within one execution wave. Small on purpose:
# independent branches overlap their I/O, but a local single-model LLM serialises
# anyway, so a large pool buys little and risks resource pressure.
DEFAULT_WAVE_WORKERS = 4

# Handlers may return either a plain string or a (text, extra_metadata) tuple.
# The tuple form lets handlers surface structured data (e.g. token usage) without
# requiring a separate return type.
HandlerResult = str | tuple[str, dict[str, object]]


@dataclass(frozen=True)
class ActivityChunk:
    """Short, ephemeral status line for the live UI ("Running script…").

    Activity chunks update the live status indicator (spinner replacement) but
    are NOT appended to the final user-visible response. Use for progress hints
    that should disappear once the turn finishes.
    """

    text: str


@dataclass(frozen=True)
class TraceChunk:
    """Raw trace data buffered for opt-in inspection (e.g. /trace command).

    Trace chunks are never streamed live to the user; they accumulate in the
    client and surface only when the user explicitly asks to see the detailed
    activity (cmd, stdout, stderr, JSON tool calls, etc.).
    """

    text: str


# Streaming handlers yield text chunks (str) followed by an optional final
# metadata dict (e.g. token usage). The dict — if yielded — must be the last
# item produced by the iterator. Activity/Trace chunks travel alongside text.
StreamChunk = str | dict[str, object] | ActivityChunk | TraceChunk
StreamHandler = Callable[["AgentTask"], Iterator[StreamChunk]]


@dataclass
class AgentTask:
    """A unit of work dispatched to a registered agent."""

    query: str
    agent_type: str
    memory_context: MemoryContext | None = None
    session_id: str | None = None
    params: dict[str, str] = field(default_factory=dict)
    # The gateway this work came from ("console", "web", "telegram", ...). Carried so a
    # governance halt can be delivered back to the surface that caused it: the approval
    # queue has always had a `channel` column and, with nothing to stamp it from, has
    # always recorded the "cli" default.
    origin_channel: str = "console"
    # ADR-0106 Tier B (M5.C5c): continue a halted run rather than start a new one.
    # A first-class field, not a `params` entry, because `params` is str-valued
    # prompt/routing bias that handlers may pass through to a model — a resume point
    # is control flow, and a resume that silently degraded to a string in a prompt is
    # exactly the class of bug the continuation work exists to remove.
    resume_run_id: str | None = None
    resume_step_id: int | None = None
    # What to inject as the observation of the step that paused, when anything should
    # be. Tier B sets the user's reply here; an approved *governance* halt leaves it
    # None, because that step's observation is a real tool result and overwriting it
    # would destroy the very work the resume exists to continue.
    resume_reply: str | None = None
    # ADR-0106 ``choice`` continuation: the option the user picked from a numbered list
    # this agent offered ("which email did you mean? 1. ... 2. ..."), exactly as the
    # agent stored it when it asked. First-class for the same reason as the resume
    # fields: it is control data naming what to act on, not prompt bias for a model.
    selected_choice: dict[str, Any] | None = None


@dataclass
class AgentResult:
    """Structured result from one agent execution."""

    agent_type: str
    output: str
    success: bool
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class AgentExecutor:
    """Dispatch tasks to registered agent handlers with memory injection."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[AgentTask], HandlerResult]] = {}
        self._stream_handlers: dict[str, StreamHandler] = {}

    def register(self, agent_type: str, handler: Callable[[AgentTask], HandlerResult]) -> None:
        self._handlers[agent_type] = handler
        # The trace view names the code that answered from this callable.
        register_agent_source(agent_type, handler)

    def register_stream(self, agent_type: str, handler: StreamHandler) -> None:
        """Register an optional streaming variant of an agent handler."""
        self._stream_handlers[agent_type] = handler
        if agent_type not in self._handlers:
            register_agent_source(agent_type, handler)

    def has_stream_handler(self, agent_type: str) -> bool:
        return agent_type in self._stream_handlers or "system" in self._stream_handlers

    def registered_agents(self) -> frozenset[str]:
        """Return currently registered agent types.

        Includes both sync and stream registrations so runtime planning can
        constrain task outputs to capabilities that exist in this process.
        """
        return frozenset(set(self._handlers) | set(self._stream_handlers))

    def execute(self, task: AgentTask) -> AgentResult:
        handler = self._handlers.get(task.agent_type)
        if handler is None and not task.agent_type:
            handler = self._handlers.get("system")
        if handler is None:
            return AgentResult(
                agent_type=task.agent_type,
                output="",
                success=False,
                error=f"unsupported agent_type '{task.agent_type}'",
            )
        start = time.monotonic()
        try:
            with agent_scope(task.agent_type or "system"):
                raw = handler(task)
            if isinstance(raw, tuple):
                output, extra_meta = raw
            else:
                output, extra_meta = raw, {}
            return AgentResult(
                agent_type=task.agent_type,
                output=output,
                success=True,
                latency_ms=(time.monotonic() - start) * 1000,
                metadata=extra_meta,
            )
        except Exception as exc:
            # No caller logs a failed AgentResult, so without this a crashing handler
            # left only its error text in the answer. Agent + exception type, never the
            # task's message.
            logger.warning(
                "agent %s handler failed (%s)",
                task.agent_type or "system",
                type(exc).__name__,
                exc_info=True,
            )
            return AgentResult(
                agent_type=task.agent_type,
                output="",
                success=False,
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
            )

    def execute_plan(self, tasks: list[AgentTask]) -> list[AgentResult]:
        """Execute a list of tasks sequentially and return all results."""
        return [self.execute(task) for task in tasks]

    def execute_wave(
        self, tasks: Sequence[AgentTask], *, max_workers: int = DEFAULT_WAVE_WORKERS
    ) -> list[AgentResult]:
        """Execute independent tasks concurrently; results keep input order.

        A single task runs inline (no thread). ``execute`` never raises — it
        captures failures into ``AgentResult`` — so one task's error can't
        abort its siblings.
        """
        if len(tasks) <= 1:
            return [self.execute(t) for t in tasks]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
            # Each task gets its own copy of the turn's context, so its LLM calls
            # still log against this session and turn.
            futures = [pool.submit(bind_context(self.execute), t) for t in tasks]
            return [f.result() for f in futures]

    def execute_waves(
        self, waves: Sequence[Sequence[AgentTask]], *, max_workers: int = DEFAULT_WAVE_WORKERS
    ) -> list[AgentResult]:
        """Run dependency-ordered waves: tasks within a wave run concurrently,
        waves run in order. Flattened results follow wave then in-wave order."""
        results: list[AgentResult] = []
        for wave in waves:
            results.extend(self.execute_wave(wave, max_workers=max_workers))
        return results

    def execute_stream(
        self, task: AgentTask
    ) -> Iterator[str | AgentResult | ActivityChunk | TraceChunk]:
        """Streaming variant — yields text chunks, then a final ``AgentResult``.

        Falls back to a single-chunk emit when no streaming handler is
        registered for the task's agent type.
        """
        handler = self._stream_handlers.get(task.agent_type)
        if handler is None and not task.agent_type:
            handler = self._stream_handlers.get("system")
        if handler is None:
            # No streaming handler for this agent type. Rather than fail, fall
            # back to the non-streaming handler (planner, coding_agent, etc.
            # register only ``execute``) and emit its result as one chunk, so
            # every registered agent works on the streaming path too.
            yield from self._stream_via_execute(task)
            return

        start = time.monotonic()
        chunks: list[str] = []
        extra_meta: dict[str, object] = {}
        try:
            with agent_scope(task.agent_type or "system"):
                for item in handler(task):
                    if isinstance(item, str):
                        if item:
                            chunks.append(item)
                            yield item
                    elif isinstance(item, dict):
                        extra_meta.update(item)
                    elif isinstance(item, ActivityChunk | TraceChunk):
                        yield item
            yield AgentResult(
                agent_type=task.agent_type,
                output="".join(chunks),
                success=True,
                latency_ms=(time.monotonic() - start) * 1000,
                metadata=extra_meta,
            )
        except Exception as exc:
            logger.warning("agent '%s' execution failed: %s", task.agent_type, exc, exc_info=True)
            yield AgentResult(
                agent_type=task.agent_type,
                output="".join(chunks),
                success=False,
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
                metadata=extra_meta,
            )

    def _stream_via_execute(self, task: AgentTask) -> Iterator[str | AgentResult]:
        """Adapt a non-streaming handler to the streaming protocol.

        Runs the synchronous ``execute`` handler and yields its output as a
        single chunk followed by the ``AgentResult``. Used when an agent type
        is registered for ``execute`` but not ``execute_stream`` — without this,
        such agents (planner, coding_agent) would report
        ``unsupported agent_type`` on the streaming chat path.
        """
        result = self.execute(task)
        if result.success and result.output:
            yield result.output
        yield result
