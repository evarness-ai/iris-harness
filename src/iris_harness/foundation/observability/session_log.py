"""Session-scoped JSONL log capturing every LLM call, tool run, and error.

Each conversation turn populates `~/.iris/logs/session-<session_id>.jsonl`
with chronological events: ``user_message`` → one or more ``llm_call`` and
``tool_run`` records → ``agent_response``. ``error`` records may appear at
any point.

Context is propagated via ContextVars rather than threaded through every
function signature: a top-level ``session_scope(session_id)`` covers the
whole turn, and inner ``agent_scope(agent_type, iteration)`` blocks tag
which agent (intent_router, general, code_exec, …) is currently running.
``llm_call_scope`` reads those context vars automatically when wrapping
``CodingLLMClient.invoke*`` calls.

The log is best-effort: append failures are swallowed via ``logger.exception``
so logging can never break the runtime.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_P = ParamSpec("_P")

# Honor IRIS_HOME (and an explicit IRIS_SESSION_LOG_DIR override) so the test suite
# — which relocates IRIS_HOME to a throwaway temp dir in conftest BEFORE any import
# — never writes session logs into the developer's real ``~/.iris/logs``. Previously
# this used ``Path.home()`` directly, so every pytest run appended events under the
# fixed test session-ids (clarify-session, undo-session, …), accumulating millions
# of lines over days and crowding real traces out of the workload scans.
#
# Resolved on every use (:func:`session_log_dir`), not once at import, so a process that
# relocates IRIS_HOME after importing IRIS -- ``iris_harness.testing``'s harness --
# logs into the new home. ``LOG_DIR`` is an override slot (``None``: resolve from the
# environment) for a caller that points the logs somewhere else outright.
LOG_DIR: Path | None = None


def session_log_dir() -> Path:
    """Where session logs go: the ``LOG_DIR`` override, else ``$IRIS_SESSION_LOG_DIR``,
    else ``$IRIS_HOME/logs`` (``~/.iris/logs``)."""
    if LOG_DIR is not None:
        return LOG_DIR
    return Path(
        os.environ.get("IRIS_SESSION_LOG_DIR")
        or (Path(os.environ.get("IRIS_HOME") or (Path.home() / ".iris")) / "logs")
    )


_INPUT_CONTENT_CAP = 8192
# IRIS_LOG_PROMPT_CAP raises the cap on a logged LLM prompt (default 8192 chars). The
# middle cut drops a ReAct prompt's tool menu and rules, so a model bake-off that replays
# logged prompts needs the whole text. Read per call; below the default is ignored.
_PROMPT_CAP_ENV = "IRIS_LOG_PROMPT_CAP"
_OUTPUT_CONTENT_CAP = 16384
# A full portfolio / brief observation can exceed 4 KB; log enough to verify the model
# saw the complete tool output (issue 0032). Logging only — does not affect the loop.
_TOOL_OUTPUT_CAP = 16384
_TIMELINE_TEXT_CAP = 4096

_session_id_var: ContextVar[str | None] = ContextVar("iris_session_id", default=None)
_agent_type_var: ContextVar[str | None] = ContextVar("iris_agent_type", default=None)
_iteration_var: ContextVar[int | None] = ContextVar("iris_iteration", default=None)
_turn_id_var: ContextVar[str | None] = ContextVar("iris_turn_id", default=None)

# In-process event subscribers. Every appended event is also fanned out to
# these callbacks (best-effort, never breaks logging). Lets in-process tooling
# — e.g. the playground scenario runner — observe the pipeline live without
# re-reading the JSONL file. Callbacks receive the same dict written to disk.
_EventSink = "Callable[[dict[str, Any]], None]"
_subscribers: list[Any] = []


def subscribe_events(callback: Any) -> Any:
    """Register *callback* to receive every logged event. Returns an unsubscribe.

    The callback must not raise; if it does, the error is swallowed so logging
    stays best-effort. Intended for short-lived, scoped observation (register,
    run a turn, unsubscribe) rather than long-lived global listeners.
    """
    _subscribers.append(callback)

    def _unsubscribe() -> None:
        with suppress(ValueError):
            _subscribers.remove(callback)

    return _unsubscribe


def session_log_path(session_id: str) -> Path:
    return session_log_dir() / f"session-{session_id}.jsonl"


@dataclass(frozen=True)
class TurnRecord:
    """A reconstructed conversation turn: the user query + the agent's answer.

    Mined from session logs for success-trace analysis (e.g. the skill
    crystallizer). ``query``/``response`` are the captured (capped) text.
    """

    session_id: str
    turn_id: str | None
    intent: str
    agent_type: str
    query: str
    response: str
    has_errors: bool


def iter_recent_turns(
    *,
    limit: int = 200,
    log_dir: Path | None = None,
    intent: str | None = None,
    scan_cap: int = 100_000,
    require_query: bool = False,
    unique_queries: bool = False,
) -> list[TurnRecord]:
    """Reconstruct recent (query, response) turns from session logs, newest first.

    Pairs each ``user_message`` with the following ``agent_response`` in the same
    session file. Best-effort: malformed lines are skipped, missing dirs yield [].

    ``intent`` (when set) filters *while scanning* and treats ``limit`` as the number
    of *matching* turns to return — so a low-volume intent (e.g. ``calendar``) isn't
    crowded out of the newest ``limit`` turns by high-volume ones (heartbeats,
    routine-authoring, finance ingest). ``require_query`` additionally skips turns
    with no paired user query (e.g. heartbeat-generated responses) so ``limit``
    counts only real user turns — without it, empty-query turns can fill the budget
    before the genuine queries are reached. ``unique_queries`` de-duplicates by query
    text while scanning so ``limit`` counts *distinct* queries — without it a heavily
    repeated query (some recur hundreds of times) can fill the budget before other
    distinct queries are reached. ``scan_cap`` bounds the number of turns examined so
    a deep filtered scan stays affordable.
    """
    root = log_dir or session_log_dir()
    if not root.exists():
        return []
    paths = sorted(root.glob("session-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    turns: list[TurnRecord] = []
    seen_queries: set[str] = set()
    examined = 0
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        pending_query: str | None = None
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("kind")
            if kind == "user_message":
                pending_query = str(event.get("text") or "")
            elif kind == "agent_response":
                examined += 1
                turn_intent = str(event.get("intent") or "")
                query_text = (pending_query or "").strip()
                skip = (
                    (intent is not None and turn_intent != intent)
                    or (require_query and not query_text)
                    or (unique_queries and query_text in seen_queries)
                )
                if skip:
                    pending_query = None
                    if examined >= scan_cap:
                        return turns
                    continue
                if unique_queries and query_text:
                    seen_queries.add(query_text)
                turns.append(
                    TurnRecord(
                        session_id=str(event.get("session_id") or ""),
                        turn_id=(str(event["turn_id"]) if event.get("turn_id") else None),
                        intent=turn_intent,
                        agent_type=str(event.get("agent_type") or ""),
                        query=pending_query or "",
                        response=str(event.get("response") or ""),
                        has_errors=bool(event.get("has_errors")),
                    )
                )
                pending_query = None
                if len(turns) >= limit or examined >= scan_cap:
                    return turns
    return turns


def current_session_id() -> str | None:
    return _session_id_var.get()


def current_turn_id() -> str | None:
    return _turn_id_var.get()


@contextmanager
def turn_scope(turn_id: str | None = None) -> Iterator[str]:
    """Tag every event inside this block with a stable per-turn id.

    One user message → one ``turn_scope``. The id threads the user message, LLM
    calls, tool runs, the agent response, and the emitted learning signal into a
    single chain (learning-observability.md §4.1 threat 2, P5), so a
    ``learning.db`` row can be joined back to its session-log turn. A uuid is
    minted when not supplied. Save/restore (not ``Token.reset``) for the same
    generator-safety reason as ``session_scope``.
    """
    minted = turn_id or uuid.uuid4().hex
    prev = _turn_id_var.get()
    _turn_id_var.set(minted)
    try:
        yield minted
    finally:
        _turn_id_var.set(prev)


@contextmanager
def session_scope(session_id: str | None) -> Iterator[None]:
    """Tag every LLM call inside this block with ``session_id``.

    Uses a save/restore pattern rather than ``Token.reset`` so the scope
    survives generator boundaries — IRIS yields ``StreamEvent`` chunks across
    async/thread boundaries and ``Token``-based reset is invalid in a
    different Context.
    """
    prev = _session_id_var.get()
    _session_id_var.set(session_id)
    try:
        yield
    finally:
        _session_id_var.set(prev)


@contextmanager
def agent_scope(agent_type: str, iteration: int | None = None) -> Iterator[None]:
    """Tag LLM calls inside this block with the running agent + optional iteration.

    Save/restore (not ``Token.reset``) for the same generator-safety reason as
    ``session_scope``.
    """
    prev_agent = _agent_type_var.get()
    prev_iter = _iteration_var.get()
    _agent_type_var.set(agent_type)
    _iteration_var.set(iteration)
    try:
        yield
    finally:
        _iteration_var.set(prev_iter)
        _agent_type_var.set(prev_agent)


def pin_context(iterator: Iterator[_T]) -> Iterator[_T]:
    """Step *iterator* inside one ``contextvars.Context`` for its whole life.

    A server that streams a sync generator advances it one ``next()`` at a time on
    a worker thread, and each step runs in a *fresh copy* of the caller's context
    (Starlette's ``iterate_in_threadpool`` → ``anyio.to_thread.run_sync``). So a
    ``session_scope`` / ``turn_scope`` entered inside the generator is gone by the
    second step: every later event lost its ``turn_id`` and every ``llm_call``
    silently no-oped, which is why no session log said which model answered.
    Pinning runs every step in the same context, so what a step sets, the next
    step still sees. ``close()`` runs in that context too, so ``finally`` blocks
    restore the vars they set.
    """
    ctx = contextvars.copy_context()
    try:
        while True:
            try:
                item = ctx.run(next, iterator)
            except StopIteration:
                return
            yield item
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            ctx.run(close)


def bind_context(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    """Return *fn* bound to a copy of the current context, for a worker thread.

    ``ThreadPoolExecutor.submit`` / ``map`` do not carry context vars into the
    worker (``asyncio.to_thread`` does), so an LLM call made there is logged
    against no session. Copy per task: one ``Context`` cannot be entered by two
    threads at once.
    """
    ctx = contextvars.copy_context()

    def _bound(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        return ctx.run(fn, *args, **kwargs)

    return _bound


def _cap(text: str | None, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "…[truncated]"


def _prompt_cap() -> int:
    """The logged-prompt cap: IRIS_LOG_PROMPT_CAP when it is a whole number above the
    default, else the default (unset, junk and smaller values all keep 8192)."""
    raw = os.environ.get(_PROMPT_CAP_ENV, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return _INPUT_CONTENT_CAP
    return max(value, _INPUT_CONTENT_CAP)


def _cap_middle(text: str | None, limit: int) -> str:
    """Cap *text* by cutting its middle, keeping the start and (mostly) the end.

    For logged prompts. A ReAct prompt is a fixed preamble — identity, tools, rules —
    followed by the scratchpad the model is actually reacting to. Cutting from the end
    kept the preamble and dropped every Observation, so a looping model's log showed
    what it replied but never what it was replying to.
    """
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n…[{omitted} chars omitted]…\n{text[-tail:]}"


def _cap_value(value: Any, limit: int = _TIMELINE_TEXT_CAP) -> Any:
    if isinstance(value, str):
        return _cap(value, limit)
    if isinstance(value, Mapping):
        return {str(k): _cap_value(v, limit) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_cap_value(item, limit) for item in value[:50]]
    return value


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _prime_cpu_sample() -> None:
    """Reset psutil's CPU window so a later ``_sample_resources`` reflects this call.

    ``psutil.cpu_percent(interval=None)`` reports usage *since the previous call*;
    priming at the start of an LLM/tool span makes the figure logged at the end a
    per-span measurement rather than a system-wide average. Best-effort.
    """
    with suppress(Exception):
        import psutil

        psutil.cpu_percent(interval=None)


def _sample_resources() -> dict[str, Any] | None:
    """Best-effort host resource snapshot for a logged step.

    Reuses the LLM tier governor's pressure sampler so the numbers match what
    drives model selection. GPU utilization is not obtainable via psutil on
    macOS/Apple Silicon (would need privileged ``powermetrics``), so it is
    reported as ``None`` — a reserved field the trace UI renders as "n/a".

    Returns ``None`` on any failure; logging must never break the runtime.
    """
    try:
        import psutil

        from iris_harness.foundation.observability.host_pressure import sample_pressure

        snap = sample_pressure()
        return {
            "cpu_percent": snap.cpu_percent,
            "ram_free_gb": snap.ram_free_gb,
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_percent": None,
            "thermal_throttled": snap.thermal_throttled,
        }
    except Exception:  # noqa: BLE001
        return None


def _append_event(session_id: str, event: dict[str, Any]) -> None:
    try:
        path = session_log_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=True, separators=(",", ":"), default=str)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        logger.exception("session_log append failed; continuing")
    # Fan out to in-process subscribers after the durable write. Best-effort:
    # a misbehaving subscriber must never break logging or the runtime.
    if _subscribers:
        for callback in list(_subscribers):
            try:
                callback(event)
            except Exception:
                logger.exception("session_log subscriber failed; continuing")


def log_user_message(
    session_id: str,
    *,
    text: str,
    intent: str | None = None,
    agent_type: str | None = None,
) -> None:
    _append_event(
        session_id,
        {
            "kind": "user_message",
            "ts": _now_iso(),
            "session_id": session_id,
            "turn_id": _turn_id_var.get(),
            "text": _cap(text, _INPUT_CONTENT_CAP),
            "intent": intent,
            "agent_type": agent_type,
        },
    )


def log_agent_response(
    session_id: str,
    *,
    response: str,
    intent: str,
    agent_type: str,
    has_errors: bool,
    total_tokens: int | None = None,
    total_duration_ms: float | None = None,
) -> None:
    _append_event(
        session_id,
        {
            "kind": "agent_response",
            "ts": _now_iso(),
            "session_id": session_id,
            "turn_id": _turn_id_var.get(),
            "response": _cap(response, _OUTPUT_CONTENT_CAP),
            "intent": intent,
            "agent_type": agent_type,
            "has_errors": has_errors,
            "total_tokens": total_tokens,
            "total_duration_ms": total_duration_ms,
            "resources": _sample_resources(),
        },
    )


def log_tool_run(
    *,
    cmd: str,
    exit_code: int,
    duration_ms: float,
    stdout: str = "",
    stderr: str = "",
    artifacts: list[str] | None = None,
) -> None:
    """Record a sandbox shell run. No-op if not inside a session_scope."""
    session_id = _session_id_var.get()
    if not session_id:
        return
    _append_event(
        session_id,
        {
            "kind": "tool_run",
            "ts": _now_iso(),
            "session_id": session_id,
            "agent_type": _agent_type_var.get() or "unknown",
            "iteration": _iteration_var.get(),
            "cmd": cmd,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "stdout": _cap(stdout, _TOOL_OUTPUT_CAP),
            "stderr": _cap(stderr, _TOOL_OUTPUT_CAP),
            "artifacts": list(artifacts or []),
            "resources": _sample_resources(),
        },
    )


def log_error(*, error_type: str, message: str, phase: str | None = None) -> None:
    """Record a free-standing error. No-op if not inside a session_scope."""
    session_id = _session_id_var.get()
    if not session_id:
        return
    _append_event(
        session_id,
        {
            "kind": "error",
            "ts": _now_iso(),
            "session_id": session_id,
            "agent_type": _agent_type_var.get() or "unknown",
            "iteration": _iteration_var.get(),
            "type": error_type,
            "message": message,
            "phase": phase,
        },
    )


def log_timeline_event(
    kind: str,
    *,
    phase: str | None = None,
    text: str | None = None,
    payload: Mapping[str, Any] | None = None,
    session_id: str | None = None,
) -> None:
    """Record a replayable pipeline event. No-op if not inside a session_scope."""
    session_id = session_id or _session_id_var.get()
    if not session_id:
        return
    event: dict[str, Any] = {
        "kind": kind,
        "ts": _now_iso(),
        "session_id": session_id,
        "turn_id": _turn_id_var.get(),
        "agent_type": _agent_type_var.get() or "unknown",
        "iteration": _iteration_var.get(),
    }
    if phase is not None:
        event["phase"] = phase
    if text is not None:
        event["text"] = _cap(text, _TIMELINE_TEXT_CAP)
    if payload is not None:
        event["payload"] = _cap_value(payload)
    _append_event(session_id, event)


@contextmanager
def llm_call_scope(
    *,
    model: str,
    provider: str,
    input_messages: list[dict[str, str]],
    tier: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Wrap an LLM invocation. Yields a state dict the caller populates.

    On normal exit, an ``llm_call`` event is logged with the populated state
    plus duration. On exception, an ``error`` event is logged and the
    exception re-raised. Outside a ``session_scope`` the whole thing is a
    no-op.
    """
    session_id = _session_id_var.get()
    state: dict[str, Any] = {"output_text": "", "tool_calls": None, "tokens": None}
    if session_id is None:
        yield state
        return

    turn_id = _turn_id_var.get()
    agent_type = _agent_type_var.get() or "unknown"
    iteration = _iteration_var.get()
    start = time.monotonic()
    _prime_cpu_sample()
    cap = _prompt_cap()
    capped_input = [
        {
            "role": str(m.get("role", "")),
            "content": _cap_middle(str(m.get("content", "")), cap),
        }
        for m in input_messages
    ]
    try:
        yield state
    except Exception as exc:
        _append_event(
            session_id,
            {
                "kind": "error",
                "ts": _now_iso(),
                "session_id": session_id,
                "turn_id": turn_id,
                "agent_type": agent_type,
                "iteration": iteration,
                "type": type(exc).__name__,
                "message": str(exc),
                "phase": "llm_call",
                "model": model,
                "provider": provider,
                "tier": tier,
            },
        )
        raise
    else:
        _append_event(
            session_id,
            {
                "kind": "llm_call",
                "ts": _now_iso(),
                "session_id": session_id,
                "turn_id": turn_id,
                "agent_type": agent_type,
                "iteration": iteration,
                "model": model,
                "provider": provider,
                "tier": tier,
                "input_messages": capped_input,
                "output": {
                    "text": _cap(state["output_text"], _OUTPUT_CONTENT_CAP),
                    "tool_calls": state["tool_calls"],
                },
                "tokens": state["tokens"],
                "duration_ms": (time.monotonic() - start) * 1000,
                "resources": _sample_resources(),
            },
        )


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_subscribers")
