"""Session/turn scopes survive the thread hops a streamed turn makes.

A web chat turn is a sync generator that Starlette steps one ``next()`` at a time on
worker threads, each in a fresh copy of the request context. Before ``pin_context``
the scopes entered inside the generator were gone after its first event, so every
``llm_call`` record (the only place a session log names the model) silently no-oped.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from starlette.concurrency import iterate_in_threadpool

from iris_harness.foundation.observability import session_log
from iris_harness.foundation.observability.session_log import (
    bind_context,
    current_session_id,
    current_turn_id,
    llm_call_scope,
    pin_context,
    session_scope,
    turn_scope,
)


def _turn() -> Iterator[tuple[str | None, str | None]]:
    with session_scope("sess-1"), turn_scope("turn-1"):
        yield current_session_id(), current_turn_id()
        yield current_session_id(), current_turn_id()
        with llm_call_scope(
            model="granite4:latest", provider="ollama", input_messages=[], tier="tier1"
        ) as state:
            state["output_text"] = "hi"
        yield current_session_id(), current_turn_id()


def _drain_like_starlette(iterator: Iterator[tuple[str | None, str | None]]) -> list:
    async def _run() -> list:
        return [item async for item in iterate_in_threadpool(iterator)]

    return asyncio.run(_run())


def test_unpinned_stream_loses_scopes_after_first_event() -> None:
    # The failure being fixed — kept so a Starlette change that stops copying
    # context per step is noticed rather than assumed.
    seen = _drain_like_starlette(_turn())
    assert seen[0] == ("sess-1", "turn-1")
    assert seen[1] == (None, None)


def test_pinned_stream_keeps_scopes_and_logs_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)

    seen = _drain_like_starlette(pin_context(_turn()))

    assert seen == [("sess-1", "turn-1")] * 3
    events = [
        json.loads(line)
        for line in (tmp_path / "session-sess-1.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    [call] = [e for e in events if e["kind"] == "llm_call"]
    assert call["model"] == "granite4:latest"
    assert call["provider"] == "ollama"
    assert call["tier"] == "tier1"
    assert call["turn_id"] == "turn-1"


def test_pinned_stream_close_runs_inner_cleanup() -> None:
    closed: list[bool] = []

    def _gen() -> Iterator[int]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = pin_context(_gen())
    assert next(stream) == 1
    stream.close()  # type: ignore[attr-defined]
    assert closed == [True]


def test_bind_context_carries_scopes_into_a_thread_pool() -> None:
    with session_scope("sess-2"), turn_scope("turn-2"), ThreadPoolExecutor(2) as pool:
        plain = pool.submit(lambda: (current_session_id(), current_turn_id())).result()
        bound = pool.submit(bind_context(lambda: (current_session_id(), current_turn_id())))
        assert plain == (None, None)
        assert bound.result() == ("sess-2", "turn-2")


def test_logged_prompts_keep_their_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ReAct prompt's end is the scratchpad the model is reacting to; the cap keeps it."""
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    prompt = "PREAMBLE " * 3000 + "Observation: Inbox today: 73 new."

    with (
        session_scope("sess-cap"),
        llm_call_scope(
            model="m", provider="p", input_messages=[{"role": "user", "content": prompt}]
        ),
    ):
        pass

    (line,) = (tmp_path / "session-sess-cap.jsonl").read_text(encoding="utf-8").splitlines()
    logged = json.loads(line)["input_messages"][0]["content"]
    assert logged.startswith("PREAMBLE")
    assert logged.endswith("Observation: Inbox today: 73 new.")
    assert "chars omitted" in logged
    assert len(logged) < len(prompt)
