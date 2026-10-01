"""Issue 0002 — the routing context fed to the (tiny num_ctx) Router model must
be hard-capped so a long conversation can never overflow the window and silently
truncate the system prompt. Tests SessionMemory.format_recent_context's bound directly."""

from __future__ import annotations

from types import SimpleNamespace

from iris_harness.runtime.session_memory import SessionMemory


def _sessions(turns: list[SimpleNamespace]) -> SessionMemory:
    sessions = SessionMemory(SimpleNamespace())  # type: ignore[arg-type]
    sessions.conversations = {"s": turns}  # type: ignore[dict-item]
    return sessions


def _turn(role: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=content)


def test_context_none_on_empty_history() -> None:
    assert _sessions([]).format_recent_context("s") is None


def test_context_is_hard_capped_for_a_huge_conversation() -> None:
    # 50 very long turns — the cap must bound the result regardless.
    turns = [_turn("user" if i % 2 == 0 else "assistant", "x" * 2000) for i in range(50)]
    ctx = _sessions(turns).format_recent_context("s")
    assert ctx is not None
    assert len(ctx) <= SessionMemory._ROUTING_CONTEXT_MAX_CHARS


def test_context_keeps_the_most_recent_turn() -> None:
    turns = [
        _turn("user", "first message long ago"),
        _turn("assistant", "old reply"),
        _turn("user", "any AI emails?"),
        _turn("assistant", "yes, an CourseHub one about AI workflows"),
    ]
    ctx = _sessions(turns).format_recent_context("s")
    assert ctx is not None
    assert "CourseHub" in ctx  # the latest exchange is what routing needs
    # chronological order preserved (older line before newer when both fit)
    assert ctx.index("any AI emails?") < ctx.index("CourseHub")
