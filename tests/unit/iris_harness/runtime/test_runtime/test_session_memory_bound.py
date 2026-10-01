"""Session memory keeps a bounded number of sessions; the rest reload from the store."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from iris_harness.memory.compactor import ConversationTurn
from iris_harness.runtime.session_memory import SessionMemory


class _Store:
    """Just enough MemoryStore: every turn is persisted and can be reloaded."""

    def __init__(self) -> None:
        self.turns: dict[str, list[tuple[str, str]]] = {}

    def save_conversation_turns_and_get_ids(
        self, session_id: str, turns: list[tuple[str, str]]
    ) -> list[int]:
        self.turns.setdefault(session_id, []).extend(turns)
        return []

    def load_conversation_summary(self, session_id: str) -> str:
        return ""

    def load_recent_turns(self, session_id: str, limit: int) -> list[tuple[str, str]]:
        return self.turns.get(session_id, [])[-limit:]


def _sessions(monkeypatch: pytest.MonkeyPatch, cap: str | None) -> tuple[SessionMemory, _Store]:
    if cap is None:
        monkeypatch.delenv("IRIS_SESSION_MEMORY_MAX_SESSIONS", raising=False)
    else:
        monkeypatch.setenv("IRIS_SESSION_MEMORY_MAX_SESSIONS", cap)
    store = _Store()
    host: Any = SimpleNamespace(
        memory_store=store,
        semantic_index=None,
        compactor=SimpleNamespace(
            needs_compaction=lambda history: False,
            token_budget=None,
            compact=lambda history, **_: SimpleNamespace(trigger="none"),
        ),
    )
    return SessionMemory(host), store


def test_the_least_recently_used_session_is_dropped_past_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions, _ = _sessions(monkeypatch, "2")
    sessions.record_turn("a", "hi a", "hello a")
    sessions.record_turn("b", "hi b", "hello b")
    sessions.compact_now("a")  # a is used again: b is now the oldest
    sessions._session_summaries["b"] = "summary"
    sessions.record_turn("c", "hi c", "hello c")

    assert set(sessions.conversations) == {"a", "c"}
    assert "b" not in sessions._session_summaries
    assert "b" not in sessions._loaded_sessions


def test_a_dropped_session_comes_back_from_the_store(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, _ = _sessions(monkeypatch, "1")
    sessions.record_turn("a", "hi a", "hello a")
    sessions.record_turn("b", "hi b", "hello b")
    assert "a" not in sessions.conversations

    sessions.compact_now("a")  # any access reloads it
    assert [t.content for t in sessions.conversations["a"]] == ["hi a", "hello a"]
    assert "b" not in sessions.conversations


def test_a_session_under_compaction_is_not_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, _ = _sessions(monkeypatch, "1")
    sessions.record_turn("a", "hi a", "hello a")
    sessions._compacting.add("a")
    sessions.record_turn("b", "hi b", "hello b")
    assert set(sessions.conversations) == {"a", "b"}


def test_a_window_opened_elsewhere_counts_as_least_recent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions, _ = _sessions(monkeypatch, "2")
    sessions.conversations["notice-only"] = [ConversationTurn(role="assistant", content="fyi")]
    sessions.record_turn("a", "hi a", "hello a")
    sessions.record_turn("b", "hi b", "hello b")
    assert set(sessions.conversations) == {"a", "b"}


def test_a_conversation_scoped_mark_survives_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, _ = _sessions(monkeypatch, "1")
    sessions._ephemeral_sessions.add("scoped")
    sessions.record_turn("scoped", "hi", "hello")
    sessions.record_turn("other", "hi", "hello")
    assert "scoped" not in sessions.conversations
    assert "scoped" in sessions._ephemeral_sessions


@pytest.mark.parametrize(("raw", "expected"), [(None, 256), ("5", 5), ("0", 1), ("x", 256)])
def test_the_cap_comes_from_the_setting(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int
) -> None:
    sessions, _ = _sessions(monkeypatch, raw)
    assert sessions._max_sessions == expected
