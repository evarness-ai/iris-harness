"""recall_conversation and memory_search find past conversations by MEANING (PR #688 follow-up).

The #688 A/B showed the pull tools matched text literally (``LIKE '%query%'``): a question
("Which hotel am I staying at in Goa?") never matches what was said ("Booked Taj Fort
Aguada, Goa"), so the model searched and found nothing that was there. They now ask the
semantic index first, keep the literal matches (exact names and numbers), never return a
removed conversation, and read back the conversation a pointer named when nothing matches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.memory.semantic_index import RetrievedTurn
from iris_harness.memory.store import MemoryStore
from iris_harness.runtime.react_tools import builtin_react_tools
from iris_harness.runtime.turn_context import set_current_session_id


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    s.save_conversation_turns(
        "goa", [("user", "Book the Taj Fort Aguada in Goa for December 20 to 24.")]
    )
    s.save_conversation_turns("laptop", [("user", "Laptop repair ticket R-58213, pickup Friday.")])
    s.save_conversation_turns("playground-x", [("user", "Taj Fort Aguada test run")])
    set_current_session_id("now")
    return s


class _Index:
    """Returns ``turns`` for any query; records how it was asked."""

    is_ready = True

    def __init__(self, turns: list[RetrievedTurn]) -> None:
        self.turns = turns
        self.calls: list[dict[str, Any]] = []

    def query_turns_detailed(self, query: str, **kw: Any) -> list[RetrievedTurn]:
        self.calls.append(kw)
        only = kw.get("only_session")
        return [t for t in self.turns if not only or t.session_id == only]


def _turn(sid: str, content: str) -> RetrievedTurn:
    return RetrievedTurn(row_id="1", session_id=sid, role="user", content=content)


def _call(store: MemoryStore, index: Any, tool: str, args: dict[str, Any]) -> str:
    specs = builtin_react_tools(semantic_index=index, wiki=None, repo_root=None, memory_store=store)
    return str({s.name: s for s in specs}[tool].call(args))  # type: ignore[attr-defined]


GOA = "Book the Taj Fort Aguada in Goa for December 20 to 24."


def test_a_question_finds_the_turn_by_meaning(store: MemoryStore) -> None:
    index = _Index([_turn("goa", GOA)])
    out = _call(
        store, index, "recall_conversation", {"query": "Which hotel am I staying at in Goa?"}
    )
    assert "Taj Fort Aguada" in out
    assert index.calls[0]["exclude_session"] == "now"  # not the conversation asking


def test_without_an_index_it_is_the_literal_search_it_was(store: MemoryStore) -> None:
    assert "Nothing stored matches" in _call(
        store, None, "recall_conversation", {"query": "Which hotel am I staying at in Goa?"}
    )
    assert "Taj Fort Aguada" in _call(store, None, "recall_conversation", {"query": "Fort Aguada"})


def test_semantic_and_literal_hits_are_merged_without_duplicates(store: MemoryStore) -> None:
    index = _Index([_turn("goa", GOA)])
    out = _call(store, index, "recall_conversation", {"query": "Fort Aguada"})
    assert out.count("Taj Fort Aguada in Goa") == 1


def test_a_session_id_searches_only_that_conversation(store: MemoryStore) -> None:
    index = _Index(
        [_turn("goa", GOA), _turn("laptop", "Laptop repair ticket R-58213, pickup Friday.")]
    )
    out = _call(
        store, index, "recall_conversation", {"query": "pickup day", "session_id": "laptop"}
    )
    assert "Friday" in out and "Fort Aguada" not in out
    assert index.calls[0]["only_session"] == "laptop"


def test_a_pointed_conversation_is_read_back_when_nothing_matches(store: MemoryStore) -> None:
    out = _call(store, _Index([]), "recall_conversation", {"query": "zzz", "session_id": "laptop"})
    assert "R-58213" in out


def test_a_removed_conversation_never_comes_back(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "removed_session_ids", lambda: {"goa"})
    out = _call(
        store, _Index([_turn("goa", GOA)]), "recall_conversation", {"query": "hotel in Goa"}
    )
    assert "Fort Aguada" not in out


def test_an_unreadable_removal_ledger_gives_no_semantic_hits(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> set[str]:
        raise OSError("locked")

    monkeypatch.setattr(store, "removed_session_ids", boom)
    out = _call(
        store, _Index([_turn("goa", GOA)]), "recall_conversation", {"query": "hotel in Goa"}
    )
    assert "Fort Aguada" not in out


def test_playground_runs_are_not_recalled(store: MemoryStore) -> None:
    out = _call(
        store,
        _Index([_turn("playground-x", "Taj Fort Aguada test run")]),
        "recall_conversation",
        {"query": "hotel"},
    )
    assert "test run" not in out


def test_memory_search_sessions_scope_finds_conversations_by_meaning(store: MemoryStore) -> None:
    store.save_conversation_summary("goa", "Goa trip: Taj Fort Aguada, Dec 20-24")
    out = _call(
        store,
        _Index([_turn("goa", GOA)]),
        "memory_search",
        {"query": "where am I staying", "scope": "sessions"},
    )
    assert "past session goa" in out and "Goa trip" in out


def test_the_index_filters_to_one_session_when_asked() -> None:
    from iris_harness.memory.semantic_index import SemanticIndex

    seen: dict[str, Any] = {}

    class _Turns:
        def count(self) -> int:
            return 5

        def query(self, **kw: Any) -> dict[str, Any]:
            seen.update(kw)
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    index = object.__new__(SemanticIndex)
    index._ok = True  # type: ignore[attr-defined]
    index._turns = _Turns()  # type: ignore[attr-defined]
    index.query_turns_detailed("q", only_session="laptop", exclude_session="now", max_distance=1.0)
    assert seen["where"] == {"session_id": "laptop"}  # only_session wins
    index.query_turns_detailed("q", exclude_session="now")
    assert seen["where"] == {"session_id": {"$ne": "now"}}
