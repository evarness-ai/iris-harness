"""IRIS_MEMORY_RECALL_MODE: push (default, unchanged) vs pointer (2026-09-27, owner decision).

pointer: turns from OTHER conversations are not put in the prompt; a one-line note names
up to two of those conversations (title, date, session_id) for recall_conversation, unless
the message clearly refers back, when the nearest two turns are pushed as before.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris_harness.memory import retriever as retriever_module
from iris_harness.memory.retriever import MemoryRetriever, recall_mode, refers_back
from iris_harness.memory.semantic_index import RetrievedTurn
from iris_harness.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _index(turns: list[RetrievedTurn]) -> Any:
    return SimpleNamespace(
        is_ready=True,
        query_facts=lambda q, n: [],
        query_turns_detailed=lambda q, **kw: list(turns),
        query_episodic=lambda q, n: [],
    )


@pytest.fixture
def turns(store: MemoryStore) -> list[RetrievedTurn]:
    """Three turns from two earlier conversations, stored so they have dates."""
    ids = store.save_conversation_turns_and_get_ids(
        "car-sess", [("user", "renew my car insurance policy"), ("assistant", "renewed")]
    )
    ids2 = store.save_conversation_turns_and_get_ids(
        "dentist-sess", [("user", "dentist bill was 240 dollars")]
    )
    store.save_conversation_summary("car-sess", "Car insurance renewal with Geico for 2027")
    return [
        RetrievedTurn(
            row_id=str(ids[0]),
            session_id="car-sess",
            role="user",
            content="renew my car insurance policy",
        ),
        RetrievedTurn(
            row_id=str(ids[1]), session_id="car-sess", role="assistant", content="renewed"
        ),
        RetrievedTurn(
            row_id=str(ids2[0]),
            session_id="dentist-sess",
            role="user",
            content="dentist bill was 240 dollars",
        ),
    ]


def _context(store: MemoryStore, turns: list[RetrievedTurn], query: str) -> Any:
    return MemoryRetriever(store=store, index=_index(turns)).build_context(
        query=query, session_id="now-sess"
    )


def test_push_is_the_default_and_unchanged(
    store: MemoryStore, turns: list[RetrievedTurn], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_MEMORY_RECALL_MODE", raising=False)
    context = _context(store, turns, "insurance")
    assert recall_mode() == "push"
    assert len(context.related_turns) == 3 and context.pointers == ()
    assert [t.session_id for t in context.reused_turn_refs] == [
        "car-sess",
        "car-sess",
        "dentist-sess",
    ]
    assert context.recall_pointer_sessions == () and not context.recall_backstop


@pytest.mark.parametrize("value", ["", "PUSH", "points", "typo"])
def test_anything_but_pointer_is_push(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_MEMORY_RECALL_MODE", value)
    assert recall_mode() == "push"


def test_pointer_names_conversations_instead_of_pushing_their_text(
    store: MemoryStore, turns: list[RetrievedTurn], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_MEMORY_RECALL_MODE", "Pointer")
    context = _context(store, turns, "how much is my insurance")
    assert context.related_turns == () and context.reused_turn_refs == ()
    [note] = context.pointers
    assert "NOT loaded" in note and "recall_conversation" in note
    assert "session_id=car-sess" in note and "session_id=dentist-sess" in note
    assert "Car insurance renewal with Geico" in note  # the summary titles it
    assert "240 dollars" in note  # no summary: the matched turn titles it
    assert note.count("session_id=car-sess") == 1  # one pointer per conversation
    assert context.recall_pointer_sessions == ("car-sess", "dentist-sess")


def test_pointer_names_at_most_two_conversations(
    store: MemoryStore, turns: list[RetrievedTurn], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_MEMORY_RECALL_MODE", "pointer")
    more = turns + [RetrievedTurn(row_id="999", session_id="third", role="user", content="x")]
    [note] = _context(store, more, "anything").pointers
    assert "third" not in note


def test_pointer_backstop_pushes_two_turns_when_the_message_refers_back(
    store: MemoryStore, turns: list[RetrievedTurn], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_MEMORY_RECALL_MODE", "pointer")
    context = _context(store, turns, "what did we decide about the insurance?")
    assert context.pointers == ()
    assert len(context.related_turns) == 2 and context.recall_backstop


def test_pointer_with_nothing_related_adds_nothing(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_MEMORY_RECALL_MODE", "pointer")
    context = _context(store, [], "plan my day")
    assert context.pointers == () and context.related_turns == () and not context.recall_backstop


def test_pointer_still_fails_closed_on_an_unreadable_removal_ledger(
    store: MemoryStore, turns: list[RetrievedTurn], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_MEMORY_RECALL_MODE", "pointer")

    def boom() -> set[str]:
        raise OSError("locked")

    monkeypatch.setattr(store, "removed_session_ids", boom)
    context = _context(store, turns, "insurance")
    assert context.pointers == () and context.related_turns == ()


def test_pointer_never_names_a_removed_conversation(
    store: MemoryStore, turns: list[RetrievedTurn], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_MEMORY_RECALL_MODE", "pointer")
    monkeypatch.setattr(store, "removed_session_ids", lambda: {"car-sess"})
    [note] = _context(store, turns, "insurance").pointers
    assert "car-sess" not in note and "dentist-sess" in note


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("what did we decide about the car insurance?", True),
        ("Remind me what you said about the dentist", True),
        ("do you remember my wife's name", True),
        ("book an earlier flight", False),
        ("any mail from the bank earlier today?", False),
        ("what's on my calendar last week", False),
        ("plan my day", False),
    ],
)
def test_refers_back_uses_the_yaml_phrases(message: str, expected: bool) -> None:
    retriever_module._REFERS_BACK = None  # re-read config/memory/recall.yaml
    assert refers_back(message) is expected
