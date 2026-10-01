"""The memory Map degrades on a broken store, and now says so (review 2026-09-26).

Each reader keeps drawing what it can — the result is unchanged — but a store failure is
logged at WARNING rather than swallowed (or kept at DEBUG), so an empty Map is never
mistaken for an empty memory. A store without the removal ledger (AttributeError) is
still the quiet "nothing removed" case.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from iris_harness.memory import graph, removal
from iris_harness.memory.graph import build_memory_graph
from iris_harness.memory.store import MemoryStore

_GRAPH = "iris_harness.memory.graph"
_REMOVAL = "iris_harness.memory.removal"


def _boom(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise sqlite3.OperationalError("database is locked")


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    s.save_conversation_turns("dc43912781ed", [("user", "hello"), ("assistant", "hi")])
    s.save_conversation_summary("dc43912781ed", "Goal: plan the week")
    return s


def _warned(caplog: pytest.LogCaptureFixture, logger: str, text: str) -> bool:
    return any(
        r.name == logger and r.levelno == logging.WARNING and text in r.getMessage()
        for r in caplog.records
    )


def _session_ids(store: Any) -> set[str]:
    return {sid for sid, *_ in graph._sessions_with_summaries(store)}


def test_an_unreadable_removal_ledger_draws_no_sessions_and_logs(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Fails closed (owner decision 2026-09-27): it used to draw every session."""
    monkeypatch.setattr(store, "removed_session_ids", _boom)

    with caplog.at_level(logging.WARNING, logger=_GRAPH):
        assert graph._removed_sessions(store) is None
        assert _session_ids(store) == set()

    assert _warned(caplog, _GRAPH, "removed-session ledger unreadable (OperationalError)")
    assert _warned(caplog, _GRAPH, "drawing no sessions")


def test_a_readable_ledger_draws_the_sessions_not_removed(store: MemoryStore) -> None:
    assert graph._removed_sessions(store) == set()
    assert _session_ids(store) == {"dc43912781ed"}


def test_a_store_without_the_ledger_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=_GRAPH):
        assert graph._removed_sessions(SimpleNamespace()) == set()

    assert not caplog.records


def test_unreadable_session_activity_draws_no_sessions_and_logs(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(store, "session_activity", _boom)

    with caplog.at_level(logging.WARNING, logger=_GRAPH):
        assert _session_ids(store) == set()

    assert _warned(caplog, _GRAPH, "session activity unreadable")


def test_an_unreadable_summary_leaves_that_session_off_and_logs(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(store, "load_conversation_summary", _boom)

    with caplog.at_level(logging.WARNING, logger=_GRAPH):
        assert _session_ids(store) == set()

    assert _warned(caplog, _GRAPH, "summary of session dc43912781ed unreadable")


def test_unreadable_statements_still_draw_the_rest_and_log(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(store, "memory_graph", _boom)

    with caplog.at_level(logging.WARNING, logger=_GRAPH):
        result = build_memory_graph(store)

    assert result["nodes"]  # the owner (and the session) are still drawn
    assert _warned(caplog, _GRAPH, "statements unavailable")


def test_an_unreadable_suppression_ledger_suppresses_nothing_and_logs(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(removal, "suppressed_keys", _boom)

    with caplog.at_level(logging.WARNING, logger=_GRAPH):
        build_memory_graph(store)

    assert _warned(caplog, _GRAPH, "removal ledger unreadable (OperationalError)")


def test_unreadable_behaviors_and_patterns_are_logged(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from iris_harness.memory import identity

    monkeypatch.setattr(identity, "list_behaviors", _boom)
    monkeypatch.setattr(identity, "list_episodic_patterns", _boom)

    with caplog.at_level(logging.WARNING, logger=_GRAPH):
        assert list(graph._read_behaviors(store, None)) == []  # type: ignore[arg-type]
        assert list(graph._read_patterns(store, None)) == []  # type: ignore[arg-type]

    assert _warned(caplog, _GRAPH, "behaviors unavailable")
    assert _warned(caplog, _GRAPH, "patterns unavailable")


def test_unreadable_removed_entities_still_suppress_names_and_log(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(store, "memory_graph", _boom)

    with caplog.at_level(logging.WARNING, logger=_REMOVAL):
        keys = removal.suppressed_keys(store)

    assert keys == set(store.suppressed_name_keys())
    assert _warned(caplog, _REMOVAL, "removed entities unreadable")
