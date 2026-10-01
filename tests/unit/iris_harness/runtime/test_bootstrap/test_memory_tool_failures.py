"""The memory tools still hand the agent their error text, and now log it (review 2026-09-26).

Before, a broken memory store reached the agent as "<tool> failed: ..." and left no
trace for anyone else. The text the agent reads is unchanged; a WARNING naming the tool
and the exception type (never the query, key or value) is added.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from iris_harness.memory.store import MemoryStore
from iris_harness.runtime.react_tools import builtin_react_tools

_LOGGER = "iris_harness.runtime.react_tools"


def _boom(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise sqlite3.OperationalError("database is locked")


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _call(store: MemoryStore, name: str, args: dict[str, object], *, wiki: Any = None) -> str:
    specs = builtin_react_tools(semantic_index=None, wiki=wiki, repo_root=None, memory_store=store)
    tool = {s.name: s for s in specs}[name]
    return str(tool.call(args))  # type: ignore[attr-defined]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == _LOGGER and r.levelno == logging.WARNING
    ]


@pytest.mark.parametrize(
    ("tool", "store_method", "args"),
    [
        ("memory_search", "search_summaries", {"query": "secret plan", "scope": "sessions"}),
        ("recall_conversation", "search_turns", {"query": "secret plan"}),
        ("memory_forget", "delete_user_fact", {"key": "location"}),
        ("memory_correct", "correct_user_fact", {"key": "location", "value": "Paris"}),
        ("memory_restore", "restore_user_fact", {"key": "location"}),
    ],
)
def test_a_store_failure_returns_the_same_text_and_logs(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tool: str,
    store_method: str,
    args: dict[str, object],
) -> None:
    monkeypatch.setattr(store, store_method, _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = _call(store, tool, args)

    assert out == f"{tool} failed: database is locked"
    [message] = _warnings(caplog)
    assert message == f"react tool {tool} failed (OperationalError)"
    assert "secret plan" not in message and "Paris" not in message


def test_a_wiki_failure_returns_the_same_text_and_logs(
    store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    wiki = SimpleNamespace(query=_boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = _call(store, "wiki_search", {"query": "budget"}, wiki=wiki)

    assert out == "wiki_search failed: database is locked"
    assert _warnings(caplog) == ["react tool wiki_search failed (OperationalError)"]


def test_a_failed_summary_fallback_still_answers_and_logs(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(store, "search_summaries", _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = _call(store, "recall_conversation", {"query": "budget"})

    assert out == "Nothing stored matches that."
    assert _warnings(caplog) == [
        "react tool recall_conversation summary fallback failed (OperationalError)"
    ]


def test_a_non_memory_tool_failure_also_logs(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The action and learning tools (#668 follow-up) used to return "X failed" silently."""
    from iris_harness.runtime import agent_console

    monkeypatch.setattr(agent_console, "render_agent_detail", _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = _call(store, "agents", {"name": "finance"})

    assert out == "agents failed: database is locked"
    assert _warnings(caplog) == ["react tool agents failed (OperationalError)"]
