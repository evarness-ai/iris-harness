"""A broken memory store must not read as "no memory" (pre-open-source review, 2026-09-26).

Each recall step that swallows a store failure keeps its degraded result — the turn goes
on — but now says so in the log. The one change of result is the removal ledger: when it
cannot be read, no cross-session turn is recalled, because the retriever can no longer
tell which sessions the owner removed (ADR-0119).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from iris_harness.memory import identity
from iris_harness.memory.retriever import MemoryRetriever
from iris_harness.memory.semantic_index import RetrievedTurn
from iris_harness.memory.store import MemoryStore, UserFact

_LOGGER = "iris_harness.memory.retriever"


def _boom(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise sqlite3.OperationalError("database is locked")


def _fact(key: str, value: str) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=0.9,
        source="test",
        first_seen=now,
        last_confirmed=now,
        times_confirmed=1,
        confirmed=True,
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    s.upsert_user_fact(_fact("name", "robin"))
    return s


def _index(turns: list[RetrievedTurn], fact_keys: list[str] | None = None) -> Any:
    return SimpleNamespace(
        is_ready=True,
        query_facts=lambda q, n: list(fact_keys or []),
        query_turns_detailed=lambda q, **kw: list(turns),
        query_episodic=lambda q, n: [],
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _LOGGER and r.levelno == logging.WARNING]


def test_unreadable_facts_recall_none_and_log(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(store, "fetch_all_user_facts", _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        context = MemoryRetriever(store=store, index=None).build_context(query="who am I")

    assert context.user_facts == ()
    [record] = _warnings(caplog)
    assert "fetch confirmed facts" in record.getMessage()
    assert "OperationalError" in record.getMessage()


def test_an_unreadable_fact_is_skipped_and_logged(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(store, "fetch_user_fact", _boom)
    retriever = MemoryRetriever(store=store, index=_index([], fact_keys=["name"]))

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        context = retriever.build_context(query="who am I", session_id="now")

    # The keyword backfill still finds it: only the per-key read failed.
    assert [f.key for f in context.user_facts] == ["name"]
    assert any("fetch a fact" in r.getMessage() for r in _warnings(caplog))


def test_unreadable_signals_are_empty_and_logged(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(store, "fetch_learning_signals", _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        signals = MemoryRetriever(store=store, index=None)._safe_fetch_signals()

    assert signals == []
    assert any("fetch learning signals" in r.getMessage() for r in _warnings(caplog))


def test_an_unreadable_removal_ledger_recalls_no_cross_session_turn(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    turns = [RetrievedTurn("1", "elsewhere", "user", "budget for June")]
    monkeypatch.setattr(store, "removed_session_ids", _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        context = MemoryRetriever(store=store, index=_index(turns)).build_context(
            query="budget", session_id="now"
        )

    # Fail closed: the retriever cannot tell a removed session from a kept one.
    assert context.related_turns == ()
    assert context.reused_turn_refs == ()
    assert any("read removed sessions" in r.getMessage() for r in _warnings(caplog))


def test_a_readable_removal_ledger_still_recalls(store: MemoryStore) -> None:
    turns = [RetrievedTurn("1", "elsewhere", "user", "budget for June")]

    context = MemoryRetriever(store=store, index=_index(turns)).build_context(
        query="budget", session_id="now"
    )

    assert context.related_turns == ("user: budget for June",)


def test_a_failed_identity_layer_keeps_the_context_and_logs(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(identity, "match_behavior", _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        context = MemoryRetriever(store=store, index=None).build_context(query="who am I")

    assert context.soul is None
    assert context.user_profile is None
    assert [f.key for f in context.user_facts] == ["name"]
    assert any("attach identity context" in r.getMessage() for r in _warnings(caplog))


def test_the_log_carries_no_query_or_memory_content(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(store, "fetch_all_user_facts", _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        MemoryRetriever(store=store, index=None).build_context(query="my secret plan")

    for record in _warnings(caplog):
        assert "my secret plan" not in record.getMessage()
        assert "robin" not in record.getMessage()
