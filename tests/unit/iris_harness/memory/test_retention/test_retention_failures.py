"""Housekeeping and the test-session review degrade on a broken store, and log it.

Review 2026-09-26: a failed housekeeping step was only a string on the report, and an
unreadable removal ledger silently re-offered removed sessions. Both keep their result
and now log a WARNING naming the step and the exception type.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from iris_harness.memory import retention
from iris_harness.memory.retention import RetentionService, flag_test_sessions
from iris_harness.memory.store import MemoryStore

_LOGGER = "iris_harness.memory.retention"


def _boom(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise sqlite3.OperationalError("database is locked")


@pytest.fixture(autouse=True)
def _fresh_config() -> Iterator[None]:
    retention.reset_config_cache()
    yield
    retention.reset_config_cache()


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    s.save_conversation_turns("cascade", [("user", "t0"), ("assistant", "t1")])
    return s


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == _LOGGER and r.levelno == logging.WARNING
    ]


def test_an_unreadable_ledger_still_flags_and_logs(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(store, "removed_session_ids", _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        flagged = {f.session_id for f in flag_test_sessions(store)}

    assert flagged == {"cascade"}
    assert any("ledger unreadable (OperationalError)" in m for m in _warnings(caplog))


def test_a_store_without_the_ledger_flags_quietly(
    store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    bare = SimpleNamespace(
        session_activity=store.session_activity,
        load_conversation_summary=store.load_conversation_summary,
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        flagged = {f.session_id for f in flag_test_sessions(bare)}

    assert flagged == {"cascade"}
    assert _warnings(caplog) == []


def test_a_failed_housekeeping_step_is_reported_and_logged(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    service = RetentionService(store)
    monkeypatch.setattr(service, "_sweep_orphan_vectors", _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        report = service.run(dry_run=True)

    assert report.errors == ["sweep: database is locked"]
    assert any("housekeeping step sweep failed (OperationalError)" in m for m in _warnings(caplog))
