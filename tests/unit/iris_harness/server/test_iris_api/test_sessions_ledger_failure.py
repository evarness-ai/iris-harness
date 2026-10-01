"""An unreadable removal ledger refuses the session list (503) and logs it.

It used to list every session, removed ones included (review 2026-09-26, #668 follow-up
2). The owner decided (2026-09-27) that it fails closed like the retriever and the Map:
a conversation the owner removed must not reappear because a read failed.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.memory.store import MemoryStore
from iris_harness.server.iris_api.main import create_app

_LOGGER = "iris_harness.server.iris_api.session_routes"


def _boom(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise sqlite3.OperationalError("database is locked")


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


@pytest.fixture
def client(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from iris_harness.foundation.observability import trace_builder

    def fake_list(*, limit: int, skip: Any) -> list[dict[str, Any]]:
        return [{"session_id": s} for s in ("mine", "cascade") if not skip(s)]

    monkeypatch.setattr(trace_builder, "list_sessions", fake_list)
    app = create_app(
        runtime=SimpleNamespace(memory_store=store, semantic_index=None),
        auto_start_runtime=False,
    )
    with TestClient(app, headers=auth_headers()) as c:
        yield c


def test_an_unreadable_ledger_refuses_the_list_and_logs(
    client: TestClient,
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(store, "removed_session_ids", _boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        response = client.get("/api/sessions")

    assert response.status_code == 503
    assert "removed conversations cannot be read" in response.json()["detail"]
    assert "mine" not in response.text and "cascade" not in response.text
    assert any(
        r.name == _LOGGER
        and r.levelno == logging.WARNING
        and "removed-session ledger unreadable (OperationalError)" in r.getMessage()
        and "refusing to list" in r.getMessage()
        for r in caplog.records
    )


def test_a_readable_ledger_still_hides_removed_sessions(
    client: TestClient, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "removed_session_ids", lambda: {"cascade"})
    listed = [s["session_id"] for s in client.get("/api/sessions").json()]
    assert listed == ["mine"]
