"""Tests for the fact-history retention endpoints (review + write-gated prune)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.foundation.persistence import sqlite_conn
from iris_harness.memory.store import MemoryStore, UserFact
from iris_harness.server.iris_api.main import create_app


def _forgotten(db: Path, key: str, value: str, days_ago: int) -> None:
    """A fact captured and forgotten ``days_ago`` — two history events, no longer holding.

    Since memris plan PR 2c-ii the history IS the statement chain, so history is made
    by using the store, then aged, rather than by inserting log rows.
    """
    store = MemoryStore(db_path=db)
    now = datetime.now(UTC)
    store.upsert_user_fact(UserFact(key, value, 0.9, "test", now, now, 1, True))
    store.delete_user_fact(key)
    ts = (now - timedelta(days=days_ago)).isoformat()
    with sqlite_conn(db) as conn:
        conn.execute(
            "UPDATE memris_statements SET recorded_at = ?, retracted_at = ? "
            "WHERE reason = 'forgot' AND recorded_at > ?",
            (ts, ts, (now - timedelta(seconds=60)).isoformat()),
        )


def _seed_store(tmp_path: Path) -> MemoryStore:
    db = tmp_path / "memory.db"
    for key, value, days in [
        ("city", "Leeds", 200),
        ("email", "a@example.org", 400),
        ("hobby", "chess", 5),
    ]:
        _forgotten(db, key, value, days)
    return MemoryStore(db_path=db)


def test_retention_endpoint_lists_candidates(tmp_path: Path) -> None:
    rt = SimpleNamespace(memory_store=_seed_store(tmp_path))
    with TestClient(create_app(runtime=rt, auto_start_runtime=False), headers=auth_headers()) as c:
        body = c.get("/memory/history/retention?older_than_days=180").json()
    assert body["total"] == 6  # three forgotten facts: a capture and a forget each
    assert body["count"] == 4
    assert [e["key"] for e in body["entries"]] == ["email", "email", "city", "city"]  # oldest first


def test_prune_endpoint_removes_selected(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    rt = SimpleNamespace(memory_store=store)
    with TestClient(create_app(runtime=rt, auto_start_runtime=False), headers=auth_headers()) as c:
        cands = c.get("/memory/history/retention?older_than_days=180").json()["entries"]
        ids = [e["id"] for e in cands]
        r = c.post("/memory/history/prune", json={"entry_ids": ids})
    assert r.status_code == 200
    assert r.json() == {"requested": 4, "removed": 2}  # two statements, two events each
    assert store.count_fact_history() == 2  # only the recent one remains


def test_prune_is_write_gated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)  # override conftest
    rt = SimpleNamespace(memory_store=_seed_store(tmp_path))
    with TestClient(create_app(runtime=rt, auto_start_runtime=False), headers=auth_headers()) as c:
        # Read still works...
        assert c.get("/memory/history/retention").status_code == 200
        # ...but the prune is blocked by the write gate.
        r = c.post("/memory/history/prune", json={"entry_ids": ["st_any"]})
    assert r.status_code == 403
