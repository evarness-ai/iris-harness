"""Tests for the fact-contradictions endpoints (review + write-gated ack)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.memory.store import MemoryStore, UserFact
from iris_harness.server.iris_api.main import create_app


def _seed_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    now = datetime.now(UTC)
    store.upsert_user_fact(_fact("city", "Berlin", 0.9, now))
    store.upsert_user_fact(_fact("city", "New York", 0.9, now))  # superseding conflict
    return store


def _fact(key: str, value: str, conf: float, now: datetime) -> UserFact:
    return UserFact(
        key=key,
        value=value,
        confidence=conf,
        source="test",
        first_seen=now,
        last_confirmed=now,
        times_confirmed=1,
    )


def test_contradictions_endpoint_lists(tmp_path: Path) -> None:
    rt = SimpleNamespace(memory_store=_seed_store(tmp_path))
    with TestClient(create_app(runtime=rt, auto_start_runtime=False), headers=auth_headers()) as c:
        body = c.get("/memory/contradictions").json()
    assert body["count"] == 1
    row = body["contradictions"][0]
    assert row["key"] == "city"
    assert row["incoming_value"] == "New York"
    assert row["resolution"] == "superseded"
    assert row["seen_count"] == 1


def test_ack_endpoint_clears_from_default_queue(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    rt = SimpleNamespace(memory_store=store)
    with TestClient(create_app(runtime=rt, auto_start_runtime=False), headers=auth_headers()) as c:
        cid = c.get("/memory/contradictions").json()["contradictions"][0]["id"]
        r = c.post("/memory/contradictions/ack", json={"ids": [cid]})
        assert r.status_code == 200
        assert r.json() == {"requested": 1, "acknowledged": 1}
        assert c.get("/memory/contradictions").json()["count"] == 0
        assert c.get("/memory/contradictions?include_acknowledged=true").json()["count"] == 1


def test_ack_is_write_gated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)  # override conftest
    rt = SimpleNamespace(memory_store=_seed_store(tmp_path))
    with TestClient(create_app(runtime=rt, auto_start_runtime=False), headers=auth_headers()) as c:
        assert c.get("/memory/contradictions").status_code == 200  # read open
        assert c.post("/memory/contradictions/ack", json={"ids": [1]}).status_code == 403  # gated
