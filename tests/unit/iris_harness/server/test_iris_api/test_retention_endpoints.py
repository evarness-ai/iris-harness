"""The retention API: housekeeping runs, forget-about-X, purge test sessions.

Every capability needs an API, not just a UI — the Memory screens land on these.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.memory.retention import RetentionService
from iris_harness.memory.store import MemoryStore
from iris_harness.server.iris_api.main import create_app


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


@pytest.fixture
def client(store: MemoryStore, tmp_path: Path) -> Iterator[TestClient]:
    runtime = SimpleNamespace(
        memory_store=store,
        semantic_index=None,
        retention=RetentionService(store, None, logs_dir=tmp_path / "logs"),
    )
    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as c:
        yield c


def test_housekeeping_run_reports_what_it_did(client: TestClient) -> None:
    body = client.post("/memory/housekeeping/run").json()

    assert body["dry_run"] is False
    assert "sessions_cooled" in body and "turns_deleted" in body


def test_runs_are_recorded(client: TestClient) -> None:
    client.post("/memory/housekeeping/run", params={"dry_run": True})

    runs = client.get("/memory/housekeeping").json()["runs"]

    assert len(runs) == 1 and runs[0]["dry_run"] is True


def test_forget_previews_before_deleting(client: TestClient, store: MemoryStore) -> None:
    store.save_conversation_turns("s1", [("user", "the Northwind statement"), ("assistant", "ok")])

    body = client.post("/memory/forget", params={"needle": "Northwind"}).json()

    assert body["deleted"] is False
    assert body["preview"]["total"] == 1
    assert len(store.fetch_turn_ids("s1")) == 2


def test_forget_with_confirm_deletes(client: TestClient, store: MemoryStore) -> None:
    store.save_conversation_turns("s1", [("user", "the Northwind statement"), ("assistant", "ok")])

    body = client.post("/memory/forget", params={"needle": "Northwind", "confirm": True}).json()

    assert body["deleted"] is True and body["turns"] == 1
    assert len(store.fetch_turn_ids("s1")) == 1  # only the matching turn went


def test_forget_rejects_an_empty_needle(client: TestClient) -> None:
    assert client.post("/memory/forget", params={"needle": "  "}).status_code == 400


def test_purge_previews_test_sessions(client: TestClient, store: MemoryStore) -> None:
    store.save_conversation_turns("playground-x", [("user", "q"), ("assistant", "a")])
    store.save_conversation_turns("default", [("user", "q"), ("assistant", "a")])

    body = client.post("/memory/sessions/purge").json()

    assert body["deleted"] is False
    assert [s["session_id"] for s in body["sessions"]] == ["playground-x"]


def test_purge_with_confirm_removes_only_those(client: TestClient, store: MemoryStore) -> None:
    store.save_conversation_turns("playground-x", [("user", "q"), ("assistant", "a")])
    store.save_conversation_turns("default", [("user", "q"), ("assistant", "a")])

    body = client.post("/memory/sessions/purge", params={"confirm": True}).json()

    assert body["deleted"] is True and body["turns"] == 2
    assert store.fetch_turn_ids("playground-x") == []
    assert len(store.fetch_turn_ids("default")) == 2


def test_writes_are_gated(store: MemoryStore, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "0")
    runtime = SimpleNamespace(
        memory_store=store,
        semantic_index=None,
        retention=RetentionService(store, None, logs_dir=tmp_path / "logs"),
    )
    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as client:
        assert client.post("/memory/forget", params={"needle": "x"}).status_code == 403
        assert client.post("/memory/sessions/purge").status_code == 403
        assert client.get("/memory/housekeeping").status_code == 200  # reads stay open


# ── the session-log archive (owner decision 2026-09-19) ──────────────────────


@pytest.fixture
def archived_client(store: MemoryStore, tmp_path: Path) -> Iterator[TestClient]:
    import os
    import time

    from cryptography.fernet import Fernet

    from iris_harness.memory.log_archive import LogArchive

    key = Fernet.generate_key()
    archive = LogArchive(tmp_path / "archive", key=lambda: key)
    src = tmp_path / "src"
    src.mkdir()
    path = src / "session-web-abc.jsonl"
    path.write_text('{"q": "hi"}\n')
    stamp = time.mktime((2026, 7, 15, 12, 0, 0, 0, 0, -1))
    os.utime(path, (stamp, stamp))
    archive.add([path])
    runtime = SimpleNamespace(
        memory_store=store,
        semantic_index=None,
        retention=RetentionService(store, None, logs_dir=tmp_path / "logs", archive=archive),
    )
    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as c:
        yield c


def test_the_archive_is_listed_and_a_session_restored(
    archived_client: TestClient, tmp_path: Path
) -> None:
    listed = archived_client.get("/logs/archive").json()
    assert listed["enabled"] is True
    assert [(m["month"], m["members"]) for m in listed["months"]] == [
        ("2026-07", ["session-web-abc.jsonl"])
    ]

    restored = archived_client.post("/logs/archive/restore", params={"session": "web-abc"})
    assert restored.status_code == 200
    assert restored.json() == {"restored": ["session-web-abc.jsonl"], "count": 1}
    assert (tmp_path / "logs" / "session-web-abc.jsonl").read_text() == '{"q": "hi"}\n'


def test_restore_needs_exactly_one_target(archived_client: TestClient) -> None:
    assert archived_client.post("/logs/archive/restore").status_code == 400
    both = archived_client.post(
        "/logs/archive/restore", params={"session": "x", "month": "2026-07"}
    )
    assert both.status_code == 400
    bad = archived_client.post("/logs/archive/restore", params={"month": "July"})
    assert bad.status_code == 400


def test_restore_is_write_gated(archived_client: TestClient, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    assert (
        archived_client.post("/logs/archive/restore", params={"month": "2026-07"}).status_code
        == 403
    )
    assert archived_client.get("/logs/archive").status_code == 200


def test_no_archive_configured_says_so(client: TestClient) -> None:
    assert client.get("/logs/archive").json()["enabled"] is False
    assert client.post("/logs/archive/restore", params={"session": "x"}).status_code == 503
