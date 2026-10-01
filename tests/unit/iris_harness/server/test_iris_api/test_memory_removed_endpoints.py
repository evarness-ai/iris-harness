"""The Removed API (ADR-0119): the contract the Memory page's Map and Removed tab use."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.memory.store import MemoryStore, UserFact
from iris_harness.server.iris_api.main import create_app

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _client(store: MemoryStore) -> Iterator[TestClient]:
    runtime = SimpleNamespace(memory_store=store, semantic_index=None)
    app = create_app(runtime=runtime, auto_start_runtime=False)
    with TestClient(app, headers=auth_headers()) as c:
        yield c


@pytest.fixture
def client(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    yield from _client(store)


def _fact(key: str, value: str) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=1.0,
        source="test",
        first_seen=now,
        last_confirmed=now,
        confirmed=True,
    )


def _node(client: TestClient, label: str) -> dict[str, Any] | None:
    nodes = client.get("/memory/graph").json()["nodes"]
    return next((n for n in nodes if n["label"] == label), None)


def test_remove_an_entity_then_restore_it_through_the_api(
    client: TestClient, store: MemoryStore
) -> None:
    store.upsert_user_fact(_fact("bank", "Northwind Bank"))
    node = _node(client, "Northwind Bank")
    assert node is not None and node["previously_removed"] is False
    target = node["ref"]
    assert target["kind"] == "entity"

    effects = client.post("/memory/removed/preview", json={"targets": [target]}).json()
    assert effects["effects"][0]["target"] == target
    assert effects["effects"][0]["lines"][0] == "forgets 1 fact: bank: Northwind Bank"

    [item] = client.post("/memory/removed", json={"targets": [target]}).json()["items"]
    assert set(item) == {"id", "kind", "label", "removed_at", "cascade", "permanent"}
    assert item["cascade"][0]["text"] == "bank: Northwind Bank" and item["permanent"] is False
    assert _node(client, "Northwind Bank") is None
    assert client.get("/memory/facts").json()["count"] == 0
    assert client.get("/memory/removed").json()["items"] == [item]

    restored = client.post(f"/memory/removed/{item['id']}/restore").json()["restored"]
    assert restored["id"] == item["id"]
    assert _node(client, "Northwind Bank") is not None
    assert client.get("/memory/removed").json()["items"] == []


def test_a_removed_session_leaves_the_chat_lists(client: TestClient, store: MemoryStore) -> None:
    store.save_conversation_turns("cascade", [("user", "q"), ("assistant", "a")])
    store.save_conversation_summary("cascade", "Goal: test\nReferenced: Budget Review")
    target = {"kind": "session", "id": "cascade"}

    client.post("/memory/removed", json={"targets": [target]})

    sessions = client.get("/memory/sessions").json()["sessions"]
    assert all(s["session_id"] != "cascade" for s in sessions)
    assert _node(client, "cascade") is None


def test_delete_needs_the_word_and_reports_refusals(client: TestClient, store: MemoryStore) -> None:
    store.save_conversation_turns("s1", [("user", "q"), ("assistant", "a")])
    store.save_conversation_summary("s1", "Goal: test\nReferenced: Walgreens")
    [item] = client.post(
        "/memory/removed", json={"targets": [{"kind": "name", "id": "Walgreens"}]}
    ).json()["items"]

    no_word = client.post("/memory/removed/delete", json={"ids": [item["id"]], "confirm": ""})
    refused = client.post(
        "/memory/removed/delete", json={"ids": [item["id"]], "confirm": "delete"}
    ).json()

    assert no_word.status_code == 422
    assert refused["deleted"] == [] and refused["refused"][0]["id"] == item["id"]


def test_errors(client: TestClient) -> None:
    assert client.post("/memory/removed", json={"targets": []}).status_code == 400
    bad_kind = client.post("/memory/removed", json={"targets": [{"kind": "x", "id": "y"}]})
    assert bad_kind.status_code == 422
    missing = client.post("/memory/removed", json={"targets": [{"kind": "session", "id": "no"}]})
    assert missing.status_code == 404
    assert client.post("/memory/removed/rm_nope/restore").status_code == 404


def test_writes_are_gated_but_a_preview_is_not(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    store.save_conversation_turns("s1", [("user", "q"), ("assistant", "a")])
    store.save_conversation_summary("s1", "Goal: test\nReferenced: Walgreens")
    body = {"targets": [{"kind": "name", "id": "Walgreens"}]}
    for client in _client(store):
        assert client.post("/memory/removed/preview", json=body).status_code == 200
        assert client.post("/memory/removed", json=body).status_code == 403
        assert client.post("/memory/removed/x/restore").status_code == 403
        assert (
            client.post("/memory/removed/delete", json={"ids": [], "confirm": "delete"}).status_code
            == 403
        )


def test_a_removed_session_leaves_the_chat_history(
    client: TestClient, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.foundation.observability import trace_builder

    def fake_list(*, limit: int, skip: Any) -> list[dict[str, Any]]:
        ids = ["mine", "cascade", "playground-1"]
        return [{"session_id": s} for s in ids if skip is None or not skip(s)]

    monkeypatch.setattr(trace_builder, "list_sessions", fake_list)
    store.save_conversation_turns("cascade", [("user", "q"), ("assistant", "a")])
    client.post("/memory/removed", json={"targets": [{"kind": "session", "id": "cascade"}]})

    listed = [s["session_id"] for s in client.get("/api/sessions").json()]
    with_runs = client.get("/api/sessions", params={"include_runs": True}).json()

    assert listed == ["mine"]
    assert [s["session_id"] for s in with_runs] == ["mine", "playground-1"]
