"""The review-queue API: /memory/facts, /memory/review, approve / reject / confirm.

Every capability needs an API, not just a UI — the web screens land on top of these.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.memory.store import MemoryStore, UserFact
from iris_harness.server.iris_api.main import create_app


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


@pytest.fixture
def client(store: MemoryStore) -> Iterator[TestClient]:
    runtime = SimpleNamespace(memory_store=store, semantic_index=None)
    app = create_app(runtime=runtime, auto_start_runtime=False)
    with TestClient(app, headers=auth_headers()) as c:
        yield c


def _fact(key: str, value: str, confirmed: bool) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=0.9,
        source="conversation:llm",
        first_seen=now,
        last_confirmed=now,
        confirmed=confirmed,
    )


def test_facts_endpoint_returns_confirmed_by_default(
    client: TestClient, store: MemoryStore
) -> None:
    store.upsert_user_fact(_fact("country", "India", True))
    store.upsert_user_fact(_fact("name", "ollama", False))

    confirmed = client.get("/memory/facts").json()
    everything = client.get("/memory/facts", params={"confirmed_only": False}).json()

    assert [f["key"] for f in confirmed["facts"]] == ["country"]
    assert {f["key"] for f in everything["facts"]} == {"country", "name"}


def test_review_is_one_queue_for_proposals_and_unconfirmed_facts(
    client: TestClient, store: MemoryStore
) -> None:
    """Since memris PR 2c a fact never reviewed IS a proposal: one queue, not two."""
    store.upsert_user_fact(_fact("name", "ollama", False))
    store.add_fact_proposal(key="city", value="Springfield", confidence=0.8, source="llm")

    body = client.get("/memory/review").json()

    assert body["pending_count"] == 2
    assert {p["key"] for p in body["proposals"]} == {"city", "name"}
    assert all(p["kind"] == "new" for p in body["proposals"])
    assert "unconfirmed_facts" not in body


def test_a_conflicting_proposal_is_flagged_as_changed(
    client: TestClient, store: MemoryStore
) -> None:
    store.upsert_user_fact(_fact("city", "Chennai", True))
    store.add_fact_proposal(key="city", value="Springfield", confidence=0.8, source="llm")

    proposal = client.get("/memory/review").json()["proposals"][0]

    assert proposal["kind"] == "changed"
    assert proposal["current_value"] == "Chennai"


def test_approve_confirms_the_fact(client: TestClient, store: MemoryStore) -> None:
    pid = store.add_fact_proposal(key="city", value="Springfield", confidence=0.8, source="llm")

    resp = client.post(f"/memory/review/{pid}/approve")

    assert resp.status_code == 200
    stored = store.fetch_user_fact("city")
    assert stored is not None and stored.confirmed and stored.value == "Springfield"
    assert client.get("/memory/review").json()["pending_count"] == 0


def test_reject_writes_nothing(client: TestClient, store: MemoryStore) -> None:
    pid = store.add_fact_proposal(key="city", value="Springfield", confidence=0.8, source="llm")

    assert client.post(f"/memory/review/{pid}/reject").status_code == 200
    assert store.fetch_all_user_facts() == []


def test_approving_twice_is_a_404(client: TestClient, store: MemoryStore) -> None:
    pid = store.add_fact_proposal(key="city", value="Springfield", confidence=0.8, source="llm")
    client.post(f"/memory/review/{pid}/approve")

    assert client.post(f"/memory/review/{pid}/approve").status_code == 404


def test_confirm_clears_a_legacy_fact(client: TestClient, store: MemoryStore) -> None:
    store.upsert_user_fact(_fact("country", "India", False))

    assert client.post("/memory/facts/country/confirm").status_code == 200

    stored = store.fetch_user_fact("country")
    assert stored is not None and stored.confirmed


def test_confirm_unknown_key_is_a_404(client: TestClient) -> None:
    assert client.post("/memory/facts/nope/confirm").status_code == 404


def test_expire_clears_stale_proposals(client: TestClient, store: MemoryStore) -> None:
    store.add_fact_proposal(key="city", value="Springfield", confidence=0.8, source="llm")

    body = client.post("/memory/review/expire", params={"older_than_days": 0}).json()

    assert body["expired"] == 1


def test_writes_are_gated(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """The approve/reject routes sit behind IRIS_WEBUI_ALLOW_WRITES like other controls."""
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "0")
    pid = store.add_fact_proposal(key="city", value="Springfield", confidence=0.8, source="llm")
    runtime = SimpleNamespace(memory_store=store, semantic_index=None)
    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as client:
        assert client.post(f"/memory/review/{pid}/approve").status_code == 403
        assert client.get("/memory/review").status_code == 200  # reads stay open


# --- look-alike entities (memris PR 4b): every capability needs an API ------------


def _lookalike(store: MemoryStore) -> tuple[str, str, str]:
    graph = store.memory_graph()
    old = graph.add_entity("mem:Organization", "Acme")
    new = graph.add_entity("mem:Organization", "Acmee")
    candidate = graph.note_candidate(old.id, new.id, score=0.9, episode="s1")
    assert candidate is not None
    return old.id, new.id, candidate.id


def test_entities_lists_open_look_alikes_with_their_names(
    client: TestClient, store: MemoryStore
) -> None:
    _, _, cid = _lookalike(store)
    body = client.get("/memory/entities").json()
    assert body["count"] == 1
    [row] = body["decisions"]
    assert (row["id"], row["decision"], row["evidence_count"]) == (cid, "candidate", 1)
    assert {row["a"]["label"], row["b"]["label"]} == {"Acme", "Acmee"}
    assert row["asked_at"] is None


def test_same_merges_and_unmerge_undoes_it(client: TestClient, store: MemoryStore) -> None:
    old, new, cid = _lookalike(store)
    resp = client.post(f"/memory/entities/{cid}/same")
    assert resp.status_code == 200 and resp.json()["decision"] == "same"
    graph = store.memory_graph()
    assert graph.canonical_id(new) == old
    assert client.get("/memory/entities").json()["count"] == 0  # no longer open
    assert client.get("/memory/entities", params={"decision": "same"}).json()["count"] == 1
    assert client.post(f"/memory/entities/{cid}/unmerge").status_code == 200
    assert graph.canonical_id(new) == new


def test_distinct_keeps_apart_and_a_second_decision_conflicts(
    client: TestClient, store: MemoryStore
) -> None:
    old, new, cid = _lookalike(store)
    assert client.post(f"/memory/entities/{cid}/distinct").json()["decision"] == "distinct"
    assert store.memory_graph().are_distinct(old, new)
    assert client.post(f"/memory/entities/{cid}/same").status_code == 409


def test_a_fact_about_someone_one_hop_away_says_whom(
    client: TestClient, store: MemoryStore
) -> None:
    """memris PR 3b: "my wife Petra works at Infosys" — reviewed as Petra's, approved as hers."""
    store.add_fact_proposal(key="spouse", value="Petra", confidence=0.9, source="llm")
    pid = store.add_fact_proposal(
        key="employer",
        value="Infosys",
        confidence=0.9,
        source="llm",
        subject="Petra",
        subject_class="Person",
    )

    rows = {p["id"]: p for p in client.get("/memory/review").json()["proposals"]}
    approved = client.post(f"/memory/review/{pid}/approve").json()

    assert rows[pid]["subject"] == "Petra"
    assert [p["subject"] for i, p in rows.items() if i != pid] == [None]
    assert approved == {"approved": True, "key": "employer", "value": "Infosys", "subject": "Petra"}
    assert store.fetch_user_fact("employer") is None
