"""The APIs behind the Memory screen: profile, fact edits, sessions, context.

Every screen action has a CLI twin and a route — the UI is a client of the same
surface, never the only way to do something.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.memory import identity
from iris_harness.memory.retention import RetentionService
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


@pytest.fixture
def iris_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "iris-home"
    (home / "workspace").mkdir(parents=True)
    (home / "workspace" / "USER.md").write_text(
        "# User Profile\n\n- Role: maintainer\n\n## Auto-detected\n\n- **bank**: Northwind\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(identity.loader, "USER_MD_PATH", home / "workspace" / "USER.md")
    return home


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


def _fact(key: str, value: str, confirmed: bool = True) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=0.9,
        source="test",
        first_seen=now,
        last_confirmed=now,
        confirmed=confirmed,
    )


class TestProfile:
    def test_get_returns_only_the_curated_head(self, client: TestClient, iris_home: Path) -> None:
        body = client.get("/memory/profile").json()

        assert "- Role: maintainer" in body["profile"]
        assert "Auto-detected" not in body["profile"]  # the store's projection, not yours

    def test_put_replaces_the_head_and_keeps_the_auto_block(
        self, client: TestClient, iris_home: Path
    ) -> None:
        resp = client.put("/memory/profile", json={"profile": "# Me\n\n- Role: owner"})

        assert resp.status_code == 200
        text = (iris_home / "workspace" / "USER.md").read_text(encoding="utf-8")
        assert "- Role: owner" in text
        assert "- **bank**: Northwind" in text  # the auto block survived
        assert "maintainer" not in text


class TestFactEdits:
    def test_forget_removes_it(self, client: TestClient, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind"))

        assert client.post("/memory/facts/bank/forget").status_code == 200
        assert store.fetch_user_fact("bank") is None

    def test_forget_unknown_is_a_404(self, client: TestClient) -> None:
        assert client.post("/memory/facts/nope/forget").status_code == 404

    def test_patch_sets_the_value_and_confirms_it(
        self, client: TestClient, store: MemoryStore
    ) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind", confirmed=False))

        resp = client.patch("/memory/facts/bank", json={"value": "Wingtip"})

        assert resp.status_code == 200
        stored = store.fetch_user_fact("bank")
        assert stored is not None and stored.value == "Wingtip" and stored.confirmed

    def test_history_shows_the_change(self, client: TestClient, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind"))
        client.patch("/memory/facts/bank", json={"value": "Wingtip"})

        body = client.get("/memory/facts/bank/history").json()

        assert body["count"] >= 1
        assert body["history"][0]["new_value"] == "Wingtip"


class TestSessions:
    def test_it_lists_sessions_with_state_and_summary(
        self, client: TestClient, store: MemoryStore
    ) -> None:
        store.save_conversation_turns("s1", [("user", "q"), ("assistant", "a")])
        store.save_conversation_summary("s1", "Goal: the test session")
        store.save_conversation_turns("playground-x", [("user", "q"), ("assistant", "a")])

        body = client.get("/memory/sessions").json()
        rows = {r["session_id"]: r for r in body["sessions"]}

        assert rows["s1"]["turns"] == 2
        assert rows["s1"]["summary"] == "Goal: the test session"
        assert rows["s1"]["state"] == "hot"
        assert rows["s1"]["is_run"] is False
        assert rows["playground-x"]["is_run"] is True


class TestContextInspection:
    def test_it_shows_each_block_with_its_token_cost(self, tmp_path: Path) -> None:
        """The anti-'shipped unwired' view: a block that is absent says so."""
        from iris_harness.memory.compactor import ConversationCompactor
        from iris_harness.runtime.session_memory import SessionMemory

        store = MemoryStore(db_path=tmp_path / "memory.db")
        store.ensure_schema()
        host = SimpleNamespace(
            memory_store=store,
            semantic_index=None,
            memory_retriever=SimpleNamespace(
                build_context=lambda **kw: __import__(
                    "iris_harness.memory.retriever", fromlist=["MemoryContext"]
                ).MemoryContext(recent_turns=kw.get("recent_turns", ())),
            ),
            compactor=ConversationCompactor(compaction_threshold=50, keep_recent=10),
            learning=SimpleNamespace(behavior_miner=None),
        )
        sessions = SessionMemory(host)  # type: ignore[arg-type]
        sessions.record_turn("s1", "the deposit was 180000", "noted")
        runtime = SimpleNamespace(memory_store=store, semantic_index=None, sessions=sessions)

        with TestClient(
            create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
        ) as client:
            body = client.get("/memory/context/s1").json()

        blocks = {b["block"]: b for b in body["blocks"]}
        assert blocks["recent turns"]["present"] is True
        assert blocks["recent turns"]["tokens"] > 0
        assert blocks["session summary"]["present"] is False  # nothing rolled yet
        assert body["total_tokens"] > 0


def test_the_action_center_carries_one_memory_row(store: MemoryStore) -> None:
    from iris_harness.runtime.action_center import memory_pending_actions

    store.upsert_user_fact(_fact("name", "ollama", confirmed=False))
    store.add_fact_proposal(key="city", value="Springfield", confidence=0.8, source="llm")

    rows = memory_pending_actions(store)

    assert len(rows) == 1
    assert "2 memory item(s)" in rows[0].title
    assert rows[0].action.command == "iris facts review"


def test_no_row_when_nothing_is_waiting(store: MemoryStore) -> None:
    from iris_harness.runtime.action_center import memory_pending_actions

    assert memory_pending_actions(store) == []


class TestMemoryGraphEndpoint:
    def test_it_opens_on_you_with_what_touches_you(
        self, client: TestClient, store: MemoryStore
    ) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        store.save_conversation_turns("s1", [("user", "q"), ("assistant", "a")])
        store.save_conversation_summary("s1", "Referenced: Petra Sutton, Northwind Bank")
        # A name that is not a memory entity needs two sessions (drawn_mentions).
        store.save_conversation_turns("s2", [("user", "q"), ("assistant", "a")])
        store.save_conversation_summary("s2", "Referenced: Petra Sutton")

        body = client.get("/memory/graph").json()
        labels = {n["label"] for n in body["nodes"]}

        assert "you" in {n["id"] for n in body["nodes"]}
        assert {"Northwind Bank", "Petra Sutton"} <= labels
        assert body["focus"] == "you"

    def test_focus_expands_one_node(self, client: TestClient, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("bank", "Northwind Bank"))
        store.upsert_user_fact(_fact("blog", "tech4talk.com"))
        bank = next(
            n["id"]
            for n in client.get("/memory/graph").json()["nodes"]
            if n["label"] == "Northwind Bank"
        )

        body = client.get("/memory/graph", params={"focus": bank}).json()
        ids = {n["id"] for n in body["nodes"]}

        assert ids == {bank, "you"}

    def test_as_of_draws_what_held_then(self, client: TestClient, store: MemoryStore) -> None:
        from datetime import UTC, datetime

        store.upsert_user_fact(_fact("employer", "Barclays"))
        between = datetime.now(UTC)
        store.upsert_user_fact(_fact("employer", "Litware"))

        then = client.get("/memory/graph", params={"as_of": between.isoformat()}).json()
        labels = {n["label"] for n in then["nodes"]}

        assert "Barclays" in labels and "Litware" not in labels

    def test_the_cap_is_bounded_by_the_server(self, client: TestClient, store: MemoryStore) -> None:
        for i in range(40):  # many facts need a property with no limit: 40 cards
            store.upsert_user_fact(_fact("credit_card", f"card {i}"))

        body = client.get("/memory/graph", params={"cap": 5}).json()

        assert len(body["nodes"]) == 11  # cap floors at 10, plus the "+N more" marker
        assert any(n["kind"] == "more" for n in body["nodes"])
