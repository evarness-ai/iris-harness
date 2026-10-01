"""Unit tests for SemanticIndex — ChromaDB-backed semantic retrieval."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.store import LearningSignal, MemoryStore, UserFact

pytestmark = pytest.mark.real_embeddings


def _now() -> datetime:
    return datetime.now(UTC)


def _fact(key: str, value: str) -> UserFact:
    now = _now()
    return UserFact(
        key=key, value=value, confidence=0.9, source="test", first_seen=now, last_confirmed=now
    )


def _signal(id: str, query: str, domain: str) -> LearningSignal:
    return LearningSignal(
        id=id,
        signal_type="success",
        domain=domain,
        agent_type="system",
        query=query,
        context="",
        outcome="ok",
        improvement_hint=None,
        timestamp=_now(),
    )


@pytest.fixture()
def index(tmp_path: Path) -> SemanticIndex:
    idx = SemanticIndex(persist_dir=tmp_path / "chroma")
    assert idx.is_ready, "ChromaDB did not initialise — check chromadb install"
    return idx


class TestSemanticIndexInit:
    def test_ready_with_valid_dir(self, tmp_path: Path) -> None:
        idx = SemanticIndex(persist_dir=tmp_path / "chroma")
        assert idx.is_ready

    def test_creates_persist_dir(self, tmp_path: Path) -> None:
        persist = tmp_path / "nested" / "chroma"
        SemanticIndex(persist_dir=persist)
        assert persist.exists()


class TestFactIndexing:
    def test_index_and_query_fact(self, index: SemanticIndex) -> None:
        index.index_fact(_fact("preferred_language", "Python"))
        keys = index.query_facts("what programming language does the user prefer")
        assert "preferred_language" in keys

    def test_unrelated_query_still_returns_results(self, index: SemanticIndex) -> None:
        index.index_fact(_fact("name", "Robin"))
        index.index_fact(_fact("location", "Springfield"))
        keys = index.query_facts("Python programming", n=5)
        # ChromaDB always returns best matches — we just care it doesn't crash
        assert isinstance(keys, list)

    def test_drop_fact_removes_from_results(self, index: SemanticIndex) -> None:
        index.index_fact(_fact("theme", "dark mode"))
        keys_before = index.query_facts("user interface theme")
        assert "theme" in keys_before
        index.drop_fact("theme")
        keys_after = index.query_facts("user interface theme")
        assert "theme" not in keys_after

    def test_upsert_updates_existing_fact(self, index: SemanticIndex) -> None:
        index.index_fact(_fact("language", "Python"))
        index.index_fact(_fact("language", "TypeScript"))
        # Should not raise; collection count stays at 1
        assert index._facts.count() == 1

    def test_query_facts_text_distance_threshold(self, index: SemanticIndex) -> None:
        # issue 0032: a relevance cutoff stops an off-topic query from dumping the
        # nearest unrelated facts (incl. PII like the user's email) to chat.
        index.index_fact(_fact("email", "user@example.com"))
        # No cutoff → ChromaDB returns the nearest match regardless of relevance.
        assert index.query_facts_text("number of stock holdings", n=5) != []
        # An impossibly-tight cutoff filters everything (mechanism check).
        assert index.query_facts_text("number of stock holdings", n=5, max_distance=0.01) == []
        # A generous cutoff still returns a genuinely-relevant match.
        assert index.query_facts_text("what is my email address", n=5, max_distance=10.0) != []


class TestSignalIndexing:
    def test_index_and_query_signal(self, index: SemanticIndex) -> None:
        index.index_signal(_signal("s1", "fetch my emails from Gmail", "communication"))
        ids = index.query_signals("retrieve emails from inbox")
        assert "s1" in ids

    def test_empty_signals_returns_empty_list(self, index: SemanticIndex) -> None:
        ids = index.query_signals("some query")
        assert ids == []


class TestEpisodicIndexing:
    def test_index_and_query_episodic(self, index: SemanticIndex) -> None:
        index.index_episodic("p1", "User prefers concise answers without preamble")
        index.index_episodic("p2", "User reads tech news every weekday morning")
        hits = index.query_episodic("morning routine reading habits")
        assert any("tech news" in h for h in hits)

    def test_empty_episodic_returns_empty_list(self, index: SemanticIndex) -> None:
        assert index.query_episodic("anything") == []

    def test_drop_episodic_removes_from_results(self, index: SemanticIndex) -> None:
        index.index_episodic("p1", "User asks about Kubernetes manifests often")
        before = index.query_episodic("kubernetes deployment yaml")
        assert any("Kubernetes" in h for h in before)
        index.drop_episodic("p1")
        after = index.query_episodic("kubernetes deployment yaml")
        assert not any("Kubernetes" in h for h in after)

    def test_sync_drops_stale_patterns(self, index: SemanticIndex) -> None:
        index.sync_episodic_patterns([("p1", "old removed pattern"), ("p2", "kept pattern")])
        # Re-sync without p1 — it should be dropped from the index.
        index.sync_episodic_patterns([("p2", "kept pattern")])
        hits = index.query_episodic("removed", n=10)
        assert not any("removed" in h for h in hits)
        kept_hits = index.query_episodic("kept pattern", n=10)
        assert any("kept" in h for h in kept_hits)


class TestTurnIndexing:
    def test_index_and_query_turn(self, index: SemanticIndex) -> None:
        index.index_turn(1, "session_A", "user", "How do I connect to Gmail?")
        index.index_turn(2, "session_A", "assistant", "You can use the Gmail skill with OAuth.")
        pairs = index.query_turns("email integration OAuth")
        assert len(pairs) > 0
        roles = {p[0] for p in pairs}
        assert roles <= {"user", "assistant"}

    def test_exclude_session_filters_current(self, index: SemanticIndex) -> None:
        index.index_turn(1, "session_A", "user", "Tell me about Python decorators")
        index.index_turn(2, "session_B", "user", "Explain Python decorators please")
        pairs = index.query_turns("Python decorators", exclude_session="session_A")
        contents = [p[1] for p in pairs]
        assert all("decorators" in c.lower() for c in contents)
        # session_A turn should be excluded
        assert not any(c == "Tell me about Python decorators" for c in contents)

    def test_empty_turns_returns_empty_list(self, index: SemanticIndex) -> None:
        pairs = index.query_turns("anything")
        assert pairs == []

    def test_query_turns_detailed_carries_provenance(self, index: SemanticIndex) -> None:
        index.index_turn(
            7, "session_A", "assistant", "Canberra is the capital of Australia.", turn_id="t-abc"
        )
        refs = index.query_turns_detailed("what is australia's capital")
        assert refs, "expected a semantic hit"
        ref = refs[0]
        assert ref.row_id == "7"
        assert ref.turn_id == "t-abc"
        assert ref.session_id == "session_A"
        assert ref.role == "assistant"

    def test_turn_without_turn_id_has_none(self, index: SemanticIndex) -> None:
        # Backfilled rows (no turn_id) still queryable; provenance is None.
        index.index_turn(8, "session_B", "user", "Tell me about quantum tunnelling")
        refs = index.query_turns_detailed("quantum tunnelling")
        assert refs and refs[0].turn_id is None
        assert refs[0].row_id == "8"


class TestSyncFromStore:
    def test_sync_indexes_facts_and_turns(self, tmp_path: Path) -> None:
        store = MemoryStore(db_path=tmp_path / "memory.db")
        store.ensure_schema()
        store.upsert_user_fact(_fact("city", "Springfield"))
        store.save_conversation_turns_and_get_ids("s1", [("user", "What is the weather like?")])

        idx = SemanticIndex(persist_dir=tmp_path / "chroma")
        new_turns = idx.sync_from_store(store)

        assert new_turns == 1
        assert "city" in idx.query_facts("Where does the user live?")
        pairs = idx.query_turns("weather forecast")
        assert len(pairs) == 1

    def test_sync_is_incremental_via_watermark(self, tmp_path: Path) -> None:
        store = MemoryStore(db_path=tmp_path / "memory.db")
        store.ensure_schema()
        store.save_conversation_turns_and_get_ids("s1", [("user", "first turn")])

        idx = SemanticIndex(persist_dir=tmp_path / "chroma")
        first_sync = idx.sync_from_store(store)
        assert first_sync == 1

        store.save_conversation_turns_and_get_ids("s1", [("user", "second turn")])
        second_sync = idx.sync_from_store(store)
        assert second_sync == 1  # only the new turn

        assert idx._turns.count() == 2

    def test_sync_with_empty_store_returns_zero(self, tmp_path: Path) -> None:
        store = MemoryStore(db_path=tmp_path / "memory.db")
        store.ensure_schema()
        idx = SemanticIndex(persist_dir=tmp_path / "chroma")
        assert idx.sync_from_store(store) == 0


class TestGracefulDegradation:
    def test_not_ready_index_returns_empty_on_all_queries(self, tmp_path: Path) -> None:
        idx = SemanticIndex(persist_dir=tmp_path / "chroma")
        idx._ok = False  # simulate init failure
        assert idx.query_facts("anything") == []
        assert idx.query_signals("anything") == []
        assert idx.query_turns("anything") == []

    def test_not_ready_index_is_silent_on_writes(self, tmp_path: Path) -> None:
        idx = SemanticIndex(persist_dir=tmp_path / "chroma")
        idx._ok = False
        idx.index_fact(_fact("k", "v"))  # should not raise
        idx.index_turn(1, "s", "user", "hello")  # should not raise
