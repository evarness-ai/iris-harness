"""Facts about someone one hop from the owner (memris PR 3b; ADR-0115 decision 6).

"My wife Petra works at Infosys" is two statements: the owner's spouse is Petra, and
Petra works at Infosys. The second is only kept when Petra is linked to the owner — by a
relation in the same message or one already confirmed — and it is a proposal about
Petra, never a fact about the owner.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_harness.memory.coordinator import FactCoordinator
from iris_harness.memory.fact_statements import Link, SubjectError
from iris_harness.memory.store import MemoryStore, UserFact
from memris.model import OWNER_ID

NOW = datetime(2026, 9, 1, tzinfo=UTC)
WIFE = [Link(None, "spouse", "Petra")]
PERSON = "mem:Person"


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _confirmed(store: MemoryStore, key: str, value: str) -> None:
    store.upsert_user_fact(UserFact(key, value, 0.9, "test", NOW, NOW, 1, True))


class TestScope:
    def test_a_relation_in_the_same_message_is_a_hop(self, store: MemoryStore) -> None:
        assert store.subject_in_scope("Petra", WIFE, max_hops=1) == PERSON

    def test_a_confirmed_relation_is_a_hop(self, store: MemoryStore) -> None:
        _confirmed(store, "spouse", "Petra")

        assert store.subject_in_scope("Petra", [], max_hops=1) == PERSON

    def test_a_proposed_relation_is_not(self, store: MemoryStore) -> None:
        store.add_fact_proposal(key="spouse", value="Petra", confidence=0.9, source="test")

        assert store.subject_in_scope("Petra", [], max_hops=1) is None

    def test_someone_nobody_linked_is_out_of_scope(self, store: MemoryStore) -> None:
        _confirmed(store, "spouse", "Petra")

        assert store.subject_in_scope("Sundar Pichai", WIFE, max_hops=1) is None

    def test_an_attribute_is_not_a_hop(self, store: MemoryStore) -> None:
        assert store.subject_in_scope("Petra", [Link(None, "name", "Petra")], max_hops=1) is None

    def test_zero_hops_keeps_capture_to_the_owner(self, store: MemoryStore) -> None:
        assert store.subject_in_scope("Petra", WIFE, max_hops=0) is None

    def test_two_hops_needs_a_limit_of_two(self, store: MemoryStore) -> None:
        links = [*WIFE, Link("Petra", "family", "Remy")]

        assert store.subject_in_scope("Remy", links, max_hops=1) is None
        assert store.subject_in_scope("Remy", links, max_hops=2) == PERSON

    def test_a_link_from_someone_out_of_scope_does_not_count(self, store: MemoryStore) -> None:
        links = [Link("Elon", "spouse", "Talulah")]

        assert store.subject_in_scope("Talulah", links, max_hops=2) is None

    def test_names_fold_like_everywhere_else(self, store: MemoryStore) -> None:
        _confirmed(store, "spouse", "Petra")

        assert store.subject_in_scope("petra", [], max_hops=1) == PERSON


class TestProposals:
    def _propose(self, store: MemoryStore) -> str:
        store.add_fact_proposal(key="spouse", value="Petra", confidence=0.9, source="test")
        pid = store.add_fact_proposal(
            key="employer",
            value="Infosys",
            confidence=0.9,
            source="test",
            evidence="my wife Petra works at Infosys",
            subject="Petra",
            subject_class="Person",
        )
        assert pid is not None
        return pid

    def test_the_fact_is_about_her_not_the_owner(self, store: MemoryStore) -> None:
        pid = self._propose(store)

        graph = store.memory_graph()
        statement = graph.store.get_statement(pid)
        assert statement is not None and statement.subject_id != OWNER_ID
        # the same Petra the owner's spouse fact names — resolved, not made twice
        [spouse] = graph.current(OWNER_ID, "spouse", include_proposed=True)
        assert statement.subject_id == spouse.object_id
        assert store.fetch_user_fact("employer") is None

    def test_review_lists_it_with_whom_it_is_about(self, store: MemoryStore) -> None:
        pid = self._propose(store)

        rows = {p.id: p for p in store.fetch_fact_proposals()}
        assert rows[pid].subject == "Petra"
        assert (rows[pid].key, rows[pid].value) == ("employer", "Infosys")
        assert [p.subject for p in rows.values() if p.id != pid] == [None]
        assert store.count_pending_review() == 2

    def test_approving_it_confirms_her_fact_and_leaves_the_owner_alone(
        self, store: MemoryStore
    ) -> None:
        pid = self._propose(store)

        fact = FactCoordinator(store, None).approve_proposal(pid)

        assert fact is not None and fact.value == "Infosys"
        assert store.fetch_user_fact("employer") is None
        statement = store.memory_graph().store.get_statement(pid)
        assert statement is not None and statement.status == "confirmed"
        assert store.fetch_fact_proposal(pid).status == "approved"  # type: ignore[union-attr]

    def test_saying_it_again_counts_instead_of_queueing_twice(self, store: MemoryStore) -> None:
        pid = self._propose(store)

        again = store.add_fact_proposal(
            key="employer",
            value="Infosys",
            confidence=0.9,
            source="test",
            subject="Petra",
            subject_class="Person",
        )

        assert again == pid
        assert store.fetch_fact_proposal(pid).seen_count == 2  # type: ignore[union-attr]

    def test_a_property_that_cannot_describe_them_is_refused(self, store: MemoryStore) -> None:
        with pytest.raises(SubjectError):
            store.add_fact_proposal(
                key="employer",
                value="Infosys",
                confidence=0.9,
                source="test",
                subject="Bruno",
                subject_class="Animal",
            )

    def test_a_name_known_only_as_something_vaguer_is_refused(self, store: MemoryStore) -> None:
        # "Petra" turned up earlier as an untyped name (a summary mention): she resolves to
        # it, and a person's property cannot describe a bare Entity — refused, not crashed.
        store.memory_graph().add_entity("Entity", "Petra")

        with pytest.raises(SubjectError):
            store.add_fact_proposal(
                key="employer",
                value="Infosys",
                confidence=0.9,
                source="test",
                subject="Petra",
                subject_class="Person",
            )


class TestHopLimit:
    def test_it_is_read_from_learning_yaml(self) -> None:
        from iris_harness.memory import fact_keys

        fact_keys.reset_cache()
        assert fact_keys.subject_max_hops() == 1  # config/memory/learning.yaml

    def test_it_follows_the_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from iris_harness.memory import fact_keys

        monkeypatch.setattr(fact_keys, "_RAW_CACHE", {"subject_scope": {"max_hops": 0}})
        assert fact_keys.subject_max_hops() == 0
        monkeypatch.setattr(fact_keys, "_RAW_CACHE", {"subject_scope": {"max_hops": "x"}})
        assert fact_keys.subject_max_hops() == 1
