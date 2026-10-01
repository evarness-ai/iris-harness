"""Owner-confirmed recall: extraction proposes, the owner confirms, recall reads.

The store this replaces held 58 facts mined from whatever was being discussed —
`name=ollama` at 1.00, `employer=Department of Justice` at 0.9, `location=IRIS`,
`age=4`, `topic=murder plot` — and every one of them was eligible for the prompt on
confidence alone.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_harness.memory.coordinator import FactCoordinator
from iris_harness.memory.retriever import MemoryRetriever
from iris_harness.memory.store import MemoryStore, UserFact


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _fact(key: str, value: str, confidence: float = 0.9, confirmed: bool = False) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=confidence,
        source="test",
        first_seen=now,
        last_confirmed=now,
        times_confirmed=1,
        confirmed=confirmed,
    )


class TestProposalQueue:
    def test_a_proposal_is_not_a_fact(self, store: MemoryStore) -> None:
        store.add_fact_proposal(key="city", value="Springfield", confidence=0.9, source="llm")

        # Not believed: recall and prompts read confirmed facts only. (Since memris PR 2c
        # a proposal is a proposed statement, so the unfiltered list does include it.)
        assert store.fetch_all_user_facts(confirmed_only=True) == []
        assert [f.confirmed for f in store.fetch_all_user_facts()] == [False]
        assert [p.key for p in store.fetch_fact_proposals()] == ["city"]

    def test_a_proposal_records_what_is_stored_today(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("city", "Chennai", confirmed=True))

        store.add_fact_proposal(key="city", value="Springfield", confidence=0.9, source="llm")

        proposal = store.fetch_fact_proposals()[0]
        assert proposal.current_value == "Chennai"  # the review reads "changed?"

    def test_restating_a_confirmed_fact_does_not_queue_anything(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("city", "Chennai", confirmed=True))

        assert (
            store.add_fact_proposal(key="city", value="Chennai", confidence=0.9, source="llm")
            is None
        )
        assert store.fetch_fact_proposals() == []
        assert (store.fetch_user_fact("city") or _fact("x", "y")).times_confirmed == 2

    def test_the_same_proposal_twice_is_one_row(self, store: MemoryStore) -> None:
        first = store.add_fact_proposal(key="city", value="Springfield", confidence=0.9, source="a")
        second = store.add_fact_proposal(
            key="city", value="Springfield", confidence=0.5, source="b"
        )

        assert first == second
        assert len(store.fetch_fact_proposals()) == 1

    def test_unreviewed_proposals_expire(self, store: MemoryStore) -> None:
        store.add_fact_proposal(key="city", value="Springfield", confidence=0.9, source="llm")
        assert store.expire_fact_proposals(older_than_days=30) == 0

        assert store.expire_fact_proposals(older_than_days=0) == 1
        assert store.fetch_fact_proposals() == []
        assert len(store.fetch_fact_proposals(status="expired")) == 1

    def test_the_queue_counts_legacy_facts_too(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("name", "ollama", 1.0))  # pre-confirmation row
        store.add_fact_proposal(key="city", value="Springfield", confidence=0.9, source="llm")

        assert store.count_pending_review() == 2


class TestRecallIsConfirmedOnly:
    def test_an_unconfirmed_fact_never_reaches_context(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("name", "ollama", 1.0))  # confidence 1.0, still junk

        ctx = MemoryRetriever(store=store).build_context(query="what is my name?")

        assert ctx.user_facts == ()

    def test_a_confirmed_fact_does(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("profession", "solutions architect", 0.9, confirmed=True))

        ctx = MemoryRetriever(store=store).build_context(query="what do i do for work?")

        assert [f.value for f in ctx.user_facts] == ["solutions architect"]

    def test_confidence_still_filters_confirmed_facts(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("city", "Chennai", 0.1, confirmed=True))

        ctx = MemoryRetriever(store=store).build_context(query="where do i live?")

        assert ctx.user_facts == ()


class TestCoordinator:
    def test_approving_makes_it_recallable(self, store: MemoryStore) -> None:
        pid = store.add_fact_proposal(key="city", value="Springfield", confidence=0.6, source="llm")
        assert pid is not None

        fact = FactCoordinator(store).approve_proposal(pid)

        assert fact is not None and fact.confirmed
        assert store.fetch_fact_proposals() == []
        ctx = MemoryRetriever(store=store).build_context(query="where do i live?")
        assert [f.value for f in ctx.user_facts] == ["Springfield"]

    def test_approving_overrides_the_confidence_gate(self, store: MemoryStore) -> None:
        """The owner's yes outranks a higher-confidence value already stored."""
        store.upsert_user_fact(_fact("city", "Chennai", 0.95, confirmed=True))
        pid = store.add_fact_proposal(key="city", value="Springfield", confidence=0.4, source="llm")
        assert pid is not None

        FactCoordinator(store).approve_proposal(pid)

        assert (store.fetch_user_fact("city") or _fact("x", "y")).value == "Springfield"

    def test_rejecting_writes_nothing(self, store: MemoryStore) -> None:
        pid = store.add_fact_proposal(key="city", value="Springfield", confidence=0.9, source="llm")
        assert pid is not None

        assert FactCoordinator(store).reject_proposal(pid) is True
        assert store.fetch_all_user_facts() == []
        assert FactCoordinator(store).reject_proposal(pid) is False  # only once

    def test_confirming_a_legacy_fact_clears_it_from_the_queue(self, store: MemoryStore) -> None:
        store.upsert_user_fact(_fact("country", "India", 0.9))

        fact = FactCoordinator(store).confirm("country")

        assert fact is not None and fact.confirmed
        assert store.count_pending_review() == 0

    def test_a_blocked_write_no_longer_leaks_to_the_derived_homes(self, store: MemoryStore) -> None:
        """The store's confidence gate refuses the write; the index must agree.

        `record()` used to index and project the INCOMING value even when the store
        kept the old one (ADR-0105 drift).
        """
        indexed: list[tuple[str, str]] = []

        class _Index:
            def index_fact(self, fact: UserFact) -> None:
                indexed.append((fact.key, fact.value))

            def drop_fact(self, key: str) -> None:
                indexed.append((key, "<dropped>"))

        store.upsert_user_fact(_fact("blog", "real.example", 0.9, confirmed=True))
        coordinator = FactCoordinator(store, _Index(), project_md=False)  # type: ignore[arg-type]

        coordinator.record("blog", "site", 0.3, "llm", confirmed=True)

        assert (store.fetch_user_fact("blog") or _fact("x", "y")).value == "real.example"
        assert indexed == [("blog", "real.example")]

    def test_an_unconfirmed_fact_is_dropped_from_the_recall_index(self, store: MemoryStore) -> None:
        dropped: list[str] = []

        class _Index:
            def index_fact(self, fact: UserFact) -> None:
                raise AssertionError("an unconfirmed fact must not be indexed")

            def drop_fact(self, key: str) -> None:
                dropped.append(key)

        FactCoordinator(store, _Index(), project_md=False).record(  # type: ignore[arg-type]
            "city", "Springfield", 0.9, "llm"
        )

        assert dropped == ["city"]
