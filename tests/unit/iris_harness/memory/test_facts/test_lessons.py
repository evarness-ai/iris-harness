"""Lessons need evidence, and reach the prompt only as an approved behavior.

What this replaces: `learning_signals` written on every successful turn, all of the
shape "For 'finance' requests like this, the finance agent using finance answered it
cleanly." Four rows in the real store, no information in any of them — fetched every
turn and dropped unless an opt-in flag was set, which it never was.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.memory import identity
from iris_harness.memory.lessons import SOURCE_CORRECTION, LessonCurator
from iris_harness.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


@pytest.fixture
def iris_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "iris-home"
    (home / "behaviors").mkdir(parents=True)
    monkeypatch.setattr(identity.loader, "IRIS_HOME", home)
    monkeypatch.setattr(identity.loader, "BEHAVIORS_DIR", home / "behaviors")
    return home


class TestProposing:
    def test_a_correction_becomes_a_pending_lesson(self, store: MemoryStore) -> None:
        curator = LessonCurator(store)

        pid = curator.propose(
            trigger="what are my card dues",
            lesson="use finance_lookup before search_inbox",
            source=SOURCE_CORRECTION,
            evidence="user said: no, check the statement",
        )

        assert pid is not None
        pending = store.fetch_lesson_proposals()
        assert [(p.trigger, p.lesson) for p in pending] == [
            ("what are my card dues", "use finance_lookup before search_inbox")
        ]

    def test_a_report_that_things_went_fine_is_not_a_lesson(self, store: MemoryStore) -> None:
        """The exact shape of all four rows in the real store."""
        curator = LessonCurator(store)

        pid = curator.propose(
            trigger="what's my net worth?",
            lesson="For 'finance' requests like this, the finance agent answered it cleanly.",
            source=SOURCE_CORRECTION,
        )

        assert pid is None
        assert store.fetch_lesson_proposals() == []

    def test_an_empty_lesson_is_dropped(self, store: MemoryStore) -> None:
        assert LessonCurator(store).propose(trigger="x", lesson="  ", source="t") is None

    def test_the_same_lesson_twice_is_one_row(self, store: MemoryStore) -> None:
        curator = LessonCurator(store)
        first = curator.propose(trigger="card dues", lesson="check finance first", source="t")
        second = curator.propose(trigger="card dues", lesson="check finance first", source="t")

        assert first == second
        assert len(store.fetch_lesson_proposals()) == 1


class TestApproving:
    def test_approving_writes_a_matchable_behavior(
        self, store: MemoryStore, iris_home: Path
    ) -> None:
        pid = LessonCurator(store).propose(
            trigger="asked about credit card dues",
            lesson="call finance_lookup before searching the inbox",
            source=SOURCE_CORRECTION,
        )
        assert pid is not None

        name = LessonCurator(store).approve(pid)

        assert name is not None
        behaviors = identity.list_behaviors()
        assert [b.name for b in behaviors] == [name]
        assert "call finance_lookup before searching the inbox" in behaviors[0].body
        # and it is reachable by the matcher on the next turn
        matched = identity.match_behavior("finance", "what are my credit card dues this month?")
        assert matched is not None and matched.name == name
        assert store.fetch_lesson_proposals() == []

    def test_rejecting_writes_nothing(self, store: MemoryStore, iris_home: Path) -> None:
        pid = LessonCurator(store).propose(
            trigger="card dues", lesson="do the other thing", source="t"
        )
        assert pid is not None

        assert LessonCurator(store).reject(pid) is True
        assert identity.list_behaviors() == []
        assert LessonCurator(store).reject(pid) is False  # only once

    def test_approving_twice_is_a_no_op(self, store: MemoryStore, iris_home: Path) -> None:
        pid = LessonCurator(store).propose(
            trigger="card dues", lesson="check finance first", source="t"
        )
        assert pid is not None
        LessonCurator(store).approve(pid)

        assert LessonCurator(store).approve(pid) is None


def test_unreviewed_lessons_expire(store: MemoryStore) -> None:
    LessonCurator(store).propose(trigger="card dues", lesson="check finance first", source="t")

    assert store.expire_lesson_proposals(older_than_days=30) == 0
    assert store.expire_lesson_proposals(older_than_days=0) == 1
    assert store.fetch_lesson_proposals() == []
