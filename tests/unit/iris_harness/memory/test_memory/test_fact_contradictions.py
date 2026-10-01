"""Tests for write-time contradiction detection (review queue for same-key conflicts)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from iris_harness.memory.store import MemoryStore, UserFact


def _fact(key: str, value: str, confidence: float) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=confidence,
        source="conversation:llm",
        first_seen=now,
        last_confirmed=now,
        times_confirmed=1,
    )


def _store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def test_no_contradiction_on_first_capture_or_reconfirm(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("city", "Berlin", 0.9))  # first capture
    s.upsert_user_fact(_fact("city", "Berlin", 0.95))  # reconfirm (same value)
    assert s.fetch_contradictions() == []


def test_superseding_conflict_is_recorded(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("city", "Berlin", 0.9))
    s.upsert_user_fact(_fact("city", "New York", 0.9))  # equal conf → new wins
    contras = s.fetch_contradictions()
    assert len(contras) == 1
    c = contras[0]
    assert (c.stored_value, c.incoming_value, c.resolution) == ("Berlin", "New York", "superseded")
    assert s.fetch_user_fact("city").value == "New York"


def test_blocked_conflict_is_recorded_even_though_value_unchanged(tmp_path: Path) -> None:
    """The key win: a lower-confidence conflicting write is normally silently dropped."""
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("blog", "web3notes.example", 0.9))
    s.upsert_user_fact(_fact("blog", "wrong.com", 0.3))  # lower conf → blocked
    assert s.fetch_user_fact("blog").value == "web3notes.example"  # unchanged
    contras = s.fetch_contradictions()
    assert len(contras) == 1
    assert (contras[0].incoming_value, contras[0].resolution) == ("wrong.com", "blocked")


def test_acknowledge_clears_from_default_queue(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("city", "Berlin", 0.9))
    s.upsert_user_fact(_fact("city", "New York", 0.9))
    [c] = s.fetch_contradictions()
    assert s.acknowledge_contradictions([c.id]) == 1
    assert s.fetch_contradictions() == []  # default hides acknowledged
    assert len(s.fetch_contradictions(include_acknowledged=True)) == 1


def test_acknowledge_empty_is_noop(tmp_path: Path) -> None:
    assert _store(tmp_path).acknowledge_contradictions([]) == 0


def test_repeated_identical_conflict_dedups_and_counts(tmp_path: Path) -> None:
    """A model re-extracting the same bad value every turn must not flood the queue."""
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("blog", "web3notes.example", 0.9))
    for _ in range(4):
        s.upsert_user_fact(_fact("blog", "wrong.com", 0.3))  # same blocked conflict x4
    contras = s.fetch_contradictions()
    assert len(contras) == 1  # collapsed to one row
    assert contras[0].seen_count == 4
    assert contras[0].incoming_value == "wrong.com"


def test_distinct_conflicts_stay_separate(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("blog", "web3notes.example", 0.9))
    s.upsert_user_fact(_fact("blog", "wrong.com", 0.3))
    s.upsert_user_fact(_fact("blog", "other.com", 0.3))  # different incoming → separate
    assert len(s.fetch_contradictions()) == 2


def test_recurrence_after_ack_opens_fresh_row(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("blog", "web3notes.example", 0.9))
    s.upsert_user_fact(_fact("blog", "wrong.com", 0.3))
    [c] = s.fetch_contradictions()
    s.acknowledge_contradictions([c.id])
    assert s.fetch_contradictions() == []
    s.upsert_user_fact(_fact("blog", "wrong.com", 0.3))  # recurs after ack
    fresh = s.fetch_contradictions()
    assert len(fresh) == 1 and fresh[0].seen_count == 1 and fresh[0].id != c.id
