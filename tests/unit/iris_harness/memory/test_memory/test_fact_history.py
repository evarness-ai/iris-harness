"""Tests for the fact history / supersession / reversible correct-forget trail."""

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


def test_capture_records_history(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("blog", "web3notes.example", 0.9))
    hist = s.fetch_fact_history("blog")
    assert len(hist) == 1
    assert hist[0].reason == "capture"
    assert hist[0].old_value is None
    assert hist[0].new_value == "web3notes.example"


def test_supersede_records_history(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("city", "New York", 0.9))
    s.upsert_user_fact(_fact("city", "Berlin", 0.9))  # equal confidence → new value wins
    assert s.fetch_user_fact("city").value == "Berlin"
    hist = s.fetch_fact_history("city")
    assert [h.reason for h in hist] == ["supersede", "capture"]
    assert hist[0].old_value == "New York"
    assert hist[0].new_value == "Berlin"


def test_blocked_low_confidence_does_not_log_or_change(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("blog", "web3notes.example", 0.9))
    s.upsert_user_fact(_fact("blog", "site", 0.3))  # lower conf, different value → blocked
    assert s.fetch_user_fact("blog").value == "web3notes.example"
    hist = s.fetch_fact_history("blog")
    assert len(hist) == 1  # only the capture; the blocked write left no trail
    assert hist[0].reason == "capture"


def test_reconfirm_same_value_does_not_log(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("name", "Robin", 0.9))
    s.upsert_user_fact(_fact("name", "Robin", 0.95))  # same value, reconfirm
    hist = s.fetch_fact_history("name")
    assert len(hist) == 1  # reconfirms are not noise in the trail
    assert hist[0].reason == "capture"


def test_correct_overrides_confidence_gate(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("city", "Berlin", 0.95))
    replaced = s.correct_user_fact("city", "New York")  # explicit user correction wins
    assert replaced is True
    assert s.fetch_user_fact("city").value == "New York"
    hist = s.fetch_fact_history("city")
    assert hist[0].reason == "correct"
    assert hist[0].old_value == "Berlin"
    assert hist[0].new_value == "New York"


def test_forget_is_reversible(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("city", "Berlin", 0.9))
    assert s.delete_user_fact("city") is True
    assert s.fetch_user_fact("city") is None
    # The forget is recorded (value preserved in history)...
    hist = s.fetch_fact_history("city")
    assert hist[0].reason == "forget"
    assert hist[0].old_value == "Berlin"
    assert hist[0].new_value is None
    # ...and restore brings it back.
    restored = s.restore_user_fact("city")
    assert restored == "Berlin"
    assert s.fetch_user_fact("city").value == "Berlin"
    assert s.fetch_fact_history("city")[0].reason == "restore"


def test_restore_undoes_last_correction(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("city", "Berlin", 0.9))
    s.correct_user_fact("city", "New York")
    assert s.restore_user_fact("city") == "Berlin"
    assert s.fetch_user_fact("city").value == "Berlin"


def test_forget_missing_returns_false(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.delete_user_fact("nope") is False


def test_restore_with_no_history_returns_none(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.restore_user_fact("nope") is None
