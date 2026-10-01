"""Human-reviewed history retention (never auto-deletes).

Since memris plan PR 2c-ii the history IS the statement chain, and the owner decided
(2026-09-18) that pruning may delete only statements that no longer hold: a value that
was replaced or forgotten. A current fact is never a candidate, however old.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from iris_harness.foundation.persistence import sqlite_conn
from iris_harness.memory.store import MemoryStore, UserFact


def _store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


def _fact(key: str, value: str) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(key, value, 0.9, "test", now, now, 1, True)


def _replaced(s: MemoryStore, key: str, old: str, new: str, days_ago: int) -> None:
    """``old`` held until ``new`` replaced it; age the old value's history by ``days_ago``."""
    s.upsert_user_fact(_fact(key, old))
    [held] = [e for e in s.fetch_fact_history(key) if e.new_value == old]
    s.upsert_user_fact(_fact(key, new))
    _age(s, held.id, days_ago)


def _age(s: MemoryStore, statement_id: str, days_ago: int) -> None:
    ts = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    with sqlite_conn(s.db_path) as conn:
        conn.execute(
            "UPDATE memris_statements SET recorded_at = ? WHERE id = ?", (ts, statement_id)
        )


def test_candidates_only_past_the_window(tmp_path: Path) -> None:
    s = _store(tmp_path)
    _replaced(s, "city", "Berlin", "Paris", days_ago=200)
    _replaced(s, "email", "a@example.org", "b@example.org", days_ago=365)
    _replaced(s, "hobby", "chess", "go", days_ago=10)
    cands = s.fetch_history_retention_candidates(older_than_days=180)
    assert [(c.key, c.new_value) for c in cands] == [
        ("email", "a@example.org"),
        ("city", "Berlin"),
    ]  # oldest first; recent and current excluded
    assert s.count_fact_history() == 6  # three captures, three supersedes


def test_window_is_configurable(tmp_path: Path) -> None:
    s = _store(tmp_path)
    _replaced(s, "city", "Berlin", "Paris", days_ago=30)
    assert s.fetch_history_retention_candidates(older_than_days=180) == []
    assert len(s.fetch_history_retention_candidates(older_than_days=7)) == 1


def test_prune_by_ids(tmp_path: Path) -> None:
    s = _store(tmp_path)
    _replaced(s, "city", "Berlin", "Paris", days_ago=200)
    _replaced(s, "email", "a@example.org", "b@example.org", days_ago=200)
    cands = s.fetch_history_retention_candidates(older_than_days=180)
    removed = s.prune_history_entries([cands[0].id])
    assert removed == 1
    assert len(s.fetch_history_retention_candidates(older_than_days=180)) == 1
    assert s.fetch_user_fact("city").value == "Paris"  # type: ignore[union-attr]
    assert s.fetch_user_fact("email").value == "b@example.org"  # type: ignore[union-attr]


def test_prune_empty_is_noop(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.prune_history_entries([]) == 0


def test_a_current_fact_is_never_pruned_however_old(tmp_path: Path) -> None:
    """Pruning clears history; forgetting a fact is a retraction, never a prune."""
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("name", "Robin"))
    [capture] = s.fetch_fact_history("name")
    _age(s, capture.id, 400)
    assert s.fetch_history_retention_candidates(older_than_days=180) == []
    assert s.prune_history_entries([capture.id]) == 0
    assert s.fetch_user_fact("name").value == "Robin"  # type: ignore[union-attr]
    assert s.count_fact_history() == 1


def test_a_forgotten_fact_can_be_pruned_with_its_forget(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert_user_fact(_fact("email", "old@example.org"))
    s.delete_user_fact("email")
    [forget, capture] = s.fetch_fact_history("email")
    assert (forget.reason, capture.reason) == ("forget", "capture")
    _age(s, capture.id, 400)
    with sqlite_conn(s.db_path) as conn:  # the forget happened long ago too
        conn.execute(
            "UPDATE memris_statements SET retracted_at = recorded_at WHERE id = ?", (capture.id,)
        )
    assert {c.reason for c in s.fetch_history_retention_candidates(older_than_days=180)} == {
        "capture",
        "forget",
    }
    assert s.prune_history_entries([forget.id]) == 1  # both events go with the statement
    assert s.fetch_fact_history("email") == []
