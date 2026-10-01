"""Tests for the recall quality filter (Task 2: clean, non-polluted context).

Static confidence thresholds (no decay): facts < min_fact_confidence are dropped,
facts in [min, uncertain_below) are marked uncertain, and curated USER.md facts are
never filtered (that boundary is enforced upstream — these tests cover the store-fact path).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from iris_harness.memory.retriever import MemoryRetriever
from iris_harness.memory.store import MemoryStore, UserFact


def _fact(key: str, value: str, confidence: float) -> UserFact:
    now = datetime.now(UTC)
    return UserFact(
        key=key,
        value=value,
        confidence=confidence,
        source="test",
        first_seen=now,
        last_confirmed=now,
        times_confirmed=1,
        # These cases are about the CONFIDENCE filter, so they start owner-confirmed.
        # The confirmation gate itself is covered in test_fact_confirmation.py.
        confirmed=True,
    )


def _retriever(tmp_path: Path) -> MemoryRetriever:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    store.upsert_user_fact(_fact("hobby", "noise", 0.3))
    store.upsert_user_fact(_fact("interest", "maybe-berlin", 0.5))
    store.upsert_user_fact(_fact("name", "robin", 0.9))
    # Defaults: min_fact_confidence=0.35, uncertain_below=0.6 (no index → keyword path).
    return MemoryRetriever(store=store, index=None)


def test_low_confidence_fact_dropped(tmp_path: Path) -> None:
    ctx = _retriever(tmp_path).build_context(query="who am I")
    keys = {f.key for f in ctx.user_facts}
    assert "hobby" not in keys  # 0.3 < 0.35 → dropped
    assert "interest" in keys
    assert "name" in keys


def test_mid_confidence_fact_marked_uncertain(tmp_path: Path) -> None:
    ctx = _retriever(tmp_path).build_context(query="who am I")
    by_key = {f.key: f for f in ctx.user_facts}
    assert by_key["interest"].uncertain is True  # 0.35 <= 0.5 < 0.6
    assert by_key["name"].uncertain is False  # 0.9 >= 0.6


def test_filter_disabled_via_threshold(tmp_path: Path) -> None:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    store.upsert_user_fact(_fact("hobby", "noise", 0.3))
    # min=0.0 disables dropping; uncertain_below=0.0 disables marking.
    retr = MemoryRetriever(store=store, index=None, min_fact_confidence=0.0, uncertain_below=0.0)
    ctx = retr.build_context(query="x")
    by_key = {f.key: f for f in ctx.user_facts}
    assert "hobby" in by_key
    assert by_key["hobby"].uncertain is False


def test_env_overrides_thresholds(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IRIS_MEMORY_MIN_FACT_CONFIDENCE", "0.55")
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    store.upsert_user_fact(_fact("interest", "maybe", 0.5))
    store.upsert_user_fact(_fact("name", "sure", 0.9))
    retr = MemoryRetriever(store=store, index=None)  # picks up env default
    ctx = retr.build_context(query="x")
    keys = {f.key for f in ctx.user_facts}
    assert "interest" not in keys  # 0.5 < 0.55 → dropped under the override
    assert "name" in keys
