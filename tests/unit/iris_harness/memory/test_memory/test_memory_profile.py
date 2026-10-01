"""Behavioral tests for the DB-backed user profile."""

from __future__ import annotations

import tempfile
from pathlib import Path

from iris_harness.memory.profile import UserProfile
from iris_harness.memory.store import MemoryStore


def _profile() -> UserProfile:
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(db_path=Path(tmp) / "test.db")
        return UserProfile(store=store)


def test_upsert_and_retrieve_fact() -> None:
    profile = _profile()
    profile.upsert("name", "Alice")

    assert profile.get("name") == "Alice"


def test_upsert_increments_confidence_on_repeat() -> None:
    profile = _profile()
    f1 = profile.upsert("timezone", "UTC", confidence=0.8)
    f2 = profile.upsert("timezone", "UTC")

    assert f2.confidence > f1.confidence
    assert f2.times_confirmed == 2


def test_remove_fact() -> None:
    profile = _profile()
    profile.upsert("hobby", "chess")
    removed = profile.remove("hobby")

    assert removed
    assert profile.get("hobby") is None


def test_remove_nonexistent_returns_false() -> None:
    profile = _profile()
    assert not profile.remove("does_not_exist")


def test_all_facts_returns_persisted_entries() -> None:
    profile = _profile()
    profile.upsert("hobby", "chess")
    profile.upsert("interest", "ontologies")

    facts = profile.all_facts()
    keys = {f.key for f in facts}
    assert {"hobby", "interest"} <= keys


def test_to_summary_string_returns_key_value_pairs() -> None:
    profile = _profile()
    profile.upsert("name", "Carol")
    summary = profile.to_summary_string()

    assert "name=Carol" in summary
