"""Tests for the fact-store coherence doctor (diagnose + repair).

Drift is injected by writing to one home and not the others (simulating the
pre-coordinator legacy paths), then asserting diagnose() catches it and repair()
reconciles the derived homes back to the SQLite truth — never mutating the truth.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_harness.memory import coherence
from iris_harness.memory.coordinator import FactCoordinator
from iris_harness.memory.identity import loader
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.store import MemoryStore, UserFact

pytestmark = pytest.mark.real_embeddings


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(loader, "USER_MD_PATH", tmp_path / "workspace" / "USER.md")
    monkeypatch.setattr(loader, "MEMORY_DIR", tmp_path / "memory")
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    index = SemanticIndex(persist_dir=tmp_path / "chroma")
    assert index.is_ready
    return store, index


def _store_only_fact(store: MemoryStore, key: str, value: str) -> None:
    """Write straight to the SQLite truth, bypassing the coordinator (legacy path)."""
    now = datetime.now(UTC)
    store.upsert_user_fact(
        UserFact(
            key=key, value=value, confidence=0.9, source="t", first_seen=now, last_confirmed=now
        )
    )


def test_coordinated_writes_are_coherent(env) -> None:
    store, index = env
    FactCoordinator(store, index).record("a", "1", 0.9, "t")
    assert coherence.diagnose(store, index).is_coherent


def test_legacy_store_only_forget_creates_orphans_then_repair_fixes(env) -> None:
    store, index = env
    FactCoordinator(store, index).record("location", "Berlin", 0.9, "t")

    # Simulate the pre-fix bug: delete from the truth only.
    store.delete_user_fact("location")
    report = coherence.diagnose(store, index)
    assert "location" in report.index_orphans
    assert "location" in report.md_orphans
    assert not report.is_coherent

    coherence.repair(store, index)
    after = coherence.diagnose(store, index)
    assert after.is_coherent
    assert "location" not in index.fact_keys()
    assert "location" not in loader.read_auto_fact_keys()


def test_index_missing_detected_and_repaired(env) -> None:
    store, index = env
    _store_only_fact(store, "role", "engineer")  # never indexed

    report = coherence.diagnose(store, index, include_md=False)
    assert "role" in report.index_missing

    coherence.repair(store, index, reproject_md=False)
    assert "role" in index.fact_keys()


def test_repair_reprojects_when_user_md_absent(env) -> None:
    # Store-only facts with no USER.md yet (the workspace dir does not exist):
    # repair must create the projection, not crash on a missing parent dir.
    store, index = env
    _store_only_fact(store, "home_city", "Springfield")
    assert not loader.USER_MD_PATH.exists()

    coherence.repair(store, index)  # reproject_md=True by default

    assert loader.USER_MD_PATH.exists()
    assert "home_city" in loader.read_auto_fact_keys()
    assert coherence.diagnose(store, index).is_coherent


def test_md_value_mismatch_detected_and_repaired(env) -> None:
    store, index = env
    FactCoordinator(store, index).record("city", "Berlin", 0.9, "t")

    # Legacy store-only correction: the projection keeps the stale value.
    store.correct_user_fact("city", "Paris")
    report = coherence.diagnose(store, index)
    assert any(key == "city" for key, _, _ in report.md_value_mismatches)

    coherence.repair(store, index)
    assert coherence.diagnose(store, index).is_coherent
    assert loader.read_auto_fact_keys()["city"][0] == "Paris"


def test_curated_head_fact_is_not_flagged_missing(env) -> None:
    store, index = env
    # A fact the user curated by hand, above the auto block.
    loader.USER_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    loader.USER_MD_PATH.write_text(
        "# User Profile\n\n- **employer**: Acme\n\n## Auto-detected\n", encoding="utf-8"
    )
    _store_only_fact(store, "employer", "Acme")

    report = coherence.diagnose(store, index)
    assert "employer" not in report.md_missing  # curated, not a projection gap


def test_repair_never_mutates_the_store(env) -> None:
    store, index = env
    FactCoordinator(store, index).record("a", "1", 0.9, "t")
    store.delete_user_fact("a")  # truth now empty; index + md are orphaned

    before = {f.key for f in store.fetch_all_user_facts()}
    coherence.repair(store, index)
    after = {f.key for f in store.fetch_all_user_facts()}
    assert before == after == set()  # repair touched only derived homes
