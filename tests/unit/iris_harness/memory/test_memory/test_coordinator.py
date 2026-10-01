"""Tests for FactCoordinator — the single write-seam across the three fact-homes.

The regression these lock in: before the coordinator, the *write* path fanned out
to store + index + USER.md but *forget/correct* only touched the store, silently
desyncing the derived homes. Every test asserts all three homes move together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.memory.coordinator import FactCoordinator
from iris_harness.memory.identity import loader
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.store import MemoryStore

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


def test_record_fans_out_to_all_three_homes(env) -> None:
    store, index = env
    FactCoordinator(store, index).record("location", "Springfield", 0.9, "test")

    assert store.fetch_user_fact("location").value == "Springfield"  # truth
    assert "location" in index.fact_keys()  # derived index
    assert "location" in loader.read_auto_fact_keys()  # derived projection


def test_forget_removes_from_all_three_homes(env) -> None:
    store, index = env
    coord = FactCoordinator(store, index)
    coord.record("location", "Springfield", 0.9, "test")

    assert coord.forget("location") is True
    assert store.fetch_user_fact("location") is None
    assert "location" not in index.fact_keys()
    assert "location" not in loader.read_auto_fact_keys()


def test_correct_updates_value_in_all_three_homes(env) -> None:
    store, index = env
    coord = FactCoordinator(store, index)
    coord.record("location", "Berlin", 0.9, "test")

    coord.correct("location", "New York")
    assert store.fetch_user_fact("location").value == "New York"
    assert loader.read_auto_fact_keys()["location"][0] == "New York"


def test_restore_re_derives_homes(env) -> None:
    store, index = env
    coord = FactCoordinator(store, index)
    coord.record("location", "Berlin", 0.9, "test")
    coord.forget("location")

    assert coord.restore("location") == "Berlin"
    assert store.fetch_user_fact("location").value == "Berlin"
    assert "location" in index.fact_keys()
    assert "location" in loader.read_auto_fact_keys()


def test_project_md_false_skips_user_md(env) -> None:
    store, index = env
    FactCoordinator(store, index, project_md=False).record("x", "1", 0.9, "test")

    assert store.fetch_user_fact("x").value == "1"
    assert "x" in index.fact_keys()
    assert loader.read_auto_fact_keys() == {}  # projection suppressed


def test_record_without_index_still_writes_store_and_md(env) -> None:
    store, _ = env
    FactCoordinator(store, None).record("role", "engineer", 0.9, "test")

    assert store.fetch_user_fact("role").value == "engineer"
    assert "role" in loader.read_auto_fact_keys()
