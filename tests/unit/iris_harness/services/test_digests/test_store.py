"""The stored-digest store (loop-proof plan PR 2): the one full web copy."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.digests import (
    DigestStore,
    FailedSection,
    default_db_path,
    shared_digest_store,
)


def test_save_then_get_round_trips_body_and_failures(tmp_path: Path) -> None:
    store = DigestStore(db_path=tmp_path / "d.db")
    failed = (FailedSection(name="portfolio", title="Portfolio", reason="timeout"),)
    saved = store.save(
        "## Bills due\n- AT&T",
        heartbeat="morning-digest",
        skill_id="morning-brief",
        subject="Morning brief",
        failed_sections=failed,
    )
    got = store.get(saved.id)
    assert got == saved
    assert got is not None and got.failed_sections == failed
    assert got.as_dict()["failed_sections"] == [
        {"name": "portfolio", "title": "Portfolio", "reason": "timeout"}
    ]


def test_latest_is_the_newest_and_can_filter_by_skill(tmp_path: Path) -> None:
    store = DigestStore(db_path=tmp_path / "d.db")
    first = store.save("one", skill_id="morning-brief")
    second = store.save("two", skill_id="evening-brief")
    assert store.latest() == second
    assert store.latest(skill_id="morning-brief") == first
    assert store.latest(skill_id="nope") is None


def test_an_empty_store_has_no_latest(tmp_path: Path) -> None:
    store = DigestStore(db_path=tmp_path / "d.db")
    assert store.latest() is None
    assert store.get("0" * 32) is None
    assert store.list() == ()


def test_old_rows_are_pruned_past_keep(tmp_path: Path) -> None:
    store = DigestStore(db_path=tmp_path / "d.db", keep=3)
    ids = [store.save(f"digest {i}").id for i in range(5)]
    listed = [d.id for d in store.list(limit=10)]
    assert listed == list(reversed(ids[-3:]))


def test_the_shared_store_lives_in_the_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    assert default_db_path() == tmp_path / "digests.db"
    store = shared_digest_store()
    assert store.db_path == tmp_path / "digests.db"
    assert shared_digest_store() is store
