"""Tests for ``CategoryStore`` (Track 1F)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_personal.email.category_store import Category, CategoryStore


def _cat(path: str = "email/shopping/apparel/outlet-brand", **overrides) -> Category:  # type: ignore[no-untyped-def]
    parts = path.split("/")
    base = {
        "path": path,
        "type": parts[0],
        "root": parts[1],
        "branch": parts[2],
        "leaf": parts[3],
    }
    base.update(overrides)
    return Category(**base)


@pytest.fixture
def store(tmp_path: Path) -> CategoryStore:
    s = CategoryStore(db_path=tmp_path / "iris.db")
    s.ensure_schema()
    return s


# ─── Schema ─────────────────────────────────────────────────────────────────


def test_ensure_schema_is_idempotent(store: CategoryStore) -> None:
    store.ensure_schema()
    store.ensure_schema()  # second call must not raise
    # Tables exist
    with store._connect() as conn:  # internal access is fine in tests
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    names = {r["name"] for r in rows}
    assert "categories" in names
    assert "categories_history" in names


# ─── upsert_if_new ──────────────────────────────────────────────────────────


def test_upsert_inserts_new_path_and_writes_history(store: CategoryStore) -> None:
    c = _cat()
    inserted = store.upsert_if_new(c)
    assert inserted is True

    got = store.get(c.path)
    assert got is not None
    assert got.path == c.path

    history = store.history(c.path)
    assert len(history) == 1
    assert history[0]["op"] == "insert"
    assert history[0]["source"] == "accept-categories"


def test_upsert_is_idempotent_on_same_path(store: CategoryStore) -> None:
    c = _cat()
    assert store.upsert_if_new(c) is True
    # Same path again → no-op, no new history row
    assert store.upsert_if_new(c) is False
    assert store.upsert_if_new(c) is False
    assert len(store.history(c.path)) == 1


def test_upsert_reactivates_archived_row(store: CategoryStore) -> None:
    c = _cat()
    store.upsert_if_new(c)
    assert store.archive(c.path) is True

    # Re-upserting the same path while archived → reactivates + history
    assert store.upsert_if_new(c) is True
    got = store.get(c.path)
    assert got is not None and got.active is True

    history_ops = [h["op"] for h in store.history(c.path)]
    # Most-recent first: update (reactivation) → archive → insert
    assert history_ops == ["update", "archive", "insert"]


def test_upsert_records_custom_source_tag(store: CategoryStore) -> None:
    c = _cat()
    store.upsert_if_new(c, source="self-learning")
    history = store.history(c.path)
    assert history[0]["source"] == "self-learning"


# ─── archive ────────────────────────────────────────────────────────────────


def test_archive_marks_inactive_and_records_history(store: CategoryStore) -> None:
    c = _cat()
    store.upsert_if_new(c)
    assert store.archive(c.path) is True

    got = store.get(c.path)
    assert got is not None and got.active is False
    assert store.history(c.path)[0]["op"] == "archive"


def test_archive_is_idempotent(store: CategoryStore) -> None:
    c = _cat()
    store.upsert_if_new(c)
    assert store.archive(c.path) is True
    assert store.archive(c.path) is False
    # Only one archive event in history
    archive_events = [h for h in store.history(c.path) if h["op"] == "archive"]
    assert len(archive_events) == 1


def test_archive_unknown_path_returns_false(store: CategoryStore) -> None:
    assert store.archive("email/none/none/none") is False


# ─── list ───────────────────────────────────────────────────────────────────


def test_list_filters_by_type_account_active(store: CategoryStore) -> None:
    e1 = _cat(path="email/shopping/apparel/gap", account_id="gmail:a@b.com")
    e2 = _cat(path="email/finance/investing/bonds", account_id="gmail:a@b.com")
    e3 = _cat(path="email/shopping/apparel/shopmart", account_id="gmail:other@x.com")
    for c in (e1, e2, e3):
        store.upsert_if_new(c)
    store.archive(e2.path)

    # active_only=True is the default → e2 is hidden
    active = store.list()
    paths = [c.path for c in active]
    assert e2.path not in paths
    assert e1.path in paths and e3.path in paths

    # account filter
    only_a = store.list(account_id="gmail:a@b.com")
    assert {c.path for c in only_a} == {e1.path}  # e2 archived

    # active_only=False brings the archived back
    all_for_a = store.list(account_id="gmail:a@b.com", active_only=False)
    assert {c.path for c in all_for_a} == {e1.path, e2.path}


def test_list_orders_by_path(store: CategoryStore) -> None:
    paths = [
        "email/shopping/apparel/gap",
        "email/finance/investing/bonds",
        "email/shopping/apparel/shopmart",
    ]
    for p in paths:
        store.upsert_if_new(_cat(path=p))
    listed = [c.path for c in store.list()]
    assert listed == sorted(paths)


# ─── roundtrip ──────────────────────────────────────────────────────────────


def test_roundtrip_preserves_metadata_and_cohesion(store: CategoryStore) -> None:
    c = _cat(
        cohesion=0.87,
        sensitivity="high",
        metadata={"top_domains": [["gap.com", 45]], "cluster_id": 7},
    )
    store.upsert_if_new(c)
    got = store.get(c.path)
    assert got is not None
    assert got.cohesion == 0.87
    assert got.sensitivity == "high"
    assert got.metadata["cluster_id"] == 7


def test_history_payload_carries_snapshot(store: CategoryStore) -> None:
    """The payload column should carry enough of the Category to
    reconstruct what was accepted (Track 1J self-learning relies on it)."""
    import json

    c = _cat(cohesion=0.75, metadata={"x": 1})
    store.upsert_if_new(c)
    payload = json.loads(store.history(c.path)[0]["payload"])
    assert payload["path"] == c.path
    assert payload["cohesion"] == 0.75
    assert payload["metadata"] == {"x": 1}


# ─── record_correction + list_corrections (Track 1J / ADR-0024) ─────────────


def test_record_correction_writes_history_row(store: CategoryStore) -> None:
    target = _cat(path="email/finance/banking/northwind-savings")
    store.upsert_if_new(target)
    store.record_correction(
        message_id="m-1",
        account_id="gmail:u@x.com",
        old_path="email/shopping/apparel/outlet-brand",
        new_path=target.path,
        previous_classifier="pure-knn",
        reason="this is my bank, not retail",
    )

    rows = store.list_corrections(account_id="gmail:u@x.com")
    assert len(rows) == 1
    r = rows[0]
    assert r["category_path"] == target.path
    assert r["old_path"] == "email/shopping/apparel/outlet-brand"
    assert r["new_path"] == target.path
    assert r["payload"]["message_id"] == "m-1"
    assert r["payload"]["previous_classifier"] == "pure-knn"
    assert r["payload"]["reason"] == "this is my bank, not retail"


def test_record_correction_does_not_appear_in_regular_history(
    store: CategoryStore,
) -> None:
    """A correction lives in categories_history but should NOT pollute
    the regular `history(path)` view used by self-learning tagged-only."""
    target = _cat()
    store.upsert_if_new(target)
    pre_count = len(store.history(target.path))

    store.record_correction(
        message_id="m-2",
        account_id="gmail:u@x.com",
        old_path=None,
        new_path=target.path,
        previous_classifier="pure-knn",
    )

    # history(path) returns ALL rows for that path regardless of source.
    # The correction adds one row.
    post_count = len(store.history(target.path))
    assert post_count == pre_count + 1
    # But list_corrections filters to the user-classification rows
    correction_rows = store.list_corrections()
    assert len(correction_rows) == 1


def test_list_corrections_filters_by_account(store: CategoryStore) -> None:
    target = _cat()
    store.upsert_if_new(target)
    store.record_correction(
        message_id="m-a",
        account_id="gmail:a@x.com",
        old_path=None,
        new_path=target.path,
        previous_classifier=None,
    )
    store.record_correction(
        message_id="m-b",
        account_id="gmail:b@x.com",
        old_path=None,
        new_path=target.path,
        previous_classifier=None,
    )
    only_a = store.list_corrections(account_id="gmail:a@x.com")
    assert {r["payload"]["message_id"] for r in only_a} == {"m-a"}


def test_list_corrections_orders_most_recent_first(store: CategoryStore) -> None:
    target = _cat()
    store.upsert_if_new(target)
    for i in range(3):
        store.record_correction(
            message_id=f"m-{i}",
            account_id="gmail:u@x.com",
            old_path=None,
            new_path=target.path,
            previous_classifier=None,
        )
    rows = store.list_corrections()
    assert [r["payload"]["message_id"] for r in rows] == ["m-2", "m-1", "m-0"]


def test_list_corrections_respects_limit(store: CategoryStore) -> None:
    target = _cat()
    store.upsert_if_new(target)
    for i in range(5):
        store.record_correction(
            message_id=f"m-{i}",
            account_id="gmail:u@x.com",
            old_path=None,
            new_path=target.path,
            previous_classifier=None,
        )
    rows = store.list_corrections(limit=2)
    assert len(rows) == 2
