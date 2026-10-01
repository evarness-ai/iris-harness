"""``EmailStore`` vendor classifications: the ``classified_source`` column, the
precedence IRIS > vendor on upsert, and the vendor backfill (promo grill, PR 2)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import VENDOR_CONFIDENCE, EmailStore

ACCOUNT = "gmail:user@gmail.com"


def _msg(
    id: str = "m-1",
    *,
    vendor_category: str | None = None,
    labels: tuple[str, ...] = ("INBOX",),
    subject: str = "hello",
) -> EmailMessage:
    return EmailMessage(
        id=id,
        provider="gmail",
        account_id=ACCOUNT,
        from_address="shop@example.com",
        subject=subject,
        received_at=datetime(2026, 9, 1, tzinfo=UTC),
        labels=labels,
        vendor_category=vendor_category,
    )


def _row(store: EmailStore, id: str) -> sqlite3.Row:
    with store._connect() as conn:
        return conn.execute(
            "SELECT classified_category, classified_confidence, classified_at, "
            "classified_source, triage_state FROM emails WHERE id = ?",
            (id,),
        ).fetchone()


@pytest.fixture
def store(tmp_path: Path) -> EmailStore:
    s = EmailStore(db_path=tmp_path / "email.db")
    s.ensure_schema()
    return s


# ─── Migration ─────────────────────────────────────────────────────────


def test_migration_adds_column_and_stamps_existing_classifications_iris(tmp_path: Path) -> None:
    """A pre-existing DB gains ``classified_source``; rows triage already classified
    become 'iris' (so no vendor write can replace them), unclassified stay NULL."""
    db = tmp_path / "email.db"
    old = EmailStore(db_path=db)
    old.ensure_schema()
    old.upsert(_msg("classified"))
    old.upsert(_msg("unclassified"))
    old.mark_classified("classified", category="email/shopping/apparel", confidence=0.9)
    with sqlite3.connect(db) as conn:  # rewind to the pre-column schema
        conn.execute("ALTER TABLE emails DROP COLUMN classified_source")

    EmailStore(db_path=db).ensure_schema()
    EmailStore(db_path=db).ensure_schema()  # idempotent

    assert _row(old, "classified")["classified_source"] == "iris"
    assert _row(old, "unclassified")["classified_source"] is None


# ─── Upsert precedence ────────────────────────────────────────────────


def test_upsert_writes_vendor_category_on_insert(store: EmailStore) -> None:
    store.upsert(_msg(vendor_category="email/promotions"))
    row = _row(store, "m-1")
    assert row["classified_category"] == "email/promotions"
    assert row["classified_source"] == "vendor"
    assert row["classified_confidence"] == VENDOR_CONFIDENCE
    assert row["classified_at"] is not None
    assert row["triage_state"] is None  # triage still owes it a verdict
    got = store.get("m-1")
    assert got is not None and got.classified_source == "vendor"


def test_upsert_without_vendor_category_leaves_row_unclassified(store: EmailStore) -> None:
    store.upsert(_msg())
    row = _row(store, "m-1")
    assert row["classified_category"] is None
    assert row["classified_source"] is None


def test_upsert_fills_vendor_on_an_existing_unclassified_row(store: EmailStore) -> None:
    store.upsert(_msg())
    store.upsert(_msg(vendor_category="email/social"))
    assert _row(store, "m-1")["classified_category"] == "email/social"


def test_upsert_never_overwrites_an_iris_classification(store: EmailStore) -> None:
    store.upsert(_msg(vendor_category="email/promotions"))
    store.mark_classified("m-1", category="email/shopping/apparel", confidence=0.9)
    store.upsert(_msg(vendor_category="email/promotions"))
    row = _row(store, "m-1")
    assert row["classified_category"] == "email/shopping/apparel"
    assert row["classified_source"] == "iris"
    assert row["classified_confidence"] == 0.9
    assert row["triage_state"] == "classified"


def test_a_changed_tab_replaces_a_stale_vendor_path(store: EmailStore) -> None:
    store.upsert(_msg(vendor_category="email/promotions"))
    store.upsert(_msg(vendor_category="email/updates"))
    row = _row(store, "m-1")
    assert row["classified_category"] == "email/updates"
    assert row["classified_source"] == "vendor"


def test_a_message_without_a_vendor_path_does_not_clear_one(store: EmailStore) -> None:
    """Re-upserting an envelope rebuilt from the store (no vendor_category) keeps it."""
    store.upsert(_msg(vendor_category="email/promotions"))
    store.upsert(_msg())
    assert _row(store, "m-1")["classified_category"] == "email/promotions"


def test_upsert_many_applies_the_same_precedence(store: EmailStore) -> None:
    store.upsert(_msg("a"))
    store.upsert(_msg("b"))
    store.mark_classified("b", category="email/work", confidence=0.8)
    store.upsert_many(
        [_msg("a", vendor_category="email/forums"), _msg("b", vendor_category="email/forums")]
    )
    assert _row(store, "a")["classified_category"] == "email/forums"
    assert _row(store, "b")["classified_category"] == "email/work"


# ─── Triage still sees vendor rows ────────────────────────────────────


def test_list_unclassified_includes_vendor_rows_until_triage_classifies(
    store: EmailStore,
) -> None:
    store.upsert(_msg("vendor", vendor_category="email/promotions"))
    store.upsert(_msg("plain"))
    store.upsert(_msg("done"))
    store.mark_classified("done", category="email/work", confidence=0.8)

    assert {m.id for m in store.list_unclassified(ACCOUNT)} == {"vendor", "plain"}

    store.mark_classified("vendor", category="email/shopping", confidence=0.7)
    assert {m.id for m in store.list_unclassified(ACCOUNT)} == {"plain"}
    row = _row(store, "vendor")
    assert row["classified_source"] == "iris"
    assert row["classified_category"] == "email/shopping"


def test_list_pending_review_includes_vendor_rows(store: EmailStore) -> None:
    store.upsert(_msg("vendor", vendor_category="email/promotions"))
    store.mark_pending_review("vendor")
    assert [m.id for m in store.list_pending_review(ACCOUNT)] == ["vendor"]
    assert store.list_unclassified(ACCOUNT) == []  # queued, not first-pass work


# ─── Backfill ─────────────────────────────────────────────────────────


def _derive(labels: tuple[str, ...]) -> str | None:
    return "email/promotions" if "PROMO" in labels else None


def test_backfill_fills_only_unclassified_and_vendor_rows(store: EmailStore) -> None:
    store.upsert(_msg("new", labels=("PROMO",)))
    store.upsert(_msg("iris", labels=("PROMO",)))
    store.mark_classified("iris", category="email/work", confidence=0.8)
    store.upsert(_msg("primary", labels=("INBOX",)))
    store.upsert(_msg("stale", labels=("PROMO",), vendor_category="email/social"))

    dry = store.backfill_vendor_categories(ACCOUNT, _derive, dry_run=True)
    assert (dry.scanned, dry.set_new, dry.replaced, dry.skipped_iris) == (4, 1, 1, 1)
    assert dry.no_vendor_category == 1
    assert dry.by_category == {"email/promotions": 2}
    assert _row(store, "new")["classified_category"] is None  # dry run wrote nothing

    done = store.backfill_vendor_categories(ACCOUNT, _derive)
    assert done.written == 2
    assert _row(store, "new")["classified_category"] == "email/promotions"
    assert _row(store, "new")["classified_source"] == "vendor"
    assert _row(store, "stale")["classified_category"] == "email/promotions"
    assert _row(store, "iris")["classified_category"] == "email/work"

    again = store.backfill_vendor_categories(ACCOUNT, _derive)
    assert again.written == 0
    assert again.unchanged == 2
