"""Held mail is invisible to every EmailStore read (loop-proof PR 5, stream M).

A plugin holds a synced message back until it has processed it (the email judge's
``waiting`` row). The fixture has three emails: one released (its hold row says
``judged``), one held (``waiting``), one never queued (a promo: no hold row at all).
Every public read returns the released and never-queued ones and never the held one;
``include_held`` returns all three; with no hold table at all nothing is held.

The hold table is created here with plain SQL (the core store knows only its name and
the held status), so this suite never imports the plugin that owns it.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import (
    HELD_STATUS,
    HELD_TABLE,
    CategoryFilter,
    EmailStore,
    visible_condition,
)

ACCOUNT = "gmail:owner@example.com"
RELEASED, HELD, PROMO = "m-released", "m-held", "m-promo"
ALL = {RELEASED, HELD, PROMO}
VISIBLE = {RELEASED, PROMO}
_NOW = datetime.now(UTC)


def _msg(mid: str, minutes_ago: int) -> EmailMessage:
    return EmailMessage(
        id=mid,
        provider="gmail",
        account_id=ACCOUNT,
        thread_id="t-1",
        from_address=f"Sender <{mid}@shop.example.com>",
        to=(f"{mid}-to@example.org",),
        subject=f"Quarterly invoice {mid}",
        snippet="invoice attached",
        received_at=_NOW - timedelta(minutes=minutes_ago),
    )


def _hold(db: Path, mid: str, status: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {HELD_TABLE} "
            "(message_id TEXT PRIMARY KEY, status TEXT NOT NULL)"
        )
        conn.execute(
            f"INSERT OR REPLACE INTO {HELD_TABLE} (message_id, status) VALUES (?, ?)",  # noqa: S608
            (mid, status),
        )


def _store(tmp_path: Path, *, with_holds: bool = True) -> EmailStore:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert_many([_msg(RELEASED, 3), _msg(HELD, 2), _msg(PROMO, 1)])
    for mid in (RELEASED, HELD, PROMO):
        store.mark_classified(mid, category="Finance/Bills", confidence=0.9)
    # Leave triage state open on all three so the triage queues see them.
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE emails SET classified_category = NULL, classified_source = NULL, "
            "triage_state = NULL"
        )
    if with_holds:
        _hold(store.db_path, RELEASED, "judged")
        _hold(store.db_path, HELD, HELD_STATUS)
    return store


def _classify_all(store: EmailStore) -> None:
    for mid in ALL:
        store.mark_classified(mid, category="Finance/Bills", confidence=0.9)


@pytest.fixture
def store(tmp_path: Path) -> EmailStore:
    return _store(tmp_path)


def _ids(messages: list[EmailMessage]) -> set[str]:
    return {m.id for m in messages}


def test_get_hides_held_and_escape_hatch_reads_it(store: EmailStore) -> None:
    assert store.get(RELEASED) is not None
    assert store.get(PROMO) is not None
    assert store.get(HELD) is None
    held = store.get(HELD, include_held=True)
    assert held is not None and held.id == HELD


def test_list_reads_hide_held(store: EmailStore) -> None:
    assert _ids(store.list_recent(ACCOUNT)) == VISIBLE
    assert _ids(store.list_recent(ACCOUNT, since=_NOW - timedelta(hours=1))) == VISIBLE
    assert _ids(store.list_recent(ACCOUNT, include_held=True)) == ALL
    assert _ids(store.list_by_thread(ACCOUNT, "t-1")) == VISIBLE
    assert _ids(store.list_by_thread(ACCOUNT, "t-1", include_held=True)) == ALL
    assert _ids(store.list_by_sender_domain("example.com")) == VISIBLE
    assert _ids(store.list_unclassified(ACCOUNT)) == VISIBLE
    for mid in ALL:
        store.mark_pending_review(mid)
    assert _ids(store.list_pending_review(ACCOUNT)) == VISIBLE


def test_category_reads_and_counts_hide_held(store: EmailStore) -> None:
    _classify_all(store)
    assert _ids(store.list_by_category(ACCOUNT, "Finance")) == VISIBLE
    assert _ids(store.list_by_category(ACCOUNT, CategoryFilter(paths=("Finance",)))) == VISIBLE
    assert store.count_by_category(ACCOUNT, "Finance") == 2
    assert store.category_counts() == {"Finance/Bills": 2}
    assert store.category_counts(ACCOUNT) == {"Finance/Bills": 2}
    assert store.path_counts(ACCOUNT) == {"Finance/Bills": 2}
    assert store.count() == 2
    assert store.count(ACCOUNT) == 2
    assert store.count(ACCOUNT, include_held=True) == 3
    rows = store.sender_category_counts(ACCOUNT, "Finance")
    assert {addr for addr, *_ in rows} == {
        f"Sender <{RELEASED}@shop.example.com>",
        f"Sender <{PROMO}@shop.example.com>",
    }


def test_search_and_address_known_hide_held(store: EmailStore) -> None:
    assert {h.id for h in store.search("invoice")} == VISIBLE
    assert {h.id for h in store.search("invoice", account_id=ACCOUNT)} == VISIBLE
    assert store.address_known(f"{RELEASED}-to@example.org") is True
    assert store.address_known(f"{HELD}-to@example.org") is False


def test_account_with_only_held_mail_is_not_listed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for mid in (RELEASED, PROMO):
        _hold(store.db_path, mid, HELD_STATUS)
    assert store.list_accounts() == []
    _hold(store.db_path, PROMO, "judged")
    assert store.list_accounts() == [ACCOUNT]


def test_released_message_appears(store: EmailStore) -> None:
    assert store.get(HELD) is None
    _hold(store.db_path, HELD, "judged")
    assert store.get(HELD) is not None
    assert _ids(store.list_recent(ACCOUNT)) == ALL


def test_no_hold_table_everything_visible(tmp_path: Path) -> None:
    store = _store(tmp_path, with_holds=False)
    _classify_all(store)
    assert _ids(store.list_recent(ACCOUNT)) == ALL
    assert store.get(HELD) is not None
    assert {h.id for h in store.search("invoice")} == ALL
    assert store.count() == 3
    assert store.list_accounts() == [ACCOUNT]
    with sqlite3.connect(store.db_path) as conn:
        assert visible_condition(conn) == "1"


def test_hold_table_created_after_first_read_is_honoured(tmp_path: Path) -> None:
    """The store caches only "the table exists"; a table added later still counts."""
    store = _store(tmp_path, with_holds=False)
    assert store.get(HELD) is not None
    _hold(store.db_path, HELD, HELD_STATUS)
    assert store.get(HELD) is None


def test_visible_condition_for_raw_sql_readers(store: EmailStore) -> None:
    with sqlite3.connect(store.db_path) as conn:
        cond = visible_condition(conn, "e.id")
        rows = conn.execute(
            f"SELECT e.id FROM emails AS e WHERE {cond}"  # noqa: S608 — test SQL
        ).fetchall()
    assert {r[0] for r in rows} == VISIBLE


def test_held_filter_uses_the_status_index(store: EmailStore) -> None:
    """The hold lookup must stay an index probe, not a scan of the hold table."""
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_held_status ON {HELD_TABLE}(status)")
        plan = " | ".join(
            str(r[3])
            for r in conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM emails WHERE account_id = ? AND "  # noqa: S608
                + visible_condition(conn),
                (ACCOUNT,),
            )
        )
    assert "idx_held_status" in plan, plan


def test_semantic_index_backfill_and_event_skip_held(
    store: EmailStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Held mail is not embedded; it is indexed after release (by the release event)."""
    from iris_personal.email import semantic_index as si
    from iris_personal.email.events import EmailNewArrivedPayload

    monkeypatch.delenv("IRIS_TEST_NULL_EMBEDDINGS", raising=False)
    index = si.EmailSemanticIndex(
        persist_dir=tmp_path / "vec", embed_fn=lambda texts: [[1.0, float(len(t))] for t in texts]
    )
    assert si.backfill_semantic_index(email_store=store, index=index) == 2
    assert index.count() == 2

    # The arrival handler re-reads ids from the store: a held id is skipped.
    monkeypatch.setattr(si, "_lazy_index", lambda: index)
    monkeypatch.setattr(si, "EmailStore", lambda: store)
    held_only = EmailNewArrivedPayload(
        account_id=ACCOUNT, new_message_ids=(HELD,), count=1, fell_back_to_cold_start=False
    )
    si._handle_email_new_arrived_index(held_only)
    assert index.count() == 2
    _hold(store.db_path, HELD, "judged")
    si._handle_email_new_arrived_index(held_only)
    assert index.count() == 3
