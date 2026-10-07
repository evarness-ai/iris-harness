"""Tests for ``EmailStore`` — persistence over ``data/email.db``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_personal.email.contracts import EmailAttachment, EmailMessage
from iris_personal.email.store import EmailStore


def _now() -> datetime:
    return datetime.now(UTC)


def _make_message(
    *,
    id: str = "gmail-msg-1",
    provider: str = "gmail",
    account_id: str = "gmail:user@gmail.com",
    thread_id: str | None = "thr-1",
    from_address: str = "Bob <bob@example.com>",
    subject: str = "hello",
    received_at: datetime | None = None,
    snippet: str = "Hello there...",
    body_text: str | None = "Hello there, the longer body content",
    labels: tuple[str, ...] = ("INBOX",),
    attachments: tuple[EmailAttachment, ...] = (),
) -> EmailMessage:
    return EmailMessage(
        id=id,
        provider=provider,  # type: ignore[arg-type]
        account_id=account_id,
        thread_id=thread_id,
        from_address=from_address,
        to=("user@gmail.com",),
        subject=subject,
        received_at=received_at or _now(),
        snippet=snippet,
        body_text=body_text,
        body_html="<p>...</p>",
        labels=labels,
        attachments=attachments,
    )


@pytest.fixture
def store(tmp_path: Path) -> EmailStore:
    s = EmailStore(db_path=tmp_path / "email.db")
    s.ensure_schema()
    return s


# ─── Schema ───────────────────────────────────────────────────────────


def test_ensure_schema_is_idempotent(tmp_path: Path) -> None:
    s = EmailStore(db_path=tmp_path / "email.db")
    s.ensure_schema()
    s.ensure_schema()  # must not raise
    assert s.count() == 0


# ─── upsert / get ─────────────────────────────────────────────────────


def test_upsert_returns_input_message(store: EmailStore) -> None:
    m = _make_message()
    assert store.upsert(m) == m


def test_get_missing_returns_none(store: EmailStore) -> None:
    assert store.get("ghost-id") is None


def test_upsert_then_get_strips_body_fields(store: EmailStore) -> None:
    """Persisted row must not retain body_text or body_html."""
    m = _make_message(body_text="real body" * 100)
    store.upsert(m)
    fetched = store.get(m.id)
    assert fetched is not None
    assert fetched.body_text is None
    assert fetched.body_html is None
    assert fetched.snippet == "Hello there..."  # snippet survives


def test_upsert_idempotent_on_same_id(store: EmailStore) -> None:
    """Re-upserting the same id replaces the row, not duplicates it."""
    m1 = _make_message(subject="first")
    m2 = _make_message(subject="second (updated)")  # same id
    store.upsert(m1)
    store.upsert(m2)
    assert store.count() == 1
    fetched = store.get(m1.id)
    assert fetched is not None
    assert fetched.subject == "second (updated)"


def test_from_domain_auto_derived(store: EmailStore) -> None:
    """If from_domain is unset, it's derived from from_address."""
    m = _make_message(from_address="Bob <bob@chase.com>")
    # Override to make sure auto-derive runs
    m = m.model_copy(update={"from_domain": None})
    store.upsert(m)
    fetched = store.get(m.id)
    assert fetched is not None
    assert fetched.from_domain == "chase.com"


def test_attachments_roundtrip(store: EmailStore) -> None:
    atts = (
        EmailAttachment(
            filename="stmt.pdf",
            mime_type="application/pdf",
            size_bytes=42000,
            attachment_id="att-1",
        ),
        EmailAttachment(
            filename="receipt.png",
            mime_type="image/png",
            size_bytes=3000,
            attachment_id="att-2",
        ),
    )
    m = _make_message(attachments=atts)
    store.upsert(m)
    fetched = store.get(m.id)
    assert fetched is not None
    assert fetched.attachments == atts


def test_labels_and_headers_roundtrip(store: EmailStore) -> None:
    m = _make_message(labels=("INBOX", "IMPORTANT", "CATEGORY_UPDATES"))
    # Add custom headers via model_copy
    m = m.model_copy(update={"headers_subset": {"Message-ID": "<a@b>", "X-Spam": "0"}})
    store.upsert(m)
    fetched = store.get(m.id)
    assert fetched is not None
    assert fetched.labels == ("INBOX", "IMPORTANT", "CATEGORY_UPDATES")
    assert fetched.headers_subset == {"Message-ID": "<a@b>", "X-Spam": "0"}


# ─── upsert_many ──────────────────────────────────────────────────────


def test_upsert_many_inserts_all(store: EmailStore) -> None:
    msgs = [_make_message(id=f"msg-{i}") for i in range(5)]
    n = store.upsert_many(msgs)
    assert n == 5
    assert store.count() == 5


def test_upsert_many_with_empty_input(store: EmailStore) -> None:
    assert store.upsert_many([]) == 0
    assert store.count() == 0


# ─── list_recent / list_by_thread ─────────────────────────────────────


def test_list_recent_orders_descending_by_received_at(store: EmailStore) -> None:
    now = _now()
    msgs = [
        _make_message(id="old", received_at=now - timedelta(days=2)),
        _make_message(id="new", received_at=now),
        _make_message(id="mid", received_at=now - timedelta(days=1)),
    ]
    store.upsert_many(msgs)
    listed = store.list_recent("gmail:user@gmail.com", limit=10)
    assert [m.id for m in listed] == ["new", "mid", "old"]


def test_list_recent_respects_limit(store: EmailStore) -> None:
    store.upsert_many(_make_message(id=f"msg-{i}") for i in range(10))
    listed = store.list_recent("gmail:user@gmail.com", limit=3)
    assert len(listed) == 3


def test_list_recent_filters_by_since(store: EmailStore) -> None:
    now = _now()
    store.upsert_many(
        [
            _make_message(id="old", received_at=now - timedelta(days=7)),
            _make_message(id="recent", received_at=now - timedelta(hours=1)),
        ]
    )
    cutoff = now - timedelta(hours=12)
    listed = store.list_recent("gmail:user@gmail.com", since=cutoff)
    assert [m.id for m in listed] == ["recent"]


def test_list_recent_filters_by_account(store: EmailStore) -> None:
    store.upsert_many(
        [
            _make_message(id="us-1", account_id="gmail:us@gmail.com"),
            _make_message(id="in-1", account_id="gmail:in@gmail.com"),
        ]
    )
    listed = store.list_recent("gmail:us@gmail.com")
    assert [m.id for m in listed] == ["us-1"]


def test_list_by_thread(store: EmailStore) -> None:
    now = _now()
    store.upsert_many(
        [
            _make_message(id="a1", thread_id="A", received_at=now - timedelta(days=1)),
            _make_message(id="a2", thread_id="A", received_at=now),
            _make_message(id="b1", thread_id="B", received_at=now),
        ]
    )
    listed = store.list_by_thread("gmail:user@gmail.com", "A")
    assert [m.id for m in listed] == ["a1", "a2"]  # ascending within thread


# ─── mark_classified ──────────────────────────────────────────────────


def test_mark_classified_updates_columns(store: EmailStore) -> None:
    m = _make_message()
    store.upsert(m)
    store.mark_classified(m.id, category="finance.banking", confidence=0.92, sensitivity="high")
    # We don't have a public field for classified columns in EmailMessage yet;
    # verify via raw DB inspection.
    import sqlite3

    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT classified_category, classified_confidence, sensitivity "
            "FROM emails WHERE id = ?",
            (m.id,),
        ).fetchone()
    assert row["classified_category"] == "finance.banking"
    assert row["classified_confidence"] == pytest.approx(0.92)
    assert row["sensitivity"] == "high"


def test_mark_classified_on_missing_id_is_noop(store: EmailStore) -> None:
    """Stamping classification on a ghost id does not raise."""
    store.mark_classified("ghost", category="x", confidence=0.5)  # no error


def test_mark_classified_returns_true_when_row_updated(store: EmailStore) -> None:
    m = _make_message()
    store.upsert(m)
    result = store.mark_classified(m.id, category="email/shopping/apparel/gap", confidence=0.85)
    assert result is True


def test_mark_classified_returns_false_for_ghost_id(store: EmailStore) -> None:
    """Caller can detect ghost ids via the bool return without a separate get."""
    result = store.mark_classified("ghost-id", category="x", confidence=0.5)
    assert result is False


# ─── list_unclassified ────────────────────────────────────────────────


def test_list_unclassified_returns_only_unclassified(store: EmailStore) -> None:
    """Only rows where classified_category IS NULL come back."""
    acct = "gmail:user@gmail.com"
    classified = _make_message(id="m-classified", account_id=acct)
    unclassified = _make_message(id="m-unclassified", account_id=acct)
    store.upsert(classified)
    store.upsert(unclassified)
    store.mark_classified(classified.id, category="email/shopping/apparel/gap", confidence=0.85)

    result = store.list_unclassified(acct)
    ids = [m.id for m in result]
    assert ids == ["m-unclassified"]


def test_list_unclassified_orders_recent_first_respects_limit(store: EmailStore) -> None:
    from datetime import UTC, datetime, timedelta

    acct = "gmail:user@gmail.com"
    base = datetime(2026, 5, 1, tzinfo=UTC)
    for i in range(5):
        store.upsert(
            _make_message(id=f"m-{i}", account_id=acct, received_at=base + timedelta(days=i))
        )

    result = store.list_unclassified(acct, limit=3)
    assert len(result) == 3
    # Most-recent first → m-4, m-3, m-2
    assert [m.id for m in result] == ["m-4", "m-3", "m-2"]


def test_list_unclassified_filters_by_account(store: EmailStore) -> None:
    store.upsert(_make_message(id="m-a", account_id="gmail:a@b.com"))
    store.upsert(_make_message(id="m-b", account_id="gmail:other@x.com"))

    result = store.list_unclassified("gmail:a@b.com")
    assert [m.id for m in result] == ["m-a"]


# ─── triage_state state machine (ADR-0022) ────────────────────────────


def _read_triage_state(store: EmailStore, message_id: str) -> str | None:
    import sqlite3

    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute("SELECT triage_state FROM emails WHERE id = ?", (message_id,)).fetchone()
    return row[0] if row else None


def test_mark_classified_sets_triage_state_to_classified(store: EmailStore) -> None:
    """mark_classified must also write triage_state per ADR-0022."""
    m = _make_message()
    store.upsert(m)
    store.mark_classified(m.id, category="email/shopping/apparel/gap", confidence=0.85)
    assert _read_triage_state(store, m.id) == "classified"


def test_mark_pending_review_sets_state_without_classification(store: EmailStore) -> None:
    """Queue path leaves classified_category NULL but sets triage_state."""
    import sqlite3

    m = _make_message()
    store.upsert(m)
    assert store.mark_pending_review(m.id) is True
    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            "SELECT triage_state, classified_category FROM emails WHERE id = ?",
            (m.id,),
        ).fetchone()
    assert row[0] == "pending_review"
    assert row[1] is None


def test_mark_triage_error_sets_state_for_retry(store: EmailStore) -> None:
    m = _make_message()
    store.upsert(m)
    assert store.mark_triage_error(m.id) is True
    assert _read_triage_state(store, m.id) == "error"


def test_mark_pending_review_returns_false_for_ghost(store: EmailStore) -> None:
    assert store.mark_pending_review("ghost") is False


def test_list_unclassified_excludes_pending_review(store: EmailStore) -> None:
    """Queued rows must not be re-processed by the first-pass classifier."""
    acct = "gmail:user@gmail.com"
    pending = _make_message(id="m-pending", account_id=acct)
    fresh = _make_message(id="m-fresh", account_id=acct)
    store.upsert(pending)
    store.upsert(fresh)
    store.mark_pending_review(pending.id)

    result = store.list_unclassified(acct)
    ids = [m.id for m in result]
    assert ids == ["m-fresh"]


def test_list_unclassified_includes_error_state(store: EmailStore) -> None:
    """Soft-fail rows are retry candidates and must show up again."""
    acct = "gmail:user@gmail.com"
    errored = _make_message(id="m-error", account_id=acct)
    store.upsert(errored)
    store.mark_triage_error(errored.id)

    result = store.list_unclassified(acct)
    assert [m.id for m in result] == ["m-error"]


def test_list_pending_review_returns_only_queued(store: EmailStore) -> None:
    acct = "gmail:user@gmail.com"
    queued = _make_message(id="m-queued", account_id=acct)
    classified = _make_message(id="m-classified", account_id=acct)
    fresh = _make_message(id="m-fresh", account_id=acct)
    for m in (queued, classified, fresh):
        store.upsert(m)
    store.mark_pending_review(queued.id)
    store.mark_classified(classified.id, category="email/x/y/z", confidence=0.9)

    result = store.list_pending_review(acct)
    assert [m.id for m in result] == ["m-queued"]


def test_list_pending_review_orders_recent_first_respects_limit(store: EmailStore) -> None:
    from datetime import timedelta

    acct = "gmail:user@gmail.com"
    base = _now()
    for i in range(5):
        m = _make_message(id=f"q-{i}", account_id=acct, received_at=base + timedelta(days=i))
        store.upsert(m)
        store.mark_pending_review(m.id)

    result = store.list_pending_review(acct, limit=3)
    assert [m.id for m in result] == ["q-4", "q-3", "q-2"]


def test_migration_adds_triage_state_when_missing(tmp_path: Path) -> None:
    """The migration helper must add triage_state to an existing
    emails table that pre-dates the column."""
    import sqlite3

    from iris_personal.email.store import _migrate_columns

    db_path = tmp_path / "legacy.db"
    # Build the prior-version emails table (no triage_state). Mirrors
    # what the v1 _SCHEMA_SQL would have created — we just omit the
    # one new column the migration adds.
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE emails (
                id              TEXT PRIMARY KEY,
                provider        TEXT NOT NULL,
                account_id      TEXT NOT NULL,
                thread_id       TEXT,
                from_address    TEXT NOT NULL,
                from_domain     TEXT,
                to_addrs        TEXT NOT NULL DEFAULT '[]',
                cc_addrs        TEXT NOT NULL DEFAULT '[]',
                subject         TEXT NOT NULL DEFAULT '',
                snippet         TEXT NOT NULL DEFAULT '',
                received_at     TEXT NOT NULL,
                labels          TEXT NOT NULL DEFAULT '[]',
                attachments     TEXT NOT NULL DEFAULT '[]',
                headers_subset  TEXT NOT NULL DEFAULT '{}',
                classified_category    TEXT,
                classified_confidence  REAL,
                classified_at          TEXT,
                sensitivity            TEXT,
                processed_at           TEXT,
                created_at             TEXT NOT NULL,
                updated_at             TEXT NOT NULL
            )
            """)
        cols_before = {r[1] for r in conn.execute("PRAGMA table_info(emails)").fetchall()}
    assert "triage_state" not in cols_before
    _migrate_columns(db_path)
    with sqlite3.connect(db_path) as conn:
        cols_after = {r[1] for r in conn.execute("PRAGMA table_info(emails)").fetchall()}
    assert "triage_state" in cols_after


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Running ensure_schema twice on the same DB must not error."""
    s = EmailStore(db_path=tmp_path / "email.db")
    s.ensure_schema()
    s.ensure_schema()  # second call must be a no-op, not double-ALTER
    # Insert + read still works
    m = _make_message()
    s.upsert(m)
    assert s.get(m.id) is not None


# ─── count ────────────────────────────────────────────────────────────


def test_count_unfiltered(store: EmailStore) -> None:
    store.upsert_many(_make_message(id=f"msg-{i}") for i in range(4))
    assert store.count() == 4


def test_count_filtered_by_account(store: EmailStore) -> None:
    store.upsert_many(
        [
            _make_message(id="us-1", account_id="gmail:us@gmail.com"),
            _make_message(id="us-2", account_id="gmail:us@gmail.com"),
            _make_message(id="in-1", account_id="gmail:in@gmail.com"),
        ]
    )
    assert store.count("gmail:us@gmail.com") == 2
    assert store.count("gmail:in@gmail.com") == 1


def test_list_accounts_orders_by_most_recent_activity(store: EmailStore) -> None:
    older = _now() - timedelta(days=2)
    newer = _now()
    store.upsert_many(
        [
            _make_message(id="a1", account_id="gmail:a@gmail.com", received_at=older),
            _make_message(id="b1", account_id="gmail:b@gmail.com", received_at=newer),
        ]
    )
    # Most-recently-active mailbox leads.
    assert store.list_accounts() == ["gmail:b@gmail.com", "gmail:a@gmail.com"]


def test_list_accounts_empty(store: EmailStore) -> None:
    assert store.list_accounts() == []


# ─── sync_cursors ─────────────────────────────────────────────────────


def test_cursor_get_missing_returns_none(store: EmailStore) -> None:
    assert store.get_cursor("gmail", "gmail:user@gmail.com", "history_id") is None


def test_cursor_set_and_get(store: EmailStore) -> None:
    store.set_cursor("gmail", "gmail:user@gmail.com", "history_id", "12345")
    assert store.get_cursor("gmail", "gmail:user@gmail.com", "history_id") == "12345"


def test_cursor_overwrite_on_repeat_set(store: EmailStore) -> None:
    """set_cursor on existing (provider, account, kind) replaces the value."""
    store.set_cursor("gmail", "gmail:user@gmail.com", "history_id", "100")
    store.set_cursor("gmail", "gmail:user@gmail.com", "history_id", "200")
    assert store.get_cursor("gmail", "gmail:user@gmail.com", "history_id") == "200"


def test_cursors_are_independent_per_account(store: EmailStore) -> None:
    store.set_cursor("gmail", "gmail:a@gmail.com", "history_id", "100")
    store.set_cursor("gmail", "gmail:b@gmail.com", "history_id", "200")
    assert store.get_cursor("gmail", "gmail:a@gmail.com", "history_id") == "100"
    assert store.get_cursor("gmail", "gmail:b@gmail.com", "history_id") == "200"


def test_cursors_are_independent_per_kind(store: EmailStore) -> None:
    store.set_cursor("gmail", "gmail:user@gmail.com", "history_id", "100")
    store.set_cursor("gmail", "gmail:user@gmail.com", "uid_validity", "777")
    assert store.get_cursor("gmail", "gmail:user@gmail.com", "history_id") == "100"
    assert store.get_cursor("gmail", "gmail:user@gmail.com", "uid_validity") == "777"


# ─── FTS5 (ADR-0026) ──────────────────────────────────────────────────────


def _fts_rowids_for(store: EmailStore, term: str) -> list[int]:
    """Return rowids matching the FTS query — used by the trigger tests."""
    import sqlite3

    with sqlite3.connect(store.db_path) as conn:
        return [
            r[0]
            for r in conn.execute(
                "SELECT rowid FROM emails_fts WHERE emails_fts MATCH ?", (term,)
            ).fetchall()
        ]


def test_fts_table_created_on_ensure_schema(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "email.db"
    store = EmailStore(db_path=db)
    store.ensure_schema()
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='emails_fts'"
        ).fetchone()
    assert row is not None


def test_fts_triggers_created_on_ensure_schema(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "email.db"
    store = EmailStore(db_path=db)
    store.ensure_schema()
    with sqlite3.connect(db) as conn:
        triggers = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'emails_fts%'"
            ).fetchall()
        }
    assert triggers == {"emails_fts_ai", "emails_fts_au", "emails_fts_ad"}


def test_fts_insert_trigger_indexes_new_email(store: EmailStore) -> None:
    """After upsert, the FTS table should contain the new row."""
    store.upsert(_make_message(id="msg-fts-1", subject="Northwind: May statement"))
    rowids = _fts_rowids_for(store, "northwind")
    assert len(rowids) == 1


def test_fts_update_trigger_reflects_new_subject(store: EmailStore) -> None:
    """An upsert with the same id but new subject should drop the old
    FTS row and insert a new one."""
    store.upsert(_make_message(id="msg-fts-2", subject="Northwind May statement"))
    assert _fts_rowids_for(store, "northwind") != []
    assert _fts_rowids_for(store, "shein") == []

    # Re-upsert with a totally different subject
    store.upsert(_make_message(id="msg-fts-2", subject="SHEIN summer sale"))
    assert _fts_rowids_for(store, "northwind") == []
    assert _fts_rowids_for(store, "shein") != []


def test_fts_delete_trigger_removes_row(store: EmailStore) -> None:
    import sqlite3

    store.upsert(_make_message(id="msg-fts-3", subject="SHEIN summer"))
    assert _fts_rowids_for(store, "shein") != []
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DELETE FROM emails WHERE id = ?", ("msg-fts-3",))
    assert _fts_rowids_for(store, "shein") == []


def test_fts_searches_subject_from_address_and_snippet(store: EmailStore) -> None:
    """Each of the three indexed columns is searchable."""
    store.upsert(
        _make_message(
            id="msg-fts-4",
            subject="Statement",
            from_address="alice@northwindbank.test",
            snippet="Your account balance is fine",
        )
    )
    # subject match
    assert len(_fts_rowids_for(store, "statement")) == 1
    # from_address match
    assert len(_fts_rowids_for(store, "northwindbank")) == 1
    # snippet match
    assert len(_fts_rowids_for(store, "balance")) == 1


def test_fts_backfills_existing_rows_on_first_ensure(tmp_path: Path) -> None:
    """A pre-existing emails table (no FTS table yet) gets indexed when
    ensure_schema is called by newer code. Mirrors the on-disk upgrade
    path users will hit when this commit lands."""
    import sqlite3

    db = tmp_path / "legacy.db"

    # Build the emails table manually WITHOUT the FTS surface, then
    # insert a row directly. Simulates a pre-1M database.
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE emails (
                id TEXT PRIMARY KEY, provider TEXT NOT NULL, account_id TEXT NOT NULL,
                thread_id TEXT, from_address TEXT NOT NULL, from_domain TEXT,
                to_addrs TEXT NOT NULL DEFAULT '[]', cc_addrs TEXT NOT NULL DEFAULT '[]',
                subject TEXT NOT NULL DEFAULT '', snippet TEXT NOT NULL DEFAULT '',
                received_at TEXT NOT NULL,
                labels TEXT NOT NULL DEFAULT '[]', attachments TEXT NOT NULL DEFAULT '[]',
                headers_subset TEXT NOT NULL DEFAULT '{}',
                classified_category TEXT, classified_confidence REAL, classified_at TEXT,
                sensitivity TEXT, processed_at TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            """)
        conn.execute(
            "INSERT INTO emails (id, provider, account_id, from_address, subject, snippet, "
            "received_at, created_at, updated_at) VALUES "
            "('legacy-1', 'gmail', 'gmail:u@x.com', 'a@b.com', "
            "'Legacy Northwind statement', 'legacy snippet', "
            "'2026-01-01', '2026-01-01', '2026-01-01')"
        )

    # Now `ensure_schema()` from the new EmailStore runs migrations,
    # creates the FTS surface, and backfills the legacy row.
    store = EmailStore(db_path=db)
    store.ensure_schema()

    rowids = _fts_rowids_for(store, "northwind")
    assert len(rowids) == 1


def test_ensure_schema_is_idempotent_with_fts(tmp_path: Path) -> None:
    """Re-running ensure_schema must not error (CREATE VIRTUAL TABLE IF
    NOT EXISTS + CREATE TRIGGER IF NOT EXISTS take care of duplicates)."""
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.ensure_schema()
    store.ensure_schema()
    # Still functional
    store.upsert(_make_message(id="m", subject="Northwind statement"))
    assert _fts_rowids_for(store, "northwind") != []


# ─── search() — full-text search via FTS5 ────────────────────────────────


def _seed_for_search(store: EmailStore) -> None:
    """Seed a small corpus with realistic content across multiple categories."""
    store.upsert(
        _make_message(
            id="northwind-1",
            subject="Northwind: Your May statement is ready",
            from_address="alerts@northwindbank.test",
            snippet="Account balance details inside",
        )
    )
    store.upsert(
        _make_message(
            id="gap-1",
            subject="Acme Apparel: 60% off summer sale",
            from_address="acme-apparel@email.acme-apparel.com",
            snippet="Shop summer styles today",
        )
    )
    store.upsert(
        _make_message(
            id="shopmart-1",
            subject="Memorial Day deals at Shopmart",
            from_address="shopmart@s.shopmart.com",
            snippet="Save 30 percent this weekend",
        )
    )
    # Classify two of them for the --category filter tests
    store.mark_classified(
        "northwind-1", category="email/finance/banking/northwind-savings", confidence=0.9
    )
    store.mark_classified("gap-1", category="email/shopping/apparel/acme-apparel", confidence=0.8)


def test_search_returns_matching_hits_with_bm25_rank(store: EmailStore) -> None:
    _seed_for_search(store)
    hits = store.search("northwind")
    assert len(hits) == 1
    assert hits[0].id == "northwind-1"
    # snippet contains <mark> highlighting
    assert "<mark>" in hits[0].snippet_highlighted
    # BM25 rank is a float (smaller = better)
    assert isinstance(hits[0].rank, float)


def test_search_supports_phrase_queries(store: EmailStore) -> None:
    """FTS5 phrase queries with double quotes match adjacent tokens."""
    _seed_for_search(store)
    hits = store.search('"acme apparel"')
    assert len(hits) == 1
    assert hits[0].id == "gap-1"


def test_search_supports_boolean_operators(store: EmailStore) -> None:
    """`memorial AND deals` should match the Shopmart email only."""
    _seed_for_search(store)
    hits = store.search("memorial AND deals")
    assert {h.id for h in hits} == {"shopmart-1"}


def test_search_supports_prefix_match(store: EmailStore) -> None:
    """`state*` should match `statement`."""
    _seed_for_search(store)
    hits = store.search("state*")
    assert {h.id for h in hits} == {"northwind-1"}


def test_search_filters_by_account(store: EmailStore) -> None:
    _seed_for_search(store)
    # Upsert one for a different account
    store.upsert(
        _make_message(
            id="other-1",
            account_id="gmail:other@x.com",
            subject="Northwind offer for other account",
            from_address="x@northwindbank.test",
        )
    )
    hits = store.search("northwind", account_id="gmail:user@gmail.com")
    assert {h.id for h in hits} == {"northwind-1"}


def test_search_filters_by_category_prefix(store: EmailStore) -> None:
    """--category accepts a path prefix; LIKE on classified_category."""
    _seed_for_search(store)
    # "shop summer" should match both gap-1 and the Northwind snippet? No —
    # only gap-1's subject mentions both. Use a different query to be sure.
    hits = store.search("summer OR sale OR statement OR balance", category_prefix="email/shopping/")
    # Only gap-1 (under email/shopping/) survives the filter
    assert {h.id for h in hits} == {"gap-1"}


def test_search_filters_by_since(store: EmailStore) -> None:
    from datetime import timedelta

    _seed_for_search(store)
    # All seeds use _now(); a future cutoff returns nothing
    future = _now() + timedelta(days=1)
    assert store.search("northwind", since=future) == []
    # A past cutoff returns everything matching
    past = _now() - timedelta(days=1)
    hits = store.search("northwind", since=past)
    assert {h.id for h in hits} == {"northwind-1"}


def test_search_respects_limit(store: EmailStore) -> None:
    # Insert 10 emails that all match
    for i in range(10):
        store.upsert(
            _make_message(
                id=f"northwind-{i}",
                subject=f"Northwind offer {i}",
                from_address="x@northwindbank.test",
            )
        )
    hits = store.search("northwind", limit=3)
    assert len(hits) == 3


def test_search_empty_result_for_no_match(store: EmailStore) -> None:
    _seed_for_search(store)
    # Bare alphanumeric token that won't be in any indexed field. Avoid
    # `-` in test strings because FTS5 parses `-X` as "exclude X".
    assert store.search("zzznomatch") == []


def test_search_malformed_query_raises_operational_error(store: EmailStore) -> None:
    """FTS5 syntax error → sqlite3.OperationalError; CLI maps to exit 2."""
    import sqlite3

    _seed_for_search(store)
    with pytest.raises(sqlite3.OperationalError):
        # Unbalanced quote is a known FTS5 parse error
        store.search('"unterminated')


# ─── path_counts (the digest's per-mailbox Inbox summary) ─────────────


def test_path_counts_keeps_unclassified_and_honours_the_window(store: EmailStore) -> None:
    now = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
    for mid, account, hours_ago, category in (
        ("f1", "gmail:a@x.com", 1, "email/finance/cards"),
        ("f2", "gmail:a@x.com", 2, "email/finance/cards"),
        ("u1", "gmail:a@x.com", 3, None),
        ("old", "gmail:a@x.com", 30, "email/finance/cards"),
        ("other", "gmail:b@x.com", 1, "email/updates"),
    ):
        store.upsert(
            _make_message(id=mid, account_id=account, received_at=now - timedelta(hours=hours_ago))
        )
        if category:
            store.mark_classified(mid, category=category, confidence=0.9)

    assert store.path_counts("gmail:a@x.com", since=now - timedelta(hours=24)) == {
        "email/finance/cards": 2,
        None: 1,
    }
    assert store.path_counts("gmail:a@x.com")["email/finance/cards"] == 3
    assert store.path_counts("gmail:ghost@x.com") == {}
