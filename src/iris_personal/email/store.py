"""SQLite-backed CRUD for emails + per-account sync cursors.

Persists ``EmailMessage`` rows from ``iris_personal.email.contracts`` to
``data/email.db``. Body fields (``body_text``, ``body_html``) are
**not** persisted per canonical doc §3.1 — only the snippet survives.

Soft FK from ``emails.account_id`` to ``email_accounts.id`` (which
lives in ``data/iris.db``) — SQLite does not enforce FKs across DB
files; integrity is the caller's responsibility.

Schema lives in ``ensure_schema()`` for now; if/when migrations matter
we'll split it out (same pattern as ``iris_harness.services.tasks.store``).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parseaddr
from pathlib import Path
from typing import Any

from iris_harness.sdk.persistence import connect, data_path, ensure_columns
from iris_personal.email.contracts import EmailAttachment, EmailMessage


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime | None) -> str | None:
    """ISO text for a column or a comparison, in UTC when the time is zone-aware.

    Times are compared as TEXT in SQL, so every stored and compared value must carry
    the same offset. The providers store UTC, but callers build windows in local time
    (``datetime.now().astimezone() - timedelta(days=7)``), and "…T14:41-05:00" sorts
    before "…T19:41+00:00": the owner's "last 7 days" came back 5 hours wide (24
    extra promotions, 2026-09-22 end-to-end run).
    """
    if dt is None:
        return None
    return (dt.astimezone(UTC) if dt.tzinfo is not None else dt).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def domain_of_address(from_address: str) -> str | None:
    """Bare lowercase domain from ``"Name <foo@bar.com>"`` or ``"foo@bar.com"``.

    Parsed as an address, not split at the first ``@``: a bank that mails from
    ``"alerts@bank.example" <alerts@bank.example>`` was read by the old split as
    ``bank.example"``, so those alerts never matched the registry (2026-09-24).
    Returns None if there is no address.
    """
    _, address = parseaddr(from_address or "")
    if "@" not in address:
        address = from_address or ""
    if "@" not in address:
        return None
    domain = address.rsplit("@", 1)[1].strip().strip('"<>').split()[0] if address else ""
    return domain.strip('"<>()').lower() or None


_derive_domain = domain_of_address


def _attachments_to_json(attachments: tuple[EmailAttachment, ...]) -> str:
    return json.dumps([a.model_dump(mode="json") for a in attachments])


def _attachments_from_json(raw: str | None) -> tuple[EmailAttachment, ...]:
    if not raw:
        return ()
    items = json.loads(raw)
    return tuple(EmailAttachment(**item) for item in items)


def _migrate_columns(db_path: Path) -> None:
    """Additive column migrations for existing databases.

    ``ensure_schema()`` calls this BEFORE running ``_SCHEMA_SQL`` so that index statements in
    ``_SCHEMA_SQL`` which reference newly-added columns don't fail on a partially-migrated
    emails table. On a fresh database the tables don't exist yet: there is nothing to migrate,
    and ``_SCHEMA_SQL`` creates them with the new columns included.

    Each ``ensure_columns`` call adds what is missing on a connection of its own under
    ``BEGIN IMMEDIATE``: this used to read ``PRAGMA table_info`` and then ``ALTER``, and two
    processes opening an older ``email.db`` at once (the API, the CLI, a heartbeat) both saw the
    column missing, so the loser's ``ALTER`` raised ``duplicate column name`` (#201). ALTER TABLE ADD
    COLUMN is the only schema change SQLite supports cheaply; we don't drop or rename columns here.
    """
    # The provider labels a message had when it was trashed (JSON), so a restore can put back what
    # the trash removed. Rows from before stay NULL.
    ensure_columns(db_path, "trashed", {"labels_before": "TEXT"})
    ensure_columns(
        db_path,
        "emails",
        {"triage_state": "TEXT", "classified_source": "TEXT"},
        on_added=_stamp_iris_classifications,
    )
    # Future migrations append here.


def _stamp_iris_classifications(conn: sqlite3.Connection, added: list[str]) -> None:
    """``ensure_columns`` callback: every classification written before ``classified_source``
    existed came from IRIS (triage or a user correction). Stamp them so a vendor category arriving
    on the next sync never overwrites one. Runs once, in the transaction that added the column."""
    if "classified_source" in added:
        conn.execute(
            "UPDATE emails SET classified_source = 'iris' "
            "WHERE classified_category IS NOT NULL AND classified_source IS NULL"
        )


def _setup_fts(conn: sqlite3.Connection) -> None:
    """Ensure the FTS5 virtual table + sync triggers exist; backfill
    once when the table is first created.

    Per ADR-0026 §3: detection via sqlite_master; if emails_fts is
    missing we create it and run ``INSERT INTO emails_fts(emails_fts)
    VALUES('rebuild')`` to populate from the existing rows. The triggers
    keep it in sync afterward. All DDL is IF NOT EXISTS — safe to
    re-run on every ``ensure_schema``.
    """
    existed = (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'emails_fts'"
        ).fetchone()
        is not None
    )
    conn.executescript(_FTS_SQL)
    if not existed:
        # First-time creation — populate from any pre-existing emails rows.
        # Idempotent; safe to re-run if a manual rebuild is ever needed.
        conn.execute("INSERT INTO emails_fts(emails_fts) VALUES('rebuild')")


#: ``classified_source`` values. IRIS's own verdict always wins over the mailbox's.
IRIS_SOURCE = "iris"
VENDOR_SOURCE = "vendor"
#: Fixed confidence for a vendor classification: the provider's bucket is a hint,
#: not a verdict, and triage still reclassifies the row.
VENDOR_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# Held mail
# ---------------------------------------------------------------------------
#
# A plugin may take a freshly synced message in for processing before IRIS acts on
# it (the email judge queues non-promo mail until it has read it). Until that plugin
# releases it the message is *held*: every read here skips it, so no digest, search,
# tool, index or finance reader sees half-processed mail. The hold lives in the
# plugin's own table beside ``emails`` (same message id); the core only knows its
# name and the status that means "held". When the table does not exist (the plugin
# is off, an older database) nothing is held.

#: The table that holds messages back, and the status of a held row.
HELD_TABLE = "email_judgments"
HELD_STATUS = "waiting"


def held_table_exists(conn: sqlite3.Connection) -> bool:
    """True when the database has a hold table (so some mail may be held)."""
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (HELD_TABLE,)
        ).fetchone()
        is not None
    )


def _held_condition(column: str) -> str:
    # Uncorrelated: SQLite builds the (small) held-id set once per statement from the
    # hold table's status index, then probes it per row.
    return (
        f"{column} NOT IN (SELECT message_id FROM {HELD_TABLE} "  # noqa: S608 — constants
        f"WHERE status = '{HELD_STATUS}')"
    )


def visible_condition(conn: sqlite3.Connection, column: str = "id") -> str:
    """A SQL condition that is true for every message not held back.

    For readers that query ``email.db`` with their own SQL: splice it into the WHERE
    clause (``... AND {visible_condition(conn)}``). ``column`` is the message-id column
    as the query names it (``e.id`` under an alias). ``1`` when nothing can be held.
    """
    return _held_condition(column) if held_table_exists(conn) else "1"


@dataclass(frozen=True)
class CategoryFilter:
    """Emails filed under any of ``paths`` (the path itself or anything below it), or
    carrying any of the provider ``labels``.

    The labels are how a mailbox's own bucket still matches after IRIS triage moved
    the row to a path of its own: the caller maps its category to the provider labels
    (the provider plugin owns that vocabulary); the store only compares strings.
    """

    paths: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()


def _like_escape(value: str) -> str:
    # '!' as the escape char avoids Python/SQL backslash gymnastics; % and _ are
    # escaped so a category name can never widen the match.
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _category_clause(
    category: str | CategoryFilter, column_prefix: str = ""
) -> tuple[str, list[Any]]:
    """The SQL condition (and its parameters) for a category filter.

    A plain string is the original prefix match (``Newsletters`` matches
    ``Newsletters/AI``). A :class:`CategoryFilter` matches each path exactly or as a
    parent (``email/promotions`` and ``email/promotions/…``, not ``email/promotionsx``),
    or any of its labels in the row's ``labels`` JSON. An empty filter matches nothing.
    """
    col = f"{column_prefix}classified_category"
    if isinstance(category, str):
        return f"{col} LIKE ? ESCAPE '!'", [f"{_like_escape(category)}%"]
    terms: list[str] = []
    params: list[Any] = []
    for path in category.paths:
        terms.append(f"({col} = ? OR {col} LIKE ? ESCAPE '!')")
        params.extend([path, f"{_like_escape(path.rstrip('/'))}/%"])
    if category.labels:
        marks = ", ".join("?" for _ in category.labels)
        terms.append(
            f"EXISTS (SELECT 1 FROM json_each({column_prefix}labels) "  # noqa: S608
            f"WHERE value IN ({marks}))"
        )
        params.extend(category.labels)
    if not terms:
        return "0", []
    return "(" + " OR ".join(terms) + ")", params


@dataclass(frozen=True)
class VendorBackfillResult:
    """What ``EmailStore.backfill_vendor_categories`` did (or would do, on a dry run)."""

    scanned: int = 0
    set_new: int = 0  # unclassified rows that gained a vendor path
    replaced: int = 0  # vendor rows whose path changed (the tab moved)
    unchanged: int = 0  # vendor rows already holding the derived path
    skipped_iris: int = 0  # IRIS classified these; never overwritten
    no_vendor_category: int = 0  # the mapping gives no path (e.g. a Primary-tab mail)
    by_category: dict[str, int] = field(default_factory=dict)  # set_new + replaced

    @property
    def written(self) -> int:
        return self.set_new + self.replaced


def _row_to_message(row: sqlite3.Row) -> EmailMessage:
    """Reconstruct an EmailMessage from a DB row.

    ``body_text`` and ``body_html`` always come back as None — they're
    not stored.
    """
    return EmailMessage(
        id=row["id"],
        provider=row["provider"],
        account_id=row["account_id"],
        thread_id=row["thread_id"],
        from_address=row["from_address"],
        from_domain=row["from_domain"],
        to=tuple(json.loads(row["to_addrs"] or "[]")),
        cc=tuple(json.loads(row["cc_addrs"] or "[]")),
        subject=row["subject"] or "",
        received_at=_parse_dt(row["received_at"]) or _utc_now(),
        snippet=row["snippet"] or "",
        body_text=None,
        body_html=None,
        labels=tuple(json.loads(row["labels"] or "[]")),
        attachments=_attachments_from_json(row["attachments"]),
        headers_subset=json.loads(row["headers_subset"] or "{}"),
        classified_category=row["classified_category"],
        classified_source=row["classified_source"],
    )


def _message_to_row(message: EmailMessage) -> dict[str, Any]:
    """Serialize an EmailMessage for INSERT/REPLACE. Body fields dropped.

    A message carrying a ``vendor_category`` lands as a ``"vendor"`` classification;
    ``_UPSERT_SQL`` decides whether it may replace what an existing row holds.
    """
    from_domain = message.from_domain or _derive_domain(message.from_address)
    now = _utc_now()
    vendor = message.vendor_category
    return {
        "id": message.id,
        "provider": message.provider,
        "account_id": message.account_id,
        "thread_id": message.thread_id,
        "from_address": message.from_address,
        "from_domain": from_domain,
        "to_addrs": json.dumps(list(message.to)),
        "cc_addrs": json.dumps(list(message.cc)),
        "subject": message.subject,
        "snippet": message.snippet,
        "received_at": _iso(message.received_at),
        "labels": json.dumps(list(message.labels)),
        "attachments": _attachments_to_json(message.attachments),
        "headers_subset": json.dumps(message.headers_subset),
        "classified_category": vendor,
        "classified_confidence": VENDOR_CONFIDENCE if vendor else None,
        "classified_at": _iso(now) if vendor else None,
        "classified_source": VENDOR_SOURCE if vendor else None,
        "sensitivity": None,
        "triage_state": None,
        "processed_at": None,
        "created_at": _iso(now),
        "updated_at": _iso(now),
    }


@dataclass(frozen=True)
class SearchHit:
    """One row of an FTS search result (Track 1M / ADR-0026 §5).

    ``snippet_highlighted`` carries the FTS5 ``snippet()`` output with
    ``<mark>...</mark>`` tags around the matched terms; the CLI's
    Rich renderer styles those.
    """

    id: str
    subject: str
    from_address: str
    from_domain: str | None
    received_at: datetime
    classified_category: str | None
    snippet_highlighted: str
    rank: float  # FTS5 bm25(); smaller = better match


@dataclass
class EmailStore:
    """SQLite-backed CRUD over ``data/email.db``.

    Schema is created lazily via ``ensure_schema()``.

    Every read that hands messages (or counts of them) to a caller skips *held* mail
    (see ``HELD_TABLE``). ``get``, ``list_recent``, ``list_by_thread`` and ``count``
    take ``include_held=True`` for the plugin that holds a message and must read it;
    writes, the trash ledger and the maintenance passes (``backfill_vendor_categories``,
    ``repair_from_domains``) always see every row.
    """

    db_path: Path = field(default_factory=lambda: data_path("email.db"))
    # Once the hold table exists it stays; until then every read re-checks (cheap).
    _held_table_seen: bool = field(default=False, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Migrate first — _SCHEMA_SQL's CREATE INDEX statements reference columns the
        # migration may need to add.
        _migrate_columns(self.db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            _setup_fts(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = connect(self.db_path, row_factory=sqlite3.Row)
        return conn

    def _visible(
        self, conn: sqlite3.Connection, column: str = "id", *, include_held: bool = False
    ) -> str:
        """The not-held condition for this store's reads (see :func:`visible_condition`);
        ``1`` when ``include_held`` or when nothing can be held."""
        if include_held:
            return "1"
        if not self._held_table_seen:
            if not held_table_exists(conn):
                return "1"
            self._held_table_seen = True
        return _held_condition(column)

    # ------------------------------------------------------------------
    # Emails
    # ------------------------------------------------------------------

    def upsert(self, message: EmailMessage) -> EmailMessage:
        """INSERT OR REPLACE one message. Body fields are NOT persisted.

        Idempotent: same (provider, id) → same final row state. Returns
        the input message unchanged for chaining (body fields included).

        A ``message.vendor_category`` is stored as a ``"vendor"`` classification
        when the row is unclassified or vendor-classified (a changed tab replaces a
        stale vendor path); a row IRIS classified is never overwritten, and
        ``triage_state`` is left alone so triage still classifies vendor rows.
        """
        with self._connect() as conn:
            conn.execute(_UPSERT_SQL, _message_to_row(message))
        return message

    def upsert_many(self, messages: Iterable[EmailMessage]) -> int:
        """Bulk upsert. Returns the number of rows touched."""
        rows = [_message_to_row(m) for m in messages]
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(_UPSERT_SQL, rows)
        return len(rows)

    # ------------------------------------------------------------------
    # Trash (ADR-0118 step 5)
    # ------------------------------------------------------------------

    def move_to_trashed(
        self,
        messages: Iterable[EmailMessage],
        *,
        batch_id: str,
        labels_before: Mapping[str, Sequence[str]] | None = None,
    ) -> int:
        """Record messages the provider has trashed and drop them from ``emails``.

        One transaction: the ledger row and the removal land together, so a message
        is never both gone and forgotten. The FTS delete trigger keeps search in step.
        ``labels_before`` is what the provider reported just before the trash (the
        stored ``labels`` can be days stale), kept for ``trashed_labels``.
        """
        now = datetime.now(UTC).isoformat()
        before = labels_before or {}
        rows = [
            (
                m.id,
                m.account_id,
                m.subject,
                m.from_address,
                now,
                batch_id,
                json.dumps(list(before[m.id])) if m.id in before else None,
            )
            for m in messages
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO trashed "
                "(id, account_id, subject, from_address, trashed_at, batch_id, labels_before) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.executemany("DELETE FROM emails WHERE id = ?", [(r[0],) for r in rows])
        return len(rows)

    def trashed(self, message_ids: Iterable[str] | None = None) -> list[tuple[str, str, str, str]]:
        """``(id, account_id, subject, from_address)`` for the given trashed ids, or for
        the most recent trash batch when none are given (what "undo that" means)."""
        with self._connect() as conn:
            if message_ids is None:
                last = conn.execute(
                    "SELECT batch_id FROM trashed ORDER BY trashed_at DESC LIMIT 1"
                ).fetchone()
                if last is None:
                    return []
                rows = conn.execute(
                    "SELECT id, account_id, subject, from_address FROM trashed WHERE batch_id = ?",
                    (last[0],),
                ).fetchall()
            else:
                ids = list(message_ids)
                if not ids:
                    return []
                # Only "?" placeholders are interpolated; the ids are bound parameters.
                marks = ",".join("?" for _ in ids)
                sql = f"SELECT id, account_id, subject, from_address FROM trashed WHERE id IN ({marks})"  # noqa: S608
                rows = conn.execute(sql, ids).fetchall()
        return [(str(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in rows]

    def trashed_labels(self, message_ids: Iterable[str]) -> dict[str, list[str]]:
        """``{id: labels before the trash}`` for trashed ids that recorded them."""
        ids = list(message_ids)
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        # Only "?" placeholders are interpolated; the ids are bound parameters.
        sql = f"SELECT id, labels_before FROM trashed WHERE id IN ({marks})"  # noqa: S608
        with self._connect() as conn:
            rows = conn.execute(sql, ids).fetchall()
        return {str(r[0]): list(json.loads(r[1])) for r in rows if r[1]}

    def restore_from_trashed(self, messages: Iterable[EmailMessage]) -> int:
        """Put restored messages back in ``emails`` and forget them as trashed."""
        msgs = list(messages)
        with self._connect() as conn:
            conn.executemany(_UPSERT_SQL, [_message_to_row(m) for m in msgs])
            conn.executemany("DELETE FROM trashed WHERE id = ?", [(m.id,) for m in msgs])
        return len(msgs)

    def get(self, message_id: str, *, include_held: bool = False) -> EmailMessage | None:
        """Return a single EmailMessage by primary key; body fields are None.

        A held message reads as absent unless ``include_held`` (the plugin holding it
        reads it this way to process it).
        """
        with self._connect() as conn:
            visible = self._visible(conn, include_held=include_held)
            row = conn.execute(
                f"SELECT * FROM emails WHERE id = ? AND {visible}",  # noqa: S608
                (message_id,),
            ).fetchone()
        return _row_to_message(row) if row else None

    def set_labels(self, message_id: str, labels: Sequence[str]) -> bool:
        """Replace one row's provider labels (a label change read back from the mailbox).
        Returns ``False`` when the row does not exist."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE emails SET labels = ?, updated_at = ? WHERE id = ?",
                (json.dumps(list(labels)), _iso(_utc_now()), message_id),
            )
            return cursor.rowcount > 0

    def list_recent(
        self,
        account_id: str,
        *,
        limit: int = 100,
        since: datetime | None = None,
        include_held: bool = False,
    ) -> list[EmailMessage]:
        """Newest first. Held messages are skipped unless ``include_held``."""
        sql = "SELECT * FROM emails WHERE account_id = ?"
        params: list[Any] = [account_id]
        if since is not None:
            sql += " AND received_at >= ?"
            params.append(_iso(since))
        with self._connect() as conn:
            sql += f" AND {self._visible(conn, include_held=include_held)}"
            sql += " ORDER BY received_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_message(r) for r in rows]

    def repair_from_domains(self, *, apply: bool = False) -> list[tuple[str, str, str]]:
        """Recompute ``from_domain`` where the old parser stored it wrong.

        Before 2026-09-24 a quoted address display name ('"alerts@bank.example"
        <alerts@bank.example>') was stored as ``bank.example"``. Returns
        ``(message_id, stored, correct)`` for each row that differs; writes only with
        ``apply``.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT id, from_address, from_domain FROM emails").fetchall()
            fixes = [
                (r["id"], r["from_domain"] or "", correct)
                for r in rows
                if (correct := domain_of_address(r["from_address"] or "") or "")
                and correct != (r["from_domain"] or "")
            ]
            if apply:
                conn.executemany(
                    "UPDATE emails SET from_domain = ? WHERE id = ?",
                    [(new, mid) for mid, _, new in fixes],
                )
        return fixes

    def list_by_sender_domain(self, domain: str, *, limit: int = 50) -> list[EmailMessage]:
        """Mail from ``domain`` or any of its subdomains, newest first, across inboxes.

        The finance sweep reads a newly trusted sender's past mail with this
        (ADR-0121); ``list_recent`` only reaches the newest few hundred per inbox.
        """
        d = domain.strip().lower().lstrip("@")
        if not d:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM emails "  # noqa: S608 — the condition is fixed SQL
                "WHERE (lower(from_domain) = ? OR lower(from_domain) LIKE ?) "
                f"AND {self._visible(conn)} ORDER BY received_at DESC LIMIT ?",
                (d, "%." + d, limit),
            ).fetchall()
        return [_row_to_message(r) for r in rows]

    def list_by_category(
        self,
        account_id: str,
        category_prefix: str | CategoryFilter,
        *,
        limit: int = 20,
        since: datetime | None = None,
    ) -> list[EmailMessage]:
        """Recent emails in a category, newest first.

        A category *browse* (no text query) — complements ``search`` (FTS5 text)
        and ``list_recent`` (everything). A string is prefix-matched (``Newsletters``
        matches ``Newsletters/AI``); a :class:`CategoryFilter` matches its paths or
        its provider labels.
        """
        clause, clause_params = _category_clause(category_prefix)
        # ``clause`` is built from fixed SQL and ``?`` placeholders only.
        sql = f"SELECT * FROM emails WHERE account_id = ? AND {clause}"  # noqa: S608
        params: list[Any] = [account_id, *clause_params]
        if since is not None:
            sql += " AND received_at >= ?"
            params.append(_iso(since))
        with self._connect() as conn:
            sql += f" AND {self._visible(conn)} ORDER BY received_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_message(r) for r in rows]

    def count_by_category(
        self,
        account_id: str,
        category: str | CategoryFilter,
        *,
        since: datetime | None = None,
    ) -> int:
        """How many emails ``list_by_category`` would return with no limit."""
        clause, clause_params = _category_clause(category)
        sql = f"SELECT COUNT(*) FROM emails WHERE account_id = ? AND {clause}"  # noqa: S608
        params: list[Any] = [account_id, *clause_params]
        if since is not None:
            sql += " AND received_at >= ?"
            params.append(_iso(since))
        with self._connect() as conn:
            sql += f" AND {self._visible(conn)}"
            (n,) = conn.execute(sql, params).fetchone()
        return int(n)

    def sender_category_counts(
        self, account_id: str, category: str | CategoryFilter
    ) -> list[tuple[str, str, int, int]]:
        """Per sender in the account: ``(from_address, from_domain, total, in_category)``.

        ``in_category`` counts the sender's emails ``category`` matches — the share a
        caller needs to call a sender "mostly promotions" without a list of shops.
        """
        clause, clause_params = _category_clause(category)
        with self._connect() as conn:
            sql = (
                "SELECT from_address, COALESCE(from_domain, ''), COUNT(*), "  # noqa: S608
                f"SUM(CASE WHEN {clause} THEN 1 ELSE 0 END) "
                f"FROM emails WHERE account_id = ? AND {self._visible(conn)} "
                "GROUP BY from_address, from_domain"
            )
            rows = conn.execute(sql, [*clause_params, account_id]).fetchall()
        return [(str(a), str(d), int(n), int(m or 0)) for a, d, n, m in rows]

    def category_counts(self, account_id: str | None = None) -> dict[str, int]:
        """Every ``classified_category`` in use, with how many emails carry it."""
        sql = (
            "SELECT classified_category, COUNT(*) FROM emails WHERE classified_category IS NOT NULL"
        )
        params: list[Any] = []
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            sql += f" AND {self._visible(conn)} GROUP BY classified_category"
            rows = conn.execute(sql, params).fetchall()
        return {str(path): int(n) for path, n in rows}

    def path_counts(
        self, account_id: str, *, since: datetime | None = None
    ) -> dict[str | None, int]:
        """Emails per ``classified_category`` in one account, ``None`` = not yet classified.

        Unlike :meth:`category_counts` this keeps the unclassified rows (under ``None``)
        and takes a ``since`` window — the digest's per-account "last 24 h" mix.
        """
        sql = "SELECT classified_category, COUNT(*) FROM emails WHERE account_id = ?"
        params: list[Any] = [account_id]
        if since is not None:
            sql += " AND received_at >= ?"
            params.append(_iso(since))
        with self._connect() as conn:
            sql += f" AND {self._visible(conn)} GROUP BY classified_category"
            rows = conn.execute(sql, params).fetchall()
        return {(str(path) if path is not None else None): int(n) for path, n in rows}

    def list_by_thread(
        self, account_id: str, thread_id: str, *, include_held: bool = False
    ) -> list[EmailMessage]:
        """A thread oldest first. Held messages are skipped unless ``include_held``."""
        with self._connect() as conn:
            visible = self._visible(conn, include_held=include_held)
            rows = conn.execute(
                "SELECT * FROM emails WHERE account_id = ? AND thread_id = ? "  # noqa: S608
                f"AND {visible} ORDER BY received_at ASC",
                (account_id, thread_id),
            ).fetchall()
        return [_row_to_message(r) for r in rows]

    def mark_classified(
        self,
        message_id: str,
        *,
        category: str,
        confidence: float,
        sensitivity: str | None = None,
    ) -> bool:
        """Stamp classification metadata on a row.

        Called by ``email-triage`` (Track 1G+). Returns ``True`` if the
        row was updated, ``False`` if no row with that id existed
        (callers can use the boolean to detect ghost ids without a
        separate ``get`` round-trip).

        Per ADR-0022 also sets ``triage_state='classified'`` so the
        row is excluded from both ``list_unclassified`` (because
        ``classified_category`` is now set) and ``list_pending_review``
        (because ``triage_state`` is no longer ``'pending_review'``).

        Stamps ``classified_source='iris'``, replacing any vendor path.
        """
        now = _iso(_utc_now())
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE emails SET "
                "  classified_category = ?, "
                "  classified_confidence = ?, "
                "  classified_at = ?, "
                "  classified_source = ?, "
                "  sensitivity = ?, "
                "  triage_state = 'classified', "
                "  updated_at = ? "
                "WHERE id = ?",
                (category, confidence, now, IRIS_SOURCE, sensitivity, now, message_id),
            )
            return cursor.rowcount > 0

    def mark_pending_review(self, message_id: str) -> bool:
        """Mark a row as queued for batch LLM review per ADR-0022.

        The row stays ``classified_category IS NULL`` — it is not
        classified yet. ``triage_state='pending_review'`` is the
        queue signal that ``iris email triage-batch`` reads.

        Returns ``True`` if a row was updated, ``False`` for ghost id.
        """
        now = _iso(_utc_now())
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE emails SET "
                "  triage_state = 'pending_review', "
                "  updated_at = ? "
                "WHERE id = ?",
                (now, message_id),
            )
            return cursor.rowcount > 0

    def mark_triage_error(self, message_id: str) -> bool:
        """Mark a row as soft-failed during triage per ADR-0022.

        Distinct from ``pending_review``: error rows are retry
        candidates (the classifier will try them again next pass),
        queued rows are LLM-review candidates (only ``triage-batch``
        moves them out of the queue).
        """
        now = _iso(_utc_now())
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE emails SET "
                "  triage_state = 'error', "
                "  updated_at = ? "
                "WHERE id = ?",
                (now, message_id),
            )
            return cursor.rowcount > 0

    def list_unclassified(self, account_id: str, *, limit: int = 100) -> list[EmailMessage]:
        """Most-recent unclassified messages awaiting first-pass triage.

        Used by ``iris email triage --account ...`` to find work and
        by event-driven triage to backfill if it was off when mail
        arrived.

        Per ADR-0022: rows with ``triage_state='pending_review'`` are
        excluded — they've already been triaged once (kNN was
        ambiguous, they're queued for the LLM batch). Rows with
        ``triage_state='error'`` are included as retry candidates.

        A vendor-classified row (the mailbox's own bucket) still counts as
        unclassified: triage gives it IRIS's verdict and overwrites the vendor path.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM emails "  # noqa: S608 — the condition is fixed SQL
                "WHERE account_id = ? "
                "  AND (classified_category IS NULL OR classified_source = 'vendor') "
                "  AND (triage_state IS NULL OR triage_state = 'error') "
                f"  AND {self._visible(conn)} "
                "ORDER BY received_at DESC LIMIT ?",
                (account_id, limit),
            ).fetchall()
        return [_row_to_message(r) for r in rows]

    def list_pending_review(self, account_id: str, *, limit: int = 100) -> list[EmailMessage]:
        """Most-recent rows queued for LLM batch review per ADR-0022.

        Vendor-classified rows count as unclassified here too (see
        ``list_unclassified``).
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM emails "  # noqa: S608 — the condition is fixed SQL
                "WHERE account_id = ? "
                "  AND triage_state = 'pending_review' "
                "  AND (classified_category IS NULL OR classified_source = 'vendor') "
                f"  AND {self._visible(conn)} "
                "ORDER BY received_at DESC LIMIT ?",
                (account_id, limit),
            ).fetchall()
        return [_row_to_message(r) for r in rows]

    def backfill_vendor_categories(
        self,
        account_id: str,
        derive: Callable[[tuple[str, ...]], str | None],
        *,
        dry_run: bool = False,
    ) -> VendorBackfillResult:
        """Re-derive vendor classifications from each row's stored ``labels``.

        ``derive`` maps a message's provider labels to a topic path (or None); the
        provider plugin supplies it, so this store never learns a provider's label
        names. The same precedence as ``upsert``: only rows whose source is NULL or
        ``"vendor"`` are written, IRIS rows are counted and left alone. Idempotent —
        a second run finds every row ``unchanged``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, labels, classified_category, classified_source "
                "FROM emails WHERE account_id = ?",
                (account_id,),
            ).fetchall()
        scanned = set_new = replaced = unchanged = skipped_iris = no_vendor = 0
        by_category: dict[str, int] = {}
        updates: list[tuple[str, str]] = []
        for row in rows:
            scanned += 1
            source = row["classified_source"]
            current = row["classified_category"]
            if source == IRIS_SOURCE or (source is None and current is not None):
                skipped_iris += 1
                continue
            path = derive(tuple(json.loads(row["labels"] or "[]")))
            if not path:
                no_vendor += 1
                continue
            if source == VENDOR_SOURCE and current == path:
                unchanged += 1
                continue
            if source == VENDOR_SOURCE:
                replaced += 1
            else:
                set_new += 1
            by_category[path] = by_category.get(path, 0) + 1
            updates.append((path, row["id"]))
        if updates and not dry_run:
            now = _iso(_utc_now())
            with self._connect() as conn:
                conn.executemany(
                    "UPDATE emails SET "
                    "  classified_category = ?, "
                    "  classified_confidence = ?, "
                    "  classified_at = ?, "
                    "  classified_source = ?, "
                    "  updated_at = ? "
                    "WHERE id = ? "
                    "  AND (classified_source IS NULL OR classified_source = ?)",
                    [
                        (path, VENDOR_CONFIDENCE, now, VENDOR_SOURCE, now, mid, VENDOR_SOURCE)
                        for path, mid in updates
                    ],
                )
        return VendorBackfillResult(
            scanned=scanned,
            set_new=set_new,
            replaced=replaced,
            unchanged=unchanged,
            skipped_iris=skipped_iris,
            no_vendor_category=no_vendor,
            by_category=by_category,
        )

    def search(
        self,
        query: str,
        *,
        account_id: str | None = None,
        category_prefix: str | CategoryFilter | None = None,
        since: datetime | None = None,
        limit: int = 20,
    ) -> list[SearchHit]:
        """Full-text search over ``emails`` via the FTS5 virtual table.

        Per ADR-0026 §5+§6:
          - ``query`` is handed straight to FTS5 ``MATCH`` — supports
            phrases, AND/OR/NOT, prefix ``*``, and column qualifiers.
          - Filters compose with the MATCH at SQL level.
          - Results ranked by BM25 (smaller = better); the CLI hides
            the raw rank but exposes it on ``SearchHit`` for debug.
          - ``snippet_highlighted`` carries ``<mark>...</mark>`` tags
            around matched terms.

        Returns an empty list when nothing matches. A malformed FTS5
        query raises ``sqlite3.OperationalError``; the CLI maps that
        to exit 2.
        """
        sql = (
            "SELECT e.id, e.subject, e.from_address, e.from_domain, "
            "       e.received_at, e.classified_category, "
            # FTS5 column index -1 = "pick the column with the highest BM25
            # contribution"; ensures the snippet highlights the actually-
            # matched column (subject vs from_address vs snippet).
            "       snippet(emails_fts, -1, '<mark>', '</mark>', '...', 32) AS hl, "
            "       bm25(emails_fts) AS rank "
            "FROM emails_fts "
            "JOIN emails AS e ON e.rowid = emails_fts.rowid "
            "WHERE emails_fts MATCH ?"
        )
        params: list[Any] = [query]
        if account_id is not None:
            sql += " AND e.account_id = ?"
            params.append(account_id)
        if category_prefix is not None:
            clause, clause_params = _category_clause(category_prefix, "e.")
            sql += f" AND {clause}"
            params.extend(clause_params)
        if since is not None:
            sql += " AND e.received_at >= ?"
            params.append(_iso(since))
        with self._connect() as conn:
            sql += f" AND {self._visible(conn, 'e.id')} ORDER BY rank LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()

        return [
            SearchHit(
                id=r["id"],
                subject=r["subject"] or "",
                from_address=r["from_address"],
                from_domain=r["from_domain"],
                received_at=_parse_dt(r["received_at"]) or _utc_now(),
                classified_category=r["classified_category"],
                snippet_highlighted=r["hl"] or "",
                rank=float(r["rank"]),
            )
            for r in rows
        ]

    def address_known(self, address: str) -> bool:
        """True when ``address`` has sent, received or been copied on any stored email.

        Matched whole: a bare address, or one inside ``Name <address>``, never a
        substring ("x.ks@gmail.com" does not match "ks@gmail.com").
        """
        addr = address.strip().lower()
        if not addr:
            return False
        bracketed = f"%<{_like_escape(addr)}>"
        with self._connect() as conn:
            sql = (
                "SELECT 1 FROM emails WHERE (lower(from_address) = ? "  # noqa: S608
                "OR lower(from_address) LIKE ? ESCAPE '!' "
                "OR EXISTS (SELECT 1 FROM json_each(to_addrs) "
                "           WHERE lower(value) = ? OR lower(value) LIKE ? ESCAPE '!') "
                "OR EXISTS (SELECT 1 FROM json_each(cc_addrs) "
                "           WHERE lower(value) = ? OR lower(value) LIKE ? ESCAPE '!')) "
                f"AND {self._visible(conn)} LIMIT 1"
            )
            row = conn.execute(sql, (addr, bracketed) * 3).fetchone()
        return row is not None

    def count(self, account_id: str | None = None, *, include_held: bool = False) -> int:
        """How many messages are stored; held ones only with ``include_held``."""
        with self._connect() as conn:
            sql = f"SELECT COUNT(*) FROM emails WHERE {self._visible(conn, include_held=include_held)}"  # noqa: S608
            params: list[Any] = []
            if account_id is not None:
                sql += " AND account_id = ?"
                params.append(account_id)
            (n,) = conn.execute(sql, params).fetchone()
        return int(n)

    def has_synced(self, account_id: str) -> bool:
        """True once any fetch reached ``account_id``: a message is stored for it (held
        ones too) or a provider kept a sync cursor for it."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM emails WHERE account_id = ? UNION ALL "
                "SELECT 1 FROM sync_cursors WHERE account_id = ? LIMIT 1",
                (account_id, account_id),
            ).fetchone()
        return row is not None

    def list_accounts(self) -> list[str]:
        """Distinct ``account_id`` values present in the store, newest-active first.

        Ordered by most-recent message so the primary mailbox leads. Used by
        the chat email agent to summarise every connected inbox without a
        separate accounts registry.
        """
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT account_id FROM emails WHERE {self._visible(conn)} "  # noqa: S608
                "GROUP BY account_id ORDER BY MAX(received_at) DESC"
            ).fetchall()
        return [row[0] for row in rows]

    # ------------------------------------------------------------------
    # Sync cursors
    # ------------------------------------------------------------------

    def get_cursor(self, provider: str, account_id: str, cursor_kind: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cursor_value FROM sync_cursors "
                "WHERE provider = ? AND account_id = ? AND cursor_kind = ?",
                (provider, account_id, cursor_kind),
            ).fetchone()
        return row["cursor_value"] if row else None

    def set_cursor(
        self,
        provider: str,
        account_id: str,
        cursor_kind: str,
        cursor_value: str,
    ) -> None:
        now = _iso(_utc_now())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sync_cursors "
                "  (provider, account_id, cursor_kind, cursor_value, last_synced_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(provider, account_id, cursor_kind) DO UPDATE SET "
                "  cursor_value = excluded.cursor_value, "
                "  last_synced_at = excluded.last_synced_at",
                (provider, account_id, cursor_kind, cursor_value, now),
            )

    def clear_cursor(self, provider: str, account_id: str, cursor_kind: str) -> None:
        """Forget a sync cursor, so the provider's next fetch cold-starts. The one
        deliberate "start from scratch" signal: no fetch path clears its own cursor."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM sync_cursors "
                "WHERE provider = ? AND account_id = ? AND cursor_kind = ?",
                (provider, account_id, cursor_kind),
            )


# ---------------------------------------------------------------------------
# Schema + SQL
# ---------------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS emails (
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
    classified_source      TEXT,
    sensitivity            TEXT,
    triage_state           TEXT,
    processed_at           TEXT,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_emails_account_received
    ON emails(account_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_emails_account_thread
    ON emails(account_id, thread_id);
CREATE INDEX IF NOT EXISTS idx_emails_classified
    ON emails(classified_category);
CREATE INDEX IF NOT EXISTS idx_emails_triage_state
    ON emails(triage_state, account_id);

-- ADR-0118 step 5: mail IRIS moved to the provider's Trash. The message leaves `emails`
-- (it is not in the inbox any more, and the sync only adds new mail, so it will not
-- come back on its own); this remembers enough to restore it and to say what it was.
CREATE TABLE IF NOT EXISTS trashed (
    id           TEXT PRIMARY KEY,
    account_id   TEXT NOT NULL,
    subject      TEXT NOT NULL,
    from_address TEXT NOT NULL,
    trashed_at   TEXT NOT NULL,
    batch_id     TEXT NOT NULL,
    labels_before TEXT
);

CREATE TABLE IF NOT EXISTS sync_cursors (
    provider       TEXT NOT NULL,
    account_id     TEXT NOT NULL,
    cursor_kind    TEXT NOT NULL,
    cursor_value   TEXT NOT NULL,
    last_synced_at TEXT NOT NULL,
    PRIMARY KEY (provider, account_id, cursor_kind)
);
"""


# Per ADR-0026: contentless FTS5 virtual table + triggers.
# Held separately from _SCHEMA_SQL so _setup_fts can detect first-time
# creation and run the one-shot rebuild.
_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS emails_fts USING fts5(
    subject,
    from_address,
    snippet,
    content='emails',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS emails_fts_ai
AFTER INSERT ON emails BEGIN
    INSERT INTO emails_fts(rowid, subject, from_address, snippet)
        VALUES (new.rowid, new.subject, new.from_address, new.snippet);
END;

CREATE TRIGGER IF NOT EXISTS emails_fts_au
AFTER UPDATE OF subject, from_address, snippet ON emails BEGIN
    INSERT INTO emails_fts(emails_fts, rowid, subject, from_address, snippet)
        VALUES('delete', old.rowid, old.subject, old.from_address, old.snippet);
    INSERT INTO emails_fts(rowid, subject, from_address, snippet)
        VALUES (new.rowid, new.subject, new.from_address, new.snippet);
END;

CREATE TRIGGER IF NOT EXISTS emails_fts_ad
AFTER DELETE ON emails BEGIN
    INSERT INTO emails_fts(emails_fts, rowid, subject, from_address, snippet)
        VALUES('delete', old.rowid, old.subject, old.from_address, old.snippet);
END;
"""


_UPSERT_SQL = """
INSERT INTO emails (
    id, provider, account_id, thread_id,
    from_address, from_domain,
    to_addrs, cc_addrs,
    subject, snippet, received_at,
    labels, attachments, headers_subset,
    classified_category, classified_confidence, classified_at, classified_source,
    sensitivity, triage_state, processed_at, created_at, updated_at
) VALUES (
    :id, :provider, :account_id, :thread_id,
    :from_address, :from_domain,
    :to_addrs, :cc_addrs,
    :subject, :snippet, :received_at,
    :labels, :attachments, :headers_subset,
    :classified_category, :classified_confidence, :classified_at, :classified_source,
    :sensitivity, :triage_state, :processed_at, :created_at, :updated_at
)
ON CONFLICT(id) DO UPDATE SET
    provider = excluded.provider,
    account_id = excluded.account_id,
    thread_id = excluded.thread_id,
    from_address = excluded.from_address,
    from_domain = excluded.from_domain,
    to_addrs = excluded.to_addrs,
    cc_addrs = excluded.cc_addrs,
    subject = excluded.subject,
    snippet = excluded.snippet,
    received_at = excluded.received_at,
    labels = excluded.labels,
    attachments = excluded.attachments,
    headers_subset = excluded.headers_subset,
    classified_category = CASE WHEN {vendor_may_write}
        THEN excluded.classified_category ELSE emails.classified_category END,
    classified_confidence = CASE WHEN {vendor_may_write}
        THEN excluded.classified_confidence ELSE emails.classified_confidence END,
    classified_at = CASE WHEN {vendor_may_write}
        AND emails.classified_category IS NOT excluded.classified_category
        THEN excluded.classified_at ELSE emails.classified_at END,
    classified_source = CASE WHEN {vendor_may_write}
        THEN excluded.classified_source ELSE emails.classified_source END,
    updated_at = excluded.updated_at
""".format(
    # A vendor path may land on a row only when the incoming message carries one and
    # the row is unclassified or vendor-classified: IRIS's verdict is never replaced,
    # and a message without a vendor path never clears one.
    vendor_may_write=(
        "excluded.classified_source = 'vendor' "
        "AND (emails.classified_source = 'vendor' "
        "OR (emails.classified_source IS NULL AND emails.classified_category IS NULL))"
    )
)
