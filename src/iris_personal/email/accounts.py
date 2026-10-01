"""Cross-cutting registry of email accounts IRIS is connected to.

Each row represents one ``(provider, address)`` pair — e.g.
``gmail:user.in@example.com``. Optional ``currency_default`` and
``country_hint`` per account power the multi-currency aggregation
rule (ADR-0007) without forcing each statement to declare currency
explicitly.

The table lives in ``data/iris.db`` (NOT ``data/email.db``) because
it's cross-cutting — referenced by future email, calendar, and
finance skills. Putting it under one domain's DB would confuse
ownership.

See ADR-0016 for the implementation-shape decisions.

Moved out of the core (``iris_harness.foundation.data``) by the core/SDK boundary plan,
PR 2: no core module read it except ``iris system status``, which now asks a counter
this module registers (:func:`register_account_count`).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from iris_harness.sdk.persistence import data_path, sqlite_conn
from iris_harness.sdk.process_state import track_globals

# Account kinds that are not mailboxes. The table is shared: a calendar or storage
# plugin records its connected accounts here too. A plugin whose accounts are not
# mailboxes declares its provider name, and mail consumers — the email sweep — pass
# those rows by without comment, while a mail account whose provider plugin is
# missing is still reported. Declared by the owning plugin, so the core names none.
_non_mailbox_providers: set[str] = set()


def declare_non_mailbox_provider(provider: str) -> None:
    """Mark ``provider``'s accounts as not mailboxes (idempotent)."""
    _non_mailbox_providers.add(provider.strip().lower())


def is_non_mailbox_provider(provider: str) -> bool:
    """True when a plugin declared ``provider``'s accounts are not mailboxes."""
    return provider.strip().lower() in _non_mailbox_providers


def clear_non_mailbox_providers() -> None:
    """Forget every declaration (tests)."""
    _non_mailbox_providers.clear()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _account_id(provider: str, address: str) -> str:
    """Canonical slug used as the primary key.

    Lowercased throughout so case-only differences don't create
    distinct rows.
    """
    return f"{provider.strip().lower()}:{address.strip().lower()}"


def ledger_account_label(account_id: str) -> str:
    """How a governance ledger row's free-text ``reason`` names ``account_id``: by its
    provider only ("a gmail account"), never the address. The id itself goes in the
    row's payload, which the proof bundle pseudonymises; a reason is copied verbatim
    wherever the ledger is shown, so it must not carry a mailbox address."""
    provider, sep, _address = account_id.partition(":")
    provider = provider.strip().lower()
    if not sep or not provider or not provider.replace("_", "").replace("-", "").isalnum():
        return "an account"
    article = "an" if provider[0] in "aeiou" else "a"
    return f"{article} {provider} account"


class EmailAccount(BaseModel):
    """One email account IRIS is connected to."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    address: str = Field(..., min_length=3)  # "a@b" minimum
    currency_default: str | None = Field(default=None, min_length=3, max_length=3)
    country_hint: str | None = Field(default=None, min_length=2, max_length=2)
    active: bool = True
    added_at: datetime = Field(default_factory=_utc_now)

    @field_validator("address")
    @classmethod
    def _validate_address_has_at(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError(f"address must contain '@': got {v!r}")
        return v

    @field_validator("currency_default")
    @classmethod
    def _validate_currency_uppercase(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.isupper() or not v.isalpha():
            raise ValueError(f"currency_default must be 3 uppercase letters (ISO-4217): got {v!r}")
        return v

    @field_validator("country_hint")
    @classmethod
    def _validate_country_uppercase(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.isupper() or not v.isalpha():
            raise ValueError(
                f"country_hint must be 2 uppercase letters (ISO-3166-1 alpha-2): got {v!r}"
            )
        return v


@dataclass
class EmailAccountStore:
    """SQLite-backed registry of connected email accounts."""

    db_path: Path = field(default_factory=lambda: data_path("iris.db"))

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Commit on success, roll back on error -- what ``with sqlite3.connect()`` did --
        # and close the handle, which it did not (every call leaked one).
        with sqlite_conn(self.db_path, row_factory=sqlite3.Row) as conn:
            yield conn

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        provider: str,
        address: str,
        currency_default: str | None = None,
        country_hint: str | None = None,
    ) -> EmailAccount:
        """Register a new email account. Raises if (provider, address)
        is already present."""
        account = EmailAccount(
            id=_account_id(provider, address),
            provider=provider.strip().lower(),
            address=address.strip().lower(),
            currency_default=currency_default,
            country_hint=country_hint,
        )
        with self._connect() as conn:
            conn.execute(_INSERT_SQL, _to_row(account))
        return account

    def get(self, account_id: str) -> EmailAccount | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM email_accounts WHERE id = ?", (account_id,)
            ).fetchone()
        return _from_row(row) if row else None

    def get_by_address(self, provider: str, address: str) -> EmailAccount | None:
        return self.get(_account_id(provider, address))

    def list(self, *, active_only: bool = True) -> list[EmailAccount]:
        sql = "SELECT * FROM email_accounts"
        params: list[Any] = []
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY added_at ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_from_row(r) for r in rows]

    def deactivate(self, account_id: str) -> EmailAccount:
        current = self.get(account_id)
        if current is None:
            raise KeyError(f"email account not found: {account_id}")
        updated = current.model_copy(update={"active": False})
        with self._connect() as conn:
            conn.execute("UPDATE email_accounts SET active = 0 WHERE id = ?", (account_id,))
        return updated

    def activate(self, account_id: str) -> EmailAccount:
        """Undo :meth:`deactivate` (an account connected again after a logout)."""
        current = self.get(account_id)
        if current is None:
            raise KeyError(f"email account not found: {account_id}")
        with self._connect() as conn:
            conn.execute("UPDATE email_accounts SET active = 1 WHERE id = ?", (account_id,))
        return current.model_copy(update={"active": True})

    def update_currency_country(
        self,
        account_id: str,
        *,
        currency_default: str | None = None,
        country_hint: str | None = None,
    ) -> EmailAccount:
        """Set or clear currency/country hints on an existing account.

        Passing ``None`` for a field leaves it unchanged. To clear a
        previously-set value, callers should fetch the row, build a
        new EmailAccount with the cleared field, and call ``add`` after
        deleting — but for MVP this method only sets, doesn't clear.
        """
        current = self.get(account_id)
        if current is None:
            raise KeyError(f"email account not found: {account_id}")
        updates: dict[str, Any] = {}
        if currency_default is not None:
            updates["currency_default"] = currency_default
        if country_hint is not None:
            updates["country_hint"] = country_hint
        if not updates:
            return current
        # Trigger validators by constructing a fresh model.
        updated = current.model_copy(update=updates)
        # Re-validate via Pydantic
        updated = EmailAccount(**updated.model_dump())
        with self._connect() as conn:
            conn.execute(
                "UPDATE email_accounts SET currency_default = ?, country_hint = ? WHERE id = ?",
                (updated.currency_default, updated.country_hint, account_id),
            )
        return updated


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def _to_row(account: EmailAccount) -> dict[str, Any]:
    return {
        "id": account.id,
        "provider": account.provider,
        "address": account.address,
        "currency_default": account.currency_default,
        "country_hint": account.country_hint,
        "active": int(account.active),
        "added_at": _iso(account.added_at),
    }


def _from_row(row: sqlite3.Row) -> EmailAccount:
    return EmailAccount(
        id=row["id"],
        provider=row["provider"],
        address=row["address"],
        currency_default=row["currency_default"],
        country_hint=row["country_hint"],
        active=bool(row["active"]),
        added_at=_parse_dt(row["added_at"]) or _utc_now(),
    )


# ---------------------------------------------------------------------------
# Schema + SQL
# ---------------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS email_accounts (
    id               TEXT PRIMARY KEY,
    provider         TEXT NOT NULL,
    address          TEXT NOT NULL,
    currency_default TEXT,
    country_hint     TEXT,
    active           INTEGER NOT NULL DEFAULT 1,
    added_at         TEXT NOT NULL,
    UNIQUE (provider, address)
);
"""


_INSERT_SQL = """
INSERT INTO email_accounts (
    id, provider, address, currency_default, country_hint, active, added_at
) VALUES (
    :id, :provider, :address, :currency_default, :country_hint, :active, :added_at
)
"""


def count_by_provider() -> dict[str, int]:
    """Connected accounts by provider, inactive ones included: ``iris system status``."""
    store = EmailAccountStore()
    store.ensure_schema()
    counts: dict[str, int] = {}
    for account in store.list(active_only=False):
        counts[account.provider] = counts.get(account.provider, 0) + 1
    return counts


def register_account_count() -> None:
    """Fill the core's account-count seam (``iris system status``). Idempotent.

    Called by every plugin that records accounts in this table (gmail, calendar,
    file_organizer), from ``setup()`` and from its CLI registration, since the status
    command runs with no runtime built. It touches no database until status asks.
    """
    from iris_harness.sdk.health import register_account_counter

    register_account_counter(count_by_provider)


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_non_mailbox_providers")
