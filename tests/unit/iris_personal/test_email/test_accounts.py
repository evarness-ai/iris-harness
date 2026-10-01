"""Tests for the cross-cutting email_accounts registry (ADR-0016)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from iris_personal.email.accounts import EmailAccount, EmailAccountStore


@pytest.fixture
def store(tmp_path: Path) -> EmailAccountStore:
    s = EmailAccountStore(db_path=tmp_path / "iris.db")
    s.ensure_schema()
    return s


# ─── Model validation ────────────────────────────────────────────────


def test_model_rejects_address_without_at() -> None:
    with pytest.raises(ValidationError):
        EmailAccount(id="x:y", provider="x", address="notanemail")


def test_model_rejects_lowercase_currency() -> None:
    with pytest.raises(ValidationError):
        EmailAccount(id="x:a@b", provider="x", address="a@b", currency_default="inr")


def test_model_rejects_lowercase_country() -> None:
    with pytest.raises(ValidationError):
        EmailAccount(id="x:a@b", provider="x", address="a@b", country_hint="in")


def test_model_accepts_iso_codes() -> None:
    m = EmailAccount(
        id="x:a@b",
        provider="x",
        address="a@b",
        currency_default="INR",
        country_hint="IN",
    )
    assert m.currency_default == "INR"
    assert m.country_hint == "IN"


# ─── Store basics ────────────────────────────────────────────────────


def test_ensure_schema_is_idempotent(tmp_path: Path) -> None:
    s = EmailAccountStore(db_path=tmp_path / "iris.db")
    s.ensure_schema()
    s.ensure_schema()  # must not raise


def test_add_and_get(store: EmailAccountStore) -> None:
    acc = store.add(
        provider="gmail",
        address="user.in@example.com",
        currency_default="INR",
        country_hint="IN",
    )
    assert acc.id == "gmail:user.in@example.com"
    assert acc.currency_default == "INR"
    fetched = store.get(acc.id)
    assert fetched == acc


def test_add_lowercases_provider_and_address(store: EmailAccountStore) -> None:
    acc = store.add(provider="Gmail", address="user.us@example.com")
    assert acc.provider == "gmail"
    assert acc.address == "user.us@example.com"


def test_get_missing_returns_none(store: EmailAccountStore) -> None:
    assert store.get("nonexistent") is None


def test_get_by_address_is_case_insensitive(store: EmailAccountStore) -> None:
    store.add(provider="gmail", address="x@y.com")
    assert store.get_by_address("GMAIL", "X@Y.COM") is not None


def test_unique_provider_address(store: EmailAccountStore) -> None:
    store.add(provider="gmail", address="x@y.com")
    with pytest.raises(sqlite3.IntegrityError):
        store.add(provider="gmail", address="x@y.com")


def test_same_address_across_providers_allowed(store: EmailAccountStore) -> None:
    """Same address on different providers is fine — unique is on the pair."""
    a = store.add(provider="gmail", address="x@y.com")
    b = store.add(provider="outlook", address="x@y.com")
    assert a.id != b.id


# ─── List + deactivate ───────────────────────────────────────────────


def test_list_default_returns_active_only(store: EmailAccountStore) -> None:
    a = store.add(provider="gmail", address="a@x.com")
    store.add(provider="gmail", address="b@x.com")
    store.deactivate(a.id)

    actives = store.list()
    addrs = {acc.address for acc in actives}
    assert addrs == {"b@x.com"}


def test_list_with_active_only_false_returns_all(store: EmailAccountStore) -> None:
    a = store.add(provider="gmail", address="a@x.com")
    store.add(provider="gmail", address="b@x.com")
    store.deactivate(a.id)

    everyone = store.list(active_only=False)
    assert len(everyone) == 2


def test_deactivate_sets_active_false(store: EmailAccountStore) -> None:
    acc = store.add(provider="gmail", address="x@y.com")
    deactivated = store.deactivate(acc.id)
    assert deactivated.active is False
    refetched = store.get(acc.id)
    assert refetched is not None
    assert refetched.active is False


def test_deactivate_missing_raises(store: EmailAccountStore) -> None:
    with pytest.raises(KeyError):
        store.deactivate("nonexistent")


# ─── update_currency_country ─────────────────────────────────────────


def test_update_currency_country_sets_fields(store: EmailAccountStore) -> None:
    acc = store.add(provider="gmail", address="x@y.com")
    updated = store.update_currency_country(acc.id, currency_default="USD", country_hint="US")
    assert updated.currency_default == "USD"
    assert updated.country_hint == "US"


def test_update_currency_country_partial_leaves_other_field(store: EmailAccountStore) -> None:
    """Passing None for a field leaves it unchanged (MVP behavior; can't clear)."""
    acc = store.add(
        provider="gmail",
        address="x@y.com",
        currency_default="INR",
        country_hint="IN",
    )
    updated = store.update_currency_country(acc.id, currency_default="USD")
    assert updated.currency_default == "USD"
    assert updated.country_hint == "IN"  # untouched


def test_update_currency_country_validates_iso_codes(store: EmailAccountStore) -> None:
    acc = store.add(provider="gmail", address="x@y.com")
    with pytest.raises(ValidationError):
        store.update_currency_country(acc.id, currency_default="usd")


def test_update_missing_raises(store: EmailAccountStore) -> None:
    with pytest.raises(KeyError):
        store.update_currency_country("nonexistent", currency_default="USD")


def test_non_mailbox_declarations_ignore_case_and_whitespace() -> None:
    """A plugin declares its account provider is not a mailbox; lookups normalise."""
    from iris_personal.email import accounts as email_accounts

    email_accounts.clear_non_mailbox_providers()
    try:
        assert not email_accounts.is_non_mailbox_provider("cal")
        email_accounts.declare_non_mailbox_provider("cal")
        email_accounts.declare_non_mailbox_provider("cal")  # idempotent
        assert email_accounts.is_non_mailbox_provider(" CAL ")
        assert not email_accounts.is_non_mailbox_provider("mail")
    finally:
        email_accounts.clear_non_mailbox_providers()


# ─── the status seam (core/SDK boundary plan, PR 2) ──────────────────────────────


def test_the_registered_count_is_what_system_status_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The account store left the core; ``iris system status`` counts through this seam."""
    from iris_harness.services.system import status as status_module
    from iris_personal.email.accounts import register_account_count

    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    store = EmailAccountStore()
    store.ensure_schema()
    store.add(provider="gmail", address="a@x.com")
    store.add(provider="gmail", address="b@x.com")
    store.deactivate(store.add(provider="gdrive", address="a@x.com").id)
    try:
        register_account_count()
        status = status_module.iris_status(
            skills_dir=tmp_path / "none",
            heartbeats_path=tmp_path / "none.yaml",
            data_dir=tmp_path / "nodata",
        )
        # Inactive accounts count too, as they did when the core read the store itself.
        assert status.accounts == {"gmail": 2, "gdrive": 1}
    finally:
        status_module.register_account_counter(None)
