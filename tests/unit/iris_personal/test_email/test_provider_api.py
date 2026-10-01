"""``email.provider_api``: the facade is the whole stable mail-provider surface, and it
is enough to write one.

``pigeon_provider.py`` is a provider a third party could ship: it imports only the
stable tier. Mounted as a plugin in a harness, it registers, a sweep syncs its mail
into the record through the narrow store, its cursor resets, and its mailbox writes wait
for the owner's approval -- the provider never touched ``EmailStore``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.heartbeat.models import HeartbeatDefinition, HeartbeatStatus
from iris_harness.testing import check_stable_imports, harness, plugin, stable_tier
from iris_personal.email import provider_api
from iris_personal.email.providers import LabellingProvider, MailProvider, mail_provider_for

from . import pigeon_provider

FACADE = "iris_personal.email.provider_api"


def test_the_facade_exports_exactly_the_declared_stable_names() -> None:
    assert set(provider_api.__all__) == stable_tier().names[FACADE]
    # The record itself is the slice's own; a provider gets the narrow sync store.
    assert "EmailStore" not in provider_api.__all__
    assert all(module == FACADE for module in stable_tier().names if "email" in module)


def test_the_pigeon_provider_imports_only_the_stable_tier() -> None:
    assert check_stable_imports([Path(pigeon_provider.__file__)]) == []


def test_the_email_store_is_a_mail_sync_store() -> None:
    from iris_personal.email.store import EmailStore

    assert isinstance(EmailStore(), provider_api.MailSyncStore)


def _sweep() -> HeartbeatDefinition:
    return HeartbeatDefinition(
        name="email_sweep", handler="email_sweep", schedule="interval:600", enabled=True
    )


def test_a_provider_built_from_the_facade_registers_and_syncs_in_a_harness() -> None:
    from iris_personal.email.store import EmailStore
    from iris_personal.email.sweep import EmailSweepHandler
    from iris_personal.email.write_approvals import approve_mailbox_writes

    assert mail_provider_for(pigeon_provider.NAME) is None
    with harness(plugins=[plugin(pigeon_provider.setup, name="pigeon")]) as h:
        assert h.plugin_loaded("pigeon")
        provider = mail_provider_for(pigeon_provider.NAME)
        assert isinstance(provider, MailProvider)
        assert isinstance(provider, LabellingProvider)

        # Connected through the facade too -- no account store in the provider's hands.
        account_id = pigeon_provider.connect()

        # The core's provider-agnostic sweep drives it, like any registered provider.
        first = EmailSweepHandler(bus=None)(_sweep())
        assert first.status is HeartbeatStatus.SUCCESS, first.output
        assert f"{account_id}: +2" in (first.output or "")
        store = EmailStore()
        stored = store.get("p-1")
        assert stored is not None and stored.provider == pigeon_provider.NAME
        assert store.count(account_id) == 2

        # The cursor it saved through the narrow store holds: nothing new next time.
        again = EmailSweepHandler(bus=None)(_sweep())
        assert f"{account_id}: +0" in (again.output or "")
        # ...until it is cleared (bootstrap's re-fetch), through the same store.
        provider.reset_cursor(account_id, store=store)
        assert store.get_cursor(pigeon_provider.NAME, account_id, pigeon_provider.CURSOR) is None
        assert provider.fetch_new(account_id, store=store).fell_back_to_cold_start

        # Its writes wait for the owner's approval (R4), through the facade's
        # mailbox_write -- and a refused write leaves no write row.
        with pytest.raises(PermissionError, match="iris email writes approve"):
            provider.trash_messages(account_id, ["p-1"])
        assert h.audit_rows(hook_point="mailbox_write_performed") == []
        approve_mailbox_writes(account_id, "test")
        assert provider.trash_messages(account_id, ["p-1"]) == ["p-1"]
        assert provider.modify_labels(account_id, ["p-1", "p-2"], ["IRIS/Done"], []) == 2
        # Each approved write is recorded where the proof bundle reads it (R14).
        rows = h.audit_rows(hook_point="mailbox_write_performed")
        assert [(r.decision, r.reason) for r in rows] == [
            ("allow", "mailbox write: trash x1 on a pigeon account"),
            ("allow", "mailbox write: label x2 on a pigeon account"),
        ]

    # The harness put the registry back: the next run does not inherit the provider.
    assert mail_provider_for(pigeon_provider.NAME) is None
    with harness() as h:
        assert mail_provider_for(pigeon_provider.NAME) is None


def test_connect_account_records_the_account_once_and_reconnects_a_disconnected_one() -> None:
    from iris_personal.email.accounts import EmailAccountStore

    with harness():
        account_id = provider_api.connect_account("Pigeon", "Owner@Pigeon.example")
        assert account_id == "pigeon:owner@pigeon.example"
        # Connecting again is not an error: the same account, still one row.
        assert provider_api.connect_account("pigeon", "owner@pigeon.example") == account_id
        accounts = EmailAccountStore()
        assert [a.id for a in accounts.list(active_only=False)] == [account_id]
        # A disconnected account is connected again.
        accounts.deactivate(account_id)
        assert provider_api.connect_account("pigeon", "owner@pigeon.example") == account_id
        account = accounts.get(account_id)
        assert account is not None and account.active
        with pytest.raises(ValueError, match="@"):
            provider_api.connect_account("pigeon", "not-an-address")
