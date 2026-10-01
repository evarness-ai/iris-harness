"""Auth failure (clear error + a red System Health row), egress lines that name the host
and never the credential, TLS (implicit and STARTTLS) and the plaintext rule."""

from __future__ import annotations

import logging
import ssl
from dataclasses import replace
from pathlib import Path

import pytest

from iris_harness.sdk.health import HealthState
from iris_personal.email.store import EmailStore
from iris_personal.plugins.imap.account import (
    ImapAccount,
    ImapAccountError,
    load_account,
    save_account,
)
from iris_personal.plugins.imap.connection import ImapAuthError, ImapConnectionError
from iris_personal.plugins.imap.health import imap_credential_checks
from iris_personal.plugins.imap.provider import ImapProvider

from .conftest import PASSWORD, USER, seed
from .fake_imap_server import FakeImapServer, FakeMailbox, make_certificate


def _accounts() -> list[str]:
    return [USER]


def test_a_refused_login_is_a_clear_error_and_a_red_health_row(
    provider: ImapProvider,
    account: ImapAccount,
    store: EmailStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    save_account(account.with_password("wrong-password-xyz"))

    with caplog.at_level(logging.INFO, logger="iris.egress"):
        with pytest.raises(ImapAuthError) as err:
            provider.fetch_new(account.account_id, store=store)

    message = str(err.value)
    assert "refused the login" in message and "iris auth imap login --user" in message
    assert "wrong-password-xyz" not in message
    (row,) = imap_credential_checks(provider, accounts=_accounts)
    assert row.state is HealthState.RED and row.subject == USER
    assert row.action == f"iris auth imap login --user {USER}"
    egress = [r.getMessage() for r in caplog.records if r.name == "iris.egress"]
    assert egress and "status=auth_failed" in egress[-1]


def test_health_rows_follow_the_last_login(
    provider: ImapProvider, account: ImapAccount, mailbox: FakeMailbox, store: EmailStore
) -> None:
    (fresh,) = imap_credential_checks(provider, accounts=_accounts)
    assert fresh.state is HealthState.GREEN and "not used yet" in fresh.detail

    seed(mailbox, 1)
    provider.fetch_new(account.account_id, store=store)
    (ok,) = imap_credential_checks(provider, accounts=_accounts)
    assert ok.state is HealthState.GREEN and "signed in" in ok.detail

    save_account(account.with_password("revoked"))
    (probed,) = imap_credential_checks(provider, net_probe=True, accounts=_accounts)
    assert probed.state is HealthState.RED


def test_no_vault_entry_is_red_and_no_accounts_is_no_rows(provider: ImapProvider) -> None:
    (missing,) = imap_credential_checks(provider, accounts=_accounts)
    assert missing.state is HealthState.RED and "no app password" in missing.detail
    assert imap_credential_checks(provider, accounts=list) == []


def test_an_unreachable_server_is_yellow(provider: ImapProvider, state: object) -> None:
    dead = ImapAccount(
        address=USER, host="127.0.0.1", username=USER, password=PASSWORD, port=1,
        security="plain",
    )  # fmt: skip
    save_account(dead)
    with pytest.raises(ImapConnectionError):
        provider.check_login(dead)
    (row,) = imap_credential_checks(provider, accounts=_accounts)
    assert row.state is HealthState.YELLOW


def test_every_session_logs_one_egress_line_with_the_host_only(
    provider: ImapProvider,
    account: ImapAccount,
    mailbox: FakeMailbox,
    store: EmailStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seed(mailbox, 2)
    with caplog.at_level(logging.INFO, logger="iris.egress"):
        ids = provider.fetch_new(account.account_id, store=store).new_message_ids
        provider.fetch_message_body(account.account_id, ids[0])

    lines = [r.getMessage() for r in caplog.records if r.name == "iris.egress"]
    assert len(lines) == 2
    assert "purpose=imap.fetch_new" in lines[0] and "purpose=imap.fetch_body" in lines[1]
    for line in lines:
        assert "-> 127.0.0.1" in line and "status=ok" in line
        assert USER not in line and PASSWORD not in line and "owner" not in line


def test_the_password_never_appears_in_a_repr_and_lives_in_the_vault(
    account: ImapAccount,
) -> None:
    assert PASSWORD not in repr(account)
    loaded = load_account(account.account_id)
    assert loaded == account


def test_plaintext_is_refused_off_loopback() -> None:
    with pytest.raises(ImapAccountError, match="plaintext"):
        ImapAccount(address=USER, host="imap.example.test", username=USER, password="x",
                    port=143, security="plain")  # fmt: skip


@pytest.fixture
def tls_files(tmp_path: Path) -> tuple[str, str]:
    return make_certificate(tmp_path)


def _server_ctx(cert: str, key: str) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    return ctx


@pytest.mark.parametrize("mode", ["ssl", "starttls"])
def test_tls_modes_verify_the_certificate_and_sync(
    mode: str, tls_files: tuple[str, str], state: object, store: EmailStore
) -> None:
    cert, key = tls_files
    box = FakeMailbox(users={USER: PASSWORD})
    seed(box, 2)
    with FakeImapServer(box, tls=mode, ssl_context=_server_ctx(cert, key)) as srv:
        acct = ImapAccount(
            address=USER, host=srv.host, username=USER, password=PASSWORD, port=srv.port,
            security=mode,  # type: ignore[arg-type]
        )  # fmt: skip
        save_account(acct)
        trusting = ImapProvider(
            state=state,  # type: ignore[arg-type]
            ssl_context_factory=lambda: ssl.create_default_context(cafile=cert),
            timeout=5.0,
        )
        assert trusting.fetch_new(acct.account_id, store=store).fetched == 2
        if mode == "starttls":
            # The password goes over TLS: STARTTLS comes before LOGIN, and the
            # capabilities are read again over TLS (the plaintext ones are not trusted).
            # Whether a CAPABILITY precedes STARTTLS is imaplib's business: newer
            # releases (3.13.15 does) take them from the greeting and send none.
            starttls, login = box.commands.index("STARTTLS"), box.commands.index("LOGIN")
            assert starttls < login
            assert "CAPABILITY" in box.commands[starttls + 1 : login]

        # The default context does not trust a self-signed certificate: refused.
        default = ImapProvider(state=state, timeout=5.0)  # type: ignore[arg-type]
        with pytest.raises(ImapConnectionError):
            default.check_login(replace(acct))
