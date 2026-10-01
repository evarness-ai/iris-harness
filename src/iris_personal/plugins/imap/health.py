"""The IMAP rows of System Health's credentials (``api.register_credential_check``).

One row per active ``imap:`` account, target ``IMAP``, subject the address:

* red -- no app password in the vault, or the server refused the last login (a
  revoked or changed app password); the action is the command that replaces it;
* yellow -- the server could not be reached at the last try;
* green -- the last login worked, or the password is stored and not used yet.

Local by default: the verdict is the last login the provider recorded. With the owner's
net-probe opt-in, each account is logged into now (one ``EGRESS`` line per account).
No IMAP accounts, no rows: IMAP is an optional on-ramp, not a missing credential.
"""

from __future__ import annotations

from collections.abc import Callable

from iris_harness.sdk.health import CheckKind, HealthCheck, HealthState
from iris_harness.sdk.vault import SecretStore

from .account import IMAP_PROVIDER, load_account
from .connection import ImapError
from .provider import ImapProvider

TARGET = "IMAP"


def login_command(address: str) -> str:
    return f"iris auth imap login --user {address}"


def _accounts() -> list[str]:
    from iris_personal.email.accounts import EmailAccountStore

    store = EmailAccountStore()
    store.ensure_schema()
    return [a.address for a in store.list(active_only=True) if a.provider == IMAP_PROVIDER]


def imap_credential_checks(
    provider: ImapProvider,
    *,
    net_probe: bool = False,
    secret_store: SecretStore | None = None,
    accounts: Callable[[], list[str]] = _accounts,
) -> list[HealthCheck]:
    rows: list[HealthCheck] = []
    for address in accounts():
        account_id = f"{IMAP_PROVIDER}:{address}"
        creds = load_account(account_id, store=secret_store)
        if creds is None:
            rows.append(
                HealthCheck(
                    TARGET,
                    CheckKind.CREDENTIAL,
                    HealthState.RED,
                    f"{address}: no app password in the vault",
                    action=login_command(address),
                    subject=address,
                )
            )
            continue
        if net_probe:
            try:
                provider.check_login(creds)
            except ImapError:
                pass  # the outcome is recorded in the provider's state; read below
        status = provider.state.status(account_id)
        if status is not None and status.auth_failed:
            state, detail, action = (
                HealthState.RED,
                f"{address}: {creds.host} refused the login ({status.last_error_at}); "
                "the app password may be revoked or changed",
                login_command(address),
            )
        elif status is not None and status.last_error:
            state, detail, action = (
                HealthState.YELLOW,
                f"{address}: could not reach {creds.host}:{creds.port} at the last try "
                f"({status.last_error_at})",
                None,
            )
        elif status is not None and status.last_ok_at:
            state, detail, action = (
                HealthState.GREEN,
                f"{address}: signed in to {creds.host} ({status.last_ok_at})",
                None,
            )
        else:
            state, detail, action = (
                HealthState.GREEN,
                f"{address}: app password stored for {creds.host}; not used yet",
                None,
            )
        rows.append(
            HealthCheck(TARGET, CheckKind.CREDENTIAL, state, detail, action=action, subject=address)
        )
    return rows


__all__ = ["TARGET", "imap_credential_checks", "login_command"]
