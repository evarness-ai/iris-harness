"""System Health rows for the mailbox-write gate (R4): which mailboxes IRIS may not change.

One yellow row per active mailbox account with no write approval, target
``Mailbox writes``, subject the address, action the one-time command that approves it
(``iris email writes approve --account <id>``). Without it the owner would see labels
stop and trash refuse with no row saying why. Yellow, not red: read-only is a choice the
owner may keep, so the health watch (which pages on red) leaves it alone.

A mailbox account is an active ``email_accounts`` row whose provider has a mounted mail
provider (calendar and Drive rows share the table and have none). Local only: reads the
account table and the approvals table, no network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from iris_harness.sdk.health import CheckKind, HealthCheck, HealthState

TARGET = "Mailbox writes"


def _active_accounts() -> list[tuple[str, str]]:
    from iris_personal.email.accounts import EmailAccountStore

    store = EmailAccountStore()
    store.ensure_schema()
    return [(a.id, a.address) for a in store.list(active_only=True)]


def _provider_for(account_id: str) -> Any:
    from iris_personal.email.providers import mail_provider_for

    return mail_provider_for(account_id)


def write_approval_checks(
    *,
    accounts: Callable[[], list[tuple[str, str]]] = _active_accounts,
    provider_for: Callable[[str], Any] = _provider_for,
) -> list[HealthCheck]:
    """The rows; ``accounts`` returns ``(account id, address)`` pairs (tests pass one)."""
    from iris_personal.email.write_approvals import approve_command, mailbox_writes_approved

    rows: list[HealthCheck] = []
    for account_id, address in accounts():
        if provider_for(account_id) is None or mailbox_writes_approved(account_id):
            continue
        rows.append(
            HealthCheck(
                TARGET,
                CheckKind.CREDENTIAL,
                HealthState.YELLOW,
                f"{address}: mailbox writes not approved, so IRIS's labels, Trash and "
                "restore are blocked for this account",
                action=approve_command(account_id),
                subject=address,
            )
        )
    return rows


__all__ = ["TARGET", "write_approval_checks"]
