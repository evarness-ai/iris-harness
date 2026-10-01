"""``iris email writes`` -- the owner's command line over the mailbox-write gate (R4).

One per-account approval gates every change IRIS makes to a mailbox, for every provider
(``iris_personal.email.write_approvals``): labels, moves to Trash, restores. Email
setup's label-preview approval records it for a new account; these commands are the
same approval for an account connected before onboarding existed, and the way to see or
take it back:

* ``approve --account <id|address> [--ref TEXT] [--yes]`` -- asks first (``--yes`` for
  scripts), writes a governance audit row, then records the approval naming that row;
* ``status [--account ...]`` -- each mailbox account's approval (when, reference) or
  "not approved -- writes blocked";
* ``revoke --account ...`` -- removes it and writes an audit row.

The ledger rows name the account's provider, not its address (the id is in the payload);
what these commands print to the owner's own terminal still shows the address.

Provider-agnostic: an account is an ``email_accounts`` row, found by its id
(``gmail:owner@example.com``) or by its address when that names one mailbox.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Annotated

import typer

from iris_harness.sdk.cli import console, print_error
from iris_personal.email.write_approvals import AUDIT_PLUGIN

if TYPE_CHECKING:
    from iris_personal.email.accounts import EmailAccount

writes_app = typer.Typer(
    name="writes",
    help=(
        "Approve, show or revoke IRIS's permission to change a mailbox (labels, "
        "Trash, restore). One approval per account, for every provider."
    ),
    no_args_is_help=True,
)

_DEFAULT_REF = "iris email writes approve"


def _mailbox_accounts(*, active_only: bool) -> list[EmailAccount]:
    """``email_accounts`` rows that are mailboxes (calendar and Drive share the table
    and declare themselves non-mailbox providers)."""
    from iris_personal.email.accounts import EmailAccountStore, is_non_mailbox_provider

    store = EmailAccountStore()
    store.ensure_schema()
    return [
        a for a in store.list(active_only=active_only) if not is_non_mailbox_provider(a.provider)
    ]


def resolve_account(account: str) -> EmailAccount:
    """The mailbox account ``account`` names: an exact id, else a unique address.
    Raises ``LookupError`` with the owner's next step when it names none or several."""
    wanted = account.strip()
    accounts = _mailbox_accounts(active_only=False)
    for acc in accounts:
        if acc.id == wanted:
            return acc
    matches = [a for a in accounts if a.address.lower() == wanted.lower()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        ids = ", ".join(a.id for a in matches)
        raise LookupError(f"{wanted} is more than one account ({ids}); pass the account id")
    known = ", ".join(a.id for a in accounts) or "none"
    raise LookupError(f"no mailbox account {wanted!r} (known accounts: {known})")


def _run_id() -> str:
    return f"mailbox-writes-{uuid.uuid4().hex[:12]}"


def _account_or_exit(account: str) -> EmailAccount:
    try:
        return resolve_account(account)
    except LookupError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc


@writes_app.command("approve")
def cmd_writes_approve(
    account: Annotated[
        str, typer.Option("--account", help="Account id (gmail:you@example.com) or address.")
    ],
    ref: Annotated[
        str | None,
        typer.Option("--ref", help="Why, or which approval this is (kept with the row)."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask (scripts).")] = False,
) -> None:
    """Allow IRIS to change this mailbox: labels, moves to Trash, restores."""
    from iris_personal.email.write_approvals import grant_mailbox_writes

    acc = _account_or_exit(account)
    if not yes:
        console.print(
            f"  IRIS will be allowed to change [cyan]{acc.address}[/cyan] ({acc.id}): add "
            "and remove its IRIS labels, move mail to Trash when you ask, and restore it.\n"
            "  It never deletes mail permanently, archives, or marks mail read."
        )
        if not typer.confirm("  Approve mailbox writes for this account?", default=False):
            console.print("  [yellow]not approved[/yellow]  nothing changed")
            raise typer.Exit(1)
    note = (ref or _DEFAULT_REF).strip() or _DEFAULT_REF
    try:
        approval, _row = grant_mailbox_writes(
            acc.id,
            note,
            actor="owner (cli)",
            run_id=_run_id(),
            agent_type="cli",
        )
    except Exception as exc:  # no audit trail, no approval (R14's invariant)
        print_error(f"could not write the audit row, so nothing was approved: {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"  [bold green]OK[/bold green]  mailbox writes approved for [cyan]{acc.id}[/cyan] "
        f"[dim]({approval.approved_at}; {approval.approval_ref})[/dim]"
    )


@writes_app.command("status")
def cmd_writes_status(
    account: Annotated[
        str | None, typer.Option("--account", help="One account (id or address).")
    ] = None,
) -> None:
    """Show each mailbox account's write approval."""
    from iris_personal.email.write_approvals import approve_command, mailbox_write_approval

    accounts = [_account_or_exit(account)] if account else _mailbox_accounts(active_only=True)
    if not accounts:
        console.print("  [yellow]no mailbox accounts connected[/yellow]")
        return
    for acc in accounts:
        approval = mailbox_write_approval(acc.id)
        if approval is None:
            console.print(
                f"  [yellow]not approved[/yellow]  [cyan]{acc.id}[/cyan] -- writes blocked  "
                f"[dim](run `{approve_command(acc.id)}`)[/dim]"
            )
        else:
            console.print(
                f"  [bold green]approved[/bold green]  [cyan]{acc.id}[/cyan]  "
                f"[dim]at {approval.approved_at}; ref: {approval.approval_ref}[/dim]"
            )


@writes_app.command("revoke")
def cmd_writes_revoke(
    account: Annotated[
        str, typer.Option("--account", help="Account id (gmail:you@example.com) or address.")
    ],
) -> None:
    """Take back IRIS's permission to change this mailbox."""
    from iris_personal.email.write_approvals import mailbox_write_approval, revoke_mailbox_writes

    acc = _account_or_exit(account)
    if mailbox_write_approval(acc.id) is None:
        console.print(f"  [dim]{acc.id} had no write approval; nothing to revoke[/dim]")
        return
    # Less permission never waits on a ledger: the approval goes first, then its row.
    if (
        revoke_mailbox_writes(acc.id, actor="owner (cli)", agent_type="cli", run_id=_run_id())
        is None
    ):
        print_error("revoked, but the audit row could not be written (see the log)")
    console.print(
        f"  [bold green]OK[/bold green]  mailbox writes revoked for [cyan]{acc.id}[/cyan]; "
        "IRIS no longer changes this mailbox"
    )


__all__ = [
    "AUDIT_PLUGIN",
    "cmd_writes_approve",
    "cmd_writes_revoke",
    "cmd_writes_status",
    "resolve_account",
    "writes_app",
]
