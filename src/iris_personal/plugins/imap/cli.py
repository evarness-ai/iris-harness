"""``iris auth imap`` -- add, check and remove an IMAP account (app password).

``login`` asks for what it was not given (the app password always hidden, or read from
stdin with ``--password-stdin`` for scripts), logs in once to prove it works, then puts
the account in the vault and registers ``imap:<address>`` in the accounts table -- the
row the email sweep syncs. Nothing is saved when the login fails. The onboarding flow
(``iris email setup``) wraps the same :func:`connect_account`.

The CLI seam runs with no runtime built, so each command builds what it needs.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Annotated

import typer

from iris_harness.sdk.cli import console, print_error

from .account import (
    DEFAULT_FOLDER,
    DEFAULT_PORTS,
    IMAP_PROVIDER,
    SECURITIES,
    ImapAccount,
    ImapAccountError,
    Security,
    account_id_for,
    delete_account,
    load_account,
    save_account,
)

if TYPE_CHECKING:
    from iris_harness.sdk import PluginCLI
    from iris_harness.sdk.vault import SecretStore
    from iris_personal.email.accounts import EmailAccount

    from .provider import ImapProvider


def _clean_hidden_input(raw: str) -> str:
    """A hidden (no-echo) prompt's capture: drop non-printable characters (a
    terminal/paste artifact ``getpass``-style reads don't expect) and the ordinary
    leading/trailing whitespace ``--password-stdin`` already strips for itself.
    Issue #69: a pasted app password failed once through this prompt and worked
    immediately via ``--password-stdin`` with the same bytes, not reproduced
    reliably enough to name a cause -- this is a cheap, safe-regardless mitigation,
    not a claimed fix."""
    return "".join(ch for ch in raw if ch.isprintable()).strip()


#: Well-known hosts by address domain, so the common case asks only for the password.
#: ``--host`` always wins. (Outlook.com is absent on purpose: it no longer takes app
#: passwords over IMAP.)
HOST_PRESETS: dict[str, str] = {
    "gmail.com": "imap.gmail.com",
    "googlemail.com": "imap.gmail.com",
    "icloud.com": "imap.mail.me.com",
    "me.com": "imap.mail.me.com",
    "mac.com": "imap.mail.me.com",
    "yahoo.com": "imap.mail.yahoo.com",
    "fastmail.com": "imap.fastmail.com",
    "aol.com": "imap.aol.com",
}


def register(cli: PluginCLI) -> None:
    imap_app = typer.Typer(
        name="imap",
        help="IMAP accounts with an app password: login / status / logout.",
        no_args_is_help=True,
    )
    imap_app.command("login")(cmd_auth_imap_login)
    imap_app.command("status")(cmd_auth_imap_status)
    imap_app.command("logout")(cmd_auth_imap_logout)
    cli.group("auth").add_typer(imap_app, name="imap")
    from iris_personal.email.accounts import register_account_count
    from iris_personal.email.providers import offer_cli_mail_provider

    register_account_count()
    # A command with no runtime (`iris email setup`) mounts this when it needs a mailbox.
    offer_cli_mail_provider(IMAP_PROVIDER, _build_provider)


def _build_provider() -> ImapProvider:
    from .provider import ImapProvider

    return ImapProvider()


def preset_host(address: str) -> str | None:
    return HOST_PRESETS.get(address.rsplit("@", 1)[-1].strip().lower())


def connect_account(
    account: ImapAccount,
    *,
    provider: ImapProvider | None = None,
    check: bool = True,
    secret_store: SecretStore | None = None,
) -> EmailAccount:
    """Prove the login (unless ``check`` is off), then store the account in the vault
    and register its ``imap:`` row (re-activating one a logout left). Raises the
    provider's :class:`ImapError` on a failed login, having saved nothing."""
    from iris_personal.email.accounts import EmailAccountStore

    from .provider import ImapProvider

    if check:
        (provider or ImapProvider(secret_store=secret_store)).check_login(account)
    save_account(account, store=secret_store)
    store = EmailAccountStore()
    store.ensure_schema()
    existing = store.get(account.account_id)
    if existing is None:
        return store.add(provider=IMAP_PROVIDER, address=account.address)
    return existing if existing.active else store.activate(existing.id)


def cmd_auth_imap_login(
    user: Annotated[str, typer.Option("--user", help="The mailbox address, e.g. you@example.com.")],
    host: Annotated[
        str | None,
        typer.Option("--host", help="IMAP server (default: known for common providers)."),
    ] = None,
    port: Annotated[int | None, typer.Option("--port", help="Default: 993 (ssl), 143.")] = None,
    security: Annotated[
        str,
        typer.Option(
            "--security",
            help="ssl (993), starttls (143), or plain (a local bridge on localhost only).",
        ),
    ] = "ssl",
    username: Annotated[
        str | None, typer.Option("--username", help="Login name, if not the address.")
    ] = None,
    folder: Annotated[
        str, typer.Option("--folder", help="The folder IRIS syncs.")
    ] = DEFAULT_FOLDER,
    password_stdin: Annotated[
        bool,
        typer.Option("--password-stdin", help="Read the app password from stdin (scripts)."),
    ] = False,
) -> None:
    """Add an IMAP account: log in once with an app password, then keep it in the vault."""
    from .connection import ImapAuthError, ImapError

    address = user.strip().lower()
    if security not in SECURITIES:
        print_error(f"--security must be one of {', '.join(SECURITIES)}")
        raise typer.Exit(2)
    sec: Security = security
    resolved_host = host or preset_host(address)
    if not resolved_host:
        resolved_host = typer.prompt("IMAP server (host)").strip()
    if password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = _clean_hidden_input(typer.prompt(f"App password for {address}", hide_input=True))
    if not password:
        print_error("an app password is required")
        raise typer.Exit(2)
    try:
        account = ImapAccount(
            address=address,
            host=resolved_host,
            username=username or address,
            password=password,
            port=port or DEFAULT_PORTS[sec],
            security=sec,
            folder=folder,
        )
        row = connect_account(account)
    except ImapAccountError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc
    except ImapAuthError as exc:
        print_error(str(exc))
        raise typer.Exit(3) from exc
    except ImapError as exc:
        print_error(str(exc))
        raise typer.Exit(4) from exc
    console.print(
        f"  [bold green]OK[/bold green]  signed in to [cyan]{account.host}[/cyan] as "
        f"[cyan]{row.address}[/cyan] [dim](account {row.id})[/dim]"
    )
    console.print(
        "  [dim]App password stored in the vault. IRIS reads this mailbox; it changes "
        "nothing in it until you approve the label preview in email setup.[/dim]"
    )


def cmd_auth_imap_status() -> None:
    """Show every IMAP account: vault entry, last login, write approval."""
    from iris_personal.email.accounts import EmailAccountStore
    from iris_personal.email.write_approvals import mailbox_write_approval

    from .state import ImapState

    store = EmailAccountStore()
    store.ensure_schema()
    accounts = [a for a in store.list(active_only=False) if a.provider == IMAP_PROVIDER]
    if not accounts:
        console.print("  [yellow]no IMAP accounts configured[/yellow]")
        console.print("  [dim]run `iris auth imap login --user <address>` to add one[/dim]")
        return
    state = ImapState()
    for acc in accounts:
        creds = load_account(acc.id)
        marker = "[bold green]ok[/bold green]" if creds else "[red]missing[/red]"
        active = "active" if acc.active else "[dim]inactive[/dim]"
        console.print(f"  {marker}  [cyan]{acc.address}[/cyan]  [dim]({active})[/dim]")
        if creds is None:
            console.print("      [yellow]no app password in the vault; log in again[/yellow]")
            continue
        console.print(
            f"      [dim]server:[/dim] {creds.host}:{creds.port} ({creds.security})   "
            f"[dim]folder:[/dim] {creds.folder}"
        )
        status = state.status(acc.id)
        if status is not None and status.auth_failed:
            console.print(f"      [red]login refused at {status.last_error_at}[/red]")
        elif status is not None and status.last_ok_at:
            console.print(f"      [dim]last login:[/dim] {status.last_ok_at}")
        approval_row = mailbox_write_approval(acc.id)
        approval = approval_row.approval_ref if approval_row else None
        console.print(
            "      [dim]mailbox writes:[/dim] "
            + (f"approved ({approval})" if approval else "not approved (read-only)")
        )


def cmd_auth_imap_logout(
    user: Annotated[str, typer.Option("--user", help="The IMAP address to remove.")],
) -> None:
    """Remove the account's app password from the vault and deactivate its row."""
    from iris_personal.email.accounts import EmailAccountStore
    from iris_personal.email.write_approvals import revoke_mailbox_writes

    from .state import ImapState

    account_id = account_id_for(user)
    delete_account(account_id)
    ImapState().forget_account(account_id)
    # A new login is a new approval; the revoke leaves its own audit row.
    revoke_mailbox_writes(account_id, actor="owner (imap logout)", agent_type="cli")
    store = EmailAccountStore()
    store.ensure_schema()
    if store.get(account_id) is None:
        console.print(
            f"  [dim]no account row for[/dim] [cyan]{user}[/cyan]  "
            "[dim](vault entry removed if present)[/dim]"
        )
        return
    store.deactivate(account_id)
    console.print(
        f"  [bold green]OK[/bold green]  removed the app password for [cyan]{user}[/cyan] "
        "and deactivated its account row"
    )
