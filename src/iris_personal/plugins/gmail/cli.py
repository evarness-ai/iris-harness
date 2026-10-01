"""``iris auth gmail`` and ``iris email label-from-vendor`` — the Gmail plugin's commands.

The harness publishes the ``auth`` group; this plugin attaches ``gmail`` under it
with ``login`` / ``status`` / ``logout``, bodies unchanged from ``main.py``. The CLI
seam runs with no runtime built, so each body builds what it needs itself.

``label-from-vendor`` joins the ``iris email`` group: it maps the Gmail labels already
stored in email.db through this plugin's tab table. It lives here, not in
``email_workflows``, because the table is Gmail's and that plugin never imports this
one; a CLI process runs no ``setup``, so the mail-provider registry is empty there.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from iris_harness.sdk.cli import console, print_error
from iris_harness.sdk.persistence import data_path

if TYPE_CHECKING:
    from iris_harness.sdk import PluginCLI

    from .provider import GmailProvider


def register(cli: PluginCLI) -> None:
    gmail_app = typer.Typer(
        name="gmail",
        help="Gmail OAuth (installed-app flow). See docs/usage-guides/gmail-auth.md for one-time setup.",
        no_args_is_help=True,
    )
    gmail_app.command("login")(cmd_auth_gmail_login)
    gmail_app.command("status")(cmd_auth_gmail_status)
    gmail_app.command("logout")(cmd_auth_gmail_logout)
    cli.group("auth").add_typer(gmail_app, name="gmail")
    # Profile order mounts this plugin before email_workflows, so this call may be the
    # one that creates `iris email`; it carries that group's help so `iris --help`
    # reads the same either way.
    cli.group("email", help=_EMAIL_GROUP_HELP).command("label-from-vendor")(
        cmd_email_label_from_vendor
    )
    # `iris system status` is a core command with no runtime to ask, so the account
    # count seam is filled here as well as in plugin.setup().
    from iris_personal.email.accounts import register_account_count
    from iris_personal.email.providers import offer_cli_mail_provider

    register_account_count()
    # A command with no runtime (`iris email setup`) mounts this when it needs a mailbox;
    # offering it imports nothing, so `iris --help` never loads the Google client.
    offer_cli_mail_provider("gmail", _build_provider)


def _build_provider() -> GmailProvider:
    from .provider import GmailProvider

    return GmailProvider()


_EMAIL_GROUP_HELP = (
    "Email domain commands (Phase 1+). Operates on the locally-stored "
    "email.db; per-account opt-in."
)


def cmd_auth_gmail_login(
    user: Annotated[str, typer.Option("--user", help="Gmail address to authorize.")],
    client_secrets: Annotated[
        Path | None,
        typer.Option(
            "--client-secrets",
            help="Path to OAuth client_secret.json (default: "
            "$IRIS_HOME/workspace/credentials/google_oauth_client.json).",
        ),
    ] = None,
) -> None:
    """Run the installed-app OAuth flow and persist Gmail tokens."""
    from . import gmail_oauth

    try:
        account = gmail_oauth.login(user, client_secrets_path=client_secrets)
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(3) from exc

    console.print(
        f"  [bold green]✓[/bold green]  authorized [cyan]{account.address}[/cyan]"
        f" [dim](provider={account.provider})[/dim]"
    )
    console.print(
        "  [dim]Tokens stored in macOS Keychain. Account registered in data/iris.db.[/dim]"
    )


def cmd_auth_gmail_status() -> None:
    """Show the auth state for every configured Gmail account."""
    from . import gmail_oauth

    snapshots = gmail_oauth.status()
    if not snapshots:
        console.print("  [yellow]no Gmail accounts configured[/yellow]")
        console.print("  [dim]run `iris auth gmail login --user <address>` to add one[/dim]")
        return

    for s in snapshots:
        acc = s.email_account
        marker = "[bold green]✓[/bold green]" if s.has_keychain_token else "[red]✗[/red]"
        active = "active" if acc.active else "[dim]inactive[/dim]"
        console.print(f"  {marker}  [cyan]{acc.address}[/cyan]  [dim]({active})[/dim]")
        if s.has_keychain_token:
            expiry = s.token_expiry.isoformat() if s.token_expiry else "<unknown>"
            refresh = "yes" if s.refresh_token_present else "no"
            console.print(
                f"      [dim]expires:[/dim] {expiry}   [dim]refresh token:[/dim] {refresh}"
            )
            if s.scopes:
                console.print(f"      [dim]scopes:[/dim] {', '.join(s.scopes)}")
        else:
            console.print(
                "      [yellow]no token in Keychain — run `iris auth gmail login` again[/yellow]"
            )


def cmd_auth_gmail_logout(
    user: Annotated[str, typer.Option("--user", help="Gmail address to deactivate.")],
) -> None:
    """Remove cached Gmail tokens and deactivate the account row."""
    from . import gmail_oauth

    deactivated = gmail_oauth.logout(user)
    if deactivated:
        console.print(
            f"  [bold green]✓[/bold green]  removed token for [cyan]{user}[/cyan]"
            " and deactivated email_account row"
        )
    else:
        console.print(
            f"  [dim]no account row for[/dim] [cyan]{user}[/cyan]"
            "  [dim](keychain token removed if present)[/dim]"
        )


def cmd_email_label_from_vendor(
    account: Annotated[
        str,
        typer.Option("--account", help="email_accounts.id slug, e.g. gmail:user@gmail.com"),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Count what would change; write nothing."),
    ] = False,
    email_db_path: Annotated[
        Path | None,
        typer.Option(
            "--email-db",
            help="Override email.db location (default: email.db in the IRIS data dir).",
        ),
    ] = None,
) -> None:
    """Label stored mail from its Gmail tab (Promotions, Social, Updates, Forums).

    Re-derives each row's vendor category from the Gmail labels already stored in
    email.db, through the same table the fetcher applies to new mail
    (``vendor_categories.yaml``). Only rows IRIS has not classified are written; a
    row whose tab changed gets the new path. Idempotent: a second run changes nothing.
    """
    from iris_personal.email.store import EmailStore

    from .vendor_categories import vendor_category_for

    email_db = email_db_path or data_path("email.db")
    if not email_db.exists():
        print_error(f"email.db not found at {email_db}")
        raise typer.Exit(2)
    store = EmailStore(db_path=email_db)
    store.ensure_schema()
    result = store.backfill_vendor_categories(account, vendor_category_for, dry_run=dry_run)

    verb = "would label" if dry_run else "labelled"
    console.print(
        f"  [bold green]✓[/bold green]  {verb} [cyan]{result.written}[/cyan] email(s) "
        f"for [cyan]{account}[/cyan]"
        f"  [dim](new {result.set_new}, tab changed {result.replaced})[/dim]"
    )
    for path, n in sorted(result.by_category.items()):
        console.print(f"      [dim]{path}:[/dim] {n}")
    console.print(
        f"  [dim]scanned {result.scanned}; already labelled {result.unchanged}; "
        f"IRIS-classified, left alone {result.skipped_iris}; "
        f"no Gmail tab {result.no_vendor_category}[/dim]"
    )
    if dry_run:
        console.print("  [dim]dry run: no classification written[/dim]")
