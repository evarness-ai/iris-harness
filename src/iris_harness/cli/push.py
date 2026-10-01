"""``iris push`` — see who is subscribed, and send a real notification.

This is the command that makes track 2b provable. Everything else can be
tested locally; whether *iOS* actually delivers a push to a home-screen web
app on a tailnet-only origin cannot, and the plan holds PR 10 until the owner
has seen a banner on the real phone. ``iris push test`` is how.

Unlike ``iris device``, this one opens the store directly rather than going
through the API: sending a push is not a capability the console should expose
— a "notify everyone" button is a foghorn — and the operator running this is
already on the host with the data directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from iris_harness.services.channels.web_push import PushSubscriptionStore

console = Console()

push_app = typer.Typer(
    name="push",
    help="Web Push: who is subscribed, and a test notification.",
    no_args_is_help=True,
)


def _store() -> PushSubscriptionStore:
    # Imported here, not at module scope: `iris --help` should not open a
    # database, and every CLI group in this package is imported at startup.
    from iris_harness.services.channels.web_push import (
        PushSubscriptionStore as Store,
    )

    return Store()


@push_app.command("list")
def list_subscriptions() -> None:
    """Every browser subscribed to notifications."""
    subs = _store().list()
    if not subs:
        console.print(
            "[yellow]Nothing is subscribed.[/yellow] On the phone: open the console, "
            "Add to Home Screen, then Devices → Turn on notifications."
        )
        return

    table = Table(title=f"{len(subs)} subscription(s)")
    table.add_column("label")
    table.add_column("device")
    table.add_column("service")
    table.add_column("last sent")
    table.add_column("fails", justify="right")
    for sub in subs:
        # The endpoint is a bearer capability — whoever holds it can ask the
        # push service to deliver — so only its host is ever printed.
        from urllib.parse import urlparse

        table.add_row(
            sub.label[:40] or "—",
            (sub.device_id or "—")[:12],
            urlparse(sub.endpoint).netloc,
            sub.last_sent_at or "never",
            str(sub.failures),
        )
    console.print(table)


@push_app.command("key")
def show_key() -> None:
    """The VAPID public key this harness signs with."""
    from iris_harness.services.channels.web_push import keys

    console.print(f"public key: [cyan]{keys.public_key_b64()}[/cyan]")
    console.print(f"subject:    {keys.subject()}")
    console.print(f"private key at [dim]{keys.key_path()}[/dim] (0600, never rotate casually)")


@push_app.command("test")
def send_test(
    body: Annotated[str, typer.Option(help="What the notification says.")] = (
        "If you can read this, Web Push works on this device."
    ),
    title: Annotated[str, typer.Option(help="The notification title.")] = "IRIS",
    url: Annotated[str, typer.Option(help="Where a tap should land.")] = "/chat",
) -> None:
    """Send a notification to every subscribed browser, now.

    The proof that PR 10 waits on: a banner on a locked phone, arriving over
    the push service rather than the tailnet.
    """
    from iris_harness.services.channels.models import ChannelMessage
    from iris_harness.services.channels.web_push import WebPushConnector

    connector = WebPushConnector(store=_store())
    receipt = connector.send(
        ChannelMessage(recipient="*", subject=title, body=body, metadata={"url": url})
    )

    if receipt.status == "sent":
        console.print(f"[green]sent[/green] to {receipt.message_id} subscription(s)")
        if receipt.error:
            console.print(f"[yellow]some failed:[/yellow] {receipt.error}")
        return
    if receipt.status == "skipped":
        console.print(f"[yellow]{receipt.error}[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"[red]failed:[/red] {receipt.error}")
    raise typer.Exit(code=1)


__all__ = ["push_app"]
