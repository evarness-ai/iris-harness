"""``iris device`` — pair, list and revoke the devices that may use the console (ADR-0117).

Thin renderers over ``/api/v1/devices`` on ``iris_api``, authenticated with the service
secret. The CLI never opens ``devices.db``: the API process holds the one
``DeviceService`` (and its failed-claim throttle), and "every capability has an API"
means the terminal is a client of it like the web console is. This is also how the
FIRST device gets paired — from the host's shell, where the secret already is.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from iris_harness.cli.api_client import harness_urlopen
from iris_harness.foundation.auth import auth_headers, expected_secret

console = Console()

device_app = typer.Typer(
    name="device",
    help="Pair, list and revoke the phones and browsers that may use the console.",
    no_args_is_help=True,
)

_DEFAULT_API = "http://127.0.0.1:8003"
_DEVICES = "/api/v1/devices"

ApiOption = Annotated[str | None, typer.Option("--api", metavar="URL", help="IRIS API base URL")]
JsonOption = Annotated[bool, typer.Option("--json", help="Print the raw JSON")]


def _fail(message: str) -> typer.Exit:
    console.print(f"[bold red]✗[/bold red] {message}")
    return typer.Exit(1)


def _detail(exc: urllib.error.HTTPError) -> str:
    raw = exc.read().decode(errors="replace")[:300]
    try:
        detail = json.loads(raw).get("detail")
    except (ValueError, AttributeError):
        return raw
    return detail if isinstance(detail, str) else raw


def _call(api: str | None, path: str, *, method: str = "GET", body: Any = None) -> Any:
    base = (api or os.environ.get("IRIS_API_URL", _DEFAULT_API)).rstrip("/")
    headers = auth_headers()
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(  # noqa: S310 — the operator's own API URL
        f"{base}{path}", headers=headers, method=method, data=data
    )
    try:
        with harness_urlopen(req, purpose="devices", timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            why = (
                "IRIS_AUTH_SECRET is not set in this shell, so no credential was sent"
                if expected_secret() is None
                else "IRIS_AUTH_SECRET in this shell is not the secret the API runs with "
                "(a common cause: re-running the generator command in each shell makes a "
                "*different* secret each time -- write it to a file once and export that)"
            )
            raise _fail(f"the API at {base} refused the request (401): {why}.") from exc
        raise _fail(f"HTTP {exc.code} from {base}: {_detail(exc)}") from exc
    except OSError as exc:
        raise _fail(f"IRIS API unreachable at {base} — is it running?") from exc


def _resolve(api: str | None, wanted: str) -> dict[str, Any]:
    """The one device ``wanted`` names: a full ID, or a prefix only one device has."""
    devices: list[dict[str, Any]] = _call(api, _DEVICES)["devices"]
    exact = [d for d in devices if d["device_id"] == wanted]
    matches = exact or [d for d in devices if d["device_id"].startswith(wanted)]
    if not matches:
        raise _fail(f"no device ID starts with {wanted!r} — see `iris device list`.")
    if len(matches) > 1:
        console.print(f"[bold red]✗[/bold red] {wanted!r} matches {len(matches)} devices:")
        for d in matches:
            console.print(f"   {d['device_id']}  {escape(d['name'])}")
        console.print("Give more of the ID.")
        raise typer.Exit(1)
    return matches[0]


@device_app.command("pair")
def cmd_pair(
    scope: Annotated[
        str,
        typer.Option("--scope", help="read = look only; control = may also act and approve."),
    ] = "control",
    api: ApiOption = None,
    as_json: JsonOption = False,
) -> None:
    """Start pairing: print a one-time code to type on the new device."""
    if scope not in ("read", "control"):
        raise _fail("--scope must be 'read' or 'control'.")
    started = _call(api, f"{_DEVICES}/pair/start", method="POST", body={"scope": scope})
    if as_json:
        console.print_json(data=started)
        return
    console.print(f"Pairing code  [bold cyan]{started['code']}[/bold cyan]")
    console.print(f"  scope    {started['scope']}")
    console.print(f"  expires  {started['expires_at'][:19]}Z  (5 minutes, single use)")
    console.print(
        "Open the console on the new device and enter the code there. "
        "Five wrong guesses void it; run this again for a new one."
    )


@device_app.command("list")
def cmd_list(api: ApiOption = None, as_json: JsonOption = False) -> None:
    """List paired devices, newest first, revoked ones included."""
    devices: list[dict[str, Any]] = _call(api, _DEVICES)["devices"]
    if as_json:
        console.print_json(data={"devices": devices})
        return
    if not devices:
        console.print("[dim]No devices paired. Start with `iris device pair`.[/dim]")
        return
    table = Table(title="Paired devices", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Kind", style="dim")
    table.add_column("Scope", style="yellow")
    table.add_column("Paired", style="dim")
    table.add_column("Last seen", style="dim")
    table.add_column("Status")
    for d in devices:
        revoked = d.get("revoked_at")
        table.add_row(
            d["device_id"][:8],
            escape(d["name"]),  # typed on the device being paired: text, not markup
            d["kind"],
            d["scope"],
            d["created_at"][:19],
            (d.get("last_seen_at") or "never")[:19],
            f"[red]revoked {revoked[:19]}[/red]" if revoked else "[green]active[/green]",
        )
    console.print(table)
    console.print("[dim]`iris device revoke <ID>` takes the short ID shown here.[/dim]")


@device_app.command("revoke")
def cmd_revoke(
    device: Annotated[str, typer.Argument(help="Device ID, or a prefix only one device has.")],
    api: ApiOption = None,
) -> None:
    """Revoke a device: its token stops working on its next request."""
    target = _resolve(api, device)
    if target.get("revoked_at"):
        console.print(
            f"[dim]{escape(target['name'])} ({target['device_id']}) was already revoked "
            f"at {target['revoked_at'][:19]}.[/dim]"
        )
        return
    path = f"{_DEVICES}/{urllib.parse.quote(target['device_id'], safe='')}"
    revoked = _call(api, path, method="DELETE")["device"]
    console.print(
        f"[green]✓[/green] Revoked {escape(revoked['name'])} "
        f"[dim]({revoked['device_id']}, at {str(revoked['revoked_at'])[:19]})[/dim]"
    )
