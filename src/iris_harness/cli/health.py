"""``iris health`` — external connectivity and the health watch's incidents (ADR-0116).

Thin renderers over the API (``GET /health/connectors``, ``GET /health/incidents``,
``POST /health/watch``): the watch runs inside the API process, so that is where its
state lives. ``iris status`` stays the one-line "is the API up" check.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Annotated, Any

import typer
from rich.console import Console

from iris_harness.cli.api_client import harness_urlopen
from iris_harness.foundation.auth import auth_headers

console = Console()

health_app = typer.Typer(
    name="health",
    help="External connectivity (Gmail, Calendar, …) and what the health watch did.",
    no_args_is_help=True,
)

_DEFAULT_API = "http://127.0.0.1:8003"
_STATE_STYLE = {"green": "green", "yellow": "yellow", "red": "bold red", "grey": "dim"}

ApiOption = Annotated[str | None, typer.Option("--api", metavar="URL", help="IRIS API base URL")]
JsonOption = Annotated[bool, typer.Option("--json", help="Print the raw JSON")]


def _call(api: str | None, path: str, *, method: str = "GET", timeout: float = 30) -> Any:
    base = api or os.environ.get("IRIS_API_URL", _DEFAULT_API)
    req = urllib.request.Request(  # noqa: S310
        f"{base}{path}",
        headers=auth_headers(),
        method=method,
        data=b"" if method == "POST" else None,
    )
    try:
        with harness_urlopen(req, purpose="health", timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        console.print(f"[bold red]✗[/bold red] HTTP {exc.code}: {exc.read().decode()[:300]}")
        raise typer.Exit(1) from exc
    except OSError as exc:
        console.print(f"[bold red]✗[/bold red] IRIS API unreachable at {base} — is it running?")
        raise typer.Exit(1) from exc


def _state(value: str, width: int = 0) -> str:
    """Colour a state word. Pad the word itself, never the markup around it: the
    tags differ in length per state, so padding the marked-up string misaligns rows."""
    padded = value.ljust(width)
    style = _STATE_STYLE.get(value, "")
    return f"[{style}]{padded}[/{style}]" if style else padded


def _print_incident(item: dict[str, Any]) -> None:
    who = f"{item['target']} ({item['subject']})" if item.get("subject") else item["target"]
    end = item.get("resolution") or item.get("state")
    console.print(f"#{item['id']} {who}  [dim]{item['opened_at'][:19]}[/dim]  {end}")
    console.print(f"   {item['detail']}")
    for repair in item.get("repairs") or []:
        mark = "✓" if repair.get("ok") else "✗"
        extra = f" — {repair['detail']}" if repair.get("detail") and not repair.get("ok") else ""
        console.print(f"   {mark} {repair.get('tried')}{extra}")
    if item.get("notify_count"):
        console.print(f"   told you {item['notify_count']}× (last {item['notified_at'][:19]})")
    if item.get("action") and not item.get("resolved_at"):
        console.print(f"   fix: [bold]{item['action']}[/bold]")


@health_app.command("connectors")
def cmd_connectors(
    live: Annotated[
        bool, typer.Option("--live", help="Exercise each token against its provider now")
    ] = False,
    api: ApiOption = None,
    as_json: JsonOption = False,
) -> None:
    """Is every external connection (Gmail, Calendar, Drive, API keys) working?"""
    body = _call(api, f"/health/connectors?live={'true' if live else 'false'}", timeout=60)
    if as_json:
        console.print_json(data=body)
        return
    console.print(f"connectors: {_state(body['state'])}  [dim]{body['sampled_at'][:19]}[/dim]")
    for row in body["connectors"]:
        console.print(f"  {_state(row['state'], 6)} {row['target']:<10} {row['detail']}")
        if row.get("action") and row["state"] in {"red", "yellow"}:
            console.print(f"  {'':<17} fix: [bold]{row['action']}[/bold]")
    if body["state"] == "red":
        raise typer.Exit(1)


@health_app.command("incidents")
def cmd_incidents(
    open_only: Annotated[bool, typer.Option("--open", help="Only what is still broken")] = False,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    api: ApiOption = None,
    as_json: JsonOption = False,
) -> None:
    """What broke, what IRIS tried, and whether it asked you."""
    body = _call(
        api, f"/health/incidents?open_only={'true' if open_only else 'false'}&limit={limit}"
    )
    if as_json:
        console.print_json(data=body)
        return
    if not body.get("enabled", True):
        console.print("[yellow]health watch is off (IRIS_HEALTH_WATCH_ENABLED=0)[/yellow]")
    if not body["incidents"]:
        console.print("no incidents" + (" open" if open_only else ""))
        return
    for item in body["incidents"]:
        _print_incident(item)


@health_app.command("watch")
def cmd_watch(api: ApiOption = None, as_json: JsonOption = False) -> None:
    """Run one watch pass now: refresh, repair, notify (needs IRIS_WEBUI_ALLOW_WRITES=1)."""
    body = _call(api, "/health/watch", method="POST", timeout=120)
    if as_json:
        console.print_json(data=body)
        return
    console.print(f"health: {_state(body['state'])} — {body['summary']}")
    for event in body["events"]:
        console.print(f"  {event}")
    if body["open"]:
        console.print(f"{len(body['open'])} open incident(s):")
        for item in body["open"]:
            _print_incident(item)


__all__ = ["health_app"]
