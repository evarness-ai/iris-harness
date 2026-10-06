"""``iris plugins`` — what the running IRIS actually loaded, from ``GET /plugins``.

``iris --dump-config`` answers "what would load": it reads manifests and never runs a
plugin's ``setup`` (release gate 3), so it cannot know what a plugin registered at
boot. This group reads the live API instead: each plugin's status, what it
registered (agents, tools, heartbeats…), which bus topics it subscribed to and on
which bus, and which core seams it filled (API routers, public callbacks, agent
panels, learned sources). Read-only.

``new`` goes the other way: it writes a standalone, installable plugin of one kind, with
tests that pass out of the box (OSS plan R10). The logic is ``cli/plugin_scaffold.py``;
this module renders it. ``iris plugin new`` is the same command (``main.py``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from iris_harness.cli.api_client import harness_api_client
from iris_harness.cli.plugin_scaffold import KINDS, PLUGIN_KINDS, ScaffoldError, new_plugin
from iris_harness.cli.render import console, print_error

plugins_app = typer.Typer(
    name="plugins",
    help="What the running IRIS loaded: each plugin's status, registrations, "
    "subscriptions and seams (GET /plugins).",
    invoke_without_command=True,
    no_args_is_help=False,
)

ApiUrl = Annotated[str | None, typer.Option("--api-url", help="IRIS API base URL.")]
AsJson = Annotated[bool, typer.Option("--json", help="Print the API's JSON as is.")]


def _fetch(api_url: str | None, path: str) -> dict[str, Any]:
    base = api_url or os.environ.get("IRIS_API_URL") or "http://localhost:8003"
    try:
        with harness_api_client(purpose="plugins", timeout=30.0) as client:
            resp = client.get(f"{base}{path}")
    except httpx.HTTPError as exc:
        print_error(f"could not reach IRIS API at {base} ({exc}). Is the stack up?")
        raise typer.Exit(1) from None
    if resp.status_code == 404:
        print_error(str(resp.json().get("detail") or f"not found: {path}"))
        raise typer.Exit(1)
    if resp.status_code >= 400:
        print_error(f"{path} answered HTTP {resp.status_code}")
        raise typer.Exit(1)
    data: dict[str, Any] = resp.json()
    return data


@plugins_app.callback()
def cmd_list(
    ctx: typer.Context,
    api_url: ApiUrl = None,
    as_json: AsJson = False,
) -> None:
    """List every plugin the running profile named, with what each registered."""
    if ctx.invoked_subcommand is not None:
        return
    data = _fetch(api_url, "/plugins")
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    profile = data.get("profile") or {}
    totals = ", ".join(f"{n} {k}" for k, n in (data.get("totals") or {}).items() if n)
    console.print(f"  [bold]profile[/bold] {profile.get('name', '?')}  [dim]({totals})[/dim]")
    for row in data.get("plugins") or []:
        counts = row.get("registration_counts") or {}
        parts = [f"{n} {kind}" for kind, n in sorted(counts.items())]
        if row.get("subscription_count"):
            parts.append(f"{row['subscription_count']} subscription(s)")
        if row.get("seam_count"):
            parts.append(f"{row['seam_count']} seam(s)")
        status = row.get("status", "?")
        colour = {"loaded": "green", "degraded": "yellow"}.get(status, "red")
        console.print(
            f"  [{colour}]{status:<9}[/{colour}] {row.get('name', '?'):<22} "
            f"[dim]{', '.join(parts) or 'nothing registered'}[/dim]  "
            f"[dim]party={row.get('party') or '?'}[/dim]"
        )
        if row.get("degraded_reason"):
            console.print(f"            [yellow]degraded: {row['degraded_reason']}[/yellow]")
        if row.get("last_error") or row.get("load_error"):
            console.print(
                f"            [red]{row.get('load_error') or row.get('last_error')}[/red]"
            )


def _capability_lines(caps: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Who provides and who uses what, from this plugin's side (plugin-capabilities §2)."""
    lines = [
        f"provides {c['name']}"
        + ("" if c.get("provided") else "  [yellow](never provided)[/yellow]")
        + f"  [dim]used by {', '.join(c.get('used_by') or []) or '-'}[/dim]"
        for c in caps.get("provides") or []
    ]
    for role, missing in (("uses", "none: degraded"), ("requires", "none")):
        lines.extend(
            f"{role} {c['name']}  [dim]provided by {', '.join(c.get('providers') or []) or missing}"
            "[/dim]"
            for c in caps.get(role) or []
        )
    return lines


def egress_lines(egress: dict[str, Any] | None) -> list[str]:
    """The hosts a plugin may contact, one line each (``open_web`` says so loudly)."""
    if not egress:
        return []
    if egress.get("open_web"):
        return ["[yellow]any host (open_web)[/yellow]  [dim]every call is recorded[/dim]"]
    lines = []
    for h in egress.get("hosts") or []:
        ports = f":{','.join(str(p) for p in h['ports'])}" if h.get("ports") else ""
        lines.append(f"{'/'.join(h['schemes'])}://{h['host']}{ports}  [dim]data: {h['data']}[/dim]")
    return lines


@plugins_app.command("show")
def cmd_show(
    name: Annotated[str, typer.Argument(help="Plugin name, as `iris plugins` lists it.")],
    api_url: ApiUrl = None,
    as_json: AsJson = False,
) -> None:
    """One plugin: registrations, subscriptions (with their bus), seams and drift."""
    data = _fetch(api_url, f"/plugins/{name}")
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    console.print(
        f"  [bold]{data.get('name')}[/bold]  {data.get('status')}  "
        f"[dim]{data.get('source')}  v{data.get('version') or '?'}  "
        f"trust={data.get('trust')}  party={data.get('party') or '?'}[/dim]"
    )
    if data.get("degraded_reason"):
        console.print(f"  [yellow]degraded: {data['degraded_reason']}[/yellow]")
    if data.get("last_error") or data.get("load_error"):
        console.print(f"  [red]{data.get('load_error') or data.get('last_error')}[/red]")
    sections: list[tuple[str, list[str]]] = [
        (
            "registrations",
            [f"{r['kind']}: {r['name']}" for r in data.get("registrations") or []],
        ),
        (
            "subscriptions",
            [f"{s['topic']}  [dim]@{s['scope']}[/dim]" for s in data.get("subscriptions") or []],
        ),
        ("seams", [f"{s['seam']}: {s['key']}" for s in data.get("seams") or []]),
        (
            "search providers",
            [
                p["name"] + ("" if p.get("registered") else "  [dim](not registered)[/dim]")
                for p in data.get("search_providers") or []
            ],
        ),
        ("capabilities", _capability_lines(data.get("capabilities") or {})),
        ("egress", egress_lines(data.get("egress"))),
        # OSS plan R17: the console screens it owns, in the nav while it is mounted.
        (
            "screens",
            [
                f"{s['route']}  {s['label']}"
                + ("" if s.get("nav", True) else "  [dim](no nav entry)[/dim]")
                for s in ((data.get("manifest") or {}).get("webui") or {}).get("screens") or []
            ],
        ),
    ]
    for title, lines in sections:
        console.print(f"  [bold]{title}[/bold] ({len(lines)})")
        for line in lines:
            console.print(f"    {line}")
        if not lines:
            empty = (
                "none declared (the governed client contacts no host)"
                if title == "egress"
                else "none"
            )
            console.print(f"    [dim]{empty}[/dim]")
    drift = {k: v for k, v in (data.get("drift") or {}).items() if v}
    if drift:
        console.print("  [bold yellow]drift[/bold yellow]")
        for key, values in drift.items():
            console.print(f"    {key}: {', '.join(values)}")


_KIND_HELP = "; ".join(f"{kind.name}: {kind.summary}" for kind in PLUGIN_KINDS.values())


@plugins_app.command("new")
def cmd_new(
    name: Annotated[str, typer.Argument(help="The plugin's name, e.g. weather-now.")],
    kind: Annotated[str, typer.Option("--kind", "-k", help=f"What it is. {_KIND_HELP}.")],
    directory: Annotated[
        Path,
        typer.Option("--dir", help="Where to create the plugin's directory (<dir>/<name>)."),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite the scaffold's files in an existing directory."),
    ] = False,
) -> None:
    """Write a new plugin package: pyproject, manifest, setup(api), README and tests."""
    if kind not in KINDS:
        print_error(f"unknown kind {kind!r}: one of {', '.join(KINDS)}")
        raise typer.Exit(2)
    try:
        result = new_plugin(name, kind, parent=directory, force=force)
    except ScaffoldError as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None
    console.print(f"  [bold]{result.names.name}[/bold]  {kind} plugin  [dim]{result.root}[/dim]")
    for path in result.files:
        console.print(f"    {path.relative_to(result.root)}")
    console.print(
        f"\n  next:  cd {result.root} && pip install -e '.[test]' && pytest\n"
        f"  then list it in a profile (~/.iris/profile.yaml): plugins: [{{name: {name}}}]",
        markup=False,
    )


__all__ = ["plugins_app"]
