"""IRIS CLI entry point.

Usage patterns
--------------
iris                          # interactive REPL (default)
iris --resume                 # resume most recent session for this directory
iris --session <id-prefix>    # resume a specific session by ID prefix
iris --print "your message"   # one-shot, Rich-formatted output
iris --print "..." --json     # one-shot, raw JSON output (scriptable)
iris --stdio                  # JSON-RPC over stdin/stdout (portability bridge)
iris sessions                 # list recent sessions
iris status                   # check API health
iris doctor                   # install preflight: can IRIS run here, what to fix
iris serve                    # run the IRIS API (scheduled jobs + web console)
iris auth copilot login       # device-flow OAuth for the Copilot backend
iris auth copilot status      # show cached Copilot OAuth token state
iris auth copilot logout      # delete cached Copilot OAuth token
iris --version                # show version
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv

from iris_harness.cli.render import console, print_error, print_sessions
from iris_harness.cli.session import Session as _Session
from iris_harness.cli.session import SessionManager
from iris_harness.foundation.paths import data_dir
from iris_harness.foundation.settings.env_overrides import apply_env_overrides

# Load .env from the user's cwd before any subcommand runs so flags like
# IRIS_ENABLE_COPILOT_BACKEND, IRIS_API_URL, and provider keys are visible to
# `iris ...` invocations the same way they are to the FastAPI services.
load_dotenv()
# The owner's saved setting changes win over .env, as they do in the servers (ADR-0120).
apply_env_overrides()

_DEFAULT_API = "http://localhost:8003"
_VERSION = "0.1.0"

app = typer.Typer(
    name="iris",
    help="[bold cyan]IRIS[/bold cyan] — Intelligent Reasoning & Integration System",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
)

_sessions = SessionManager()
_DEFAULT_VAULT_IMPORT_VARS = (
    "GITHUB_TOKEN",
    "OPENROUTER_API_KEY",
    "GITHUB_PAT_CODING_AGENT",
)


def _api_url(override: str | None) -> str:
    return override or os.environ.get("IRIS_API_URL", _DEFAULT_API)


def _resolve_session(
    session_id: str | None,
    resume: bool,
    session_manager: SessionManager,
) -> _Session:

    if session_id:
        sess = session_manager.load(session_id)
        if sess is None:
            print_error(f"No session matching '{session_id}'")
            raise typer.Exit(1)
        return sess
    if resume:
        return session_manager.continue_recent()
    return session_manager.create()


# ---------------------------------------------------------------------------
# Root command (also the default: `iris` launches the REPL)
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    api: Annotated[
        str | None, typer.Option("--api", metavar="URL", help="IRIS API base URL")
    ] = None,
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", metavar="ID", help="Resume session by ID prefix"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", "-r", help="Resume the most recent session for this directory"),
    ] = False,
    message: Annotated[
        str | None,
        typer.Option("--print", "-p", metavar="MSG", help="Send one message and print response"),
    ] = None,
    json_out: Annotated[
        bool, typer.Option("--json", help="Output raw JSON (use with --print)")
    ] = False,
    stdio: Annotated[
        bool, typer.Option("--stdio", help="JSON-RPC over stdin/stdout for programmatic use")
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Enable strict pre-response checks (runs heavy curator signals synchronously).",
        ),
    ] = False,
    version: Annotated[bool, typer.Option("--version", "-V", help="Show version and exit")] = False,
    dump_config: Annotated[
        bool,
        typer.Option(
            "--dump-config",
            help=(
                "Print the effective profile + plugin tree (layers, sources, kinds) and exit."
                " Discovery only; `iris plugins` shows what the running IRIS registered."
            ),
        ),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            metavar="NAME",
            help="Profile for --dump-config (default: $IRIS_PROFILE, else 'default' "
            "or the profile it prefers when that one's plugins are installed).",
        ),
    ] = None,
    json_config: Annotated[
        bool, typer.Option("--json-config", help="Emit --dump-config as JSON")
    ] = False,
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    if version:
        console.print(f"IRIS [cyan]{_VERSION}[/cyan]")
        raise typer.Exit()

    if dump_config:
        # Discovery only (manifests read, no plugin setup, no runtime build): fast
        # and safe anywhere. OSS plan release gate 3.
        from iris_harness.foundation.paths import config_dir as _config_dir
        from iris_harness.runtime.plugin_host.dump import dump_config as _dump
        from iris_harness.runtime.plugin_host.dump import render_json, render_text

        tree = _dump(_config_dir(), profile_name=profile)
        print(render_json(tree) if json_config else render_text(tree))  # noqa: T201 — CLI output
        raise typer.Exit()

    url = _api_url(api)

    # ── stdio / RPC bridge ─────────────────────────────────────────────────
    if stdio:
        from iris_harness.cli.modes import run_stdio_mode

        raise typer.Exit(run_stdio_mode(api_url=url, session_manager=_sessions, strict=strict))

    sess = _resolve_session(session, resume, _sessions)

    # ── one-shot modes ─────────────────────────────────────────────────────
    if message is not None:
        if json_out:
            from iris_harness.cli.modes import run_json_mode

            raise typer.Exit(
                run_json_mode(
                    message,
                    session=sess,
                    session_manager=_sessions,
                    api_url=url,
                    strict=strict,
                )
            )
        from iris_harness.cli.modes import run_print_mode

        raise typer.Exit(
            run_print_mode(
                message,
                session=sess,
                session_manager=_sessions,
                api_url=url,
                strict=strict,
            )
        )

    # ── interactive REPL (default) ─────────────────────────────────────────
    from iris_harness.cli.repl import run_repl
    from iris_harness.llm.providers import ProviderManager

    raise typer.Exit(
        run_repl(
            session=sess,
            session_manager=_sessions,
            api_url=url,
            provider_manager=ProviderManager(),
            strict=strict,
        )
    )


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


@app.command(name="sessions")
def cmd_sessions() -> None:
    """List recent IRIS sessions."""
    all_sessions = _sessions.list()
    print_sessions(all_sessions)


@app.command(name="status")
def cmd_status(
    api: Annotated[
        str | None, typer.Option("--api", metavar="URL", help="IRIS API base URL")
    ] = None,
) -> None:
    """Check IRIS API and service health."""
    import json
    import urllib.request

    from iris_harness.cli.api_client import harness_urlopen
    from iris_harness.foundation.auth import auth_headers

    url = _api_url(api)
    try:
        # `/health` is the System Health snapshot and needs the secret like any other
        # data route (ADR-0117); only `/healthz` is an open probe.
        request = urllib.request.Request(f"{url}/health", headers=auth_headers())  # noqa: S310
        with harness_urlopen(request, purpose="status", timeout=5) as resp:
            body = json.loads(resp.read())
        console.print(f"[bold green]✓[/bold green]  IRIS API online  [dim]{url}[/dim]")
        if isinstance(body, dict):
            for k, v in body.items():
                console.print(f"  [dim]{k}:[/dim] {v}")
    except OSError as exc:
        console.print(f"[bold red]✗[/bold red]  IRIS API unreachable  [dim]{url}[/dim]")
        raise typer.Exit(1) from exc


# `iris doctor`: the install preflight (OSS plan R5). Runs in-process -- nothing is up yet.
from iris_harness.cli.doctor import cmd_doctor  # noqa: E402

app.command(name="doctor")(cmd_doctor)

# `iris serve`: the IRIS API in the foreground (the scheduled jobs and the web console).
from iris_harness.cli.serve import cmd_serve  # noqa: E402

app.command(name="serve")(cmd_serve)


# ---------------------------------------------------------------------------
# `iris auth ...` — credential management for backends that need OAuth
# ---------------------------------------------------------------------------


auth_app = typer.Typer(
    name="auth",
    help="Manage credentials for LLM backends that need device-flow OAuth.",
    no_args_is_help=True,
)
copilot_app = typer.Typer(
    name="copilot",
    help="GitHub Copilot device-flow OAuth (requires IRIS_ENABLE_COPILOT_BACKEND=1).",
    no_args_is_help=True,
)
auth_app.add_typer(copilot_app, name="copilot")
app.add_typer(auth_app, name="auth")


from iris_harness.cli.mission import mission_app  # noqa: E402

app.add_typer(mission_app, name="mission")

# `iris plan` is the planner plugin's whole group (OSS plan M5.7 track A).

from iris_harness.cli.docs import docs_app  # noqa: E402

app.add_typer(docs_app, name="docs")

from iris_harness.cli.facts import facts_app  # noqa: E402

app.add_typer(facts_app, name="facts")

from iris_harness.cli.memory import memory_app  # noqa: E402

app.add_typer(memory_app, name="memory")

from iris_harness.cli.ontology import ontology_app  # noqa: E402

app.add_typer(ontology_app, name="ontology")

from iris_harness.cli.behaviors import behaviors_app  # noqa: E402

app.add_typer(behaviors_app, name="behaviors")

from iris_harness.cli.signals import signals_app  # noqa: E402

app.add_typer(signals_app, name="signals")

from iris_harness.cli.intentions import intentions_app  # noqa: E402

app.add_typer(intentions_app, name="intentions")

from iris_harness.cli.context_health import context_health_app  # noqa: E402

app.add_typer(context_health_app, name="context-health")

from iris_harness.cli.governance import governance_app  # noqa: E402

app.add_typer(governance_app, name="governance")

from iris_harness.cli.learning import learning_app  # noqa: E402

app.add_typer(learning_app, name="learning")

from iris_harness.cli.plugins import plugins_app  # noqa: E402

app.add_typer(plugins_app, name="plugins")
# OSS plan R10 spells the scaffold `iris plugin new`: the same group, so `iris plugins new`
# works too, and the singular stays out of `iris --help`.
app.add_typer(plugins_app, name="plugin", hidden=True)

from iris_harness.cli.health import health_app  # noqa: E402
from iris_harness.cli.logs import logs_app  # noqa: E402

app.add_typer(logs_app, name="logs")

app.add_typer(health_app, name="health")

from iris_harness.cli.playground import playground_app  # noqa: E402

app.add_typer(playground_app, name="playground")

# OSS plan M2.6: let the profile's plugins add their own subcommands. Discovery
# only -- manifests are read and one module per plugin is imported; no runtime is
# built and no `setup` runs, so `iris --help` stays as cheap as it was. A plugin
# whose CLI module fails is logged and skipped rather than breaking the CLI.
from iris_harness.sdk import PluginCLI, register_plugin_commands  # noqa: E402

_plugin_cli = PluginCLI(root=app)
# `iris calendar` is the calendar plugin's whole group as of M6.1b: the domain left
# the core tree (OSS plan M6, decision 2), so the plugin publishes the group itself.
# It also owns `iris auth gcalendar`, and the gmail plugin owns `iris auth gmail`, so
# the `auth` group is published here for plugins that bring their own OAuth.
_plugin_cli.register_group("auth", auth_app)
register_plugin_commands(_plugin_cli)


system_app = typer.Typer(
    name="system",
    help="System agent — host resources + IRIS self-introspection.",
    no_args_is_help=True,
)
app.add_typer(system_app, name="system")


@system_app.command("status")
def cmd_system_status() -> None:
    """Show host resources + what IRIS has connected/running."""
    from iris_harness.services.system.status import host_status, iris_status

    host = host_status()
    iris = iris_status()
    console.print(f"  [bold]host[/bold]   {host.summary()}")
    accounts = ", ".join(f"{p}×{n}" for p, n in sorted(iris.accounts.items())) or "none"
    console.print(f"  [bold]accounts[/bold]   {accounts}")
    console.print(
        f"  [bold]iris[/bold]   {iris.skill_count} skills, {iris.heartbeat_count} heartbeats, "
        f"{iris.filemanager_roots} file root(s)"
    )
    if iris.database_sizes:
        dbs = ", ".join(f"{name} {size // 1024}KB" for name, size in iris.database_sizes.items())
        console.print(f"  [dim]databases:[/dim] {dbs}")


mcp_app = typer.Typer(
    name="mcp",
    help="Expose IRIS's own tools AS an MCP server; sign + verify external servers.",
    no_args_is_help=True,
)
app.add_typer(mcp_app, name="mcp")

# Phase 6 (6b.3): attach `mcp keygen|sign|verify` to the same group as `mcp serve`.
from iris_harness.cli.mcp import register_signing_commands  # noqa: E402

register_signing_commands(mcp_app)


@mcp_app.command("serve")
def cmd_mcp_serve(
    tool: Annotated[
        list[str] | None,
        typer.Option(
            "--tool",
            help="Registered tool to serve (repeatable; default: every contained read -- "
            "no egress, no third-party text, no code execution).",
        ),
    ] = None,
    skill: Annotated[
        list[str] | None,
        typer.Option(
            "--skill",
            help="Skill package whose read tools to serve too (repeatable; none by default).",
        ),
    ] = None,
    client: Annotated[
        str,
        typer.Option(
            "--client",
            help="A client listed in the owner's mcp-serve.yaml; calls are governed and "
            "audited as mcp:<client>.",
        ),
    ] = "stdio",
) -> None:
    """Serve IRIS tools over MCP (newline-delimited JSON-RPC on stdio), every call governed.

    See ``iris_harness.runtime.mcp_serve``: each call runs through PRE_TOOL_USE /
    POST_TOOL_USE and is audited as ``mcp:<client>``; a call that needs the owner's
    approval is refused; a personal result reaches only a client the owner declared
    local, a secret one no client.
    """
    import sys

    from iris_harness.runtime.bootstrap import build_runtime
    from iris_harness.runtime.mcp_serve import (
        McpServeError,
        client_is_local,
        load_mcp_clients,
        mcp_caller,
        select_tools,
        serve,
    )

    # stdout is the JSON-RPC channel and nothing else may write to it: everything the
    # runtime, a plugin or a tool prints goes to stderr for the life of the server.
    channel = sys.stdout
    sys.stdout = sys.stderr
    try:
        try:
            caller = mcp_caller(client)
            local = client_is_local(client, load_mcp_clients())
        except McpServeError as exc:
            print(f"iris-mcp: {exc}", file=sys.stderr)  # noqa: T201
            raise typer.Exit(2) from exc
        runtime = build_runtime(use_background_scheduler=False)
        if runtime.tool_service is None:
            print("iris-mcp: no tool service: no plugins mounted", file=sys.stderr)  # noqa: T201
            raise typer.Exit(1)
        if runtime.governance_kernel is None:
            print(  # noqa: T201
                "iris-mcp: the governance kernel is off, so every call will be refused",
                file=sys.stderr,
            )
        try:
            selection = select_tools(
                runtime.plugin_registry.tools(),
                tools=tuple(tool or ()),
                skills=tuple(skill or ()),
            )
        except McpServeError as exc:
            print(f"iris-mcp: {exc}", file=sys.stderr)  # noqa: T201
            raise typer.Exit(2) from exc
        for reason in selection.skipped:
            print(f"iris-mcp: not served: {reason}", file=sys.stderr)  # noqa: T201
        names = ", ".join(served.spec.name for served in selection.served) or "nothing"
        where = "local" if local else "not local: personal results are withheld"
        print(  # noqa: T201
            f"iris-mcp: serving {names} over stdio as {caller} ({where})", file=sys.stderr
        )
        serve(runtime.tool_service, selection.served, client=client, local=local, stdout=channel)
    finally:
        sys.stdout = channel


vault_app = typer.Typer(
    name="vault",
    help="Manage secrets in the local IRIS credential vault.",
    no_args_is_help=True,
)
app.add_typer(vault_app, name="vault")

from iris_harness.cli.approvals import approvals_app  # noqa: E402

app.add_typer(approvals_app, name="approvals")

from iris_harness.cli.device import device_app  # noqa: E402
from iris_harness.cli.push import push_app  # noqa: E402

app.add_typer(device_app, name="device")
app.add_typer(push_app, name="push")


@vault_app.command("add")
def cmd_vault_add(
    handle: Annotated[
        str, typer.Argument(help="Vault handle (for example: github-token or vault://github-token)")
    ],
    secret_value: Annotated[
        str,
        typer.Option(
            "--value",
            "-v",
            prompt=True,
            hide_input=True,
            help="Secret value to store (hidden input).",
        ),
    ],
    route: Annotated[
        str | None,
        typer.Option("--route", help="Optional governor route this credential is scoped for."),
    ] = None,
    replace: Annotated[
        bool,
        typer.Option("--replace", help="Replace the existing secret if the handle already exists."),
    ] = False,
) -> None:
    from iris_harness.kernel.governance.vault import VaultStore

    try:
        store = VaultStore()
        store.add(
            handle=handle,
            secret_value=secret_value,
            governor_route=route,
            replace=replace,
        )
    except Exception as exc:
        print_error(f"vault add failed: {exc}")
        raise typer.Exit(1) from exc

    normalized = handle if handle.startswith("vault://") else f"vault://{handle}"
    console.print(f"  [bold green]✓[/bold green]  stored [cyan]{normalized}[/cyan]")


@vault_app.command("list")
def cmd_vault_list() -> None:
    from iris_harness.kernel.governance.vault import VaultStore

    try:
        rows = VaultStore().list_metadata()
    except Exception as exc:
        print_error(f"vault list failed: {exc}")
        raise typer.Exit(1) from exc

    if not rows:
        console.print("  [dim]vault is empty[/dim]")
        return

    console.print("  [bold]Stored handles[/bold]")
    for row in rows:
        route = row.governor_route or "-"
        console.print(
            f"  - [cyan]{row.handle}[/cyan]  route=[dim]{route}[/dim]  "
            f"updated=[dim]{row.updated_at}[/dim]"
        )


@vault_app.command("remove")
def cmd_vault_remove(
    handle: Annotated[str, typer.Argument(help="Vault handle to remove.")],
) -> None:
    from iris_harness.kernel.governance.vault import VaultStore

    try:
        removed = VaultStore().remove(handle)
    except Exception as exc:
        print_error(f"vault remove failed: {exc}")
        raise typer.Exit(1) from exc

    normalized = handle if handle.startswith("vault://") else f"vault://{handle}"
    if removed:
        console.print(f"  [bold green]✓[/bold green]  removed [cyan]{normalized}[/cyan]")
    else:
        console.print(f"  [dim]no such handle:[/dim] [cyan]{normalized}[/cyan]")


@vault_app.command("import-env")
def cmd_vault_import_env(
    names: Annotated[
        list[str] | None,
        typer.Option(
            "--name",
            help=(
                "Environment variable name to import. Repeat --name multiple times. "
                "Defaults to GITHUB_TOKEN, OPENROUTER_API_KEY, GITHUB_PAT_CODING_AGENT."
            ),
        ),
    ] = None,
    replace: Annotated[
        bool,
        typer.Option("--replace", help="Replace existing handles when present."),
    ] = False,
) -> None:
    from iris_harness.kernel.governance.vault import VaultHandleAlreadyExistsError, VaultStore

    route_map = {
        "GITHUB_TOKEN": "coding/git",
        "GITHUB_PAT_CODING_AGENT": "coding/git",
        "OPENROUTER_API_KEY": "llm/cloud/*",
    }
    targets = tuple(names) if names else _DEFAULT_VAULT_IMPORT_VARS

    try:
        store = VaultStore()
    except Exception as exc:
        print_error(f"vault import-env failed: {exc}")
        raise typer.Exit(1) from exc

    imported = 0
    skipped_missing = 0
    skipped_conflict = 0
    for env_name in targets:
        value = (os.getenv(env_name) or "").strip()
        if not value:
            skipped_missing += 1
            continue
        handle = f"vault://{env_name.lower().replace('_', '-')}"
        try:
            store.add(
                handle=handle,
                secret_value=value,
                governor_route=route_map.get(env_name),
                replace=replace,
            )
            imported += 1
        except VaultHandleAlreadyExistsError:
            skipped_conflict += 1

    console.print(
        "  [bold green]✓[/bold green]  import complete  "
        f"[dim]imported={imported} missing={skipped_missing} conflicts={skipped_conflict}[/dim]"
    )


@vault_app.command("export")
def cmd_vault_export(
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Path to write the .env-style snapshot. Plaintext — encrypt at rest.",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite the output file if it already exists."),
    ] = False,
) -> None:
    """Write a .env-style snapshot of every vault entry.

    Useful for offline backup before rotating the master key. The output
    file contains PLAINTEXT secrets — keep it encrypted at rest and
    delete it once the backup has landed in your secret store.
    """
    import os
    from pathlib import Path

    from iris_harness.kernel.governance.vault import VaultStore

    output_path = Path(output).expanduser()
    if output_path.exists() and not force:
        print_error(f"refusing to overwrite existing file: {output_path} (pass --force)")
        raise typer.Exit(1)

    try:
        store = VaultStore()
    except Exception as exc:
        print_error(f"vault export failed: {exc}")
        raise typer.Exit(1) from exc

    rows = store.iter_secret_values()
    lines = ["# IRIS vault export (PLAINTEXT) — encrypt at rest, delete after backup"]
    for handle, value in rows:
        env_name = handle.removeprefix("vault://").upper().replace("-", "_")
        escaped = value.replace('"', '\\"')
        lines.append(f'{env_name}="{escaped}"  # handle={handle}')

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(output_path, 0o600)
    except OSError:
        pass  # non-POSIX or perms issue — warning already in the file header

    console.print(
        f"  [bold yellow]![/bold yellow]  exported {len(rows)} handle(s) in PLAINTEXT to "
        f"[cyan]{output_path}[/cyan]"
    )
    console.print(
        "  [dim]This file contains raw secret values. Encrypt at rest and delete "
        "after backup.[/dim]"
    )


# ---------------------------------------------------------------------------
# `iris checkpoint ...` — Phase 3 resume-checkpoint inspection (design §10)
# ---------------------------------------------------------------------------


checkpoint_app = typer.Typer(
    name="checkpoint",
    help="Inspect / pin / remove governance resume checkpoints.",
    no_args_is_help=True,
)
app.add_typer(checkpoint_app, name="checkpoint")


@checkpoint_app.command("list")
def cmd_checkpoint_list(
    agent_type: Annotated[
        str | None,
        typer.Option("--agent-type", help="Filter by agent type (chat, coding, ...)."),
    ] = None,
    include_expired: Annotated[
        bool,
        typer.Option("--include-expired", help="Include rows past their TTL (unpinned)."),
    ] = False,
) -> None:
    """List resume checkpoints, newest first."""
    from iris_harness.memory.state import CheckpointStore

    store = CheckpointStore()
    rows = store.list(agent_type=agent_type, include_expired=include_expired)
    if not rows:
        console.print("  [dim]no checkpoints[/dim]")
        return

    console.print(f"  [bold]Checkpoints[/bold] ({len(rows)})")
    for cp in rows:
        pin_marker = "[bold yellow]📌[/bold yellow] " if cp.pinned else "  "
        signal = cp.signal or "-"
        console.print(
            f"  {pin_marker}[cyan]{cp.run_id}[/cyan]  step={cp.step_id}  "
            f"agent={cp.agent_type}  signal=[dim]{signal}[/dim]  "
            f"ts=[dim]{cp.ts}[/dim]  expires=[dim]{cp.expires_at}[/dim]"
        )


@checkpoint_app.command("show")
def cmd_checkpoint_show(
    run_id: Annotated[str, typer.Argument(help="Run ID to show the latest checkpoint for.")],
) -> None:
    """Print the latest checkpoint for a run as JSON."""
    import json as _json

    from iris_harness.memory.state import CheckpointNotFoundError, CheckpointStore

    try:
        cp = CheckpointStore().get_latest(run_id)
    except CheckpointNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc

    console.print(
        _json.dumps(
            {
                "run_id": cp.run_id,
                "step_id": cp.step_id,
                "agent_type": cp.agent_type,
                "signal": cp.signal,
                "pinned": cp.pinned,
                "ts": cp.ts,
                "expires_at": cp.expires_at,
                "payload": cp.payload,
            },
            indent=2,
            sort_keys=True,
        )
    )


@checkpoint_app.command("pin")
def cmd_checkpoint_pin(
    run_id: Annotated[str, typer.Argument(help="Run ID to pin (exempt from TTL purge).")],
) -> None:
    from iris_harness.memory.state import CheckpointStore

    updated = CheckpointStore().pin(run_id)
    if updated == 0:
        print_error(f"no checkpoints for run={run_id!r}")
        raise typer.Exit(1)
    console.print(
        f"  [bold green]✓[/bold green]  pinned {updated} checkpoint(s) for run=[cyan]{run_id}[/cyan]"
    )


@checkpoint_app.command("unpin")
def cmd_checkpoint_unpin(
    run_id: Annotated[str, typer.Argument(help="Run ID to unpin (re-subject to TTL).")],
) -> None:
    from iris_harness.memory.state import CheckpointStore

    updated = CheckpointStore().unpin(run_id)
    if updated == 0:
        print_error(f"no checkpoints for run={run_id!r}")
        raise typer.Exit(1)
    console.print(
        f"  [bold green]✓[/bold green]  unpinned {updated} checkpoint(s) for run=[cyan]{run_id}[/cyan]"
    )


@checkpoint_app.command("remove")
def cmd_checkpoint_remove(
    run_id: Annotated[str, typer.Argument(help="Run ID to delete all checkpoints for.")],
) -> None:
    from iris_harness.memory.state import CheckpointStore

    deleted = CheckpointStore().remove(run_id)
    if deleted == 0:
        print_error(f"no checkpoints for run={run_id!r}")
        raise typer.Exit(1)
    console.print(
        f"  [bold green]✓[/bold green]  removed {deleted} checkpoint(s) for run=[cyan]{run_id}[/cyan]"
    )


# ---------------------------------------------------------------------------
# `iris run ...` — Phase 3 runtime control (story 12.gov-3.5)
# ---------------------------------------------------------------------------


run_app = typer.Typer(
    name="run",
    help="Inspect / control a governance-tracked run.",
    no_args_is_help=True,
)
app.add_typer(run_app, name="run")


@run_app.command("inspect")
def cmd_run_inspect(
    run_id: Annotated[str, typer.Argument(help="Run ID to inspect (governance run_id).")],
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'text' (default, Rich tables) or 'json'.",
        ),
    ] = "text",
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            help="Cap the number of audit rows shown (newest is always last).",
            min=1,
        ),
    ] = None,
) -> None:
    """Print the trace, signal trail, and final state of a run."""
    from iris_harness.cli.run_inspect import (
        UnknownRunError,
        load_run,
        render_json,
        render_text,
    )
    from iris_harness.kernel.governance.audit import AuditLog
    from iris_harness.memory.state import CheckpointStore

    if output_format not in ("text", "json"):
        print_error(f"unknown --format {output_format!r}; expected 'text' or 'json'")
        raise typer.Exit(2)

    try:
        summary, rows, checkpoint = load_run(
            run_id,
            audit_log=AuditLog(),
            checkpoint_store=CheckpointStore(),
            limit=limit,
        )
    except UnknownRunError as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc

    if output_format == "json":
        render_json(summary, rows, checkpoint, console=console)
    else:
        render_text(summary, rows, checkpoint, console=console)


@run_app.command("resume")
def cmd_run_resume(
    run_id: Annotated[str, typer.Argument(help="Run ID to resume.")],
    db_path: Annotated[
        str | None,
        typer.Option(
            "--db",
            help="Override side-effect ledger DB path.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print resume plan without executing anything.",
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help=(
                "If an approval is still gating this run, block until someone answers "
                "it (on any channel) and then continue."
            ),
        ),
    ] = False,
) -> None:
    """Resume a halted run: verify side effects, then continue the loop from its checkpoint.

    For each pending side effect:
      - completed   → skip (already done)
      - not_completed → mark for re-execution
      - ambiguous   → enqueue a HITL approval before retrying

    Then the run itself continues. **This is what the command always claimed to do and
    never did**: before ADR-0106 Tier B there was no way to re-enter a halted loop, so
    this verified the ledger, printed "re-run the original command", and stopped — while
    the halt message it was advertised in told the user to run it to resume. Now
    ``IrisRuntime.resume_halted_run`` exists and this uses it.

    A run the evaluator halted for approval is **not** resumed until that approval is
    answered. Stepping past it would walk straight through the human the evaluator
    stopped the run to consult, which is the entire purpose of the queue. ``--wait``
    blocks on the answer instead of refusing, so approving from the web or Telegram
    continues the run in this terminal.
    """
    from pathlib import Path as _Path

    from iris_harness.kernel.governance.side_effects import SideEffectLedger, run_probe
    from iris_harness.memory.state.store import (
        CheckpointNotFoundError,
        CheckpointStore,
    )

    ledger_db = _Path(db_path) if db_path else None
    ledger = SideEffectLedger(db_path=ledger_db)
    checkpoint_store = CheckpointStore()

    try:
        cp = checkpoint_store.get_latest(run_id)
    except CheckpointNotFoundError:
        print_error(f"no checkpoint found for run_id={run_id!r}")
        raise typer.Exit(1) from None

    console.print(
        f"[bold]Resume[/bold] run=[cyan]{run_id}[/cyan]  "
        f"step={cp.step_id}  signal={cp.signal or 'n/a'}"
    )

    pending = ledger.pending(run_id)
    if not pending:
        # The common case for a chat run: it wrote no irreversible side effects at all,
        # so there is nothing to verify. This used to `return` here, which is why the
        # command did nothing at all for the runs it was most often pointed at.
        console.print("  [dim]no pending side effects — safe to resume[/dim]")
        if dry_run:
            console.print("[dim]--dry-run: stopping before the run is continued.[/dim]")
            return
        _resume_run_loop(run_id, cp.step_id, wait=wait)
        return

    re_exec: list[str] = []
    ambiguous: list[str] = []

    for row in pending:
        result = run_probe(row.verification_probe, row.probe_subject, row.probe_metadata)
        status_label = {
            "completed": "[green]completed[/green]",
            "not_completed": "[yellow]not_completed → re-exec[/yellow]",
            "ambiguous": "[red]ambiguous → approval required[/red]",
        }.get(result, result)
        console.print(f"  {row.tool:<22} {row.side_effect_id[:48]:<50} {status_label}")

        if not dry_run:
            ledger.set_status(row.side_effect_id, status=result)

        if result == "not_completed":
            re_exec.append(row.side_effect_id)
        elif result == "ambiguous":
            ambiguous.append(row.side_effect_id)

    if ambiguous:
        console.print(
            f"\n[red]{len(ambiguous)} side effect(s) are ambiguous.[/red] "
            "Use [bold]iris approvals list[/bold] to review pending approvals, "
            "or pass [bold]--dry-run[/bold] to inspect without updating."
        )

    if re_exec:
        console.print(
            f"\n[yellow]{len(re_exec)} side effect(s) need re-execution.[/yellow] "
            "Re-run the original command with the same run_id to replay them."
        )

    if not re_exec and not ambiguous:
        console.print("[green]All side effects verified — safe to continue.[/green]")

    if dry_run:
        console.print("[dim]--dry-run: stopping before the run is continued.[/dim]")
        return
    if ambiguous:
        console.print(
            "[red]Not continuing the run while side effects are ambiguous.[/red] "
            "Resolve them first — a replay that cannot tell whether it already happened "
            "is how a side effect gets done twice."
        )
        raise typer.Exit(1)

    _resume_run_loop(run_id, cp.step_id, wait=wait)


def _await_gating_approval(run_id: str, *, wait: bool) -> bool:
    """True when the run may continue; False when an approval is still in the way.

    The gate (``ApprovalGate.await_approval``) shipped in Phase 3 with no caller for its
    whole life, because the flow it was written for did not exist: an evaluator halt ends
    the turn rather than blocking, so nothing was ever *waiting* on an answer. A resume
    is the one place something is.
    """
    from iris_harness.kernel.governance.approvals import ApprovalQueue
    from iris_harness.kernel.governance.approvals.gate import (
        ApprovalGate,
        ApprovalRejectedError,
        ApprovalTimedOutError,
    )

    queue = ApprovalQueue()
    gating = queue.pending_for_run(run_id)
    if gating is None:
        return True

    console.print(
        f"[yellow]An approval is still gating this run[/yellow] — "
        f"{gating.signal}: {gating.context_summary}"
    )
    if not wait:
        console.print(
            f"  Answer it first: [bold]iris approvals approve {gating.approval_id}[/bold] "
            f"(or from the web / Telegram), or re-run with [bold]--wait[/bold]."
        )
        return False

    console.print(
        f"  [dim]Waiting for an answer on any channel (times out "
        f"{gating.timeout_at[:19]} UTC)…[/dim]"
    )
    import asyncio

    from iris_harness.kernel.governance.approvals.store import ApprovalStore

    gate = ApprovalGate(ApprovalStore())
    try:
        asyncio.run(gate.await_approval(gating.approval_id))
    except ApprovalRejectedError:
        console.print("[yellow]Rejected — the run stays halted.[/yellow]")
        return False
    except ApprovalTimedOutError:
        console.print("[red]Timed out with no answer — the run stays halted.[/red]")
        return False
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped waiting; the approval is still open.[/dim]")
        return False
    console.print("[green]Approved — continuing.[/green]")
    return True


def _resume_run_loop(run_id: str, step_id: int, *, wait: bool) -> None:
    """Continue the halted run, once nothing is gating it."""
    if not _await_gating_approval(run_id, wait=wait):
        raise typer.Exit(1)

    console.print(f"\n[bold]Continuing[/bold] run=[cyan]{run_id}[/cyan] from step {step_id}…")
    from iris_harness.runtime.bootstrap import build_runtime

    try:
        runtime = build_runtime()
        resumed = runtime.resume_halted_run(run_id=run_id, step_id=step_id)
    except Exception as exc:  # a readable failure beats a traceback here
        print_error(f"could not continue the run: {exc}")
        raise typer.Exit(1) from exc

    console.print(resumed.answer)
    if resumed.session_id:
        console.print(f"\n[dim]Also recorded in session {resumed.session_id}.[/dim]")


# ---------------------------------------------------------------------------
# `iris cost ...` — Phase 3 cost ledger inspection (story 12.gov-3.7)
# ---------------------------------------------------------------------------


cost_app = typer.Typer(
    name="cost",
    help="Inspect per-user cloud LLM spend.",
    no_args_is_help=True,
)
app.add_typer(cost_app, name="cost")


@cost_app.command("summary")
def cmd_cost_summary(
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            help="Floor date (YYYY-MM-DD, UTC). Defaults to today.",
        ),
    ] = None,
    user_id: Annotated[
        str | None,
        typer.Option(
            "--user-id",
            help="User to summarise; defaults to $IRIS_USER_ID or 'local'.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: 'text' (default) or 'json'."),
    ] = "text",
) -> None:
    """Print today's (or since's) cost ledger totals + per-tier breakdown."""
    import json as _json
    from datetime import UTC as _UTC
    from datetime import date as _date
    from datetime import datetime as _datetime

    from iris_harness.kernel.governance.cost import CostStore

    if output_format not in ("text", "json"):
        print_error(f"unknown --format {output_format!r}; expected 'text' or 'json'")
        raise typer.Exit(2)

    if since is None:
        since_date: _date = _datetime.now(_UTC).date()
    else:
        try:
            since_date = _date.fromisoformat(since)
        except ValueError as exc:
            print_error(f"--since must be YYYY-MM-DD ({exc})")
            raise typer.Exit(2) from exc

    resolved_user = user_id or os.environ.get("IRIS_USER_ID") or "local"

    store = CostStore()
    total = store.sum_since(user_id=resolved_user, since=since_date)
    per_tier = store.sum_by_tier_since(user_id=resolved_user, since=since_date)

    if output_format == "json":
        console.print(
            _json.dumps(
                {
                    "user_id": resolved_user,
                    "since": since_date.isoformat(),
                    "total_usd": total,
                    "per_tier_usd": per_tier,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    console.print()
    console.print(
        f"  [bold]cost summary[/bold] user=[cyan]{resolved_user}[/cyan]  "
        f"since=[dim]{since_date.isoformat()}[/dim]"
    )
    console.print(f"    total = [bold green]${total:.4f}[/bold green]")
    if per_tier:
        console.print()
        for tier, spend in sorted(per_tier.items()):
            console.print(f"    [dim]{tier:<10}[/dim]  ${spend:.4f}")
    else:
        console.print("    [dim]no rows[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# `iris audit ...` — Phase 5 hot/cold archive ops
# ---------------------------------------------------------------------------


audit_app = typer.Typer(
    name="audit",
    help="Compact, query, and export governance audit data across hot+cold tiers.",
    no_args_is_help=True,
)
app.add_typer(audit_app, name="audit")


def _parse_since(value: str) -> datetime:
    from datetime import UTC, date, datetime

    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        pass

    try:
        return datetime.combine(date.fromisoformat(text), datetime.min.time(), tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(
            "--since must be ISO date/datetime (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
        ) from exc


@audit_app.command("compact")
def cmd_audit_compact(
    retention_days: Annotated[
        int,
        typer.Option(
            "--retention-days",
            min=1,
            help="Rows older than this many days move from SQLite to Parquet.",
        ),
    ] = 30,
    db_path: Annotated[
        str | None,
        typer.Option("--db", help="Override governance audit SQLite path."),
    ] = None,
    archive_root: Annotated[
        str | None,
        typer.Option("--archive-root", help="Override Parquet archive root path."),
    ] = None,
) -> None:
    """Move old rows from SQLite hot tier to partitioned Parquet cold tier."""
    from pathlib import Path as _Path

    from iris_harness.kernel.governance.audit import AuditLog

    try:
        from iris_harness.kernel.governance.audit.archive import AuditArchive, AuditCompactor
    except ModuleNotFoundError as exc:
        print_error(
            f"audit compact requires optional dependency {exc.name!r}; "
            "install project dependencies from pyproject.toml first."
        )
        raise typer.Exit(2) from exc

    audit_log = AuditLog(db_path=_Path(db_path)) if db_path else AuditLog()
    archive = AuditArchive(root=_Path(archive_root) if archive_root else None)
    compactor = AuditCompactor(
        audit_log=audit_log,
        archive=archive,
        retention_days=retention_days,
    )
    result = compactor.compact()
    console.print(
        "  [bold green]✓[/bold green]  compacted audit log "
        f"(selected={result.selected_rows}, archived={result.archived_rows}, "
        f"deleted={result.deleted_rows}, partitions={result.partition_count})"
    )
    console.print(f"  [dim]cutoff:[/dim] {result.cutoff_ts}")


@audit_app.command("query")
def cmd_audit_query(
    sql: Annotated[str, typer.Argument(help="DuckDB SQL query over `audit_archive` view.")],
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: 'text' (default) or 'json'."),
    ] = "text",
    db_path: Annotated[
        str | None,
        typer.Option("--db", help="Override governance audit SQLite path."),
    ] = None,
    archive_root: Annotated[
        str | None,
        typer.Option("--archive-root", help="Override Parquet archive root path."),
    ] = None,
) -> None:
    """Run a read-only DuckDB query across hot (SQLite) and cold (Parquet) audit data."""
    import json as _json
    from pathlib import Path as _Path

    from rich.table import Table

    try:
        import duckdb

        from iris_harness.kernel.governance.audit.archive import AuditQueryEngine
    except ModuleNotFoundError as exc:
        print_error(
            f"audit query requires optional dependency {exc.name!r}; "
            "install project dependencies from pyproject.toml first."
        )
        raise typer.Exit(2) from exc

    if output_format not in ("text", "json"):
        print_error(f"unknown --format {output_format!r}; expected 'text' or 'json'")
        raise typer.Exit(2)

    engine = AuditQueryEngine(
        audit_db_path=_Path(db_path) if db_path else None,
        archive_root=_Path(archive_root) if archive_root else None,
    )
    try:
        result = engine.query(sql)
    except (ValueError, duckdb.Error) as exc:
        # DuckDB's own message already names the missing table/column and offers a
        # "did you mean" hint; the view to query is ``audit_archive``.
        print_error(str(exc))
        raise typer.Exit(2) from exc

    if output_format == "json":
        payload = [dict(zip(result.columns, row, strict=True)) for row in result.rows]
        console.print(_json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    table = Table(show_header=True, header_style="bold cyan")
    for name in result.columns:
        table.add_column(name)
    for row in result.rows:
        table.add_row(*[str(value) if value is not None else "" for value in row])
    console.print(table)
    console.print(f"  [dim]{len(result.rows)} row(s)[/dim]")


@audit_app.command("export")
def cmd_audit_export(
    since: Annotated[
        str,
        typer.Option(
            "--since",
            help="Lower time bound (YYYY-MM-DD or ISO datetime).",
        ),
    ],
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Export format: 'jsonl' (default) or 'csv'."),
    ] = "jsonl",
    output: Annotated[
        str | None,
        typer.Option("--out", help="Output file path."),
    ] = None,
    db_path: Annotated[
        str | None,
        typer.Option("--db", help="Override governance audit SQLite path."),
    ] = None,
    archive_root: Annotated[
        str | None,
        typer.Option("--archive-root", help="Override Parquet archive root path."),
    ] = None,
) -> None:
    """Export audit rows since a given timestamp across hot+cold tiers."""
    from datetime import UTC, datetime
    from pathlib import Path as _Path

    try:
        from iris_harness.kernel.governance.audit.archive import AuditQueryEngine
    except ModuleNotFoundError as exc:
        print_error(
            f"audit export requires optional dependency {exc.name!r}; "
            "install project dependencies from pyproject.toml first."
        )
        raise typer.Exit(2) from exc

    if output_format not in ("jsonl", "csv"):
        print_error(f"unknown --format {output_format!r}; expected 'jsonl' or 'csv'")
        raise typer.Exit(2)
    try:
        since_dt = _parse_since(since)
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc

    out_path = (
        _Path(output)
        if output
        else _Path(f"audit-export-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.{output_format}")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engine = AuditQueryEngine(
        audit_db_path=_Path(db_path) if db_path else None,
        archive_root=_Path(archive_root) if archive_root else None,
    )
    if output_format == "jsonl":
        count = engine.export_since(
            since=since_dt,
            output=out_path,
            fmt="jsonl",
        )
    else:
        count = engine.export_since(
            since=since_dt,
            output=out_path,
            fmt="csv",
        )
    console.print(
        f"  [bold green]✓[/bold green]  exported {count} row(s) to [cyan]{out_path}[/cyan]"
    )


# ---------------------------------------------------------------------------
# `iris evaluator ...` — Phase 3 out-of-process evaluator lifecycle (story 12.gov-3.9)
# ---------------------------------------------------------------------------


evaluator_app = typer.Typer(
    name="evaluator",
    help="Manage the out-of-process evaluator sidecar.",
    no_args_is_help=True,
)
app.add_typer(evaluator_app, name="evaluator")


_EVALUATOR_PID_FILE = Path.home() / ".local" / "state" / "iris" / "evaluator.pid"
_EVALUATOR_DEFAULT_URL = "http://127.0.0.1:8090"
_EVALUATOR_DEFAULT_PORT = 8090


def _evaluator_base_url() -> str:
    return os.environ.get("IRIS_GOVERNANCE_EVALUATOR_URL", _EVALUATOR_DEFAULT_URL).strip()


@evaluator_app.command("start")
def cmd_evaluator_start(
    port: Annotated[
        int,
        typer.Option("--port", help="Port for the evaluator sidecar."),
    ] = _EVALUATOR_DEFAULT_PORT,
    detach: Annotated[
        bool,
        typer.Option(
            "--detach",
            "-d",
            help="Spawn in the background and write a pidfile.",
        ),
    ] = False,
) -> None:
    """Spawn the evaluator sidecar (uvicorn iris_harness.server.evaluator.main:app)."""
    import shutil
    import subprocess

    uvicorn_path = shutil.which("uvicorn") or "uvicorn"
    cmd = [
        uvicorn_path,
        "iris_harness.server.evaluator.main:app",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    if not detach:
        console.print("  [dim]starting evaluator on 127.0.0.1:" f"{port} — Ctrl-C to stop[/dim]")
        os.execvp(cmd[0], cmd)  # noqa: S606 - replaces current process; cmd is built locally
        return

    _EVALUATOR_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _EVALUATOR_PID_FILE.exists():
        try:
            existing_pid = int(_EVALUATOR_PID_FILE.read_text().strip())
        except ValueError:
            existing_pid = 0
        if existing_pid and _pid_alive(existing_pid):
            print_error(
                f"evaluator already running (pid={existing_pid}); "
                "stop it first with `iris evaluator stop`"
            )
            raise typer.Exit(1)

    # cmd is built from a shutil.which-resolved uvicorn binary + a fixed
    # module path; no untrusted input flows here.
    proc = subprocess.Popen(  # noqa: S603
        cmd,  # see comment above
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _EVALUATOR_PID_FILE.write_text(str(proc.pid))
    console.print(
        f"  [bold green]✓[/bold green]  evaluator detached "
        f"pid=[cyan]{proc.pid}[/cyan] url=[cyan]http://127.0.0.1:{port}[/cyan]  "
        f"(pidfile [dim]{_EVALUATOR_PID_FILE}[/dim])"
    )


@evaluator_app.command("stop")
def cmd_evaluator_stop() -> None:
    """Terminate the detached evaluator sidecar via pidfile."""
    import signal

    if not _EVALUATOR_PID_FILE.exists():
        print_error("no evaluator pidfile — nothing to stop")
        raise typer.Exit(1)
    try:
        pid = int(_EVALUATOR_PID_FILE.read_text().strip())
    except ValueError as exc:
        print_error(f"corrupt pidfile: {exc}")
        raise typer.Exit(1) from exc

    if not _pid_alive(pid):
        _EVALUATOR_PID_FILE.unlink(missing_ok=True)
        console.print("  [dim]evaluator was not running; pidfile cleared[/dim]")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        print_error(f"could not send SIGTERM to pid={pid}: {exc}")
        raise typer.Exit(1) from exc
    _EVALUATOR_PID_FILE.unlink(missing_ok=True)
    console.print(f"  [bold green]✓[/bold green]  evaluator stopped (pid={pid})")


@evaluator_app.command("status")
def cmd_evaluator_status(
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: 'text' (default) or 'json'."),
    ] = "text",
) -> None:
    """Report whether the evaluator sidecar is reachable."""
    import json as _json

    import httpx

    from iris_harness.cli.api_client import harness_api_client

    if output_format not in ("text", "json"):
        print_error(f"unknown --format {output_format!r}; expected 'text' or 'json'")
        raise typer.Exit(2)

    url = _evaluator_base_url()
    pid: int | None = None
    if _EVALUATOR_PID_FILE.exists():
        try:
            pid = int(_EVALUATOR_PID_FILE.read_text().strip())
        except ValueError:
            pid = None

    status: dict[str, Any] = {
        "url": url,
        "pid": pid,
        "pid_alive": _pid_alive(pid) if pid else False,
        "healthz": None,
        "ok": False,
    }
    try:
        # `/healthz` is the open probe: no secret goes with it.
        with harness_api_client(purpose="evaluator-status", timeout=2.0, auth=False) as client:
            response = client.get(f"{url.rstrip('/')}/healthz")
        response.raise_for_status()
        status["healthz"] = response.json()
        status["ok"] = bool(status["healthz"].get("ok"))
    except httpx.HTTPError as exc:
        status["error"] = str(exc)

    if output_format == "json":
        console.print(_json.dumps(status, indent=2, sort_keys=True, default=str))
        return

    pid_cell = f"pid=[cyan]{pid}[/cyan]" if pid else "[dim]no pidfile[/dim]"
    if status["ok"]:
        signals = status["healthz"].get("signal_count", "?")
        console.print(
            f"  [bold green]✓[/bold green]  evaluator up at [cyan]{url}[/cyan]  "
            f"{pid_cell}  signals=[white]{signals}[/white]"
        )
    else:
        err = status.get("error", "unreachable")
        console.print(
            f"  [bold red]✗[/bold red]  evaluator down at [cyan]{url}[/cyan]  "
            f"{pid_cell}  [dim]{err}[/dim]"
        )
        raise typer.Exit(1)


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@copilot_app.command("login")
def cmd_auth_copilot_login() -> None:
    """Drive the GitHub device-flow OAuth and persist the Copilot token."""
    from iris_harness.llm import copilot_auth

    cache_path = copilot_auth.DEFAULT_CACHE_PATH
    try:
        copilot_auth._assert_copilot_enabled()
    except copilot_auth.CopilotBackendDisabledError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc

    flow = copilot_auth.CopilotDeviceFlow(cache_path=cache_path)

    def _announce(device: copilot_auth.DeviceCodeResponse) -> None:
        console.print(
            "  [dim]Open[/dim] [cyan]"
            + device.verification_uri
            + "[/cyan] [dim]and enter code[/dim] "
            + f"[bold cyan]{device.user_code}[/bold cyan]"
        )
        console.print("  [dim]Waiting for authorization…[/dim]")

    entry = flow.login(announce=_announce)
    console.print(
        f"  [bold green]✓[/bold green]  Copilot token cached at [cyan]{cache_path}[/cyan]  "
        f"[dim]{copilot_auth._mask_secret(entry.access_token)}[/dim]"
    )


@copilot_app.command("status")
def cmd_auth_copilot_status() -> None:
    """Show the currently cached Copilot OAuth token state."""
    from iris_harness.llm import copilot_auth

    cache_path = copilot_auth.DEFAULT_CACHE_PATH
    try:
        entry = copilot_auth._read_cache(cache_path)
    except copilot_auth.CopilotAuthError as exc:
        console.print(f"  [yellow]no token cached[/yellow]  [dim]{exc}[/dim]")
        raise typer.Exit(1) from exc

    console.print(
        f"  [bold green]✓[/bold green]  cached at [cyan]{cache_path}[/cyan]  "
        f"[dim]{copilot_auth._mask_secret(entry.access_token)}[/dim]"
    )


@copilot_app.command("logout")
def cmd_auth_copilot_logout() -> None:
    """Delete the cached Copilot OAuth token."""
    from iris_harness.llm import copilot_auth

    cache_path = copilot_auth.DEFAULT_CACHE_PATH
    removed = copilot_auth.logout(cache_path)
    if removed:
        console.print(
            f"  [bold green]✓[/bold green]  removed Copilot token at [cyan]{cache_path}[/cyan]"
        )
    else:
        console.print(f"  [dim]no token to remove at[/dim] [cyan]{cache_path}[/cyan]")


# ---------------------------------------------------------------------------
# Gmail OAuth (installed-app flow) — Phase 1 Track 1A
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# iris email bootstrap-categories (Track 1E.2 — ADR-0018)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# iris email accept-categories (Track 1F — ADR-0019)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# iris email triage (Track 1G — ADR-0021)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# iris email label-holdout (Track 1L — ADR-0023)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# iris email recategorize (Track 1J — ADR-0024)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# iris email reingest-wiki (Track 1K — ADR-0025 §7)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# iris email corrections (Track 1J — ADR-0024)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# iris email knn-gate (Track 1L — ADR-0023)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# iris email detect-followups (Phase 2 Track 2B)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# iris task / iris goal (Phase 2 Track 2A — ADR-0005, ADR-0014)
# ---------------------------------------------------------------------------


task_app = typer.Typer(
    name="task",
    help="User-facing tasks. Operates on data/tasks.db.",
    no_args_is_help=True,
)
app.add_typer(task_app, name="task")

goal_app = typer.Typer(
    name="goal",
    help="Long-horizon goals. Operates on data/tasks.db.",
    no_args_is_help=True,
)
app.add_typer(goal_app, name="goal")


def _open_task_store(tasks_db: Path | None) -> Any:
    from iris_harness.services.tasks.store import TaskStore

    store = TaskStore(db_path=tasks_db or data_dir() / "tasks.db")
    store.ensure_schema()
    return store


@app.command("actions")
def actions_command(
    tasks_db: Annotated[
        Path | None, typer.Option(help="Path to tasks.db (default data/tasks.db)")
    ] = None,
) -> None:
    """Show pending actions needing your attention (the Action Center).

    Reads the local task store directly — no running server needed. Health alerts
    also surface in chat (the ``pending_actions`` tool), the API (``GET /actions``)
    and the web Action Center, where the runtime's health cache is warm.
    """
    from iris_harness.runtime.action_center import collect_pending_actions, render_pending_actions

    store = _open_task_store(tasks_db)
    console.print(render_pending_actions(collect_pending_actions(store, None)))


@app.command("feedback")
def feedback_command(
    ref: Annotated[
        str,
        typer.Argument(
            help="What to give feedback on: a followup Task id, or a 'fb:...' "
            "surface token shown next to a proactively-surfaced item.",
        ),
    ],
    useful: Annotated[
        bool,
        typer.Option("--useful", help="Mark the item as wanted (lifts any suppression)."),
    ] = False,
    not_useful: Annotated[
        bool,
        typer.Option("--not-useful", help="Mark the item as noise (suppress similar). Default."),
    ] = False,
    tasks_db: Annotated[
        Path | None, typer.Option(help="Path to tasks.db (default data/tasks.db)")
    ] = None,
) -> None:
    """Tell IRIS a surfaced item was (not) useful — it learns and suppresses similar.

    Generic across subsystems: email reply-followups, finance bills, system-health
    alerts, etc. The verdict is recorded in the surface-feedback ledger
    (``learning.db``) keyed by stable dimensions, so a single "not useful" on one
    vendor's blast suppresses the rest. Same store backs the chat and ``POST
    /feedback`` surfaces (issue 0028).
    """
    from iris_harness.services.learning.suppression import (
        NOT_USEFUL,
        USEFUL,
        SurfaceFeedbackStore,
        decode_ref,
    )

    if useful and not_useful:
        print_error("Pass only one of --useful / --not-useful.")
        raise typer.Exit(2)
    verdict = USEFUL if useful else NOT_USEFUL  # not-useful is the default action

    feedback = SurfaceFeedbackStore()
    feedback.ensure_schema()

    if ref.startswith("fb:"):
        # Self-describing surface token (ephemeral items: health alerts, bills, ...).
        try:
            subsystem, surface_kind, dims = decode_ref(ref)
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(2) from exc
        feedback.record(subsystem, surface_kind, dims, verdict)
        console.print(
            f"[green]Recorded[/green] {verdict} for [cyan]{subsystem}/{surface_kind}[/cyan] "
            f"[dim]{dims}[/dim]"
        )
        return

    # Otherwise treat the ref as a Task id (the common email-followup case).
    from iris_harness.services.learning.suppression import email_followup_dims_from

    store = _open_task_store(tasks_db)
    task = store.get(ref)
    if task is None:
        print_error(f"No task {ref!r} found. Pass a followup Task id or an 'fb:' token.")
        raise typer.Exit(3)
    if task.wait_for is None or task.wait_for.kind != "reply_from":
        print_error(
            f"Task {ref[:8]} is not an email followup; feedback only applies to surfaced items."
        )
        raise typer.Exit(2)

    payload = task.wait_for.payload
    dims = email_followup_dims_from(
        str(payload.get("account_id", "")), str(payload.get("from", ""))
    )
    feedback.record("email", "followup", dims, verdict)

    if verdict == NOT_USEFUL:
        store.drop(task.id)
        console.print(
            f"[green]Got it[/green] — dropped this followup and I won't raise replies for "
            f"[cyan]{dims['from_domain']}[/cyan] again. [dim](task {task.id[:8]})[/dim]"
        )
    else:
        console.print(
            f"[green]Noted[/green] — keeping followups for [cyan]{dims['from_domain']}[/cyan]. "
            f"[dim](task {task.id[:8]})[/dim]"
        )


@app.command("agents")
def agents_command(
    name: Annotated[str | None, typer.Argument(help="Agent name for a detailed view")] = None,
    set_toggle: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="Edit a curated toggle: --set KEY=on|off (needs an agent name; repeatable)",
        ),
    ] = None,
) -> None:
    """List IRIS's agents (and what's pending for each), or one agent's detail.

    Composes from the catalog + local stores — no running server needed. The same
    view is available in chat (the ``agents`` tool) and the API (``GET /agents``).
    ``--set KEY=on|off`` edits a curated toggle (persists to the local override).
    """
    from iris_harness.runtime.agent_console import (
        config_dir,
        render_agent_detail,
        render_agents_overview_local,
    )

    if set_toggle:
        if not name:
            print_error("--set needs an agent name, e.g. `iris agents finance --set KEY=off`")
            raise typer.Exit(2)
        from iris_harness.runtime.agent_settings_store import set_toggle as apply_toggle
        from iris_harness.runtime.agent_settings_store import toggles_for_agent
        from iris_harness.runtime.settings_catalog import installed_catalog

        catalog = installed_catalog(config_dir())
        owned = toggles_for_agent(name, catalog)
        for item in set_toggle:
            key, sep, raw = item.partition("=")
            key = key.strip()
            if not sep:
                print_error(f"bad --set {item!r}; use KEY=on|off")
                raise typer.Exit(2)
            if key not in owned:
                print_error(f"{name} has no editable toggle {key!r}")
                raise typer.Exit(1)
            enabled = raw.strip().lower() in {"on", "true", "1", "yes"}
            try:
                res = apply_toggle(key, enabled, catalog=catalog, actor="cli")
            except ValueError as exc:
                print_error(str(exc))
                raise typer.Exit(1) from exc
            note = " (restart to take effect)" if res["restart_required"] else ""
            console.print(f"  [green]✓[/green] {key} = {'on' if enabled else 'off'}{note}")

    console.print(render_agent_detail(name) if name else render_agents_overview_local())


def _short_id(uuid_str: str) -> str:
    return uuid_str[:8]


def _resolve_task_id(store: Any, prefix: str) -> str:
    """Resolve a UUID prefix to a unique task id. Exits on miss or ambiguity."""
    from iris_harness.foundation.persistence.sqlite import sqlite_conn

    if len(prefix) < 4:
        print_error("task id prefix must be at least 4 characters")
        raise typer.Exit(2)
    with sqlite_conn(store.db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM tasks WHERE id LIKE ? LIMIT 5",
            (f"{prefix}%",),
        ).fetchall()
    if not rows:
        print_error(f"no task matching prefix {prefix!r}")
        raise typer.Exit(1)
    if len(rows) > 1:
        ambiguous = ", ".join(_short_id(r[0]) for r in rows)
        print_error(f"ambiguous task prefix {prefix!r}: {ambiguous}")
        raise typer.Exit(1)
    return str(rows[0][0])


def _resolve_goal_id(store: Any, prefix: str) -> str:
    from iris_harness.foundation.persistence.sqlite import sqlite_conn

    if len(prefix) < 4:
        print_error("goal id prefix must be at least 4 characters")
        raise typer.Exit(2)
    with sqlite_conn(store.db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM goals WHERE id LIKE ? LIMIT 5",
            (f"{prefix}%",),
        ).fetchall()
    if not rows:
        print_error(f"no goal matching prefix {prefix!r}")
        raise typer.Exit(1)
    if len(rows) > 1:
        ambiguous = ", ".join(_short_id(r[0]) for r in rows)
        print_error(f"ambiguous goal prefix {prefix!r}: {ambiguous}")
        raise typer.Exit(1)
    return str(rows[0][0])


def _format_due(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M")


@task_app.command("add")
def cmd_task_add(
    title: Annotated[str, typer.Argument(help="Task title.")],
    description: Annotated[
        str, typer.Option("--description", "-d", help="Longer description.")
    ] = "",
    due: Annotated[
        datetime | None,
        typer.Option(
            "--due",
            help="Due timestamp.",
            formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"],
        ),
    ] = None,
    priority: Annotated[
        int, typer.Option("--priority", "-p", help="Integer priority (higher = sooner).")
    ] = 0,
    goal: Annotated[
        str | None, typer.Option("--goal", help="Parent goal id (prefix accepted).")
    ] = None,
    parent: Annotated[
        str | None, typer.Option("--parent", help="Parent task id (prefix accepted).")
    ] = None,
    tasks_db_path: Annotated[
        Path | None,
        typer.Option("--tasks-db", help="Override tasks.db location (default: data/tasks.db)."),
    ] = None,
) -> None:
    """Create a new manual task."""
    store = _open_task_store(tasks_db_path)
    fields: dict[str, Any] = {
        "title": title,
        "description": description,
        "priority": priority,
        "source_kind": "manual",
    }
    if due is not None:
        if due.tzinfo is None:
            from datetime import UTC as _UTC

            due = due.replace(tzinfo=_UTC)
        fields["due_at"] = due
    if goal:
        fields["parent_goal_id"] = _resolve_goal_id(store, goal)
    if parent:
        fields["parent_task_id"] = _resolve_task_id(store, parent)
    task = store.create(**fields)
    console.print(f"  [green]created task[/green] [cyan]{_short_id(task.id)}[/cyan]  {task.title}")


@task_app.command("list")
def cmd_task_list(
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter by status (open|doing|done|dropped|expired)."),
    ] = "open",
    goal: Annotated[
        str | None, typer.Option("--goal", help="Filter by parent goal id (prefix accepted).")
    ] = None,
    due_before: Annotated[
        datetime | None,
        typer.Option(
            "--due-before",
            help="Only tasks with due_at on or before this point.",
            formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"],
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Cap on rows.")] = 50,
    tasks_db_path: Annotated[
        Path | None,
        typer.Option("--tasks-db", help="Override tasks.db location."),
    ] = None,
) -> None:
    """List tasks. Defaults to open tasks ordered by priority desc, due_at asc."""
    store = _open_task_store(tasks_db_path)
    goal_id = _resolve_goal_id(store, goal) if goal else None
    if status == "all":
        status_filter = None
    else:
        status_filter = status
    tasks = store.list(
        status=status_filter,
        parent_goal_id=goal_id,
        due_before=due_before,
        limit=limit,
    )
    if not tasks:
        console.print("  [yellow]no tasks match[/yellow]")
        return

    from rich.table import Table

    table = Table(title=f"{len(tasks)} task(s)", show_lines=False)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("status", style="dim", no_wrap=True)
    table.add_column("pri", justify="right", no_wrap=True)
    table.add_column("due", style="dim", no_wrap=True)
    table.add_column("wait_for", style="dim", no_wrap=True)
    table.add_column("title")
    for t in tasks:
        wait = ""
        if t.wait_for is not None:
            wait = "resolved" if t.wait_for_resolved_at else t.wait_for.kind
        table.add_row(
            _short_id(t.id),
            t.status,
            str(t.priority),
            _format_due(t.due_at),
            wait or "—",
            t.title,
        )
    console.print(table)


@task_app.command("show")
def cmd_task_show(
    task_id: Annotated[str, typer.Argument(help="Task id (prefix accepted).")],
    tasks_db_path: Annotated[
        Path | None, typer.Option("--tasks-db", help="Override tasks.db location.")
    ] = None,
) -> None:
    """Show one task with its full detail."""
    store = _open_task_store(tasks_db_path)
    resolved = _resolve_task_id(store, task_id)
    task = store.get(resolved)
    if task is None:
        print_error(f"no task with id {resolved}")
        raise typer.Exit(1)

    console.print()
    console.print(f"  [bold]{task.title}[/bold]")
    console.print(f"  [dim]id:[/dim] {task.id}")
    console.print(f"  [dim]status:[/dim] {task.status}    [dim]priority:[/dim] {task.priority}")
    if task.description:
        console.print(f"  [dim]description:[/dim] {task.description}")
    if task.due_at:
        console.print(f"  [dim]due_at:[/dim] {task.due_at.isoformat()}")
    if task.source_kind:
        console.print(f"  [dim]source:[/dim] {task.source_kind} / {task.source_id or '—'}")
    if task.parent_task_id:
        console.print(f"  [dim]parent_task:[/dim] {_short_id(task.parent_task_id)}")
    if task.parent_goal_id:
        console.print(f"  [dim]parent_goal:[/dim] {_short_id(task.parent_goal_id)}")
    if task.dedup_key:
        console.print(f"  [dim]dedup_key:[/dim] {task.dedup_key}")
    if task.wait_for:
        resolved_at = task.wait_for_resolved_at.isoformat() if task.wait_for_resolved_at else "—"
        console.print(
            f"  [dim]wait_for:[/dim] kind={task.wait_for.kind} "
            f"payload={task.wait_for.payload} resolved_at={resolved_at}"
        )
    if task.related_wikilinks:
        console.print(f"  [dim]wikilinks:[/dim] {', '.join(task.related_wikilinks)}")
    console.print(
        f"  [dim]created:[/dim] {task.created_at.isoformat()}    "
        f"[dim]updated:[/dim] {task.updated_at.isoformat()}"
    )
    if task.completed_at:
        console.print(f"  [dim]completed:[/dim] {task.completed_at.isoformat()}")


@task_app.command("complete")
def cmd_task_complete(
    task_id: Annotated[str, typer.Argument(help="Task id (prefix accepted).")],
    tasks_db_path: Annotated[
        Path | None, typer.Option("--tasks-db", help="Override tasks.db location.")
    ] = None,
) -> None:
    """Mark a task done."""
    store = _open_task_store(tasks_db_path)
    resolved = _resolve_task_id(store, task_id)
    task = store.complete(resolved)
    console.print(f"  [green]completed[/green] [cyan]{_short_id(task.id)}[/cyan]  {task.title}")


@task_app.command("drop")
def cmd_task_drop(
    task_id: Annotated[str, typer.Argument(help="Task id (prefix accepted).")],
    tasks_db_path: Annotated[
        Path | None, typer.Option("--tasks-db", help="Override tasks.db location.")
    ] = None,
) -> None:
    """Drop a task (terminal, distinct from done)."""
    store = _open_task_store(tasks_db_path)
    resolved = _resolve_task_id(store, task_id)
    task = store.drop(resolved)
    console.print(f"  [yellow]dropped[/yellow] [cyan]{_short_id(task.id)}[/cyan]  {task.title}")


@task_app.command("update")
def cmd_task_update(
    task_id: Annotated[str, typer.Argument(help="Task id (prefix accepted).")],
    title: Annotated[str | None, typer.Option("--title", help="New title.")] = None,
    description: Annotated[
        str | None, typer.Option("--description", "-d", help="New description.")
    ] = None,
    due: Annotated[
        datetime | None,
        typer.Option(
            "--due",
            help="New due_at; pass empty string to clear with --clear-due.",
            formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"],
        ),
    ] = None,
    clear_due: Annotated[
        bool, typer.Option("--clear-due", help="Clear an existing due_at.")
    ] = False,
    priority: Annotated[int | None, typer.Option("--priority", "-p", help="New priority.")] = None,
    status: Annotated[
        str | None,
        typer.Option("--status", help="New status (open|doing). Use complete/drop for terminals."),
    ] = None,
    tasks_db_path: Annotated[
        Path | None, typer.Option("--tasks-db", help="Override tasks.db location.")
    ] = None,
) -> None:
    """Update mutable fields. Use complete/drop for terminal transitions."""
    store = _open_task_store(tasks_db_path)
    resolved = _resolve_task_id(store, task_id)
    fields: dict[str, Any] = {}
    if title is not None:
        fields["title"] = title
    if description is not None:
        fields["description"] = description
    if clear_due:
        fields["due_at"] = None
    elif due is not None:
        if due.tzinfo is None:
            from datetime import UTC as _UTC

            due = due.replace(tzinfo=_UTC)
        fields["due_at"] = due
    if priority is not None:
        fields["priority"] = priority
    if status is not None:
        if status in ("done", "dropped", "expired"):
            print_error("use 'iris task complete' or 'iris task drop' for terminal status")
            raise typer.Exit(2)
        fields["status"] = status
    if not fields:
        print_error("no fields to update")
        raise typer.Exit(2)
    task = store.update(resolved, **fields)
    console.print(f"  [green]updated[/green] [cyan]{_short_id(task.id)}[/cyan]  {task.title}")


@task_app.command("group")
def cmd_task_group(
    task_ids: Annotated[
        list[str],
        typer.Argument(help="Two or more task ids (prefixes accepted) to group."),
    ],
    under: Annotated[str, typer.Option("--under", help="Title of the new parent task.")],
    tasks_db_path: Annotated[
        Path | None, typer.Option("--tasks-db", help="Override tasks.db location.")
    ] = None,
) -> None:
    """Group existing tasks under a new parent task.

    Creates one parent Task with the given title, then sets
    ``parent_task_id`` on each of the listed tasks. Per ADR-0005,
    parent_task_id is tree-only — a task that already has a parent will
    be re-parented under the new one.
    """
    if len(task_ids) < 2:
        print_error("group needs at least two task ids")
        raise typer.Exit(2)
    store = _open_task_store(tasks_db_path)
    resolved = [_resolve_task_id(store, t) for t in task_ids]
    parent = store.create(title=under, source_kind="manual")
    for child_id in resolved:
        store.update(child_id, parent_task_id=parent.id)
    console.print(
        f"  [green]grouped[/green] {len(resolved)} task(s) under "
        f"[cyan]{_short_id(parent.id)}[/cyan]  {parent.title}"
    )


# ---------------------------------------------------------------------------
# iris goal commands
# ---------------------------------------------------------------------------


@goal_app.command("add")
def cmd_goal_add(
    title: Annotated[str, typer.Argument(help="Goal title.")],
    description: Annotated[str, typer.Option("--description", "-d")] = "",
    target: Annotated[
        datetime | None,
        typer.Option(
            "--target",
            help="Target date.",
            formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"],
        ),
    ] = None,
    criteria: Annotated[str, typer.Option("--criteria", help="Success criteria.")] = "",
    tasks_db_path: Annotated[
        Path | None, typer.Option("--tasks-db", help="Override tasks.db location.")
    ] = None,
) -> None:
    """Create a new goal."""
    store = _open_task_store(tasks_db_path)
    fields: dict[str, Any] = {
        "title": title,
        "description": description,
        "success_criteria": criteria,
    }
    if target is not None:
        if target.tzinfo is None:
            from datetime import UTC as _UTC

            target = target.replace(tzinfo=_UTC)
        fields["target_date"] = target
    goal = store.create_goal(**fields)
    console.print(f"  [green]created goal[/green] [cyan]{_short_id(goal.id)}[/cyan]  {goal.title}")


@goal_app.command("list")
def cmd_goal_list(
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter by status (active|paused|achieved|dropped)."),
    ] = "active",
    limit: Annotated[int, typer.Option("--limit", help="Cap on rows.")] = 50,
    tasks_db_path: Annotated[
        Path | None, typer.Option("--tasks-db", help="Override tasks.db location.")
    ] = None,
) -> None:
    """List goals."""
    store = _open_task_store(tasks_db_path)
    status_filter = None if status == "all" else status
    goals = store.list_goals(status=status_filter, limit=limit)
    if not goals:
        console.print("  [yellow]no goals match[/yellow]")
        return

    from rich.table import Table

    table = Table(title=f"{len(goals)} goal(s)", show_lines=False)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("status", style="dim", no_wrap=True)
    table.add_column("target", style="dim", no_wrap=True)
    table.add_column("title")
    for g in goals:
        target = g.target_date.strftime("%Y-%m-%d") if g.target_date else "—"
        table.add_row(_short_id(g.id), g.status, target, g.title)
    console.print(table)


@goal_app.command("show")
def cmd_goal_show(
    goal_id: Annotated[str, typer.Argument(help="Goal id (prefix accepted).")],
    tasks_db_path: Annotated[
        Path | None, typer.Option("--tasks-db", help="Override tasks.db location.")
    ] = None,
) -> None:
    """Show one goal with its tasks."""
    store = _open_task_store(tasks_db_path)
    resolved = _resolve_goal_id(store, goal_id)
    goal = store.get_goal(resolved)
    if goal is None:
        print_error(f"no goal with id {resolved}")
        raise typer.Exit(1)
    tasks = store.list(parent_goal_id=resolved, status=None, limit=200)

    console.print()
    console.print(f"  [bold]{goal.title}[/bold]")
    console.print(f"  [dim]id:[/dim] {goal.id}")
    console.print(f"  [dim]status:[/dim] {goal.status}")
    if goal.target_date:
        console.print(f"  [dim]target:[/dim] {goal.target_date.isoformat()}")
    if goal.success_criteria:
        console.print(f"  [dim]criteria:[/dim] {goal.success_criteria}")
    if goal.description:
        console.print(f"  [dim]description:[/dim] {goal.description}")
    console.print(f"  [dim]tasks under this goal:[/dim] {len(tasks)}")
    for t in tasks:
        marker = "(x)" if t.status == "done" else "( )"
        console.print(
            f"    {marker} [cyan]{_short_id(t.id)}[/cyan]  {t.title}  [dim]{t.status}[/dim]"
        )


@goal_app.command("achieve")
def cmd_goal_achieve(
    goal_id: Annotated[str, typer.Argument(help="Goal id (prefix accepted).")],
    tasks_db_path: Annotated[
        Path | None, typer.Option("--tasks-db", help="Override tasks.db location.")
    ] = None,
) -> None:
    """Mark a goal achieved."""
    from datetime import UTC as _UTC

    store = _open_task_store(tasks_db_path)
    resolved = _resolve_goal_id(store, goal_id)
    goal = store.update_goal(resolved, status="achieved", completed_at=datetime.now(_UTC))
    console.print(f"  [green]achieved[/green] [cyan]{_short_id(goal.id)}[/cyan]  {goal.title}")


# ---------------------------------------------------------------------------
# iris reminder (Phase 2 Track 2D — ADR-0005 delivery-only)
# ---------------------------------------------------------------------------


reminder_app = typer.Typer(
    name="reminder",
    help=(
        "Delivery-only reminders pointing at Tasks/Goals/Bills/Events "
        "(ADR-0005). Operates on data/tasks.db. The legacy "
        "`src/iris_harness/reminders.py` module is unaffected — both coexist "
        "during migration."
    ),
    no_args_is_help=True,
)
app.add_typer(reminder_app, name="reminder")


def _open_reminder_store(tasks_db: Path | None) -> Any:
    from iris_harness.services.notifications.store import ReminderStore

    store = ReminderStore(db_path=tasks_db or data_dir() / "tasks.db")
    store.ensure_schema()
    return store


def _resolve_reminder_id(store: Any, prefix: str) -> str:
    from iris_harness.foundation.persistence.sqlite import sqlite_conn

    if len(prefix) < 4:
        print_error("reminder id prefix must be at least 4 characters")
        raise typer.Exit(2)
    with sqlite_conn(store.db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM notification_reminders WHERE id LIKE ? LIMIT 5",
            (f"{prefix}%",),
        ).fetchall()
    if not rows:
        print_error(f"no reminder matching prefix {prefix!r}")
        raise typer.Exit(1)
    if len(rows) > 1:
        ambiguous = ", ".join(_short_id(r[0]) for r in rows)
        print_error(f"ambiguous reminder prefix {prefix!r}: {ambiguous}")
        raise typer.Exit(1)
    return str(rows[0][0])


@reminder_app.command("add")
def cmd_reminder_add(
    target_id: Annotated[
        str,
        typer.Argument(
            help="Target id (full or prefix). Resolved against tasks then goals.",
        ),
    ],
    at: Annotated[
        datetime,
        typer.Option(
            "--at",
            help="When to fire the reminder.",
            formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"],
        ),
    ],
    target_kind: Annotated[
        str | None,
        typer.Option(
            "--kind",
            help=(
                "Target kind (task|goal|bill|event). If omitted, prefix "
                "is resolved against tasks first, then goals."
            ),
        ),
    ] = None,
    channel: Annotated[
        str, typer.Option("--channel", help="Channel name (default 'default').")
    ] = "default",
    note: Annotated[str, typer.Option("--note", help="Optional note.")] = "",
    tasks_db_path: Annotated[
        Path | None,
        typer.Option("--tasks-db", help="Override tasks.db location."),
    ] = None,
) -> None:
    """Schedule a reminder pointing at a Task or Goal.

    The reminder carries no status of its own (ADR-0005) — it's a
    "fire at time T about target X" row. When the time arrives, the
    `tick` command fires it and emits `reminder.fired` on the bus.
    """
    from datetime import UTC as _UTC

    task_store = _open_task_store(tasks_db_path)
    reminder_store = _open_reminder_store(tasks_db_path)

    if at.tzinfo is None:
        at = at.replace(tzinfo=_UTC)

    resolved_kind: str
    resolved_target: str
    if target_kind == "task":
        resolved_kind = "task"
        resolved_target = _resolve_task_id(task_store, target_id)
    elif target_kind == "goal":
        resolved_kind = "goal"
        resolved_target = _resolve_goal_id(task_store, target_id)
    elif target_kind in ("bill", "event"):
        print_error(
            f"--kind={target_kind} not yet supported " "(Phase 3+ for bills, Phase 6 for events)"
        )
        raise typer.Exit(2)
    elif target_kind is None:
        # Auto-resolve: try task first, then goal.
        from iris_harness.foundation.persistence.sqlite import sqlite_conn

        with sqlite_conn(task_store.db_path) as conn:
            t_rows = conn.execute(
                "SELECT id FROM tasks WHERE id LIKE ? LIMIT 2",
                (f"{target_id}%",),
            ).fetchall()
            g_rows = conn.execute(
                "SELECT id FROM goals WHERE id LIKE ? LIMIT 2",
                (f"{target_id}%",),
            ).fetchall()
        if t_rows and g_rows:
            print_error(
                f"prefix {target_id!r} matches both task and goal; "
                "pass --kind task|goal to disambiguate"
            )
            raise typer.Exit(1)
        if t_rows:
            if len(t_rows) > 1:
                print_error(f"ambiguous task prefix {target_id!r}")
                raise typer.Exit(1)
            resolved_kind = "task"
            resolved_target = str(t_rows[0][0])
        elif g_rows:
            if len(g_rows) > 1:
                print_error(f"ambiguous goal prefix {target_id!r}")
                raise typer.Exit(1)
            resolved_kind = "goal"
            resolved_target = str(g_rows[0][0])
        else:
            print_error(f"no task or goal matching prefix {target_id!r}")
            raise typer.Exit(1)
    else:
        print_error(f"unknown --kind: {target_kind!r}")
        raise typer.Exit(2)

    reminder = reminder_store.create(
        target_kind=resolved_kind,
        target_id=resolved_target,
        remind_at=at,
        channel=channel,
        note=note,
    )
    console.print(
        f"  [green]created reminder[/green] [cyan]{_short_id(reminder.id)}[/cyan]"
        f"  [dim]{resolved_kind}={_short_id(resolved_target)}"
        f" at {at.isoformat()} channel={channel}[/dim]"
    )


@reminder_app.command("list")
def cmd_reminder_list(
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            help="Filter by target id prefix (matches against tasks and goals).",
        ),
    ] = None,
    target_kind: Annotated[
        str | None,
        typer.Option("--kind", help="Filter by target kind."),
    ] = None,
    include_fired: Annotated[
        bool, typer.Option("--include-fired", help="Include already-delivered reminders.")
    ] = False,
    include_dismissed: Annotated[
        bool,
        typer.Option("--include-dismissed", help="Include cancelled reminders."),
    ] = False,
    limit: Annotated[int, typer.Option("--limit")] = 50,
    tasks_db_path: Annotated[
        Path | None,
        typer.Option("--tasks-db", help="Override tasks.db location."),
    ] = None,
) -> None:
    """List reminders, defaulting to active (not fired, not dismissed)."""
    store = _open_reminder_store(tasks_db_path)
    target_id_filter: str | None = None
    if target is not None:
        task_store = _open_task_store(tasks_db_path)
        if target_kind == "goal":
            target_id_filter = _resolve_goal_id(task_store, target)
        elif target_kind == "task":
            target_id_filter = _resolve_task_id(task_store, target)
        else:
            # No kind specified; pass the literal prefix and rely on
            # exact id match below — list takes target_id as exact.
            target_id_filter = target

    rows = store.list(
        target_kind=target_kind,
        target_id=target_id_filter,
        include_fired=include_fired,
        include_dismissed=include_dismissed,
        limit=limit,
    )
    if not rows:
        console.print("  [yellow]no reminders match[/yellow]")
        return

    from rich.table import Table

    table = Table(title=f"{len(rows)} reminder(s)", show_lines=False)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("kind", style="dim", no_wrap=True)
    table.add_column("target", style="dim", no_wrap=True)
    table.add_column("remind_at", style="dim", no_wrap=True)
    table.add_column("channel", style="dim", no_wrap=True)
    table.add_column("state", style="dim", no_wrap=True)
    table.add_column("note")
    for r in rows:
        if r.fired_at is not None:
            state = "fired"
        elif r.dismissed_at is not None:
            state = "dismissed"
        else:
            state = "pending"
        table.add_row(
            _short_id(r.id),
            r.target_kind,
            _short_id(r.target_id),
            r.remind_at.strftime("%Y-%m-%d %H:%M"),
            r.channel,
            state,
            r.note,
        )
    console.print(table)


@reminder_app.command("cancel")
def cmd_reminder_cancel(
    reminder_id: Annotated[str, typer.Argument(help="Reminder id (prefix accepted).")],
    tasks_db_path: Annotated[
        Path | None,
        typer.Option("--tasks-db", help="Override tasks.db location."),
    ] = None,
) -> None:
    """Dismiss a pending reminder (terminal — will never fire)."""
    store = _open_reminder_store(tasks_db_path)
    resolved = _resolve_reminder_id(store, reminder_id)
    reminder = store.cancel(resolved)
    if reminder.fired_at is not None:
        console.print(
            f"  [yellow]reminder[/yellow] [cyan]{_short_id(reminder.id)}[/cyan]"
            "  [dim]already fired — no change[/dim]"
        )
    else:
        console.print(f"  [yellow]cancelled[/yellow] [cyan]{_short_id(reminder.id)}[/cyan]")


@reminder_app.command("tick")
def cmd_reminder_tick(
    tasks_db_path: Annotated[
        Path | None,
        typer.Option("--tasks-db", help="Override tasks.db location."),
    ] = None,
) -> None:
    """Fire every reminder whose remind_at has passed.

    User-invoked in Phase 2 — the legacy reminder_tick heartbeat
    continues to drive the old subsystem. A heartbeat handler for
    this new store is a one-line follow-up (subscribe a tick handler
    in `iris_harness.runtime.bootstrap`).
    """
    store = _open_reminder_store(tasks_db_path)
    fired = store.tick()
    if not fired:
        console.print("  [yellow]no due reminders[/yellow]")
        return
    console.print(f"  [green]fired {len(fired)} reminder(s)[/green]")
    for r in fired:
        console.print(
            f"    [cyan]{_short_id(r.id)}[/cyan]  "
            f"[dim]{r.target_kind}={_short_id(r.target_id)} channel={r.channel}[/dim]"
            f"  {r.note}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
