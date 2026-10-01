"""``iris doctor`` -- renders the install preflight (``services/system/doctor.py``).

A thin renderer: the checks, the verdict and the fixes are the harness module's; this
file prints them, asks before a fix, and exits with the verdict's code (0 ready, 1 demo
only, 2 not ready). It runs in-process, not over the API: a preflight runs before any
service is up. ``GET /health/doctor`` serves the same report from a running API.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape

if TYPE_CHECKING:  # imported when a command runs, so `iris --help` stays cheap
    from iris_harness.services.system import doctor as dr

console = Console()

_STATUS_STYLE = {"pass": "green", "warn": "yellow", "fail": "bold red", "info": "dim"}


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def render_report(report: dr.DoctorReport, out: Console | None = None) -> None:
    """Every check, its fix when it has one, then the verdict and what stands in the way."""
    from iris_harness.services.system import doctor as dr

    con = out or console
    con.print("[bold]IRIS doctor[/bold]")
    for check in report.checks:
        style = _STATUS_STYLE[check.status]
        con.print(
            f"  [{style}]{check.status:<4}[/{style}]  {escape(check.name):<16} "
            f"{escape(check.detail)}"
        )
        if check.fix and check.status in ("fail", "warn"):
            con.print(f"        [dim]fix:[/dim] {escape(check.fix)}")
    con.print()
    verdict = report.verdict
    style = {"ready": "bold green", "demo_only": "bold yellow", "not_ready": "bold red"}[
        verdict.value
    ]
    con.print(f"[{style}]{verdict.headline}[/{style}]")
    blocking = [c for c in report.checks if c.status == "fail" and c.blocks != "none"]
    if verdict is dr.Verdict.DEMO_ONLY:
        con.print("  The synthetic demo runs. Real use still needs:")
    elif verdict is dr.Verdict.NOT_READY:
        con.print("  Fix these first:")
    for check in blocking:
        con.print(f"  - {escape(check.name)}: {escape(check.fix or check.detail)}")
    if report.fixable:
        con.print("  [dim]`iris doctor --fix` applies the safe fixes (models, vault key).[/dim]")


def _print_progress() -> Callable[[dr.PullProgress], None]:
    """One line per status change, and each tenth of a download."""
    seen: dict[str, int] = {}

    def show(event: dr.PullProgress) -> None:
        if event.total and event.completed is not None:
            tenth = int(event.completed * 10 / event.total)
            if seen.get(event.status) == tenth:
                return
            seen[event.status] = tenth
            console.print(f"    {escape(event.status)} {tenth * 10}%")
            return
        if event.status in seen:
            return
        seen[event.status] = -1
        console.print(f"    {escape(event.status)}")

    return show


def apply_fixes(
    report: dr.DoctorReport,
    *,
    confirm: Callable[[str], bool],
    client: Any = None,
) -> None:
    """Pull the missing starter models and create a missing vault key, each after
    ``confirm``. Nothing else is touched."""
    from iris_harness.services.system import doctor as dr

    for model in report.missing_models:
        if not confirm(f"Pull {model.name} (~{model.size_gb:g} GB download)?"):
            continue
        console.print(f"  pulling {escape(model.name)} (~{model.size_gb:g} GB) ...")
        try:
            if client is None:
                with dr.http_client() as own:
                    dr.pull_model(
                        model.name,
                        client=own,
                        root=report.ollama_url,
                        on_progress=_print_progress(),
                    )
            else:
                dr.pull_model(
                    model.name, client=client, root=report.ollama_url, on_progress=_print_progress()
                )
        except dr.DoctorFixError as exc:
            console.print(f"  [bold red]pull failed:[/bold red] {escape(str(exc))}")
        else:
            console.print(f"  [green]pulled {escape(model.name)}[/green]")

    if dr.key_needs_fix(report.key_status):
        if confirm("Create a vault master key?"):
            _render_key_fix(dr.fix_master_key())


def _render_key_fix(result: dr.KeyFix) -> None:
    if result.outcome == "stored":
        console.print(f"  [green]{escape(result.detail)}[/green]")
    elif result.outcome == "export":
        console.print(f"  {escape(result.detail)}")
        console.print()
        # Plain, unwrapped and unstyled, so it copies as one line.
        print(result.export_line)
        console.print()
    else:
        console.print(f"  {escape(result.detail)}")


def cmd_doctor(
    fix: Annotated[
        bool, typer.Option("--fix", help="Apply the safe fixes: pull models, create a key.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Apply fixes without asking (implies --fix).")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the report as JSON (applies no fixes).")
    ] = False,
) -> None:
    """Preflight: can IRIS run on this machine, and what to fix if not."""
    from iris_harness.services.system import doctor as dr

    interactive = _interactive()
    wants_fix = fix or yes
    # The OS keyring is read only when someone can answer a Keychain dialog, or when a
    # fix was asked for (the fix must read it before it may write).
    report = dr.run_doctor(read_keyring=interactive or (wants_fix and not as_json))
    if as_json:
        print(json.dumps(report.as_dict(), indent=2))
        raise typer.Exit(report.verdict.exit_code)

    render_report(report)
    if not wants_fix and interactive and report.fixable:
        console.print()
        wants_fix = typer.confirm("Apply the safe fixes now?", default=False)
    if wants_fix and report.fixable:
        if not yes and not interactive:
            console.print("[yellow]Not a terminal: pass --yes to apply fixes.[/yellow]")
        else:
            console.print()

            def confirm(question: str) -> bool:
                return True if yes else typer.confirm(question, default=True)

            apply_fixes(report, confirm=confirm)
            report = dr.run_doctor(read_keyring=True)
            console.print()
            render_report(report)
    raise typer.Exit(report.verdict.exit_code)
