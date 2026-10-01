"""``iris logs`` — the encrypted session-log archive (retention.yaml ``logs.archive``).

Session logs older than ``delete_after_days`` move to ``~/.iris/archive/logs`` as one
encrypted file per month instead of being deleted. These commands list it and put a
session (or a month) back where the Sessions view reads it. Same data as
``GET /logs/archive`` and ``POST /logs/archive/restore``.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.table import Table

from iris_harness.cli.memory import DbPath, _retention
from iris_harness.cli.render import console, print_error

logs_app = typer.Typer(
    name="logs",
    help="The encrypted archive of past session logs: list and restore.",
    no_args_is_help=True,
)


@logs_app.command("archive")
def cmd_archive(db_path: DbPath = None) -> None:
    """List the archived months and the sessions in each."""
    info = _retention(db_path).archived_logs()
    if not info["months"]:
        console.print("  the archive is empty — session logs move there after delete_after_days")
        return
    table = Table(title=f"Session-log archive ({info['root']})")
    table.add_column("Month", style="cyan")
    table.add_column("Sessions", justify="right")
    table.add_column("Stored", justify="right")
    for month in info["months"]:
        table.add_row(month["month"], str(month["files"]), f"{month['stored_bytes'] / 1024:.0f} KB")
    console.print(table)


@logs_app.command("restore")
def cmd_restore(
    session: Annotated[
        str | None, typer.Option("--session", help="Session id, e.g. web-3dedfff1")
    ] = None,
    month: Annotated[str | None, typer.Option("--month", help="A whole month: YYYY-MM")] = None,
    db_path: DbPath = None,
) -> None:
    """Put an archived session (or month) back into the live logs for 30 more days."""
    if (session is None) == (month is None):
        print_error("give exactly one of --session or --month")
        raise typer.Exit(2)
    try:
        restored = _retention(db_path).restore_logs(session=session, month=month)
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc
    if not restored:
        console.print("  nothing restored (not archived, or already live)")
        return
    for name in restored:
        console.print(f"  [green]restored[/green] {name}")


__all__ = ["logs_app"]
