"""``iris signals`` — inspect user-behavior signals (how you steer the assistant).

A read-only view of the user's explicit steering: facts you corrected/forgot, and mined
behavior patterns you confirmed/dismissed. Distinct from model-performance metrics — this
is ground truth about you, captured deterministically as you act, and the substrate for
the longitudinal intention model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from iris_harness.cli.render import console
from iris_harness.foundation.paths import data_dir
from iris_harness.services.learning.store import LearningMetricsStore

signals_app = typer.Typer(
    name="signals",
    help="Inspect user-behavior signals (corrections, confirmations) — read-only.",
    no_args_is_help=True,
)

DbPath = Annotated[
    str | None,
    typer.Option("--db-path", help="Path to learning.db (default: $IRIS_DATA_DIR/learning.db)."),
]


def _store(db_path: str | None) -> LearningMetricsStore:
    path = Path(db_path) if db_path else data_dir() / "learning.db"
    store = LearningMetricsStore(db_path=path)
    store.ensure_schema()
    return store


@signals_app.command("list")
def cmd_list(
    db_path: DbPath = None,
    kind: Annotated[
        str | None, typer.Option("--kind", help="Filter by kind (e.g. fact_corrected).")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max rows.")] = 50,
) -> None:
    """List recent user-behavior signals (newest first)."""
    rows = _store(db_path).list_user_behavior_signals(kind=kind, limit=limit)
    if not rows:
        console.print("  [green]no user-behavior signals yet[/green]")
        return
    table = Table(title=f"User-behavior signals ({len(rows)})", show_lines=False)
    table.add_column("Kind", style="magenta")
    table.add_column("Subject", style="cyan")
    table.add_column("Detail", style="dim", max_width=40)
    table.add_column("When", style="dim")
    for s in rows:
        table.add_row(s.kind, s.subject, s.detail, s.created_at[:16].replace("T", " "))
    console.print(table)


@signals_app.command("summary")
def cmd_summary(db_path: DbPath = None) -> None:
    """Show counts of user-behavior signals by kind — how you've steered IRIS."""
    counts = _store(db_path).user_behavior_summary()
    if not counts:
        console.print("  [green]no user-behavior signals yet[/green]")
        return
    table = Table(title="How you steer IRIS", show_lines=False)
    table.add_column("Kind", style="magenta")
    table.add_column("Count", justify="right")
    for kind, n in counts.items():
        table.add_row(kind, str(n))
    console.print(table)


__all__ = ["signals_app"]
