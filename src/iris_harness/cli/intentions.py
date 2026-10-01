"""``iris intentions`` — review and approve rolled-up intentions (HITL, digital-twin layer 3).

The intention rollup (opt-in `IRIS_INTENTION_ROLLUP`) aggregates your open tasks, routines,
confirmed habits, and steering signals into higher-level GOALS you are working toward over
weeks/months, and queues them for review. Nothing becomes durable until you approve it here.
Approving an intention writes it into your ACTIVE identity layer (`~/.iris/memory/active.md`),
which is injected into the agent's context every turn — so the agent works toward it. Read is
always available; approve/dismiss are your call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from iris_harness.cli.render import console, print_error
from iris_harness.foundation.paths import data_dir
from iris_harness.services.learning.store import IntentionProposal, LearningMetricsStore

intentions_app = typer.Typer(
    name="intentions",
    help="Review rolled-up intentions: list / show / approve / dismiss (HITL, propose-only).",
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


@intentions_app.command("stats")
def cmd_stats(db_path: DbPath = None) -> None:
    """Show accept/reject tallies + acceptance rate for rolled-up intentions."""
    from iris_harness.services.learning.proposal_quality import (
        build_proposal_quality,
    )

    q = build_proposal_quality(_store(db_path)).intentions
    console.print(
        f"  intentions — awaiting: [yellow]{q.awaiting}[/yellow]  "
        f"active: [green]{q.accepted}[/green]  dismissed: [red]{q.rejected}[/red]  "
        f"acceptance: [cyan]{q.acceptance_rate:.0%}[/cyan] of {q.reviewed} reviewed"
    )


@intentions_app.command("list")
def cmd_list(
    db_path: DbPath = None,
    status: Annotated[
        str, typer.Option("--status", help="proposed | active | dismissed")
    ] = "proposed",
) -> None:
    """List rolled-up intentions awaiting your review."""
    rows = _store(db_path).list_intentions(status=status)
    if not rows:
        console.print(f"  [green]no {status} intentions[/green]")
        return
    table = Table(title=f"Intentions ({status}, {len(rows)})", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Goal")
    table.add_column("Summary", style="dim", max_width=50)
    for r in rows:
        table.add_row(r.intention_id, r.title, r.summary)
    console.print(table)
    console.print(
        "  [dim]inspect one: `iris intentions show <id>`; adopt as an active goal: "
        "`iris intentions approve <id>`; discard: `iris intentions dismiss <id>`.[/dim]"
    )


@intentions_app.command("show")
def cmd_show(
    intention_id: Annotated[str, typer.Argument(help="Intention id from `intentions list`.")],
    db_path: DbPath = None,
) -> None:
    """Show a single intention with its supporting evidence."""
    proposal = _store(db_path).get_intention(intention_id)
    if proposal is None:
        print_error(f"no intention with id: {intention_id}")
        raise typer.Exit(1)
    console.print(f"[cyan]{proposal.intention_id}[/cyan]  ([magenta]{proposal.status}[/magenta])")
    console.print(f"  [bold]{proposal.title}[/bold]")
    if proposal.summary:
        console.print(f"  {proposal.summary}")
    if proposal.supporting:
        console.print("  [dim]draws on:[/dim]")
        for s in proposal.supporting:
            console.print(f"    - {s}")


@intentions_app.command("approve")
def cmd_approve(
    intention_id: Annotated[str, typer.Argument(help="Intention id from `intentions list`.")],
    db_path: DbPath = None,
) -> None:
    """Approve an intention — write it to the ACTIVE identity layer and clear the proposal."""
    from iris_harness.memory.identity.loader import add_active_item

    store = _store(db_path)
    proposal = store.get_intention(intention_id)
    if proposal is None or proposal.status != "proposed":
        print_error(f"no proposed intention with id: {intention_id}")
        raise typer.Exit(1)
    text = proposal.title if not proposal.summary else f"{proposal.title} — {proposal.summary}"
    add_active_item(text)
    store.resolve_intention(intention_id, "active")
    store.record_user_behavior_signal("intention_approved", subject=proposal.title)
    console.print(f"  [green]✓[/green] adopted as an active goal: [cyan]{proposal.title}[/cyan]")
    console.print("  [dim](injected into the agent's context on the next runtime start)[/dim]")


@intentions_app.command("dismiss")
def cmd_dismiss(
    intention_id: Annotated[str, typer.Argument(help="Intention id from `intentions list`.")],
    db_path: DbPath = None,
) -> None:
    """Dismiss a proposed intention (it won't be re-proposed)."""
    store = _store(db_path)
    proposal: IntentionProposal | None = store.get_intention(intention_id)
    if not store.resolve_intention(intention_id, "dismissed"):
        print_error(f"no intention with id: {intention_id}")
        raise typer.Exit(1)
    store.record_user_behavior_signal(
        "intention_dismissed", subject=proposal.title if proposal else intention_id
    )
    console.print(f"  [green]✓[/green] dismissed [cyan]{intention_id}[/cyan]")


__all__ = ["intentions_app"]
