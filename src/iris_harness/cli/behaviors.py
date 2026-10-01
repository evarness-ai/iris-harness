"""``iris behaviors`` — review and approve mined behavior patterns (HITL).

The behavior miner (opt-in `IRIS_BEHAVIOR_MINER`) proposes recurring habits into a
review queue; nothing becomes durable until you approve it here. Approving appends the
pattern to your episodic memory (`~/.iris/memory/episodic.md`), which the agent uses for
proactive suggestions. Read is always available; approve/reject are your call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from iris_harness.cli.render import console, print_error
from iris_harness.foundation.paths import data_dir
from iris_harness.services.learning.store import LearningMetricsStore

behaviors_app = typer.Typer(
    name="behaviors",
    help="Review mined behavior patterns: list / approve / reject (HITL, propose-only).",
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


@behaviors_app.command("list")
def cmd_list(
    db_path: DbPath = None,
    status: Annotated[
        str, typer.Option("--status", help="pending | approved | rejected")
    ] = "pending",
) -> None:
    """List proposed behavior patterns awaiting your review."""
    rows = _store(db_path).list_behavior_proposals(status=status)
    if not rows:
        console.print(f"  [green]no {status} behavior patterns[/green]")
        return
    table = Table(title=f"Behavior patterns ({status}, {len(rows)})", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Conf", style="magenta")
    table.add_column("Pattern")
    table.add_column("Evidence", style="dim", max_width=40)
    for r in rows:
        table.add_row(r.pattern_id, r.confidence, r.text, "; ".join(r.evidence))
    console.print(table)
    console.print(
        "  [dim]approve a habit into episodic memory: `iris behaviors approve <id>`; "
        "discard: `iris behaviors reject <id>`.[/dim]"
    )


@behaviors_app.command("stats")
def cmd_stats(db_path: DbPath = None) -> None:
    """Show accept/reject tallies + acceptance rate for mined behavior proposals."""
    from iris_harness.services.learning.proposal_quality import (
        build_proposal_quality,
    )

    q = build_proposal_quality(_store(db_path)).behaviors
    console.print(
        f"  behaviors — awaiting: [yellow]{q.awaiting}[/yellow]  "
        f"approved: [green]{q.accepted}[/green]  rejected: [red]{q.rejected}[/red]  "
        f"acceptance: [cyan]{q.acceptance_rate:.0%}[/cyan] of {q.reviewed} reviewed"
    )


@behaviors_app.command("approve")
def cmd_approve(
    pattern_id: Annotated[str, typer.Argument(help="Pattern id from `behaviors list`.")],
    db_path: DbPath = None,
) -> None:
    """Approve a pattern — append it to durable episodic memory and clear the proposal."""
    from iris_harness.memory.identity.loader import append_episodic_pattern

    store = _store(db_path)
    proposal = store.get_behavior_proposal(pattern_id)
    if proposal is None or proposal.status != "pending":
        print_error(f"no pending behavior pattern with id: {pattern_id}")
        raise typer.Exit(1)
    append_episodic_pattern(proposal.text)
    store.resolve_behavior_proposal(pattern_id, "approved")
    store.record_user_behavior_signal(
        "pattern_confirmed", subject=proposal.text, detail=proposal.confidence
    )
    console.print(f"  [green]✓[/green] added to episodic memory: [cyan]{proposal.text}[/cyan]")
    console.print("  [dim](recall picks it up on the next runtime start)[/dim]")


@behaviors_app.command("reject")
def cmd_reject(
    pattern_id: Annotated[str, typer.Argument(help="Pattern id from `behaviors list`.")],
    db_path: DbPath = None,
) -> None:
    """Reject a proposed pattern (it won't be re-proposed)."""
    store = _store(db_path)
    proposal = store.get_behavior_proposal(pattern_id)
    if not store.resolve_behavior_proposal(pattern_id, "rejected"):
        print_error(f"no behavior pattern with id: {pattern_id}")
        raise typer.Exit(1)
    store.record_user_behavior_signal(
        "pattern_dismissed", subject=proposal.text if proposal else pattern_id
    )
    console.print(f"  [green]✓[/green] rejected [cyan]{pattern_id}[/cyan]")


__all__ = ["behaviors_app"]
