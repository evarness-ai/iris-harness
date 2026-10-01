"""``iris ontology`` — the words memory learned on its own (ADR-0115 decision 7).

Learned terms are rows in the memory database, never YAML. These commands list them,
settle them (reject: never learned again; activate: learned now) and turn one into a
YAML snippet for a human to review and commit (``promote``) — the runtime never writes
config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from iris_harness.cli.render import console, print_error
from iris_harness.foundation.paths import data_dir
from iris_harness.memory.store import MemoryStore
from iris_harness.memory.vocabulary import export_yaml, promotion

ontology_app = typer.Typer(
    name="ontology",
    help="The vocabulary memory learned: list, reject, activate, promote, export.",
    no_args_is_help=True,
)

DbPath = Annotated[
    str | None,
    typer.Option("--db-path", help="Path to memory.db (default: $IRIS_DATA_DIR/memory.db)."),
]


def _store(db_path: str | None) -> MemoryStore:
    path = Path(db_path) if db_path else data_dir() / "memory.db"
    store = MemoryStore(db_path=path)
    store.ensure_schema()
    return store


@ontology_app.command("terms")
def cmd_terms(
    db_path: DbPath = None,
    status: Annotated[
        str | None,
        typer.Option("--status", help="candidate, alias, active or rejected (default: all)."),
    ] = None,
) -> None:
    """List learned terms with how often, and in how many conversations, each turned up."""
    terms = _store(db_path).vocabulary().terms(status)
    if not terms:
        console.print("  [dim]nothing learned yet[/dim]")
        return
    table = Table(title=f"Learned terms ({len(terms)})")
    for column in ("Term", "Status", "Seen", "Conversations", "Alias of", "Examples"):
        table.add_column(column)
    for t in terms:
        table.add_row(
            t.name,
            t.status,
            str(t.observations),
            str(len(t.episodes)),
            t.alias_of or "—",
            ", ".join(t.examples)[:60],
        )
    console.print(table)


@ontology_app.command("reject")
def cmd_reject(
    name: Annotated[str, typer.Argument(help="Term, e.g. mentors or learned:mentors.")],
    db_path: DbPath = None,
) -> None:
    """Never learn this word; anything already said with it leaves memory (kept in history)."""
    rejected = _store(db_path).vocabulary().reject(name)
    if rejected is None:
        print_error(f"no learned term {name}")
        raise typer.Exit(1)
    console.print(f"  [green]rejected[/green] {rejected.name}")


@ontology_app.command("activate")
def cmd_activate(
    name: Annotated[str, typer.Argument(help="Term, e.g. mentors or learned:mentors.")],
    db_path: DbPath = None,
) -> None:
    """Learn this word now, without waiting for it to recur."""
    active = _store(db_path).vocabulary().activate(name)
    if active is None:
        print_error(f"no candidate term {name} (an alias cannot be activated)")
        raise typer.Exit(1)
    console.print(f"  [green]active[/green] {active.name}")


@ontology_app.command("expire")
def cmd_expire(db_path: DbPath = None) -> None:
    """Forget candidates unseen for learning.yaml's expire_days."""
    gone = _store(db_path).vocabulary().expire()
    console.print(f"  expired [bold]{len(gone)}[/bold] candidate(s)")


@ontology_app.command("promote")
def cmd_promote(
    name: Annotated[str, typer.Argument(help="Term, e.g. mentors or learned:mentors.")],
    db_path: DbPath = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the snippet here instead of printing.")
    ] = None,
) -> None:
    """Print the YAML that would declare this term — for a human to review and commit."""
    vocabulary = _store(db_path).vocabulary()
    wanted = vocabulary.qualified(name)
    term = next((t for t in vocabulary.terms() if t.name == wanted), None)
    if term is None or term.status == "alias":
        print_error(f"no learned term {name} to promote")
        raise typer.Exit(1)
    text = promotion(term, vocabulary.graph.declared)
    if out is None:
        console.print(text, markup=False, highlight=False)
        return
    out.write_text(text, encoding="utf-8")
    console.print(f"  wrote {out}")


@ontology_app.command("export")
def cmd_export(
    db_path: DbPath = None,
    status: Annotated[str | None, typer.Option("--status", help="Only this status.")] = None,
) -> None:
    """Every learned term as an ontology fragment (status and counts as comments)."""
    vocabulary = _store(db_path).vocabulary()
    console.print(
        export_yaml(vocabulary.terms(status), vocabulary.graph.declared),
        markup=False,
        highlight=False,
    )
