"""``iris graphiti import`` — a Graphiti export into IRIS memory (memris plan PR 9).

The harness publishes no ``graphiti`` group, so ``cli.group("graphiti")`` creates it.
Imported claims start as PROPOSALS by default: another system's belief is a question
for the owner here (ADR-0114), not something this memory already believes. ``--confirm``
imports them as beliefs.

Locations follow the core's rule: ``$IRIS_DATA_DIR/memory.db`` (``data`` when unset) and
the resolved config directory's ``memory/`` (``IRIS_CONFIG_DIR``, else the checkout's
``config/``, else the packaged defaults), each overridable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from iris_harness.sdk.persistence import data_dir

if TYPE_CHECKING:
    from iris_harness.sdk import PluginCLI

HERE = Path(__file__).resolve().parent


def register(cli: PluginCLI) -> None:
    app = cli.group("graphiti", help="Bring a Graphiti knowledge graph into IRIS memory.")
    app.command("import")(cmd_import)


def import_export(
    document: dict[str, Any],
    graph: Any,
    *,
    confirm: bool = False,
    mappings_path: Path | None = None,
    config_path: Path | None = None,
) -> Any:
    """Read ``document`` and write what maps into ``graph``; returns the MappingReport."""
    from memris.interchange import apply_mappings
    from memris.ontology import load_mappings

    from .reader import load_config, read_export

    rules = load_mappings(mappings_path or HERE / "mappings.yaml", graph.ontology)
    read = read_export(document, load_config(config_path))
    report = apply_mappings(
        read.records, rules, graph, status="confirmed" if confirm else "proposed"
    )
    report.skipped[:0] = read.skipped
    return report


def cmd_import(
    export: Annotated[Path, typer.Argument(help="Graphiti export (JSON: nodes, edges, episodes).")],
    db: Annotated[
        Path | None,
        typer.Option("--db", help="memory.db to write (default: $IRIS_DATA_DIR/memory.db)."),
    ] = None,
    ontology_dir: Annotated[
        Path | None,
        typer.Option(
            "--ontology-dir", help="Ontology directory (default: $IRIS_CONFIG_DIR/memory)."
        ),
    ] = None,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Import as beliefs, not proposals for review.")
    ] = False,
) -> None:
    """Import a Graphiti graph export; print what was written and everything that was not."""
    from iris_harness.sdk.memory import memory_ontology
    from memris.graph import MemoryGraph
    from memris.store import SQLiteGraphStore

    db_path = db or data_dir() / "memory.db"
    document = json.loads(export.read_text(encoding="utf-8"))
    # The vocabulary IRIS runs with — the core plus installed plugins' fragments — so an
    # entity stored under a plugin's class (a bank) is recognised, not duplicated.
    graph = MemoryGraph(memory_ontology(ontology_dir), SQLiteGraphStore(db_path))
    report = import_export(document, graph, confirm=confirm)

    typer.echo(f"  {report.summary()}")
    for label, count in sorted(report.unmapped_types.items()):
        typer.echo(f"  unmapped relation {label} ×{count} — add it to mappings.yaml to keep it")
    for source_id, why in report.skipped:
        typer.echo(f"  skipped {source_id}: {why}")
    for source_id, note in report.notes:
        typer.echo(f"  approximated {source_id}: {note}")
    if not confirm and report.statements:
        typer.echo(
            "  imported as proposals: stored and queryable, not believed (--confirm to import as beliefs)"
        )
    if not report.lossless:
        raise typer.Exit(1)
