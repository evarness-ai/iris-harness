"""``iris memory`` — coherence doctor over the user-fact stores.

A user fact lives in three homes: the SQLite store (``memory.db``, the source of
truth), the ChromaDB semantic index, and the ``## Auto-detected`` block of
``USER.md``. ``iris memory doctor`` diagnoses drift between them and, with
``--repair``, rewrites the two *derived* homes back to the SQLite truth (it never
mutates the store, so it is always safe to run). See
``docs/architecture/memory-store-registry.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from iris_harness.cli.render import console, print_error
from iris_harness.foundation.paths import data_dir
from iris_harness.memory import coherence
from iris_harness.memory.retention import RetentionService
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.store import MemoryStore

memory_app = typer.Typer(
    name="memory",
    help="Inspect + repair coherence across the user-fact stores (SQLite truth / index / USER.md).",
    no_args_is_help=True,
)

DbPath = Annotated[
    str | None,
    typer.Option("--db-path", help="Path to memory.db (default: $IRIS_DATA_DIR/memory.db)."),
]


def _base_dir(db_path: str | None) -> Path:
    if db_path:
        return Path(db_path).parent
    return data_dir()


def _store(db_path: str | None) -> MemoryStore:
    path = Path(db_path) if db_path else _base_dir(db_path) / "memory.db"
    store = MemoryStore(db_path=path)
    store.ensure_schema()
    return store


def _index(db_path: str | None) -> SemanticIndex:
    return SemanticIndex(persist_dir=_base_dir(db_path) / "chroma")


def _render_drift(label: str, keys: set[str]) -> None:
    if not keys:
        return
    ordered = sorted(keys)
    sample = ", ".join(ordered[:8])
    more = "" if len(ordered) <= 8 else f" (+{len(ordered) - 8} more)"
    console.print(f"  [yellow]DRIFT[/yellow] {label}: {len(keys)} — {sample}{more}")


@memory_app.command("doctor")
def cmd_doctor(
    db_path: DbPath = None,
    repair: Annotated[
        bool,
        typer.Option("--repair", "-r", help="Rewrite the derived homes to match the SQLite truth."),
    ] = False,
    include_md: Annotated[
        bool, typer.Option("--md/--no-md", help="Include the USER.md projection leg.")
    ] = True,
) -> None:
    """Diagnose (and optionally ``--repair``) drift across the three user-fact homes.

    Read-only by default: prints per-leg drift and exits non-zero when incoherent,
    so it doubles as a CI/health check. ``--repair`` reconciles the index (and
    USER.md unless ``--no-md``) back to the SQLite truth.
    """
    store = _store(db_path)
    index = _index(db_path)
    report = coherence.diagnose(store, index, include_md=include_md)

    header = (
        f"  facts: [cyan]{report.store_count}[/cyan] in store / "
        f"[cyan]{report.index_count}[/cyan] in index"
    )
    if include_md:
        header += f" / [cyan]{report.md_count}[/cyan] in USER.md"
    console.print(header)

    _render_drift("index missing (in store, not indexed)", report.index_missing)
    _render_drift("index orphans (indexed, gone from store)", report.index_orphans)
    if include_md:
        _render_drift("USER.md missing (in store, not projected)", report.md_missing)
        _render_drift("USER.md orphans (stale bullet, gone from store)", report.md_orphans)
        for key, md_v, store_v in report.md_value_mismatches:
            console.print(
                f"  [yellow]DRIFT[/yellow] USER.md value {key}: "
                f"[dim]{md_v}[/dim] != store [cyan]{store_v}[/cyan]"
            )

    if report.is_coherent:
        console.print("  [green]OK[/green] — all fact-homes are coherent")
        return

    console.print(f"  [yellow]{report.drift_total} drift item(s)[/yellow]")
    if not repair:
        console.print(
            "  run [cyan]iris memory doctor --repair[/cyan] to reconcile the derived homes"
        )
        raise typer.Exit(1)

    actions = coherence.repair(store, index, reproject_md=include_md)
    summary = ", ".join(f"{k}={v}" for k, v in actions.items())
    console.print(f"  [green]repaired[/green] — {summary}")


# ----------------------------------------------------------------------
# Retention — see config/memory/retention.yaml
# ----------------------------------------------------------------------


def _retention(db_path: str | None) -> RetentionService:
    from iris_harness.memory.identity.loader import iris_home
    from iris_harness.memory.log_archive import LogArchive

    store = _store(db_path)
    return RetentionService(
        store,
        _index(db_path),
        logs_dir=iris_home() / "logs",
        history_path=store.db_path.parent / "housekeeping_runs.jsonl",
        archive=LogArchive(iris_home() / "archive" / "logs"),
    )


@memory_app.command("housekeeping")
def cmd_housekeeping(
    db_path: DbPath = None,
    apply: Annotated[
        bool, typer.Option("--apply", help="Actually delete. Default is a dry run.")
    ] = False,
) -> None:
    """Run the retention pass: cool old conversations, sweep vectors, rotate logs.

    Dry run by default — it reports what WOULD go. Summaries, confirmed facts and
    anything you wrote are never touched by this pass.
    """
    report = _retention(db_path).run(dry_run=not apply)
    mode = "applied" if apply else "dry run — nothing deleted"
    console.print(f"  [bold]housekeeping[/bold] ({mode})")
    console.print(
        f"  sessions closed (final summary):  {report.sessions_closed}"
        + (f" ({report.sessions_close_backlog} waiting)" if report.sessions_close_backlog else "")
    )
    console.print(f"  sessions cooled to their summary: {report.sessions_cooled}")
    console.print(f"  turns removed:                    {report.turns_deleted}")
    console.print(f"  turn vectors removed:             {report.vectors_deleted}")
    console.print(f"  orphan vectors swept:             {report.orphan_vectors_swept}")
    console.print(
        f"  logs: {report.logs_rotated} rotated, {report.logs_archived} archived, "
        f"{report.logs_compressed} compressed, {report.logs_deleted} deleted, "
        f"{report.log_bytes_reclaimed // (1024 * 1024)} MB reclaimed"
    )
    if report.errors:
        for err in report.errors:
            print_error(err)


@memory_app.command("forget")
def cmd_forget(
    needle: Annotated[
        str, typer.Argument(help="Text to forget (matches turns, summaries, facts).")
    ],
    db_path: DbPath = None,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Actually delete. Default previews.")
    ] = False,
) -> None:
    """Forget everything about something. Previews first; deletes only with --confirm."""
    service = _retention(db_path)
    result = service.forget_matching(needle, confirm=confirm)
    if result.get("deleted"):
        console.print(
            f"  [green]forgotten[/green] — {result['turns']} turn(s), "
            f"{result['vectors']} vector(s), {result['summaries']} summary(ies), "
            f"{result['facts']} fact(s), {result['log_lines']} log line(s), "
            f"{result['archived_log_lines']} archived log line(s)"
        )
        return

    preview = result["preview"]
    if not preview["total"]:
        console.print(f"  nothing stored matches [bold]{needle}[/bold]")
        return
    console.print(f"  [bold]would forget {preview['total']} item(s) matching '{needle}'[/bold]")
    if preview["facts"]:
        table = Table(title=f"Facts ({len(preview['facts'])})")
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        for fact in preview["facts"]:
            table.add_row(fact["key"], fact["value"])
        console.print(table)
    if preview["summaries"]:
        console.print(f"  session summaries: {len(preview['summaries'])}")
    if preview["log_lines"] or preview["archived_log_lines"]:
        console.print(
            f"  session log lines: {preview['log_lines']} live, "
            f"{preview['archived_log_lines']} in the encrypted archive"
        )
    if preview["turns"]:
        table = Table(title=f"Conversation turns ({len(preview['turns'])}, newest first)")
        table.add_column("Session", style="magenta", max_width=22)
        table.add_column("Role", style="dim")
        table.add_column("Content", max_width=60)
        for turn in preview["turns"][:15]:
            table.add_row(turn["session_id"], turn["role"], " ".join(turn["content"].split()))
        console.print(table)
        if len(preview["turns"]) > 15:
            console.print(f"  ... and {len(preview['turns']) - 15} more turn(s)")
    console.print("  [dim]re-run with --confirm to delete.[/dim]")


@memory_app.command("purge-sessions")
def cmd_purge_sessions(
    db_path: DbPath = None,
    session_ids: Annotated[
        str | None,
        typer.Option("--sessions", help="Comma-separated ids; default = test/eval runs."),
    ] = None,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Actually delete. Default previews.")
    ] = False,
) -> None:
    """Remove playground / eval / test sessions from your memory. Previews first.

    These are runs, not conversations: they polluted cross-session recall, which had no
    way to tell them from something you actually said.
    """
    ids = [s.strip() for s in session_ids.split(",")] if session_ids else None
    result = _retention(db_path).purge_sessions(ids, confirm=confirm)
    if result.get("deleted"):
        console.print(
            f"  [green]purged[/green] {result['total_sessions']} session(s) — "
            f"{result['turns']} turn(s), {result['vectors']} vector(s)"
        )
        return
    if not result["total_sessions"]:
        console.print("  nothing to purge — no test/eval sessions in this store")
        return
    table = Table(
        title=f"Would purge {result['total_sessions']} session(s), "
        f"{result['total_turns']} turn(s)"
    )
    table.add_column("Session", style="magenta")
    table.add_column("Turns", justify="right")
    for row in sorted(result["sessions"], key=lambda r: -r["turns"])[:25]:
        table.add_row(row["session_id"], str(row["turns"]))
    console.print(table)
    if result["total_sessions"] > 25:
        console.print(f"  ... and {result['total_sessions'] - 25} more session(s)")
    console.print("  [dim]re-run with --confirm to delete.[/dim]")


@memory_app.command("migrate-statements")
def cmd_migrate_statements(
    db_path: DbPath = None,
    out: Annotated[
        str | None,
        typer.Option(
            "--out", help="Where the report and triage file go (default: <data>/memris-migration)."
        ),
    ] = None,
) -> None:
    """Dry run: plan moving user facts onto memris statements, and report it.

    Opens memory.db read-only and writes only the report and the triage file (memris
    plan PR 2a). Facts on keys outside the allowlist are dropped by default; keep one
    by editing the triage file and running this again. Applying the plan is PR 2b.
    """
    from iris_harness.memory.ontology import fact_mappings, memory_ontology
    from iris_harness.memory.statement_migration import (
        TriageError,
        load_triage,
        plan_migration,
        read_legacy,
        render_report,
        render_triage,
    )

    db = Path(db_path) if db_path else _base_dir(db_path) / "memory.db"
    if not db.exists():
        print_error(f"no memory database at {db}")
        raise typer.Exit(1)
    out_dir = Path(out).expanduser() if out else _base_dir(db_path) / "memris-migration"
    out_dir.mkdir(parents=True, exist_ok=True)
    triage_path = out_dir / "triage.yaml"
    ontology = memory_ontology()
    try:
        triage = load_triage(triage_path, fact_mappings(ontology))
    except TriageError as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc

    data = read_legacy(db)
    plan = plan_migration(data, ontology, triage)
    triaged = {k.key for k in plan.keys if k.outcome in ("drop", "keep-as")}
    off_list = [f for f in data.facts if f.key in triaged]
    triage_path.write_text(render_triage(off_list, triage), encoding="utf-8")
    report_path = out_dir / "report.md"
    report_path.write_text(
        render_report(plan, db_path=db, triage_path=triage_path), encoding="utf-8"
    )

    console.print(
        f"  {plan.fact_rows} fact rows → mapped {len(plan.by_outcome('map'))} · "
        f"kept {len(plan.by_outcome('keep-as'))} · dropped {len(plan.by_outcome('drop'))} · "
        f"collision {len(plan.by_outcome('collision'))}"
    )
    console.print(
        f"  {len(plan.statements)} statements planned from {plan.history_rows} history rows"
    )
    for issue in plan.issues:
        console.print(f"  [yellow]⚠[/yellow] {issue}")
    console.print(f"  report: {report_path}\n  triage: {triage_path}")
    console.print(
        "  [dim]dry run — memory.db was opened read-only; applying arrives in PR 2b[/dim]"
    )


def _entity_label(graph: Any, entity_id: str) -> str:
    entity = graph.get_entity(entity_id)
    return entity.label if entity is not None else entity_id


@memory_app.command("entities")
def cmd_entities(
    db_path: DbPath = None,
    show: Annotated[
        str, typer.Option("--show", help="candidate | same | distinct | undone | all")
    ] = "all",
) -> None:
    """Which names memory treats as one thing — look-alikes, merges, and "not the same".

    A look-alike (candidate) stays a separate entity until the same pair turns up in
    enough conversations; then it merges on that evidence. Undo any merge with
    `iris memory unmerge <ID>`; settle a look-alike with `iris memory same <ID>` or
    `iris memory distinct <ID>` (chat asks about one per conversation).
    """
    graph = _store(db_path).memory_graph()
    rows = [d for d in graph.decisions() if show == "all" or d.decision == show]
    if not rows:
        console.print("  [green]no entity decisions yet[/green]")
        return
    table = Table(show_lines=False)
    table.add_column("ID", no_wrap=True)  # the id is what you copy: never cut it
    for column in ("Decision", "Kept / A", "Folded / B", "Score", "Evidence", "By"):
        table.add_column(column)
    for d in rows:
        table.add_row(
            d.id,
            d.decision,
            _entity_label(graph, d.a),
            _entity_label(graph, d.b),
            f"{d.score:.2f}" if d.score is not None else "—",
            str(len(d.evidence)),
            d.decided_by or "—",
        )
    console.print(table)


@memory_app.command("unmerge")
def cmd_unmerge(
    decision_id: Annotated[str, typer.Argument(help="A merge's ID from `iris memory entities`.")],
    db_path: DbPath = None,
) -> None:
    """Undo a merge: both names stand alone again, as they were."""
    from memris.graph import StatementError

    graph = _store(db_path).memory_graph()
    try:
        done = graph.unmerge(decision_id, decided_by="owner")
    except StatementError as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc
    console.print(
        f"  [green]✓[/green] {_entity_label(graph, done.b)} and "
        f"{_entity_label(graph, done.a)} are separate again"
    )


@memory_app.command("same")
def cmd_same(
    decision_id: Annotated[
        str, typer.Argument(help="A look-alike's ID from `iris memory entities`.")
    ],
    db_path: DbPath = None,
) -> None:
    """Two names for one thing: merge them now, the older name kept (undo: unmerge)."""
    from memris.graph import StatementError

    graph = _store(db_path).memory_graph()
    try:
        done = graph.accept_candidate(decision_id, decided_by="owner")
    except StatementError as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc
    console.print(
        f"  [green]✓[/green] {_entity_label(graph, done.b)} is now "
        f"{_entity_label(graph, done.a)} (undo: iris memory unmerge {done.id})"
    )


@memory_app.command("distinct")
def cmd_distinct(
    decision_id: Annotated[
        str, typer.Argument(help="A look-alike's ID from `iris memory entities`.")
    ],
    db_path: DbPath = None,
) -> None:
    """Two different things with similar names: never proposed or merged again."""
    from memris.graph import StatementError

    graph = _store(db_path).memory_graph()
    decision = next((d for d in graph.decisions() if d.id == decision_id), None)
    if decision is None:
        print_error(f"no entity decision with id {decision_id}")
        raise typer.Exit(1)
    try:
        graph.mark_distinct(decision.a, decision.b, decided_by="owner")
    except StatementError as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc
    console.print(
        f"  [green]✓[/green] {_entity_label(graph, decision.a)} and "
        f"{_entity_label(graph, decision.b)} will be kept apart"
    )


@memory_app.command("export")
def cmd_export(
    to: Annotated[str, typer.Option("--to", help="Directory to write the vault into.")],
    db_path: DbPath = None,
    include_unconfirmed: Annotated[
        bool,
        typer.Option("--include-unconfirmed", help="Also export what is awaiting review."),
    ] = False,
) -> None:
    """Export what IRIS remembers as a linked markdown vault (Obsidian-compatible).

    One way: nothing written in the vault is read back. IRIS believes what you confirm
    inside it (`iris facts review`); re-export to refresh the snapshot.
    """
    from iris_harness.memory.export import export_memory_vault

    out = Path(to).expanduser()
    result = export_memory_vault(_store(db_path), out, include_unconfirmed=include_unconfirmed)
    console.print(f"  exported [bold]{result.notes}[/bold] note(s) to {out}")
    if result.by_kind:
        console.print("  " + " · ".join(f"{k} {v}" for k, v in sorted(result.by_kind.items())))
    console.print(f"  {result.links} link(s) between notes")
    if result.skipped_unconfirmed:
        console.print(
            f"  [yellow]{result.skipped_unconfirmed} unconfirmed item(s) left out[/yellow] — "
            "re-run with --include-unconfirmed to include them, marked."
        )
    # Said here, not in a doc: the folder now holds personal memory.
    console.print(
        "\n  [bold]This folder contains your personal memory[/bold] — the people, "
        "institutions and accounts IRIS knows about."
    )
    console.print(
        "  [dim]An Obsidian vault stays on this machine. Uploading the folder to a cloud "
        "notebook (NotebookLM, Notion, a hosted LLM) sends that data off it.[/dim]"
    )


# -- Removed (ADR-0119): the same removal the Memory page's Map offers -----------------


def _removal(db_path: str | None) -> Any:
    from iris_harness.cli.facts import _coordinator
    from iris_harness.memory.removal import MemoryRemoval

    store = _store(db_path)

    def purge_session(session_id: str) -> None:
        _retention(db_path).purge_sessions([session_id], confirm=True)
        store.delete_conversation_summary(session_id)  # a cooled session has no turns

    return MemoryRemoval(
        store, rederive=_coordinator(store, db_path).rederive, purge_session=purge_session
    )


def _entity_by_label(db_path: str | None, name: str) -> str | None:
    """An entity id for a name typed on the command line — only when exactly one fits."""
    graph = _store(db_path).memory_graph()
    if graph.get_entity(name) is not None:
        return name
    found = [e for e in graph.store.find_entities(label=name) if not e.removed]
    return found[0].id if len(found) == 1 else None


@memory_app.command("remove")
def cmd_remove(
    kind: Annotated[str, typer.Argument(help="entity | name | session | fact")],
    target: Annotated[
        str,
        typer.Argument(help="An entity's id or name, a name, a session id, or a fact's id."),
    ],
    db_path: DbPath = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Remove something from memory — the Map, recall and the prompt together.

    Reversible: `iris memory removed` lists it and `iris memory restore <ID>` brings it
    back exactly. Removing an entity also forgets the facts that point at it.
    """
    from iris_harness.memory.removal import RemovalError, Target

    if kind == "entity":
        target = _entity_by_label(db_path, target) or target
    removal = _removal(db_path)
    try:
        wanted = Target.of({"kind": kind, "id": target})
        [effect] = removal.preview([wanted])
    except RemovalError as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc
    console.print(f"  Remove [bold]{effect['label']}[/bold]?")
    for line in effect["lines"]:
        console.print(f"    [dim]{line}[/dim]")
    if not yes and not typer.confirm("  Remove it?"):
        console.print("  [yellow]cancelled[/yellow]")
        return
    [item] = removal.remove([wanted])
    console.print(
        f"  [green]✓[/green] removed {item['label']}  "
        f"[dim](undo: iris memory restore {item['id']})[/dim]"
    )


@memory_app.command("test-sessions")
def cmd_test_sessions(
    db_path: DbPath = None,
    remove: Annotated[
        bool, typer.Option("--remove", help="Remove every listed session (restorable).")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Conversations whose id looks like a test run — review them, then remove in bulk.

    What a REAL conversation's id looks like is `test_session_review` in
    config/memory/retention.yaml; any other id is listed. `--remove` sends them all to
    `iris memory removed`, where each one can be restored.
    """
    from iris_harness.memory.removal import Target
    from iris_harness.memory.retention import (
        flag_test_sessions,
        real_session_shapes,
    )

    flagged = flag_test_sessions(_store(db_path))
    if not flagged:
        console.print("  [green]no sessions look like test runs[/green]")
        return
    table = Table(title=f"{len(flagged)} session(s) look like test runs")
    for column in ("Session", "Turns", "Last active", "Why", "Summary"):
        table.add_column(column)
    for f in flagged:
        table.add_row(
            f.session_id, str(f.turns), f.last_activity[:16], "; ".join(f.reasons), f.summary_goal
        )
    console.print(table)
    shapes = real_session_shapes()
    if shapes:
        console.print(f"  [dim]a real conversation's id is: {'; '.join(shapes)}[/dim]")
    if not remove:
        console.print("  [dim]re-run with --remove to remove them all (restorable).[/dim]")
        return
    if not yes and not typer.confirm(f"  Remove all {len(flagged)}?"):
        console.print("  [yellow]cancelled[/yellow]")
        return
    items = _removal(db_path).remove([Target("session", f.session_id) for f in flagged])
    console.print(
        f"  [green]✓[/green] removed {len(items)} session(s)  "
        "[dim](undo: iris memory removed, then iris memory restore <ID>)[/dim]"
    )


@memory_app.command("removed")
def cmd_removed(db_path: DbPath = None) -> None:
    """What was removed from memory, newest first — each restorable by its ID."""
    items = _removal(db_path).items()
    if not items:
        console.print("  [green]nothing removed[/green]")
        return
    table = Table(show_lines=False)
    table.add_column("ID", no_wrap=True)  # the id is what you copy: never cut it
    for column in ("Kind", "What", "Went with it", "When"):
        table.add_column(column)
    for item in items:
        went = "; ".join(c["text"] for c in item["cascade"] if c["text"] != item["label"])
        label = item["label"] + (" [dim](deleted)[/dim]" if item["permanent"] else "")
        table.add_row(item["id"], item["kind"], label, went or "—", item["removed_at"][:16])
    console.print(table)


@memory_app.command("restore")
def cmd_restore(
    removal_id: Annotated[str, typer.Argument(help="An ID from `iris memory removed`.")],
    db_path: DbPath = None,
) -> None:
    """Undo one removal exactly — an entity comes back with the facts it took."""
    from iris_harness.memory.removal import RemovalError

    try:
        item = _removal(db_path).restore(removal_id)
    except RemovalError as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc
    console.print(f"  [green]✓[/green] restored {item['label']}")


@memory_app.command("delete")
def cmd_delete(
    removal_ids: Annotated[list[str], typer.Argument(help="IDs from `iris memory removed`.")],
    confirm: Annotated[
        str, typer.Option("--confirm", help="Type the word `delete` to delete for good.")
    ] = "",
    db_path: DbPath = None,
) -> None:
    """Delete removed things for good. Cannot be undone; a deleted name stays hidden."""
    from iris_harness.memory.removal import RemovalError

    try:
        result = _removal(db_path).delete(removal_ids, confirm=confirm)
    except RemovalError as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc
    for removal_id in result["deleted"]:
        console.print(f"  [green]✓[/green] deleted {removal_id}")
    for refusal in result["refused"]:
        console.print(f"  [yellow]kept[/yellow] {refusal['id']}: {refusal['reason']}")
    if result["refused"]:
        raise typer.Exit(1)
