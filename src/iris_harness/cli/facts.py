"""``iris facts`` — inspect and curate the learned user facts in memory.db.

Read:  `list` / `show` / `audit` / `history`. `audit` buckets every fact by the live
recall-filter thresholds (DROP / UNCERTAIN / CLEAN) so they can be validated against
real data.

Write (all reversible, all audited): `correct` overrides a value past the confidence
gate; `forget` removes a fact but keeps its value in history; `restore` undoes the
last change. Every change lands in ``user_fact_history`` — nothing is lost or
unexplained, which is the trust property that makes the memory safe to curate.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from iris_harness.cli.render import console, print_error
from iris_harness.foundation.paths import data_dir
from iris_harness.memory.coordinator import FactCoordinator
from iris_harness.memory.retriever import MemoryRetriever
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.store import MemoryStore, UserFact

facts_app = typer.Typer(
    name="facts",
    help="Inspect and curate user facts: list / show / audit / history / correct / forget / restore / retention / prune.",
    no_args_is_help=True,
)

# Shared --db-path option: defaults to the runtime's resolution (IRIS_DATA_DIR/data),
# overridable for tests or pointing at another profile's memory.db.
DbPath = Annotated[
    str | None,
    typer.Option("--db-path", help="Path to memory.db (default: $IRIS_DATA_DIR/memory.db)."),
]


def _store(db_path: str | None) -> MemoryStore:
    path = Path(db_path) if db_path else data_dir() / "memory.db"
    store = MemoryStore(db_path=path)
    store.ensure_schema()
    return store


def _coordinator(store: MemoryStore, db_path: str | None) -> FactCoordinator:
    """Build the fact write-seam over the given store + its sibling chroma index.

    Manual curation (correct/forget/restore) must fan out to the derived homes
    (index + USER.md), or a fix here leaves stale copies behind — the exact
    desync the coherence doctor exists to catch.
    """
    base = Path(db_path).parent if db_path else data_dir()
    return FactCoordinator(store, SemanticIndex(persist_dir=base / "chroma"))


def _record_user_signal(db_path: str | None, kind: str, *, subject: str, detail: str = "") -> None:
    """Best-effort user-behavior signal for a CLI correction/forget (learning.db sibling)."""
    from iris_harness.services.learning.store import LearningMetricsStore

    base = Path(db_path).parent if db_path else data_dir()
    store = LearningMetricsStore(db_path=base / "learning.db")
    store.ensure_schema()
    store.record_user_behavior_signal(kind, subject=subject, detail=detail)


def _thresholds(store: MemoryStore) -> tuple[float, float]:
    """The live retriever thresholds (honors env overrides), so audit == reality."""
    r = MemoryRetriever(store=store, index=None)
    return r.min_fact_confidence, r.uncertain_below


def _bucket(conf: float, drop_below: float, uncertain_below: float) -> str:
    if conf < drop_below:
        return "DROP"
    if conf < uncertain_below:
        return "UNCERTAIN"
    return "CLEAN"


@facts_app.command("list")
def cmd_list(
    db_path: DbPath = None,
    sort_by_confidence: Annotated[
        bool, typer.Option("--by-confidence/--by-key", help="Sort order.")
    ] = True,
) -> None:
    """List all learned facts (lowest-confidence first by default — the suspect ones)."""
    facts = _store(db_path).fetch_all_user_facts()
    if not facts:
        console.print("  [yellow]no facts yet — IRIS learns as you use it[/yellow]")
        return
    facts = sorted(facts, key=(lambda f: f.confidence) if sort_by_confidence else (lambda f: f.key))
    table = Table(title=f"User facts ({len(facts)})", show_lines=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="dim", max_width=48)
    table.add_column("Conf", justify="right")
    table.add_column("Source", style="magenta", max_width=24)
    table.add_column("Owner-confirmed", justify="right", style="dim")
    table.add_column("Last seen", style="dim")
    for f in facts:
        table.add_row(
            f.key if f.confirmed else f"[yellow]{f.key}[/yellow]",
            f.value,
            f"{f.confidence:.2f}",
            f.source,
            ("yes" if f.confirmed else "[yellow]awaiting review[/yellow]"),
            f.last_confirmed.strftime("%Y-%m-%d"),
        )
    console.print(table)
    unconfirmed = [f for f in facts if not f.confirmed]
    if unconfirmed:
        console.print(
            f"  [yellow]{len(unconfirmed)} fact(s) are not owner-confirmed and never reach a "
            f"prompt.[/yellow] Review them with [bold]iris facts review[/bold]."
        )


@facts_app.command("show")
def cmd_show(
    key: Annotated[str, typer.Argument(help="Fact key to show.")],
    db_path: DbPath = None,
) -> None:
    """Show full detail for a single fact by key."""
    fact = _store(db_path).fetch_user_fact(key)
    if fact is None:
        print_error(f"no fact with key: {key}")
        raise typer.Exit(1)
    console.print(f"  [bold cyan]{fact.key}[/bold cyan] = {fact.value}")
    console.print(f"  [dim]confidence[/dim]      {fact.confidence:.2f}")
    console.print(f"  [dim]source[/dim]          {fact.source}")
    console.print(f"  [dim]first seen[/dim]      {fact.first_seen.isoformat()}")
    console.print(f"  [dim]last confirmed[/dim]  {fact.last_confirmed.isoformat()}")
    console.print(f"  [dim]times confirmed[/dim] {fact.times_confirmed}")
    console.print(
        f"  [dim]owner-confirmed[/dim] {'yes' if fact.confirmed else 'no — not used in prompts'}"
    )


@facts_app.command("audit")
def cmd_audit(
    db_path: DbPath = None,
    show_clean: Annotated[
        bool, typer.Option("--show-clean", help="Also list the CLEAN facts.")
    ] = False,
) -> None:
    """Show what the recall filter does to each fact: DROP / UNCERTAIN / CLEAN.

    Uses the live retriever thresholds (IRIS_MEMORY_MIN_FACT_CONFIDENCE /
    IRIS_MEMORY_UNCERTAIN_BELOW). DROP = excluded from prompts; UNCERTAIN = injected
    but marked "(unconfirmed)"; CLEAN = injected as-is.
    """
    store = _store(db_path)
    facts = store.fetch_all_user_facts()
    if not facts:
        console.print("  [yellow]no facts to audit[/yellow]")
        return
    drop_below, uncertain_below = _thresholds(store)

    buckets: dict[str, list[UserFact]] = {"DROP": [], "UNCERTAIN": [], "CLEAN": []}
    for f in facts:
        buckets[_bucket(f.confidence, drop_below, uncertain_below)].append(f)

    console.print(
        f"  thresholds: drop < [red]{drop_below:.2f}[/red], "
        f"uncertain < [yellow]{uncertain_below:.2f}[/yellow], clean ≥ [green]{uncertain_below:.2f}[/green]"
    )
    console.print(
        f"  total [bold]{len(facts)}[/bold]  →  "
        f"[red]DROP {len(buckets['DROP'])}[/red] · "
        f"[yellow]UNCERTAIN {len(buckets['UNCERTAIN'])}[/yellow] · "
        f"[green]CLEAN {len(buckets['CLEAN'])}[/green]"
    )

    styles = {"DROP": "red", "UNCERTAIN": "yellow", "CLEAN": "green"}
    order = ["DROP", "UNCERTAIN"] + (["CLEAN"] if show_clean else [])
    for label in order:
        rows = sorted(buckets[label], key=lambda f: f.confidence)
        if not rows:
            continue
        table = Table(title=f"[{styles[label]}]{label}[/{styles[label]}] ({len(rows)})")
        table.add_column("Conf", justify="right")
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="dim", max_width=56)
        table.add_column("Source", style="magenta", max_width=24)
        for f in rows:
            table.add_row(f"{f.confidence:.2f}", f.key, f.value, f.source)
        console.print(table)

    if buckets["DROP"]:
        console.print(
            "  [dim]tip: a true fact stuck in DROP means it was captured weakly — "
            "restate it in chat to reconfirm, or lower the threshold via "
            "IRIS_MEMORY_MIN_FACT_CONFIDENCE.[/dim]"
        )


@facts_app.command("history")
def cmd_history(
    key: Annotated[str, typer.Argument(help="Fact key to show history for.")],
    db_path: DbPath = None,
) -> None:
    """Show the change history for a fact (most recent first)."""
    entries = _store(db_path).fetch_fact_history(key)
    if not entries:
        console.print(f"  [yellow]no history for: {key}[/yellow]")
        return
    table = Table(title=f"History — {key} ({len(entries)})", show_lines=False)
    table.add_column("When", style="dim")
    table.add_column("Change", style="magenta")
    table.add_column("Old", style="dim")
    table.add_column("New", style="cyan")
    table.add_column("Source", style="dim")
    for e in entries:
        table.add_row(
            e.changed_at.strftime("%Y-%m-%d %H:%M"),
            e.reason,
            "—" if e.old_value is None else e.old_value,
            "—" if e.new_value is None else e.new_value,
            e.source,
        )
    console.print(table)


@facts_app.command("correct")
def cmd_correct(
    key: Annotated[str, typer.Argument(help="Fact key to set.")],
    value: Annotated[str, typer.Argument(help="The correct value.")],
    db_path: DbPath = None,
) -> None:
    """Set a fact's value explicitly (overrides the confidence gate; recorded + reversible)."""
    from iris_harness.memory.fact_statements import FactKeyError

    store = _store(db_path)
    prior = store.fetch_user_fact(key)
    try:
        _coordinator(store, db_path).correct(key, value)
    except FactKeyError as exc:  # the vocabulary is closed (ADR-0115 decision 6)
        print_error(str(exc))
        raise typer.Exit(1) from exc
    _record_user_signal(db_path, "fact_corrected", subject=key, detail=value)
    if prior is not None:
        console.print(f"  [green]✓[/green] {key}: [dim]{prior.value}[/dim] → [cyan]{value}[/cyan]")
    else:
        console.print(f"  [green]✓[/green] {key} = [cyan]{value}[/cyan] (new)")
    console.print("  [dim]undo with `iris facts restore " + key + "`[/dim]")


@facts_app.command("forget")
def cmd_forget(
    key: Annotated[str, typer.Argument(help="Fact key to forget.")],
    db_path: DbPath = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Forget a fact (removed from recall, but kept in history and restorable)."""
    store = _store(db_path)
    fact = store.fetch_user_fact(key)
    if fact is None:
        print_error(f"no fact with key: {key}")
        raise typer.Exit(1)
    if not yes and not typer.confirm(f"Forget {key} = {fact.value!r}?"):
        console.print("  [yellow]cancelled[/yellow]")
        return
    _coordinator(store, db_path).forget(key)
    _record_user_signal(db_path, "fact_forgotten", subject=key)
    console.print(
        f"  [green]✓[/green] forgot [cyan]{key}[/cyan]  "
        f"[dim](restore with `iris facts restore {key}`)[/dim]"
    )


@facts_app.command("restore")
def cmd_restore(
    key: Annotated[str, typer.Argument(help="Fact key to restore.")],
    db_path: DbPath = None,
) -> None:
    """Undo the last change to a fact, re-activating its prior value."""
    from iris_harness.memory.fact_statements import FactKeyError

    try:
        restored = _coordinator(_store(db_path), db_path).restore(key)
    except FactKeyError as exc:
        print_error(f"{exc} — an archived fact comes back through the migration triage file")
        raise typer.Exit(1) from exc
    if restored is None:
        print_error(f"nothing to restore for: {key}")
        raise typer.Exit(1)
    console.print(f"  [green]✓[/green] restored [cyan]{key}[/cyan] = {restored}")


def _retention_days(override: int | None) -> int:
    if override is not None:
        return override
    try:
        return int(os.environ.get("IRIS_FACT_HISTORY_RETENTION_DAYS", "180"))
    except ValueError:
        return 180


@facts_app.command("retention")
def cmd_retention(
    db_path: DbPath = None,
    older_than: Annotated[
        int | None, typer.Option("--older-than", help="Retention window in days (default 180).")
    ] = None,
) -> None:
    """Review history entries past the retention window (default 180d) — never auto-deleted.

    This is the human review queue: it only SHOWS what is old enough to consider pruning.
    Choose entries yourself and remove them with `iris facts prune --id <id> ...`.
    """
    store = _store(db_path)
    days = _retention_days(older_than)
    total = store.count_fact_history()
    candidates = store.fetch_history_retention_candidates(older_than_days=days)
    console.print(
        f"  history: [bold]{total}[/bold] entries · "
        f"[yellow]{len(candidates)}[/yellow] older than {days}d (review candidates)"
    )
    if not candidates:
        console.print("  [green]nothing past the retention window[/green]")
        return
    now = datetime.now(UTC)
    table = Table(title=f"Retention review — older than {days}d", show_lines=False)
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Age", justify="right", style="dim")
    table.add_column("Key", style="cyan")
    table.add_column("Change", style="magenta")
    table.add_column("Old", style="dim")
    table.add_column("New", style="dim")
    for e in candidates:
        age = (now - e.changed_at).days
        table.add_row(
            str(e.id),
            f"{age}d",
            e.key,
            e.reason,
            "—" if e.old_value is None else e.old_value,
            "—" if e.new_value is None else e.new_value,
        )
    console.print(table)
    console.print(
        "  [dim]prune chosen rows: `iris facts prune --id <id> --id <id>` "
        "(or `--older-than <days>` to prune all past a window). You verify; nothing is "
        "removed without your confirmation.[/dim]"
    )


@facts_app.command("prune")
def cmd_prune(
    db_path: DbPath = None,
    ids: Annotated[
        list[str] | None, typer.Option("--id", help="Specific history entry id(s) to prune.")
    ] = None,
    older_than: Annotated[
        int | None,
        typer.Option("--older-than", help="Prune ALL entries older than this many days."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Remove reviewed history entries (terminal). Select by --id, or bulk by --older-than.

    Pruning is destructive and not reversible — it is the human-actioned outcome of the
    retention review. You must pass --id or --older-than explicitly; there is no default.
    """
    store = _store(db_path)
    if ids:
        target = sorted(set(ids))
    elif older_than is not None:
        target = [
            e.id for e in store.fetch_history_retention_candidates(older_than_days=older_than)
        ]
    else:
        print_error("specify --id <id> (chosen rows) or --older-than <days> (bulk).")
        raise typer.Exit(2)

    if not target:
        console.print("  [green]nothing to prune[/green]")
        return
    if not yes and not typer.confirm(f"Permanently prune {len(target)} history entr(y/ies)?"):
        console.print("  [yellow]cancelled[/yellow]")
        return
    removed = store.prune_history_entries(target)
    console.print(f"  [green]✓[/green] pruned [bold]{removed}[/bold] history entr(y/ies)")


@facts_app.command("contradictions")
def cmd_contradictions(
    db_path: DbPath = None,
    show_all: Annotated[
        bool, typer.Option("--all", help="Include already-acknowledged conflicts.")
    ] = False,
) -> None:
    """Review detected same-key value conflicts (e.g. 'city' captured as both X and Y).

    Shows whether each conflict superseded the old value or was blocked. Resolve by
    correcting the fact (`iris facts correct`) and/or `iris facts ack <id>`.
    """
    rows = _store(db_path).fetch_contradictions(include_acknowledged=show_all)
    if not rows:
        console.print("  [green]no contradictions to review[/green]")
        return
    table = Table(title=f"Contradictions ({len(rows)})", show_lines=False)
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Key", style="cyan")
    table.add_column("Stored", style="dim")
    table.add_column("Conflicting", style="yellow")
    table.add_column("Outcome", style="magenta")
    table.add_column("Seen", justify="right", style="dim")
    table.add_column("Last", style="dim")
    for c in rows:
        table.add_row(
            str(c.id),
            c.key,
            (
                f"{c.stored_value} ({c.stored_confidence:.2f})"
                if c.stored_confidence
                else c.stored_value
            ),
            (
                f"{c.incoming_value} ({c.incoming_confidence:.2f})"
                if c.incoming_confidence
                else c.incoming_value
            ),
            c.resolution,
            f"{c.seen_count}×" if c.seen_count > 1 else "1",
            c.detected_at.strftime("%Y-%m-%d"),
        )
    console.print(table)
    console.print(
        "  [dim]fix with `iris facts correct <key> <value>`; mark reviewed with "
        "`iris facts ack <id>`.[/dim]"
    )


@facts_app.command("ack")
def cmd_ack(
    ids: Annotated[list[str], typer.Argument(help="Contradiction id(s) to acknowledge.")],
    db_path: DbPath = None,
) -> None:
    """Mark contradiction(s) reviewed — clears them from the default queue."""
    n = _store(db_path).acknowledge_contradictions(ids)
    console.print(f"  [green]✓[/green] acknowledged [bold]{n}[/bold] contradiction(s)")


def _live_thresholds_envonly() -> tuple[float, float]:
    def f(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    return f("IRIS_MEMORY_MIN_FACT_CONFIDENCE", 0.35), f("IRIS_MEMORY_UNCERTAIN_BELOW", 0.6)


@facts_app.command("gate-eval")
def cmd_gate_eval(
    corpus_path: Annotated[
        str | None,
        typer.Option("--corpus", help="JSON list of labelled candidates; default = built-in."),
    ] = None,
) -> None:
    """Measure the garbage-in / garbage-out guarantees on LABELLED data.

    Unlike `audit` (which buckets your real facts by the live thresholds), this scores
    the capture gates' and recall filter's precision/recall against a labelled corpus —
    false-admits (junk that passed), false-recalls (junk that reached the prompt),
    over-blocks (real facts the gates rejected) — and sweeps the drop threshold to
    suggest a tuned value. Defaults to a built-in synthetic corpus; pass --corpus to
    audit your own labelled JSON (`[{key,value,message,confidence,label}]`).
    """
    from iris_harness.memory.gate_audit import FactCandidate, run_full_audit
    from iris_harness.memory.gate_audit_corpus import DEFAULT_CORPUS

    if corpus_path:
        try:
            raw = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
            corpus = [FactCandidate(**r) for r in raw]
        except Exception as exc:  # noqa: BLE001
            print_error(f"could not load corpus {corpus_path}: {exc}")
            raise typer.Exit(1) from None
    else:
        corpus = list(DEFAULT_CORPUS)

    min_conf, uncertain = _live_thresholds_envonly()
    a = run_full_audit(corpus, min_fact_confidence=min_conf, uncertain_below=uncertain)
    g, r, s = a.gates, a.recall, a.sweep

    console.print(f"  [bold]corpus[/bold] {g.total} candidates ({g.keep} keep · {g.junk} junk)\n")
    console.print(
        "  [dim]note: an admitted candidate is PROPOSED for review, not recalled — reaching a "
        "prompt also needs owner confirmation, so false-recall here is an upper bound.[/dim]\n"
    )
    console.print(
        f"  [bold]capture gates[/bold] (garbage IN)  precision [cyan]{g.precision:.0%}[/cyan] · "
        f"recall [cyan]{g.recall:.0%}[/cyan]"
    )
    console.print(
        f"    false-admit [red]{g.false_admit}[/red] ({g.false_admit_rate:.0%} of junk slipped) · "
        f"false-reject [yellow]{g.false_reject}[/yellow] (real facts over-blocked)"
    )
    if g.by_gate:
        console.print(
            f"    blocks by gate: {', '.join(f'{k}:{v}' for k, v in sorted(g.by_gate.items()))}"
        )
    if g.false_reject_by_gate:
        console.print(
            f"    [yellow]over-blocks by gate: "
            f"{', '.join(f'{k}:{v}' for k, v in sorted(g.false_reject_by_gate.items()))}[/yellow]"
        )
    console.print(
        f"\n  [bold]recall filter[/bold] (garbage OUT) @ drop<{r.min_fact_confidence} "
        f"uncertain<{r.uncertain_below}  precision [cyan]{r.precision:.0%}[/cyan]"
    )
    console.print(
        f"    false-recall [red]{r.false_recall}[/red] (junk reached the prompt) · "
        f"false-drop [yellow]{r.false_drop}[/yellow] (real facts dropped) · "
        f"{r.dropped} dropped, {r.uncertain_keep + r.uncertain_junk} marked uncertain"
    )
    arrow = "=" if s.suggested_min_confidence == s.current_min_confidence else "->"
    console.print(
        f"\n  [bold]threshold sweep[/bold]  drop {s.current_min_confidence} {arrow} "
        f"[cyan]{s.suggested_min_confidence}[/cyan] (best balanced accuracy)"
    )
    console.print(
        "  [dim]suggestion is evidence on THIS corpus — validate on your real labelled data "
        "before changing IRIS_MEMORY_MIN_FACT_CONFIDENCE.[/dim]"
    )


# ----------------------------------------------------------------------
# Review queue — nothing reaches a prompt until the owner says yes
# ----------------------------------------------------------------------


@facts_app.command("review")
def cmd_review(
    db_path: DbPath = None,
    limit: Annotated[int, typer.Option("--limit", help="Max rows to show.")] = 50,
) -> None:
    """Show what is waiting for review: every proposed fact, newest first.

    Since memris plan PR 2c a fact stored before review existed IS a proposal, so the
    queue is one list; approve or reject each by its ID. A fact about someone one hop
    from you ("my wife Petra works at Infosys") is listed with whom it is about.
    """
    store = _store(db_path)
    proposals = store.fetch_fact_proposals(status="pending", limit=limit)

    if not proposals:
        console.print("  [green]nothing waiting — every stored fact is owner-confirmed[/green]")
        return

    table = Table(title=f"Proposed facts ({len(proposals)})", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("About", max_width=20)
    table.add_column("Key", style="cyan")
    table.add_column("Proposed value", max_width=40)
    table.add_column("Currently", style="dim", max_width=28)
    table.add_column("Conf", justify="right")
    table.add_column("Evidence", style="dim", max_width=40)
    for proposal in proposals:
        table.add_row(
            proposal.id,
            proposal.subject or "you",
            proposal.key,
            proposal.value,
            proposal.current_value or "—",
            f"{proposal.confidence:.2f}",
            " ".join(proposal.evidence.split())[:80],
        )
    console.print(table)
    console.print(
        "  [dim]approve with[/dim] iris facts approve <ID>   "
        "[dim]reject with[/dim] iris facts reject <ID>"
    )


@facts_app.command("approve")
def cmd_approve(
    proposal_id: Annotated[str, typer.Argument(help="Proposal ID from `iris facts review`.")],
    db_path: DbPath = None,
) -> None:
    """Approve a proposed fact — it becomes confirmed and starts reaching prompts."""
    store = _store(db_path)
    proposal = store.fetch_fact_proposal(proposal_id)
    fact = _coordinator(store, db_path).approve_proposal(proposal_id)
    if fact is None:
        print_error(f"no pending proposal with id {proposal_id}")
        raise typer.Exit(1)
    about = f" (about {proposal.subject})" if proposal and proposal.subject else ""
    console.print(f"  [green]confirmed[/green] {fact.key} = {fact.value}{about}")


@facts_app.command("reject")
def cmd_reject(
    proposal_id: Annotated[str, typer.Argument(help="Proposal ID from `iris facts review`.")],
    db_path: DbPath = None,
) -> None:
    """Reject a proposed fact. Nothing is written to the fact store."""
    if not _coordinator(_store(db_path), db_path).reject_proposal(proposal_id):
        print_error(f"no pending proposal with id {proposal_id}")
        raise typer.Exit(1)
    console.print(f"  [green]rejected[/green] proposal {proposal_id}")


@facts_app.command("confirm")
def cmd_confirm(
    key: Annotated[str, typer.Argument(help="Fact key to confirm.")],
    db_path: DbPath = None,
) -> None:
    """Confirm a fact already in the store (this is how the legacy rows are cleared)."""
    fact = _coordinator(_store(db_path), db_path).confirm(key)
    if fact is None:
        print_error(f"no fact with key: {key}")
        raise typer.Exit(1)
    console.print(f"  [green]confirmed[/green] {fact.key} = {fact.value}")


@facts_app.command("expire-proposals")
def cmd_expire_proposals(
    db_path: DbPath = None,
    older_than_days: Annotated[
        int, typer.Option("--older-than-days", help="Age at which a proposal expires.")
    ] = 30,
) -> None:
    """Expire proposals nobody reviewed, so the queue does not grow forever."""
    expired = _store(db_path).expire_fact_proposals(older_than_days=older_than_days)
    console.print(f"  expired [bold]{expired}[/bold] proposal(s) older than {older_than_days}d")


__all__ = ["facts_app"]


@facts_app.command("lessons")
def cmd_lessons(db_path: DbPath = None, limit: int = 50) -> None:
    """Lessons waiting for review — approving one writes a behavior IRIS will use."""
    proposals = _store(db_path).fetch_lesson_proposals(status="pending", limit=limit)
    if not proposals:
        console.print("  [green]no lessons waiting[/green]")
        return
    table = Table(title=f"Proposed lessons ({len(proposals)})")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("When", max_width=34)
    table.add_column("Do", max_width=44)
    table.add_column("From", style="magenta", max_width=16)
    for p in proposals:
        table.add_row(str(p.id), p.trigger, p.lesson, p.source)
    console.print(table)
    console.print(
        "  [dim]approve with[/dim] iris facts approve-lesson <ID>   "
        "[dim]reject with[/dim] iris facts reject-lesson <ID>"
    )


@facts_app.command("approve-lesson")
def cmd_approve_lesson(
    proposal_id: Annotated[int, typer.Argument(help="Lesson ID from `iris facts lessons`.")],
    db_path: DbPath = None,
) -> None:
    """Approve a lesson: it becomes a behavior, matchable from the next turn."""
    from iris_harness.memory.lessons import LessonCurator

    name = LessonCurator(_store(db_path)).approve(proposal_id)
    if name is None:
        print_error(f"no pending lesson with id {proposal_id}")
        raise typer.Exit(1)
    console.print(f"  [green]approved[/green] — wrote behavior '{name}'")


@facts_app.command("reject-lesson")
def cmd_reject_lesson(
    proposal_id: Annotated[int, typer.Argument(help="Lesson ID from `iris facts lessons`.")],
    db_path: DbPath = None,
) -> None:
    """Reject a lesson. Nothing is written."""
    from iris_harness.memory.lessons import LessonCurator

    if not LessonCurator(_store(db_path)).reject(proposal_id):
        print_error(f"no pending lesson with id {proposal_id}")
        raise typer.Exit(1)
    console.print(f"  [green]rejected[/green] lesson {proposal_id}")
