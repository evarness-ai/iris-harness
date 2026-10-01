"""``iris learning`` — one front door to the self-learning loop.

Unifies what used to be scattered across separate commands and API-only surfaces:
review queues (``behaviors`` / ``intentions`` / ``signals`` mounted here), a one-glance
``status``, the measured ``intelligence`` report + analyst ``recommendations`` (previously
API / agent-tool only), and a dry-run ``preview`` of the miners on your real data.

The review sub-groups are also still available top-level (``iris behaviors`` …) for muscle
memory; this group is the canonical home. Read-only except the mounted approve/reject
actions; the measurement surfaces never mutate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import typer

from iris_harness.cli.api_client import harness_api_client
from iris_harness.cli.behaviors import behaviors_app
from iris_harness.cli.intentions import intentions_app
from iris_harness.cli.render import console, print_error
from iris_harness.cli.signals import signals_app
from iris_harness.foundation.paths import data_dir
from iris_harness.services.learning.store import LearningMetricsStore

learning_app = typer.Typer(
    name="learning",
    help="Self-learning loop: status / intelligence / recommendations / preview + review queues.",
    no_args_is_help=True,
)

# Mount the HITL review queues under one roof (also available top-level).
learning_app.add_typer(behaviors_app, name="behaviors")
learning_app.add_typer(intentions_app, name="intentions")
learning_app.add_typer(signals_app, name="signals")

DbPath = Annotated[
    str | None,
    typer.Option("--db-path", help="Path to learning.db (default: $IRIS_DATA_DIR/learning.db)."),
]


def _store(db_path: str | None) -> LearningMetricsStore:
    path = Path(db_path) if db_path else data_dir() / "learning.db"
    store = LearningMetricsStore(db_path=path)
    store.ensure_schema()
    return store


@learning_app.command("status")
def cmd_status(db_path: DbPath = None) -> None:
    """One-glance health of the learning loop: proposals, signals, intelligence, experiments."""
    from iris_harness.services.learning.intelligence import build_intelligence
    from iris_harness.services.learning.proposal_quality import (
        build_proposal_quality,
    )

    store = _store(db_path)
    pq = build_proposal_quality(store)
    console.print("  [bold]proposals[/bold] (digital-twin)")
    for q in (pq.behaviors, pq.intentions):
        console.print(
            f"    {q.subsystem:<10} awaiting [yellow]{q.awaiting}[/yellow]  "
            f"accepted [green]{q.accepted}[/green]  rejected [red]{q.rejected}[/red]  "
            f"acceptance [cyan]{q.acceptance_rate:.0%}[/cyan] of {q.reviewed}"
        )

    summary = store.user_behavior_summary()
    if summary:
        kinds = ", ".join(f"{k}:{v}" for k, v in sorted(summary.items()))
        console.print(f"  [bold]signals[/bold]   {kinds}")

    report = build_intelligence(store)
    acc = report.accuracy
    prec = f"{acc.escalation_precision:.0%}" if acc.escalation_precision is not None else "n/a"
    console.print(
        f"  [bold]intel[/bold]     escalation precision [cyan]{prec}[/cyan]  "
        f"signals {acc.signals_recorded_total} ({acc.signals_dropped_total} dropped)  "
        f"outcome cells {len(report.matrix)}"
    )
    if report.experiments:
        exp = ", ".join(f"{k}:{v}" for k, v in sorted(report.experiments.items()))
        console.print(f"  [bold]experiments[/bold] {exp}")

    analysis = store.latest_analysis()
    if analysis:
        recs = analysis.get("recommendations") or []
        console.print(
            f"  [bold]analyst[/bold]   {len(recs)} recommendation(s) — "
            "`iris learning recommendations`"
        )
    else:
        console.print(
            "  [bold]analyst[/bold]   no analysis yet "
            "[dim](enable IRIS_LEARNING_ANALYST; runs on a 12h tick)[/dim]"
        )


@learning_app.command("intelligence")
def cmd_intelligence(db_path: DbPath = None) -> None:
    """The measured learning report — escalation accuracy + the per-(intent, tier) matrix."""
    from iris_harness.services.learning.intelligence import build_intelligence

    report = build_intelligence(_store(db_path))
    acc = report.accuracy
    prec = f"{acc.escalation_precision:.0%}" if acc.escalation_precision is not None else "n/a"
    console.print(
        f"  window {report.window_hours:.0f}h · escalation precision [cyan]{prec}[/cyan] "
        f"· {acc.signals_recorded_total} signals ({acc.drop_rate:.0%} dropped)"
    )
    if not report.matrix:
        console.print("  [dim]no per-(intent, tier) outcomes yet — use IRIS a while[/dim]")
        return
    from rich.table import Table

    table = Table(title="Outcomes by intent x tier", show_lines=False)
    for col in ("intent", "tier", "n", "completion", "correction", "reuse", "avg tok"):
        table.add_column(col)
    for c in report.matrix:
        corr = f"{c.correction_rate:.0%}" if c.correction_rate is not None else "-"
        avg = f"{c.avg_tokens:.0f}" if c.avg_tokens is not None else "-"
        table.add_row(
            c.intent,
            c.tier,
            str(c.samples),
            f"{c.completion_rate:.0%}",
            corr,
            str(c.reuse_count),
            avg,
        )
    console.print(table)


@learning_app.command("recommendations")
def cmd_recommendations(db_path: DbPath = None) -> None:
    """The agentic analyst's latest recommendations (advisory; promote via the API)."""
    analysis = _store(db_path).latest_analysis()
    if not analysis:
        console.print(
            "  [green]no analysis yet[/green] "
            "[dim](enable IRIS_LEARNING_ANALYST; runs on a 12h tick)[/dim]"
        )
        return
    if analysis.get("summary"):
        console.print(f"  [italic]{analysis['summary']}[/italic]\n")
    recs = analysis.get("recommendations") or []
    for i, r in enumerate(recs):
        console.print(
            f"  [bold cyan]{i}[/bold cyan] [{r.get('confidence', '?')}] {r.get('title', '(untitled)')}"
        )
        if r.get("finding"):
            console.print(f"      [dim]finding:[/dim] {r['finding']}")
        if r.get("action"):
            console.print(f"      [dim]action:[/dim]  {r['action']}")
    if recs:
        console.print(
            "\n  [dim]promote one: POST /learning/recommendations/<i>/promote "
            "(or the web Learning screen)[/dim]"
        )


@learning_app.command("preview")
def cmd_preview(
    api_url: Annotated[str | None, typer.Option("--api-url", help="IRIS API base URL.")] = None,
) -> None:
    """Dry-run the behavior + intention miners on your real data (persists nothing).

    Needs the live stack (the miners run a local LLM). Use this to eyeball proposal
    quality before enabling IRIS_BEHAVIOR_MINER / IRIS_INTENTION_ROLLUP.
    """
    base = api_url or os.environ.get("IRIS_API_URL") or "http://localhost:8003"

    def _fetch(path: str) -> dict[str, Any]:
        with harness_api_client(purpose="learning.preview", timeout=30.0) as client:
            resp = client.get(f"{base}{path}")
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    try:
        behaviors = _fetch("/learning/behaviors/preview")
        intentions = _fetch("/learning/intentions/preview")
    except Exception as exc:  # noqa: BLE001 — friendly hint, not a stack trace
        print_error(f"could not reach IRIS API at {base} ({exc}). Is the stack up?")
        raise typer.Exit(1) from None

    pats = behaviors.get("patterns") or []
    console.print(f"  [bold]would-be behaviors[/bold] ({len(pats)})")
    for p in pats:
        console.print(f"    [{p.get('confidence', '?')}] {p.get('text', '')}")
    if not pats:
        console.print("    [dim]none[/dim]")
    ints = intentions.get("intentions") or []
    console.print(f"  [bold]would-be intentions[/bold] ({len(ints)})")
    for it in ints:
        console.print(f"    {it.get('title', '')}")
    if not ints:
        console.print("    [dim]none[/dim]")
    console.print(
        "  [dim]nothing was persisted. Enable the miners to queue these for review.[/dim]"
    )


__all__ = ["learning_app"]
