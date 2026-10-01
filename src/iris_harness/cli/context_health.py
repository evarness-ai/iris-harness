"""``iris context-health`` — one view of how the harness is managing its context.

Reads the live runtime via ``GET /context-health`` (ADR-0081): the conversation-window
fill + last compaction, the in-loop transcript/memory budget split + latest eviction, and
the surface-feedback suppression roll-up. Bring the stack up first (``scripts/start_iris.sh``).
"""

from __future__ import annotations

import os
from typing import Annotated, Any

import typer

from iris_harness.cli.api_client import harness_api_client
from iris_harness.cli.render import console, print_error

context_health_app = typer.Typer(
    name="context-health",
    help="Show how the harness is managing its context (window, budgets, suppression).",
    invoke_without_command=True,
    no_args_is_help=False,
)


def _bar(pct: float, width: int = 20) -> str:
    filled = max(0, min(width, round(pct * width)))
    return "█" * filled + "░" * (width - filled)


@context_health_app.callback(invoke_without_command=True)
def show(
    ctx: typer.Context,
    session_id: Annotated[
        str, typer.Option("--session-id", help="Conversation to scope the window fill to.")
    ] = "default",
    api_url: Annotated[str | None, typer.Option("--api-url", help="IRIS API base URL.")] = None,
) -> None:
    """Render the context-health snapshot from the running IRIS API."""
    if ctx.invoked_subcommand is not None:
        return
    base = api_url or os.environ.get("IRIS_API_URL") or "http://localhost:8003"
    try:
        with harness_api_client(purpose="context-health", timeout=5.0) as client:
            resp = client.get(f"{base}/context-health", params={"session_id": session_id})
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
    except Exception as exc:  # noqa: BLE001 — surface a friendly hint, not a stack trace
        print_error(f"could not reach IRIS API at {base} ({exc}). Is the stack up?")
        raise typer.Exit(1) from None
    if not data.get("available"):
        console.print("  [yellow]context health unavailable (runtime not ready)[/yellow]")
        return

    w = data["window"]
    fill = float(w["fill_pct"])
    near = " [red](near full — next turn likely compacts)[/red]" if w["near_full"] else ""
    console.print(
        f"  [bold]window[/bold]  {_bar(fill)} {fill:.0%}  "
        f"[dim]{w['current_tokens']}/{w['budget_tokens']} tok, "
        f"compact at {w['compaction_ratio']:.0%}[/dim]{near}"
    )
    lc = w.get("last_compaction")
    if lc:
        console.print(
            f"          [dim]last compaction: {lc['trigger']} trigger, archived "
            f"{lc['archived_count']} turns, {lc['tokens_before']}->{lc['tokens_after']} tok[/dim]"
        )
    b = data["budgets"]
    ev = b.get("last_transcript_evicted")
    ev_txt = f", last evicted {ev} tok" if ev else ""
    console.print(
        f"  [bold]budgets[/bold] transcript {b['transcript_budget']} tok · "
        f"memory {b['memory_budget']} tok{ev_txt}"
    )
    s = data["suppression"]
    by = ", ".join(f"{k}:{v}" for k, v in sorted(s.get("by_subsystem", {}).items())) or "none"
    console.print(
        f"  [bold]suppress[/bold] {s['active_suppressions']} active "
        f"[dim]({by}); {s['total_feedback']} feedback total[/dim]"
    )


__all__ = ["context_health_app"]
