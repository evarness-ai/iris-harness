"""``iris run inspect <run_id>`` — render a halted run's audit + checkpoint state.

Closes the Phase 3 exit criterion: an operator can diagnose a halted run
without grepping SQLite. Pulls rows from the hot audit log (story 3.3),
joins the latest resume checkpoint (story 3.4), and renders either a
human-readable Rich layout or a stable JSON shape for scripting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final, Literal

from rich.console import Console
from rich.table import Table

from iris_harness.kernel.governance.audit import AuditLog, AuditRow
from iris_harness.memory.state import (
    Checkpoint,
    CheckpointNotFoundError,
    CheckpointStore,
)

OutputFormat = Literal["text", "json"]


# Severity → Rich style. Keeps the inspect colour palette in lockstep
# with the rest of the CLI (info default, warn yellow, error red,
# critical bold red).
_SEVERITY_STYLE: Final[dict[str, str]] = {
    "info": "",
    "warn": "yellow",
    "error": "red",
    "critical": "bold red",
}

# Decision → Rich style, applied to the cell text.
_DECISION_STYLE: Final[dict[str, str]] = {
    "allow": "green",
    "transform": "cyan",
    "deny": "bold red",
    "require_approval": "magenta",
}


class UnknownRunError(LookupError):
    """No audit rows AND no checkpoint exist for the requested run_id."""


@dataclass(frozen=True)
class RunSummary:
    """Header bag derived from the rows in audit_log for one run."""

    run_id: str
    agent_type: str
    start_ts: str
    last_ts: str
    total_rows: int
    final_decision: str
    final_severity: str
    final_reason: str


def load_run(
    run_id: str,
    *,
    audit_log: AuditLog,
    checkpoint_store: CheckpointStore,
    limit: int | None = None,
) -> tuple[RunSummary, tuple[AuditRow, ...], Checkpoint | None]:
    """Return (summary, rows, checkpoint) for ``run_id``.

    Raises ``UnknownRunError`` when the run has no audit rows and no
    checkpoint — the typical "typo in run_id" path.
    """
    rows = audit_log.query(run_id=run_id, limit=limit)
    try:
        checkpoint: Checkpoint | None = checkpoint_store.get_latest(run_id)
    except CheckpointNotFoundError:
        checkpoint = None

    if not rows and checkpoint is None:
        raise UnknownRunError(f"no audit rows or checkpoints for run_id={run_id!r}")

    summary = _summarise(run_id, rows, checkpoint)
    return summary, rows, checkpoint


def render_text(
    summary: RunSummary,
    rows: tuple[AuditRow, ...],
    checkpoint: Checkpoint | None,
    *,
    console: Console,
) -> None:
    """Print a Rich-formatted human-readable run report."""
    console.print()
    console.print(f"  [bold cyan]run[/bold cyan] [white]{summary.run_id}[/white]")
    console.print(
        f"    agent=[white]{summary.agent_type}[/white]  "
        f"steps=[white]{summary.total_rows}[/white]  "
        f"start=[dim]{summary.start_ts}[/dim]  "
        f"last=[dim]{summary.last_ts}[/dim]"
    )
    decision_style = _DECISION_STYLE.get(summary.final_decision, "")
    decision_cell = (
        f"[{decision_style}]{summary.final_decision}[/{decision_style}]"
        if decision_style
        else summary.final_decision
    )
    severity_style = _SEVERITY_STYLE.get(summary.final_severity, "")
    sev_cell = (
        f"[{severity_style}]{summary.final_severity}[/{severity_style}]"
        if severity_style
        else summary.final_severity
    )
    console.print(
        f"    final=[white]{decision_cell}[/white]  severity={sev_cell}  "
        f"reason=[dim]{summary.final_reason}[/dim]"
    )
    console.print()

    if rows:
        _render_trace_table(rows, console=console)
    else:
        console.print("  [dim]no audit rows[/dim]")
        console.print()

    if checkpoint is not None:
        _render_checkpoint(checkpoint, console=console)


def render_json(
    summary: RunSummary,
    rows: tuple[AuditRow, ...],
    checkpoint: Checkpoint | None,
    *,
    console: Console,
) -> None:
    """Print the stable JSON shape: ``{run, steps[], checkpoint}``."""
    payload: dict[str, Any] = {
        "run": {
            "run_id": summary.run_id,
            "agent_type": summary.agent_type,
            "start_ts": summary.start_ts,
            "last_ts": summary.last_ts,
            "total_rows": summary.total_rows,
            "final_decision": summary.final_decision,
            "final_severity": summary.final_severity,
            "final_reason": summary.final_reason,
        },
        "steps": [_row_to_json(row) for row in rows],
        "checkpoint": _checkpoint_to_json(checkpoint),
    }
    console.print(json.dumps(payload, indent=2, sort_keys=True))


def _summarise(
    run_id: str,
    rows: tuple[AuditRow, ...],
    checkpoint: Checkpoint | None,
) -> RunSummary:
    if rows:
        first = rows[0]
        last = rows[-1]
        return RunSummary(
            run_id=run_id,
            agent_type=first.agent_type,
            start_ts=first.ts,
            last_ts=last.ts,
            total_rows=len(rows),
            final_decision=last.decision,
            final_severity=last.severity,
            final_reason=last.reason,
        )
    # No audit rows but a checkpoint exists — synthesize a header from it.
    assert checkpoint is not None  # guarded by load_run
    return RunSummary(
        run_id=run_id,
        agent_type=checkpoint.agent_type,
        start_ts=checkpoint.ts,
        last_ts=checkpoint.ts,
        total_rows=0,
        final_decision=checkpoint.signal or "unknown",
        final_severity="info",
        final_reason="(no audit rows — header derived from checkpoint)",
    )


def _render_trace_table(rows: tuple[AuditRow, ...], *, console: Console) -> None:
    table = Table(title="trace", title_style="bold", show_lines=False)
    table.add_column("ts", style="dim", no_wrap=True)
    table.add_column("step", justify="right")
    table.add_column("hook")
    table.add_column("plugin")
    table.add_column("decision")
    table.add_column("severity")
    table.add_column("reason", overflow="fold")

    for row in rows:
        decision_style = _DECISION_STYLE.get(row.decision, "")
        severity_style = _SEVERITY_STYLE.get(row.severity, "")
        table.add_row(
            row.ts,
            "" if row.step_id is None else str(row.step_id),
            row.hook_point,
            row.plugin,
            (
                f"[{decision_style}]{row.decision}[/{decision_style}]"
                if decision_style
                else row.decision
            ),
            (
                f"[{severity_style}]{row.severity}[/{severity_style}]"
                if severity_style
                else row.severity
            ),
            row.reason,
        )
    console.print(table)
    console.print()


def _render_checkpoint(checkpoint: Checkpoint, *, console: Console) -> None:
    pin_marker = "[yellow]📌[/yellow] " if checkpoint.pinned else ""
    console.print(
        f"  [bold]checkpoint[/bold]  {pin_marker}"
        f"step=[white]{checkpoint.step_id}[/white]  "
        f"signal=[dim]{checkpoint.signal or '-'}[/dim]  "
        f"ts=[dim]{checkpoint.ts}[/dim]  "
        f"expires=[dim]{checkpoint.expires_at}[/dim]"
    )
    console.print()


def _row_to_json(row: AuditRow) -> dict[str, Any]:
    try:
        payload = json.loads(row.payload_json)
    except json.JSONDecodeError:
        payload = {}
    return {
        "id": row.id,
        "ts": row.ts,
        "step_id": row.step_id,
        "agent_type": row.agent_type,
        "hook_point": row.hook_point,
        "plugin": row.plugin,
        "decision": row.decision,
        "classification": row.classification,
        "tier": row.tier,
        "cost_usd": row.cost_usd,
        "severity": row.severity,
        "reason": row.reason,
        "payload": payload,
    }


def _checkpoint_to_json(checkpoint: Checkpoint | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    return {
        "run_id": checkpoint.run_id,
        "step_id": checkpoint.step_id,
        "ts": checkpoint.ts,
        "agent_type": checkpoint.agent_type,
        "signal": checkpoint.signal,
        "pinned": checkpoint.pinned,
        "expires_at": checkpoint.expires_at,
        "payload": checkpoint.payload,
    }
