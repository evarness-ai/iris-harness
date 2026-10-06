"""``iris governance`` -- read-only views of the governance ledger.

``pii-shadow``: what the owner-PII guards would have done (ADR-0125 PR 4), counted by
hook point x guard x kind x action from the shadow hook's audit rows. Never a literal:
the rows hold none. It is what the owner reads to choose the order PR 5 enforces kinds
in. Reads the local ledger directly (``foundation.paths.audit_db_path``, which honours
``IRIS_GOVERNANCE_AUDIT_DB_PATH``), as ``iris run inspect`` does, so it works with the
stack down; ``GET /governance/pii-shadow`` renders the same summary.

``audit``: the newest ledger rows as ``GET /governance/audit`` returns them
(``kernel/governance/audit/view.py``): who called (``--caller mcp:`` for every MCP
client), what ran (the ``by`` column: the tool's owning plugin, the capability's provider
or the model a model call was bound for), whether a deterministic handler answered,
local or cloud, the reason with email addresses masked. Never the payload.

``proof-bundle export|verify|check``: the R14 proof bundle -- the ledger's evidence for
the three onboarding invariants as one versioned JSON document, its offline check, and
both at once over a recent window without a file (what the web Governance screen shows,
``GET /governance/proof-bundle/check``) (``kernel/governance/audit/proof_bundle.py``;
docs/reference/proof-bundle.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from iris_harness.cli.render import console, print_error

governance_app = typer.Typer(
    name="governance",
    help="Read-only views of the governance ledger.",
    no_args_is_help=True,
)


@governance_app.callback()
def _group() -> None:
    """Read-only views of the governance ledger."""


@governance_app.command("pii-shadow")
def pii_shadow(
    days: Annotated[float, typer.Option("--days", help="Window, in days back from now.")] = 7.0,
    as_json: Annotated[bool, typer.Option("--json", help="Print the summary as JSON.")] = False,
) -> None:
    """What the owner-PII guards would have done (IRIS_GOVERNANCE_OWNER_PII=shadow)."""
    from iris_harness.foundation.paths import audit_db_path
    from iris_harness.kernel.governance.audit import AuditLog
    from iris_harness.kernel.governance.plugins.owner_pii_shadow import (
        ENV_FLAG,
        owner_pii_mode_from_env,
        owner_pii_shadow_summary,
    )

    if not 0 < days <= 366:
        print_error("--days must be in (0, 366]")
        raise typer.Exit(2)
    mode = owner_pii_mode_from_env().mode
    summary = owner_pii_shadow_summary(AuditLog(db_path=audit_db_path()), days=days)
    if as_json:
        console.print_json(json.dumps({"mode": mode, **summary.as_dict()}))
        return
    console.print(f"[bold]Owner-PII shadow[/bold] since {summary.since} ({summary.rows} rows)")
    if mode == "off":
        # An empty table from a kernel that is not observing reads like a quiet week.
        console.print(f"[yellow]{ENV_FLAG} is off in this environment: nothing is recorded.")
    if summary.checked:
        checked = ", ".join(f"{g} {n}" for g, n in sorted(summary.checked.items()))
        console.print(f"calls read per guard: {checked}")
    if summary.unobserved:
        missed = ", ".join(f"{k} {n}" for k, n in sorted(summary.unobserved.items()))
        console.print(f"[yellow]calls not observed: {missed}")
    if not summary.cells:
        console.print("no owner PII observed in the window")
        return
    table = Table(show_lines=False)
    for column in ("hook point", "guard", "kind", "would", "occurrences", "calls", "distinct"):
        table.add_column(column)
    for cell in summary.cells:
        kind = cell.kind + (" (first name alone)" if cell.first_name_alone else "")
        action = cell.action + (" (log-only destination)" if cell.log_only_destination else "")
        table.add_row(
            cell.hook_point,
            cell.guard,
            kind,
            action,
            str(cell.occurrences),
            str(cell.calls),
            "-" if cell.distinct is None else str(cell.distinct),
        )
    console.print(table)


@governance_app.command("audit")
def audit(
    limit: Annotated[int, typer.Option("--limit", help="How many rows, newest first.")] = 50,
    decision: Annotated[
        str | None, typer.Option("--decision", help="allow | deny | transform | ...")
    ] = None,
    caller: Annotated[
        str | None,
        typer.Option("--caller", help="Who made the call; ending in ':' is a namespace (mcp:)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the view as JSON.")] = False,
) -> None:
    """The newest decisions: hook, decision, caller, by (tool owner or model), label, where."""
    from iris_harness.foundation.paths import audit_db_path
    from iris_harness.kernel.governance.audit import AuditLog
    from iris_harness.kernel.governance.audit.view import audit_view

    view = audit_view(
        AuditLog(db_path=audit_db_path()), decision=decision, caller=caller, limit=limit
    )
    if as_json:
        console.print_json(json.dumps(view))
        return
    console.print(f"[bold]Governance ledger[/bold] {view['count']} of {view['total']} rows")
    if not view["entries"]:
        console.print("no rows match")
        return
    table = Table(show_lines=False)
    for column in ("time", "hook", "plugin", "decision", "caller", "by", "label", "where", "how"):
        table.add_column(column)
    for e in view["entries"]:
        how = f"deterministic:{e.get('handler') or '?'}" if e.get("deterministic") else ""
        table.add_row(
            e["ts"][:19],
            e["hook_point"],
            e["plugin"],
            e["decision"],
            e.get("caller") or "",
            # What ran: the tool's owning plugin, the capability's provider, or the model.
            e.get("tool_plugin") or e.get("capability_provider") or e.get("model") or "",
            e["classification"] or "",
            e["locality"] or "",
            how or e.get("tool_name") or "",
        )
    console.print(table)


proof_bundle_app = typer.Typer(
    name="proof-bundle",
    help="Export / verify the R14 proof bundle (docs/reference/proof-bundle.md).",
    no_args_is_help=True,
)
governance_app.add_typer(proof_bundle_app, name="proof-bundle")


@proof_bundle_app.command("export")
def proof_bundle_export(
    out: Annotated[Path, typer.Option("--out", help="Where to write the bundle (JSON).")],
    since: Annotated[
        str | None, typer.Option("--since", help="Only rows at or after this ISO time.")
    ] = None,
    until: Annotated[
        str | None, typer.Option("--until", help="Only rows at or before this ISO time.")
    ] = None,
    run: Annotated[
        list[str] | None, typer.Option("--run", help="Only this run id (repeatable).")
    ] = None,
    subject: Annotated[str, typer.Option("--subject", help="What the bundle covers.")] = "",
    db: Annotated[
        Path | None, typer.Option("--db", help="Audit ledger (default: this home's).")
    ] = None,
    session_logs: Annotated[
        Path | None,
        typer.Option(
            "--session-logs",
            help="Session log directory the model calls and answers are observed from "
            "(default: this home's).",
        ),
    ] = None,
    observations: Annotated[
        Path | None,
        typer.Option(
            "--observations",
            help="More observations as JSON, e.g. mailbox writes: "
            '{"mailbox_writes": [{"account": "...", "count": 3, "at": "ISO 8601"}]}. '
            "A mailbox write must carry 'at': one without a time cannot be ordered "
            "against a revoke, so verify fails it.",
        ),
    ] = None,
) -> None:
    """Write the proof bundle of the ledger rows in scope. No content leaves: ids,
    decisions, classifications, tiers and digests only; accounts are pseudonymised."""
    from iris_harness.foundation.observability.session_log import session_log_dir
    from iris_harness.foundation.paths import audit_db_path
    from iris_harness.kernel.governance.audit import proof_bundle

    ledger = db or audit_db_path()
    if not ledger.exists():
        print_error(f"no audit ledger at {ledger}")
        raise typer.Exit(2)
    try:
        seen = proof_bundle.observations_from_session_logs(
            session_logs or session_log_dir(), since=since, until=until
        )
        if observations is not None:
            raw = json.loads(observations.read_text(encoding="utf-8"))
            seen = seen + proof_bundle.observations_from_json(raw)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print_error(f"could not read the observations: {exc}")
        raise typer.Exit(2) from exc
    bundle = proof_bundle.export_bundle(
        ledger,
        since=since,
        until=until,
        run_ids=tuple(run or ()),
        observations=seen,
        subject=subject,
    )
    proof_bundle.write_bundle(bundle, out)
    obs = bundle["observations"]
    console.print(
        f"wrote {out}: {len(bundle['ledger'])} ledger row(s), "
        f"{len(obs['model_calls'])} model call(s), {len(obs['answers'])} answer(s), "
        f"{len(obs['mailbox_writes'])} mailbox write record(s) observed"
    )


@proof_bundle_app.command("verify")
def proof_bundle_verify(
    bundle: Annotated[Path, typer.Argument(help="The bundle to verify (JSON).")],
) -> None:
    """Check a bundle offline: its format, its integrity and the three invariants. Exits
    non-zero on any violation."""
    from iris_harness.kernel.governance.audit import proof_bundle

    try:
        doc = proof_bundle.read_bundle(bundle)
    except (OSError, proof_bundle.ProofBundleError) as exc:
        print_error(str(exc))
        raise typer.Exit(2) from exc
    violations = proof_bundle.verify_bundle(doc)
    if violations:
        for violation in violations:
            console.print(f"[red]FAIL[/red] {violation}", markup=True, highlight=False)
        raise typer.Exit(1)
    for key in proof_bundle.INVARIANTS:
        console.print(f"[green]ok[/green] {key}")
    console.print(f"{bundle}: verified ({len(doc['ledger'])} ledger row(s))")


@proof_bundle_app.command("check")
def proof_bundle_check(
    days: Annotated[float, typer.Option("--days", help="Window, in days back from now.")] = 7.0,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Export the last ``--days`` in memory and verify it, per invariant. Writes nothing;
    exits non-zero when an invariant fails."""
    from datetime import UTC, datetime, timedelta

    from iris_harness.foundation.observability.session_log import session_log_dir
    from iris_harness.foundation.paths import audit_db_path
    from iris_harness.kernel.governance.audit import AuditLog, proof_bundle

    if not 0 < days <= 366:
        print_error("--days must be in (0, 366]")
        raise typer.Exit(2)
    result = proof_bundle.check_window(
        AuditLog(db_path=audit_db_path()),
        since=datetime.now(UTC) - timedelta(days=days),
        session_logs=session_log_dir(),
    )
    if as_json:
        console.print_json(json.dumps(result))
    else:
        console.print(
            f"proof bundle since {result['since']}: {result['ledger_rows']} ledger row(s)"
        )
        for detail in result["integrity"]:
            console.print(f"[red]FAIL[/red] {detail}", markup=True, highlight=False)
        for item in result["invariants"]:
            mark = "[green]ok[/green]" if item["ok"] else "[red]FAIL[/red]"
            seen = ", ".join(f"{k} {v}" for k, v in item["evidence"].items())
            console.print(f"{mark} {item['id']} ({seen})", markup=True, highlight=False)
            for detail in item["violations"]:
                console.print(f"     {detail}", markup=False, highlight=False)
    if not result["ok"]:
        raise typer.Exit(1)


__all__ = ["governance_app", "proof_bundle_app"]
