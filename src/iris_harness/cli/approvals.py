"""CLI commands for the HITL approval queue (iris approvals ...).

``approve`` / ``reject`` go through ``governance.approvals.service``, the same
function the API endpoint and the Telegram handler call. They used to talk straight to
``ApprovalStore``, which meant two things silently: a governance decision made from
the CLI left **no audit row**, and approving could not continue the run it had just
approved. Both are properties of the capability, not of the channel, so they live in
one place now and every surface gets them.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from iris_harness.kernel.governance.approvals import ApprovalStore
    from iris_harness.kernel.governance.approvals.queue import ApprovalQueue

console = Console()

approvals_app = typer.Typer(
    name="approvals",
    help="Manage HITL approval requests.",
    no_args_is_help=True,
)


def _store() -> ApprovalStore:
    from iris_harness.kernel.governance.approvals import ApprovalStore as _ApprovalStore

    return _ApprovalStore()


def _queue() -> ApprovalQueue:
    """The store wrapped so a CLI decision leaves an audit row like any other."""
    from iris_harness.kernel.governance.approvals.queue import ApprovalQueue as _Queue
    from iris_harness.kernel.governance.audit import AuditLog

    return _Queue(store=_store(), audit_log=AuditLog())


def _actor() -> str:
    return f"cli:{os.environ.get('USER', 'unknown')}"


@approvals_app.command("list")
def cmd_list(
    due: Annotated[
        bool, typer.Option("--due", help="Show only overdue (past timeout) rows.")
    ] = False,
    status_filter: Annotated[
        str,
        typer.Option("--status", help="Filter by status: pending|approved|rejected|timed_out"),
    ] = "pending",
) -> None:
    """List approval requests."""
    store = _store()
    if status_filter == "pending":
        rows = store.list_pending(due_only=due)
    else:
        rows = store.list_by_status(status_filter)

    if not rows:
        console.print(f"[dim]No {status_filter} approvals.[/dim]")
        return

    table = Table(title=f"Approvals [{status_filter}]", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True, max_width=36)
    table.add_column("Run ID", style="dim", max_width=20)
    table.add_column("Signal", style="yellow")
    table.add_column("Context", max_width=40)
    table.add_column("Timeout", style="red")
    table.add_column("Status", style="green")
    for r in rows:
        table.add_row(
            r.approval_id,
            r.run_id,
            r.signal,
            r.context_summary[:40],
            r.timeout_at[:19],
            r.status,
        )
    console.print(table)


def _respond(approval_id: str, *, status: str, reason: str, resume: bool) -> None:
    """Shared body for approve/reject — one capability, rendered for a terminal."""
    from iris_harness.kernel.governance.approvals import (
        ApprovalAlreadyAnsweredError,
        ApprovalNotFoundError,
    )
    from iris_harness.kernel.governance.approvals.service import (
        parse_checkpoint_id,
        respond_to_approval,
    )

    actor = _actor()
    if reason:
        actor = f"{actor} ({reason})"

    queue = _queue()
    resumer = None
    executor = None
    existing = queue.get(approval_id)
    if existing is not None and existing.is_deferred_call and existing.status == "pending":
        # A code caller's call (plugin-capabilities decision 1): approving runs it.
        if status == "approved" and not resume:
            # --no-resume would approve a call and not run it: an approved row nothing
            # runs. Leave it waiting instead, as the service does with no executor.
            console.print(
                "[yellow]Not answered:[/yellow] this approval runs a call, and "
                "--no-resume would approve it without running it. It is still waiting."
            )
            return
        # Answer through the running IRIS API when it is up, so the call runs in the
        # server and approval.call_completed reaches the plugin that asked, there.
        if _respond_via_api(approval_id, status=status, actor=actor):
            return
        # The API is down: run it in a runtime built here. With no runtime at all the
        # service leaves an approve waiting.
        console.print(
            "[dim]IRIS API not reachable; answering in this process. The "
            "approval.call_completed event stays local: a plugin in the running server "
            "will not hear it.[/dim]"
        )
        try:
            from iris_harness.runtime.bootstrap import build_runtime

            executor = build_runtime().tool_service
        except Exception as exc:  # noqa: BLE001 — the service says what happened
            console.print(f"[dim]could not start a runtime to run the call: {exc}[/dim]")
    elif resume and status == "approved":
        # Only build a runtime when there is something for it to continue. Peeking the
        # row first keeps an ordinary approve — of a halt with no checkpoint behind it —
        # from paying for a full bootstrap it would not use. Failure is never fatal: the
        # decision is still recorded and the outcome says the run was not continued.
        if existing is not None and parse_checkpoint_id(existing.checkpoint_id) is not None:
            try:
                from iris_harness.runtime.bootstrap import build_runtime

                resumer = build_runtime()
            except Exception as exc:  # noqa: BLE001 — recording the answer matters more
                console.print(f"[dim]could not start a runtime to continue the run: {exc}[/dim]")

    try:
        outcome = respond_to_approval(
            approval_id, status=status, actor=actor, queue=queue, resumer=resumer, executor=executor
        )
    except ApprovalNotFoundError:
        console.print(f"[red]Error:[/red] approval {approval_id!r} not found.", style="red")
        raise typer.Exit(1) from None
    except ApprovalAlreadyAnsweredError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        raise typer.Exit(1) from None

    row = outcome.row
    if row.status == "pending":
        console.print(f"[yellow]Not answered:[/yellow] {outcome.detail}")
        return
    if row.status == "approved":
        console.print(f"[green]\u2713[/green] Approved {row.approval_id} (run {row.run_id})")
    else:
        console.print(f"[yellow]\u2717[/yellow] Rejected {row.approval_id} (run {row.run_id})")
    if outcome.executed:
        console.print(f"\n[bold]The call ran:[/bold]\n{outcome.detail}")
    elif outcome.resumed:
        console.print("\n[bold]The run continued:[/bold]")
        console.print(outcome.detail)
    else:
        console.print(f"[dim]{outcome.detail}[/dim]")


def _api_base() -> str:
    return (os.environ.get("IRIS_API_URL") or "http://localhost:8003").rstrip("/")


def _respond_via_api(approval_id: str, *, status: str, actor: str) -> bool:
    """Answer through ``POST /governance/approvals/{id}/respond``; False when unreachable.

    The same endpoint the web and the gateway's Telegram use (``ApiApprovalBackend``).
    Only a transport failure falls back to answering locally: an API that answered with
    an error has been reached, and the error is the answer.
    """
    import httpx

    from iris_harness.cli.api_client import harness_api_client

    try:
        with harness_api_client(purpose="approvals.respond", timeout=120.0) as client:
            response = client.post(
                f"{_api_base()}/governance/approvals/{approval_id}/respond",
                json={"status": status, "actor": actor},
            )
    except httpx.TransportError:
        return False
    if response.status_code == 404:
        console.print(f"[red]Error:[/red] approval {approval_id!r} not found.", style="red")
        raise typer.Exit(1)
    if response.status_code == 409:
        detail = response.json().get("detail", "already answered")
        console.print(f"[red]Error:[/red] {detail}", style="red")
        raise typer.Exit(1)
    if response.status_code >= 400:
        console.print(f"[red]Error:[/red] the IRIS API answered HTTP {response.status_code}")
        raise typer.Exit(1)
    body = response.json()
    answered = body.get("status")
    if answered == "pending":
        console.print(f"[yellow]Not answered:[/yellow] {body.get('detail', '')}")
        return True
    mark = (
        "[green]\u2713[/green] Approved"
        if answered == "approved"
        else "[yellow]\u2717[/yellow] Rejected"
    )
    console.print(f"{mark} {approval_id} (answered by the running IRIS)")
    if body.get("executed"):
        console.print(f"\n[bold]The call ran:[/bold]\n{body.get('detail', '')}")
    else:
        console.print(f"[dim]{body.get('detail', '')}[/dim]")
    return True


@approvals_app.command("approve")
def cmd_approve(
    approval_id: Annotated[str, typer.Argument(help="Approval ID (UUID)")],
    reason: Annotated[str, typer.Option("--reason", "-r", help="Optional reason.")] = "",
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help=(
                "Continue the halted run from its checkpoint (default: yes). A code "
                "caller's approval runs its call when approved, so --no-resume leaves "
                "it waiting instead."
            ),
        ),
    ] = True,
) -> None:
    """Approve a pending request and continue the run it halted."""
    _respond(approval_id, status="approved", reason=reason, resume=resume)


@approvals_app.command("reject")
def cmd_reject(
    approval_id: Annotated[str, typer.Argument(help="Approval ID (UUID)")],
    reason: Annotated[str, typer.Option("--reason", "-r", help="Optional reason.")] = "",
) -> None:
    """Reject a pending request; the run stays halted."""
    _respond(approval_id, status="rejected", reason=reason, resume=False)


@approvals_app.command("exemptions")
def cmd_exemptions(
    show_all: Annotated[
        bool,
        typer.Option("--all", help="Include pending and rejected rows, not just the grants."),
    ] = False,
) -> None:
    """Show the thought shapes your approvals have taught ``goal_drift`` to allow.

    An exemption is a standing grant, so it has to be readable. Each row is the task it
    was earned on and the words a later thought must share to use it.
    """
    from iris_harness.kernel.governance.evaluator.drift_exemptions import (
        DriftExemptionStore,
    )

    store = DriftExemptionStore()
    rows = store.all_rows() if show_all else store.approved()
    if not rows:
        console.print("[dim]No exemptions. goal_drift asks every time.[/dim]")
        return

    table = Table(title="goal_drift exemptions", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True, max_width=36)
    table.add_column("Earned on", max_width=34)
    table.add_column("Keywords", style="yellow", max_width=40)
    table.add_column("Status", style="green")
    table.add_column("Created", style="dim")
    for e in rows:
        table.add_row(
            e.exemption_id,
            e.original_task[:34],
            ", ".join(sorted(e.keywords))[:40],
            e.status,
            e.created_at[:19],
        )
    console.print(table)


@approvals_app.command("forget-exemptions")
def cmd_forget_exemptions(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Take every grant back; ``goal_drift`` goes back to asking each time."""
    from iris_harness.kernel.governance.evaluator.drift_exemptions import (
        DriftExemptionStore,
    )

    if not yes and not typer.confirm("Forget every goal_drift exemption?"):
        console.print("[dim]Kept.[/dim]")
        return
    removed = DriftExemptionStore().clear()
    console.print(f"[green]✓[/green] Forgot {removed} exemption(s).")
