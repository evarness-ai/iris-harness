"""ApprovalQueue — facade combining ApprovalStore + AuditLog."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from iris_harness.kernel.governance.approvals.store import (
    ApprovalCard,
    ApprovalId,
    ApprovalItem,
    ApprovalRow,
    ApprovalStore,
)

if TYPE_CHECKING:
    from iris_harness.kernel.governance.audit import AuditLog


def _with_call_id(payload: dict[str, object], call_id: str | None) -> dict[str, object]:
    """An approval row's audit payload, naming the held call attempt it is about (#134)."""
    if call_id:
        payload["call_id"] = call_id
    return payload


class ApprovalQueue:
    """Thin facade: delegates to ApprovalStore, writes an audit row on every state change."""

    def __init__(
        self,
        *,
        store: ApprovalStore | None = None,
        audit_log: AuditLog | None = None,
        db_path: Path | None = None,
    ) -> None:
        self._store = store or ApprovalStore(db_path=db_path)
        self._audit = audit_log

    def enqueue(
        self,
        run_id: str,
        checkpoint_id: str | None,
        signal: str,
        context_summary: str,
        *,
        channel: str = "cli",
        timeout_minutes: int = 10,
        policy_on_timeout: str = "fail_closed",
        session_id: str | None = None,
        items: tuple[ApprovalItem, ...] | None = None,
        card: ApprovalCard | None = None,
        caller: str | None = None,
        call_id: str | None = None,
    ) -> ApprovalId:
        approval_id = self._store.enqueue(
            run_id,
            checkpoint_id,
            signal,
            context_summary,
            channel=channel,
            timeout_minutes=timeout_minutes,
            policy_on_timeout=policy_on_timeout,
            session_id=session_id,
            items=items,
            card=card,
            caller=caller,
            call_id=call_id,
        )
        if self._audit is not None:
            self._audit.record(
                run_id=run_id,
                step_id=None,
                agent_type="approvals",
                hook_point="approval_queue",
                plugin="ApprovalQueue",
                decision="require_approval",
                severity="warn",
                reason=f"approval enqueued: signal={signal} channel={channel}",
                payload=_with_call_id(
                    {"approval_id": approval_id, "signal": signal, "channel": channel}, call_id
                ),
            )
        return approval_id

    def get(self, approval_id: str) -> ApprovalRow | None:
        return self._store.get(approval_id)

    def pending_for_run(self, run_id: str) -> ApprovalRow | None:
        """The approval still gating ``run_id``, or None."""
        return self._store.pending_for_run(run_id)

    def list_pending(self, due_only: bool = False) -> list[ApprovalRow]:
        return self._store.list_pending(due_only=due_only)

    def list_by_status(self, status: str) -> list[ApprovalRow]:
        return self._store.list_by_status(status)

    def respond(
        self,
        approval_id: str,
        *,
        status: str,
        actor: str,
    ) -> ApprovalRow:
        row = self._store.respond(approval_id, status=status, actor=actor)
        if self._audit is not None:
            self._audit.record(
                run_id=row.run_id,
                step_id=None,
                agent_type="approvals",
                hook_point="approval_queue",
                plugin="ApprovalQueue",
                decision=status,
                severity="info",
                reason=f"approval {status} by {actor}",
                payload=_with_call_id({"approval_id": approval_id, "actor": actor}, row.call_id),
            )
        return row

    def claim_execution(self, approval_id: str) -> ApprovalRow | None:
        """Claim an approved code caller's call to run it (at most once); audited."""
        row = self._store.claim_execution(approval_id)
        if row is not None and self._audit is not None:
            self._audit.record(
                run_id=row.run_id,
                step_id=None,
                agent_type="approvals",
                hook_point="approval_queue",
                plugin="ApprovalQueue",
                decision="claimed",
                severity="info",
                reason=f"approved call claimed to run for {row.caller}",
                payload=_with_call_id(
                    {"approval_id": approval_id, "caller": row.caller}, row.call_id
                ),
            )
        return row

    def record_call_outcome(self, row: ApprovalRow, *, status: str, summary: str) -> None:
        """Audit what became of a code caller's approved (or unrun) call.

        The kernel's own rows record the governed call itself; this one closes the
        approval: which caller, which tool, and whether it ran, failed, was denied at
        execution, or never ran because it was rejected or lapsed.
        """
        if self._audit is None:
            return
        tool = row.items[0].tool if row.items else None
        self._audit.record(
            run_id=row.run_id,
            step_id=None,
            agent_type="approvals",
            hook_point="approval_queue",
            plugin="ApprovalQueue",
            decision=f"call_{status}",
            severity="info" if status == "ran" else "warn",
            reason=f"approved call {status}: {tool} for {row.caller}",
            payload=_with_call_id(
                {
                    "approval_id": row.approval_id,
                    "caller": row.caller,
                    "tool": tool,
                    "status": status,
                    "summary": summary,
                },
                row.call_id,
            ),
        )

    def set_checkpoint(self, approval_id: str, checkpoint_id: str) -> ApprovalRow:
        """Link the queued approval to the checkpoint answering it resumes."""
        return self._store.set_checkpoint(approval_id, checkpoint_id)

    def expire_stale(self) -> list[ApprovalRow]:
        """Time out overdue rows, auditing each one against the run it halted.

        One row per lapse, not one per sweep. The aggregate row this used to write was
        keyed to ``run_id="system"``, so ``iris run inspect <run_id>`` on the run that
        actually stalled showed nothing — the decision not to proceed was recorded
        against nobody.
        """
        rows = self._store.expire_stale()
        if self._audit is not None:
            for row in rows:
                self._audit.record(
                    run_id=row.run_id,
                    step_id=None,
                    agent_type="approvals",
                    hook_point="approval_queue",
                    plugin="ApprovalQueue",
                    decision="timed_out",
                    severity="warn",
                    reason=(
                        f"approval timed out unanswered after {row.timeout_at[:19]} "
                        f"(signal={row.signal}, policy={row.policy_on_timeout})"
                    ),
                    payload=_with_call_id(
                        {
                            "approval_id": row.approval_id,
                            "signal": row.signal,
                            "channel": row.channel,
                            "policy_on_timeout": row.policy_on_timeout,
                        },
                        row.call_id,
                    ),
                )
        return rows
