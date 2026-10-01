"""Responding to an approval — the capability, once, for every channel.

The rule this module exists to honour is the Action Center's
(``action_center.py``): *a core capability is never isolated in one channel.* Before
it, "answer an approval" existed three times and agreed on nothing. ``iris approvals
approve`` talked straight to ``ApprovalStore``, so it wrote no audit row. The Telegram
handler in ``channel_gateway`` did the same, separately. The web had no way to do it
at all. And none of the three could continue the run they had just approved, because
the approval row had no link to its checkpoint.

So the decision, its audit trail, and the resume all live here, and the surfaces are
thin: :func:`respond_to_approval` is called by the API endpoint, the CLI command, and
the Telegram command handler alike, and anything added later gets the same behaviour
by calling the same function.

**No agent-facing tool answers an approval.** A pending approval is *visible* to the
loop, because it reaches the Action Center and the ``pending_actions`` tool reads that
— knowing it is waiting is useful context. Answering is a different thing: the
approvals queue exists precisely to put a human between the agent and an action the
evaluator would not allow on its own, and a tool that approved would hand the decision
back to the party it was taken from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from iris_harness.kernel.governance.approvals.events import ApprovedCallStatus
from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
from iris_harness.kernel.governance.approvals.store import ApprovalRow

logger = logging.getLogger(__name__)

__all__ = [
    "ApprovalOutcome",
    "ApprovedCallExecutor",
    "ApprovedCallResult",
    "LapseNotifier",
    "ResumedRun",
    "RunResumer",
    "lapse_notice",
    "parse_checkpoint_id",
    "pending_approvals",
    "respond_to_approval",
    "sweep_expired",
]


@dataclass(frozen=True)
class ResumedRun:
    """What continuing a halted run produced."""

    answer: str
    session_id: str | None = None


@runtime_checkable
class RunResumer(Protocol):
    """Whatever can continue a halted run — in production, ``IrisRuntime``.

    A protocol rather than an import: ``governance`` sits below ``runtime``, and this
    module must not reach upwards to reverse that. The caller that has a runtime passes
    it in; the caller that does not (a test, a CLI invocation with no runtime built)
    passes nothing and the decision is still recorded.
    """

    def resume_halted_run(
        self, *, run_id: str, step_id: int, channel: str = "console"
    ) -> ResumedRun: ...


@dataclass(frozen=True)
class ApprovedCallResult:
    """What became of a code caller's approved call (plugin-capabilities decision 1).

    ``status``: ``ran``, ``failed`` (the tool raised), ``denied`` (governance refused it at
    execution), ``rejected`` or ``expired`` (never ran). ``summary`` is display-masked.
    """

    status: ApprovedCallStatus
    summary: str

    def notice(self, tool: str) -> str:
        """The line the conversation the call came from is told."""
        heads = {
            "ran": f"The approved call to {tool} ran.",
            "failed": f"The approved call to {tool} ran and failed.",
            "denied": f"The approved call to {tool} was refused when it came to run.",
            "rejected": f"The call to {tool} was rejected; nothing was run.",
            "expired": f"The approval for {tool} expired; nothing was run.",
        }
        head = heads.get(self.status, f"The call to {tool}: {self.status}.")
        return f"{head}\n\n{self.summary}" if self.summary else head


@runtime_checkable
class ApprovedCallExecutor(Protocol):
    """Whatever runs a code caller's approved call — in production, the runtime's
    ``ToolService``, which runs it through the governed tool runner.

    A protocol for the reason ``RunResumer`` is one: ``governance`` sits below
    ``runtime``. It is handed the answered row; the approval hook, not the executor,
    claims it as the call passes ``PRE_TOOL_USE``, so the call runs at most once.
    """

    def execute_approved_call(self, row: ApprovalRow) -> ApprovedCallResult: ...

    def settle_unrun_call(
        self, row: ApprovalRow, status: ApprovedCallStatus
    ) -> ApprovedCallResult: ...


@runtime_checkable
class LapseNotifier(Protocol):
    """Whatever can put a notice in front of the user — in production, the runtime's
    ``ActivityNotices``.

    The split is deliberate: this module composes the *text*, because the wording is
    part of the governance behaviour, and the notifier only delivers. Conversations and
    outbound channels belong to the runtime, and ``governance`` sits below it.
    """

    def deliver_lapse_notice(self, *, session_id: str | None, channel: str, text: str) -> None: ...


def lapse_notice(approval: ApprovalRow) -> str:
    """What the user is told when their approval window closes unanswered.

    The rule the halt message already follows (ADR-0107): say what happened, say why,
    and stop. Naming the run matters here for the same reason it mattered there —
    it is the only handle the user has on it afterwards.

    A code caller's approval halted no run (decision 1), so its notice says the call was
    not run instead — the sweep sends it when no executor is here to settle the row.
    """
    if approval.is_deferred_call:
        return (
            f"The approval I was waiting on has expired, so "
            f"{approval.lapse_consequence()}. Nothing was changed.\n\n"
            f"What it was for: {approval.context_summary}\n"
            f"No answer arrived before {approval.timeout_at[:19]} UTC.\n\n"
            f"Ask again if you still want it done."
        )
    return (
        f"The approval I was waiting on has expired, so run {approval.run_id} is "
        f"still stopped where it was.\n\n"
        f"Why it stopped: {approval.context_summary}\n"
        f"No answer arrived before {approval.timeout_at[:19]} UTC "
        f"(policy: {approval.policy_on_timeout}), so nothing further has run.\n\n"
        f"Tell me how you'd like to proceed and I'll start from there."
    )


@dataclass(frozen=True)
class ApprovalOutcome:
    """The answered row, plus what became of the run it was about."""

    row: ApprovalRow
    resumed: bool
    detail: str
    # A code caller's approved call ran (decision 1); ``detail`` is its masked output.
    executed: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "approval_id": self.row.approval_id,
            "run_id": self.row.run_id,
            "status": self.row.status,
            "signal": self.row.signal,
            "responded_at": self.row.responded_at,
            "response_actor": self.row.response_actor,
            "checkpoint_id": self.row.checkpoint_id,
            "resumed": self.resumed,
            "executed": self.executed,
            "caller": self.row.caller,
            "detail": self.detail,
        }


def parse_checkpoint_id(checkpoint_id: str | None) -> tuple[str, int] | None:
    """Split a ``"<run_id>:<step_id>"`` link, or None if it is absent or malformed.

    The format is ``AgenticCore._write_checkpoint``'s (``f"{run_id}:{step_id}"``), and
    ``run_id`` is a UUID so the rightmost colon is the separator.
    """
    if not checkpoint_id:
        return None
    run_id, _, step = checkpoint_id.rpartition(":")
    if not run_id or not step.isdigit():
        logger.warning("unparseable checkpoint link %r", checkpoint_id)
        return None
    return (run_id, int(step))


def _queue(queue: ApprovalQueue | None) -> ApprovalQueue:
    """The queue, not the bare store — so every answer leaves an audit row.

    The CLI and Telegram paths both used ``ApprovalStore`` directly and so recorded
    nothing in the ledger; a governance decision that leaves no trace is the one kind
    that most needs one.
    """
    if queue is not None:
        return queue
    from iris_harness.kernel.governance.audit import AuditLog

    return ApprovalQueue(audit_log=AuditLog())


def pending_approvals(
    *, queue: ApprovalQueue | None = None, due_only: bool = False
) -> list[ApprovalRow]:
    """Every approval still waiting on a human, newest state from the store."""
    return _queue(queue).list_pending(due_only=due_only)


def respond_to_approval(
    approval_id: str,
    *,
    status: str,
    actor: str,
    queue: ApprovalQueue | None = None,
    resumer: RunResumer | None = None,
    executor: ApprovedCallExecutor | None = None,
) -> ApprovalOutcome:
    """Record a human's answer, then continue the run if they approved.

    Ordering is deliberate: **the decision is recorded first and separately.** It is
    the durable, auditable thing a person did, and it must survive a resume that fails
    for any reason — an expired checkpoint, a model outage, no runtime in this process.
    Every failure past that point degrades to "answered, not resumed" and says so in
    ``detail``, rather than losing the answer or raising at the caller.

    A code caller's approval (``row.is_deferred_call``, decision 1) has no run to
    resume: approving it hands the row to ``executor``, which runs the pinned call once
    (the approval hook claims it); rejecting it tells the caller nothing will run. See :func:`_settle_deferred_call`.

    Raises ``ApprovalNotFoundError`` / ``ApprovalAlreadyAnsweredError`` from the store,
    which every surface already renders.
    """
    q = _queue(queue)
    existing = q.get(approval_id)
    if (
        existing is not None
        and existing.is_deferred_call
        and existing.status == "pending"
        and status == "approved"
        and executor is None
    ):
        # Approving a call nobody here can run would leave an approved row that never
        # runs: the promise the queue exists not to break. Leave it waiting instead.
        return ApprovalOutcome(
            row=existing,
            resumed=False,
            detail=(
                "Not approved yet: this approval runs a call, and no runtime is available "
                "here to run it. It is still waiting; answer it where IRIS is running."
            ),
        )
    row = q.respond(approval_id, status=status, actor=actor)
    if row.is_deferred_call:
        return _settle_deferred_call(q, row, executor)
    _settle_drift_exemptions(row)

    # A destructive-tool approval (it pins items, ADR-0118) resumes on a rejection too:
    # the loop reads the row, executes nothing, and tells the owner so in the chat. An
    # evaluator halt stays halted when rejected, as it always has.
    if row.status != "approved" and row.items is None:
        return ApprovalOutcome(row=row, resumed=False, detail="Rejected — the run stays halted.")
    verb = "Approved" if row.status == "approved" else "Rejected, nothing was changed"

    point = parse_checkpoint_id(row.checkpoint_id)
    if point is None:
        return ApprovalOutcome(
            row=row,
            resumed=False,
            detail=(
                f"{verb}. No checkpoint is linked to this approval, so there is "
                "nothing to continue automatically."
            ),
        )
    if resumer is None:
        return ApprovalOutcome(
            row=row,
            resumed=False,
            detail=(
                f"{verb}. Run {point[0]} is resumable from step {point[1]}, but no "
                "runtime is available here to continue it."
            ),
        )

    run_id, step_id = point
    try:
        # The row remembers which surface raised this, and the resumed turn can halt
        # again — so the channel is carried through rather than defaulted, or a second
        # approval would go somewhere the user is not looking.
        resumed = resumer.resume_halted_run(
            run_id=run_id, step_id=step_id, channel=row.channel or "console"
        )
    except Exception as exc:  # the decision is already recorded
        logger.warning("approved run %s failed to resume", run_id, exc_info=True)
        return ApprovalOutcome(
            row=row,
            resumed=False,
            detail=f"{verb}, but the run could not be continued: {exc}",
        )
    logger.info("%s run %s resumed from step %s", row.status, run_id, step_id)
    return ApprovalOutcome(row=row, resumed=True, detail=resumed.answer)


def _settle_deferred_call(
    q: ApprovalQueue, row: ApprovalRow, executor: ApprovedCallExecutor | None
) -> ApprovalOutcome:
    """Run (or settle) a code caller's call once its approval is answered.

    Approved: run it through the executor, which governs it again. The approval hook
    claims the row as the call passes ``PRE_TOOL_USE`` — a conditional write only one
    claimant wins — so the call runs at most once however it is answered, retried or
    replayed. Rejected: nothing runs; the caller is told. Each outcome is audited
    against the approval.
    """
    if row.status != "approved":
        detail = "Rejected, nothing was run."
        if executor is None:
            q.record_call_outcome(row, status="rejected", summary=detail)
            return ApprovalOutcome(row=row, resumed=False, detail=detail)
        result = _settle_unrun(q, row, executor, "rejected")
        return ApprovalOutcome(row=row, resumed=False, detail=result.summary or detail)
    if executor is None:  # only reachable by a race; the row stays approved and unclaimed
        return ApprovalOutcome(
            row=row, resumed=False, detail="Approved, but no runtime is here to run it."
        )
    try:
        result = executor.execute_approved_call(row)
    except Exception as exc:  # the answer stands; a claimed row can never run again
        logger.warning("approved call %s failed to run", row.approval_id, exc_info=True)
        result = ApprovedCallResult(status="failed", summary=f"It could not be run: {exc}")
    q.record_call_outcome(row, status=result.status, summary=result.summary)
    return ApprovalOutcome(
        row=q.get(row.approval_id) or row,
        resumed=False,
        executed=result.status in ("ran", "failed"),
        detail=f"Approved. {result.notice(_tool_of(row))}",
    )


def _settle_unrun(
    q: ApprovalQueue, row: ApprovalRow, executor: ApprovedCallExecutor, status: ApprovedCallStatus
) -> ApprovedCallResult:
    try:
        result = executor.settle_unrun_call(row, status)
    except Exception:  # the decision is recorded; telling the caller is best-effort
        logger.warning("could not settle approval %s", row.approval_id, exc_info=True)
        result = ApprovedCallResult(status=status, summary="")
    q.record_call_outcome(row, status=status, summary=result.summary)
    return result


def _tool_of(row: ApprovalRow) -> str:
    return row.items[0].tool if row.items else ""


def sweep_expired(
    *,
    queue: ApprovalQueue | None = None,
    router: object | None = None,
    notifier: LapseNotifier | None = None,
    executor: ApprovedCallExecutor | None = None,
) -> list[ApprovalRow]:
    """Time out overdue approvals and tell the user about each one.

    **Nothing called ``expire_stale`` in production before this.** Not the runtime, not
    a heartbeat, not a read path — so an approval never actually timed out. The row sat
    ``pending`` past its deadline and ``list_pending`` kept returning it, which means a
    lapsed approval did not go quiet so much as go stale: the Action Center showed a
    dead request as though it were still live, and `policy_on_timeout` had never once
    been applied to anything.

    Two notices, because they answer different questions. The **channel** notice goes
    to the surface that was asked, so whoever was waiting for a prompt learns it closed.
    The **in-chat** notice goes to the conversation the run belongs to, which is where
    the user was told "paused for approval" in the first place and therefore the only
    place the silence is conspicuous.

    Safe to call from anywhere and as often as you like: ``expire_stale`` only
    transitions rows out of ``pending``, so each lapse is announced exactly once no
    matter how many sweeps run.
    """
    q = _queue(queue)
    rows = q.expire_stale()
    for row in rows:
        if router is not None:
            notify_timeout = getattr(router, "notify_timeout", None)
            if notify_timeout is not None:
                try:
                    notify_timeout(row)
                except Exception:  # one channel must not stop the sweep
                    logger.warning(
                        "could not announce timeout of approval %s on its channel",
                        row.approval_id,
                        exc_info=True,
                    )
        if row.is_deferred_call and executor is not None:
            # A code caller's approval lapsing ends it: the caller hears (event) and the
            # conversation gets the executor's notice, instead of "the run stays halted",
            # which is not true of a call with no run.
            _settle_unrun(q, row, executor, "expired")
            continue
        if notifier is not None:
            try:
                notifier.deliver_lapse_notice(
                    session_id=row.session_id,
                    channel=row.channel or "console",
                    text=lapse_notice(row),
                )
            except Exception:  # nor stop it from reaching the next row
                logger.warning(
                    "could not deliver the lapse notice for approval %s",
                    row.approval_id,
                    exc_info=True,
                )
    return rows


def _settle_drift_exemptions(row: ApprovalRow) -> None:
    """Teach ``goal_drift`` from the person's answer, or bury the thought they refused.

    Best-effort by design: the decision is already recorded and the run must continue
    (or stay halted) whatever the exemption store does. A learning step that can fail a
    governance response is worse than one that occasionally misses a lesson.
    """
    if row.signal != "goal_drift":
        return
    try:
        from iris_harness.kernel.governance.evaluator.drift_exemptions import (
            DriftExemptionStore,
        )

        promoted = DriftExemptionStore().settle_run(row.run_id, approved=row.status == "approved")
    except Exception:  # never let the lesson break the answer
        logger.warning("could not settle drift exemptions for run %s", row.run_id, exc_info=True)
        return
    if promoted:
        logger.info("goal_drift: %d exemption(s) %s for run %s", promoted, row.status, row.run_id)
