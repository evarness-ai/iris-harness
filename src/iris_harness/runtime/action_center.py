"""Action Center composition — the harness home for "what needs attention".

The unified pending-actions view (ADR-0073 §2b) lives here, not in any one client.
It unions persisted agent action-tasks with read-time Health-synthesized items, and
renders them as plain text. The API (``GET /actions``), the ReAct ``pending_actions``
tool, and the CLI all call these — so the capability is a first-class harness
surface, never isolated in the Web UI.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from typing import Any

from iris_harness.kernel.governance.approvals.store import ApprovalRow
from iris_harness.services.health.models import HealthSnapshot
from iris_harness.services.health.pending_actions import health_pending_actions
from iris_harness.services.tasks import TaskAction, TaskStore
from iris_harness.services.tasks.pending_actions import (
    PendingAction,
    pending_action_feedback_key,
    pending_action_from_task,
)

logger = logging.getLogger(__name__)


def memory_pending_actions(memory_store: Any) -> list[PendingAction]:
    """One row when memory is waiting on the owner — facts and lessons to review.

    One row, not one per item: the queue is reviewed in the Memory screen (or
    ``iris facts review``), and the Action Center's renderer is single-button by
    design. Its job here is to say the queue exists.
    """
    try:
        pending = int(memory_store.count_pending_review())
        lessons = len(memory_store.fetch_lesson_proposals(status="pending", limit=200))
    except Exception:  # a read model never breaks on one source
        logger.debug("memory review count failed", exc_info=True)
        return []
    total = pending + lessons
    if total <= 0:
        return []
    detail = f"{pending} fact(s)" if pending else ""
    if lessons:
        detail = f"{detail} and {lessons} lesson(s)" if detail else f"{lessons} lesson(s)"
    return [
        PendingAction(
            id="memory:review",
            origin="memory",
            source_kind="memory-review",
            title=f"{total} memory item(s) waiting for review",
            description=(
                f"IRIS noticed {detail}. Nothing reaches a prompt until you approve it — "
                "review them in Memory → Review."
            ),
            action=TaskAction(
                kind="copy_command",
                label="Review memory",
                command="iris facts review",
            ),
        )
    ]


def collect_pending_actions(
    task_store: TaskStore,
    snapshot: HealthSnapshot | None = None,
    *,
    feedback_store: Any = None,
    memory_store: Any = None,
) -> list[PendingAction]:
    """Persisted agent action-tasks (open) unioned with Health items and approvals.

    Health is recomputed on read (passed in), never copied into the task store. A
    ``None`` snapshot simply omits the Health items (e.g. health disabled).

    Each item is stamped with a ``feedback_ref`` so any channel can mark it "not
    useful", and items the user already suppressed are dropped from the read model
    (issue 0028). Health items are pre-filtered by ``health_pending_actions``; this
    applies the same gate to the task-backed items. A default ``learning.db`` store
    is used when ``feedback_store`` is omitted, so the Action Center honours feedback
    by default."""
    if feedback_store is None:
        from iris_harness.services.learning.suppression import SurfaceFeedbackStore

        feedback_store = SurfaceFeedbackStore()
        feedback_store.ensure_schema()

    actions = [
        pending_action_from_task(t)
        for t in task_store.list(has_action=True, limit=500)
        if t.status in ("open", "doing")
    ]
    if snapshot is not None:
        actions += health_pending_actions(snapshot, feedback_store=feedback_store)
    # A run halted waiting on a human is the plainest "needs your attention" there is,
    # and it used to appear only in `iris approvals list`. Read-time like Health: the
    # item lives exactly as long as its pending row. First, because a blocked run is
    # more urgent than anything the harness merely noticed.
    actions = approval_pending_actions() + actions
    if memory_store is not None:
        actions += memory_pending_actions(memory_store)

    from iris_harness.services.learning.suppression import encode_ref

    out: list[PendingAction] = []
    for pa in actions:
        key = pending_action_feedback_key(pa)
        if key is not None:
            try:
                if feedback_store.should_suppress(*key):
                    continue  # user marked this (or a sibling) not useful — hide it
            except Exception:  # suppression must never break the read
                logger.debug("action-center suppression check failed", exc_info=True)
            out.append(replace(pa, feedback_ref=encode_ref(*key)))
        else:
            out.append(pa)
    return out


def render_pending_actions(actions: list[PendingAction]) -> str:
    """Plain-text Action Center for chat / CLI (channel-agnostic)."""
    if not actions:
        return "No pending actions — nothing needs your attention right now."
    lines = [f"{len(actions)} pending action(s) need your attention:"]
    for a in actions:
        lines.append(f"- [{a.source_kind}] {a.title}")
        act = a.action
        if act.kind == "copy_command" and act.command:
            lines.append(f"    run locally: {act.command}")
        elif act.safe:
            lines.append(f"    {act.label} (from the Action Center)")
    return "\n".join(lines)


# ── Approval queue → Action Center (was governance/approvals/pending_actions.py) ──
#
# A run halted waiting on a human is the plainest case of "something needs your
# attention" there is, and until ADR-0108 it appeared on no such list: `iris approvals
# list` knew about it and nothing else did. This projection is what makes an approval
# visible on every surface at once -- `GET /actions`, the ReAct `pending_actions` tool,
# the CLI -- rather than in one UI.
#
# It lives here as of M6.2 because it is a view, not a policy: it reads the kernel's
# approval rows and speaks the Action Center's vocabulary (`PendingAction`, `TaskAction`),
# which sits ABOVE the kernel. Keeping it in governance/ meant the kernel imported
# `tasks` -- the one upward edge this module ever had.
#
# Like the Health adapter it is a **read-time projection**: nothing is persisted, the
# item exists exactly as long as its row is pending, and it disappears when the row is
# answered or times out. The action attached is `copy_command` -- display-only, the same
# shape Health uses. Answering an approval is a privileged write with two possible
# answers, which does not fit the single-CTA `invoke` lifecycle; the surfaces that can
# answer do it through `governance.approvals.service.respond_to_approval`, and everyone
# else gets the command to run. That also keeps `/actions/{id}/invoke` honest: these ids
# are not in the task store, so an invoke was never going to find them.

SOURCE_KIND = "governance-approval"


def _created_at(row: ApprovalRow) -> datetime | None:
    try:
        return datetime.fromisoformat(row.requested_at)
    except (TypeError, ValueError):
        return None


def approval_pending_action(row: ApprovalRow) -> PendingAction:
    """One pending approval as an Action Center item."""
    return PendingAction(
        id=f"approval:{row.approval_id}",
        origin="approval",
        source_kind=SOURCE_KIND,
        title=f"Approve or reject: {row.signal}",
        description=(
            f"{row.context_summary} "
            f"(run {row.run_id}; requested {row.requested_at[:19]}, "
            f"times out {row.timeout_at[:19]})"
        ),
        action=TaskAction(
            kind="copy_command",
            label="Approve from the CLI",
            command=f"iris approvals approve {row.approval_id}",
        ),
        created_at=_created_at(row),
    )


def approval_pending_actions(*, queue: Any = None) -> list[PendingAction]:
    """Every pending approval as Action Center items; never raises.

    A governance surface that erred would take the whole Action Center down with it,
    and the other sources — health, task blockers — have nothing to do with approvals.
    """
    try:
        from iris_harness.kernel.governance.approvals.service import pending_approvals

        return [approval_pending_action(r) for r in pending_approvals(queue=queue)]
    except Exception:  # one source must not break the composed read
        logger.warning("could not read pending approvals for the Action Center", exc_info=True)
        return []
