"""In-chat confirmation and the approvals timeout — the core's half of "approve/reject".

Two things live here, both about a decision the conversation owes someone:

* **The confirmation turn** (ADR-0076, ADR-0106). A domain that asks an in-chat question
  for a consequential (R3) action stashes its pending dict on a continuation with an
  ``executor_kind`` and registers an executor for that kind via
  ``PluginAPI.register_confirmation_executor``. The next turn is intercepted first: a
  plain "approve"/"reject" from any channel closes the continuation and dispatches by
  kind, so the plugin re-runs its action through its own governed path with
  ``approved=True``. Channels render the same two buttons from ``pending_options``.
* **The approvals timeout** (ADR-0108 follow-up). ``approval_timeout_tick`` sweeps the
  governance approval queue for lapsed rows and announces each lapse; a sweep that
  keeps a store's read model true is core by the OSS boundary rule.

Carved out of ``IrisRuntime`` at OSS plan M5.7 track C slice 14 as
``Confirmations(host)``, held as ``runtime.confirmations``. The state only this code
used moved with it: the lazily-built approval queue and delivery router.
:class:`ConfirmationsHost` declares the six runtime members read; the host is read
**at call time**, not captured. ``handle_confirmation_turn`` is public because the
intercept chain names it (``confirmations.handle_confirmation_turn`` in
``config/intercepts.yaml``), ``pending_options`` because the Telegram poller renders
buttons from it, ``approval_timeout_heartbeat`` because the runtime registers it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from iris_harness.runtime.nlu_parsing import _parse_confirmation_decision
from iris_harness.services.heartbeat import HeartbeatDefinition, HeartbeatRun, HeartbeatStatus

if TYPE_CHECKING:
    from pathlib import Path

    from iris_harness.runtime.activity_notices import ActivityNotices
    from iris_harness.runtime.continuations import ContinuationRegistry
    from iris_harness.runtime.plugin_host.registry import PluginRegistry
    from iris_harness.runtime.replies import DeterministicReplies
    from iris_harness.runtime.tool_service import ToolService
    from iris_harness.runtime.types import ChatResult

logger = logging.getLogger(__name__)

# An approve-executor for one in-chat confirmation kind (ADR-0076): takes the stashed
# pending dict + turn context, re-runs the action through its governed path with
# approved=True, and returns the turn's ChatResult.
ConfirmationExecutor = Callable[..., "ChatResult"]


class ConfirmationsHost(Protocol):
    """The six runtime members the confirmation turn and the timeout sweep reach."""

    continuations: ContinuationRegistry
    data_dir: Path
    plugin_registry: PluginRegistry
    replies: DeterministicReplies
    # The executor for a code caller's approved call (plugin-capabilities decision 1).
    tool_service: ToolService | None

    def _activity_notices(self) -> ActivityNotices: ...


class Confirmations:
    """The confirmation turn and the approvals timeout for one runtime. See the module
    docstring."""

    def __init__(self, host: ConfirmationsHost) -> None:
        self._host = host
        # The governance approval queue and the delivery router the timeout sweep
        # uses, each built on first use (the queue opens approvals.db).
        self._approvals_queue_cache: Any = None
        self._approvals_router_cache: Any = None

    # ------------------------------------------------------------------
    # The confirmation turn
    # ------------------------------------------------------------------

    def pending_options(self, session_id: str) -> list[str] | None:
        """Approve/reject options for a session's open confirmation, else None.

        Lets channels (e.g. Telegram inline buttons) render the same controls the
        web shows — the decision is resolved by `handle_confirmation_turn`. Only an
        *executable* continuation gets buttons: a plain question has no yes to press.
        """
        pending = self._host.continuations.pending(session_id)
        if pending is None or not pending.is_executable:
            return None
        return ["approve", "reject"]

    def handle_confirmation_turn(
        self,
        message: str,
        *,
        session_id: str,
        span: Any = None,
    ) -> ChatResult | None:
        """Resolve a pending in-chat confirmation (approve/reject) if one is open.

        Channel-agnostic: any surface (web buttons, CLI, telegram) sends a plain
        "approve"/"reject", which re-runs the stashed action through the same
        governed path with ``approved=True``. Returns None when there's nothing
        pending or the turn isn't a decision (so normal routing continues).
        """
        continuation = self._host.continuations.pending(session_id)
        if continuation is None or not continuation.is_executable:
            return None
        decision = _parse_confirmation_decision(message)
        if decision is None:
            return None  # not a decision — leave it pending, route normally
        # Closed before the action runs, and closed either way. A payload that executed
        # while its question stayed open would be re-runnable by a second "approve".
        self._host.continuations.answered(continuation.continuation_id)

        pending: dict[str, Any] = dict(continuation.payload or {})
        pending["kind"] = continuation.executor_kind or ""
        task_dedup = pending.get("task_dedup")

        if decision == "reject":
            self._resolve_approval_task(task_dedup, approved=False)
            return self._reminder_chat_result(
                message=message,
                session_id=session_id,
                response="Okay — cancelled. I won't create that.",
                metadata={"confirmation": "rejected", "kind": pending.get("kind", "")},
                span=span,
            )

        # approve → execute the stashed action through the governed path, dispatched
        # by kind (ADR-0076; each executor re-runs its action with approved=True).
        executor = self._confirmation_executors().get(str(continuation.executor_kind or ""))
        if executor is not None:
            return executor(
                pending,
                message=message,
                session_id=session_id,
                task_dedup=task_dedup,
                span=span,
            )

        # unknown kind — acknowledge without acting
        self._resolve_approval_task(task_dedup, approved=True)
        return self._reminder_chat_result(
            message=message,
            session_id=session_id,
            response="Approved.",
            metadata={"confirmation": "approved", "kind": pending.get("kind", "")},
            span=span,
        )

    def _confirmation_executors(self) -> dict[str, ConfirmationExecutor]:
        """Approve-executors by confirmation kind, all plugin-registered.

        A domain that asks an in-chat question stashes its pending dict under a
        ``kind`` and registers the executor for it via
        ``PluginAPI.register_confirmation_executor``; the core dispatches by kind and
        governs the re-run. The last core-owned entry (``calendar_event``) left with
        the calendar plugin (OSS plan M5.7 track A), which is what proves an executor
        is enough for a governed write."""
        return dict(self._host.plugin_registry.confirmation_executors())

    def _resolve_approval_task(self, dedup_key: str | None, *, approved: bool) -> None:
        """Close the Action-Center pending action once approved/rejected in chat.

        Generic across confirmation kinds: the dedup key identifies the task."""
        if not dedup_key:
            return
        try:
            from iris_harness.services.tasks import TaskStore

            store = TaskStore(db_path=self._host.data_dir / "tasks.db")
            store.ensure_schema()
            task = store.get_by_dedup_key(dedup_key)
            if task is None:
                return
            if approved:
                store.complete(task.id)
            else:
                store.drop(task.id)
        except Exception:
            logger.exception("failed to resolve calendar approval pending action")

    def _reminder_chat_result(
        self,
        *,
        message: str,
        session_id: str,
        response: str,
        metadata: dict[str, object],
        span: Any = None,
        has_errors: bool = False,
        error_summary: str | None = None,
    ) -> ChatResult:
        """The confirmation turn's calendar-flavoured reply (ADR-0076).

        The same deterministic reply the calendar plugin sends through
        ``HarnessServices.deterministic_reply``; kept as a name because the core's
        confirmation turn still answers a rejected or unknown-kind calendar question.
        """
        return self._host.replies.system_chat_result(
            message=message,
            session_id=session_id,
            response=response,
            metadata=metadata,
            span=span,
            intent="calendar",
            agent_type="calendar",
            sources=("reminders",),
            has_errors=has_errors,
            error_summary=error_summary,
        )

    # ------------------------------------------------------------------
    # The approvals timeout
    # ------------------------------------------------------------------

    def approval_timeout_heartbeat(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """Sweep overdue approvals and announce each lapse (ADR-0108 follow-up).

        This is the thing that was missing: `expire_stale` existed, was tested, and had
        no production caller, so an approval's ``timeout_at`` was decorative. A sweep
        that keeps a store's read model true is core by the OSS boundary rule, the same
        as the four filemanager sweeps and `email_sweep`.
        """
        from iris_harness.kernel.governance.approvals.service import sweep_expired

        try:
            expired = sweep_expired(
                queue=self._approvals_queue(),
                router=self._approvals_router(),
                notifier=self._host._activity_notices(),
                # A code caller's lapsed approval ends its call: the caller is told.
                executor=self._host.tool_service,
            )
        except Exception as exc:  # a failed sweep must not kill the scheduler
            logger.warning("approval timeout sweep failed: %s", exc, exc_info=True)
            return HeartbeatRun(
                name=definition.name, status=HeartbeatStatus.FAILED, output="", error=str(exc)
            )
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            output=f"approval_timeout_tick expired={len(expired)}",
        )

    def _approvals_queue(self) -> Any:
        from iris_harness.kernel.governance.approvals import ApprovalQueue
        from iris_harness.kernel.governance.audit import AuditLog

        if self._approvals_queue_cache is None:
            self._approvals_queue_cache = ApprovalQueue(audit_log=AuditLog())
        return self._approvals_queue_cache

    def _approvals_router(self) -> Any:
        """The same delivery router the evaluator notifies through.

        ``force_interactive=False`` for the same reason it is false there, and one more:
        a sweep runs on a scheduler thread with nobody at the terminal, so a blocking
        ``input()`` would hang the heartbeat rather than ask anyone anything.
        """
        from iris_harness.kernel.governance.approvals.router import ChannelRouter

        if self._approvals_router_cache is None:
            self._approvals_router_cache = ChannelRouter(
                queue=self._approvals_queue(), force_interactive=False
            )
        return self._approvals_router_cache
