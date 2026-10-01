"""Generic pending-actions mechanism (ADR-0073 slice 3).

The *mechanism* for pending actions lives here, in the tasks subsystem (the
backbone); the *policy* — what to raise — lives in each domain's provider
(``plugins_builtin.finance_workflows.pending_actions`` was the first). Any agent
can surface actionable blockers by implementing :class:`PendingActionProvider` and
letting :func:`reconcile` raise/resolve its tasks; the Action Center then shows them
uniformly. This is the extensibility seam: a new agent adds a provider, not new
plumbing.

A "pending action" is a Task carrying a ``TaskAction`` CTA. The reconciler is
deterministic and idempotent: it recomputes the provider's desired action set
from current state, upserts (dedup-keyed, first-write-wins), and completes any
active action of the provider's ``source_kind`` whose blocker no longer exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from iris_harness.foundation.process_state import track_globals

from .models import SourceKind, Task, TaskAction
from .store import TaskStore

# Live (non-terminal) statuses — terminal tasks (done/dropped) are never re-touched.
logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = ("open", "doing")


@dataclass(frozen=True)
class DesiredAction:
    """One pending action the current domain state calls for."""

    dedup_key: str
    title: str
    description: str
    source_id: str
    action: TaskAction


@dataclass(frozen=True)
class PendingActionsSummary:
    """Outcome of one reconciliation pass."""

    raised: int = 0  # newly created action tasks
    resolved: int = 0  # tasks completed because their blocker cleared
    open_total: int = 0  # action tasks currently desired (raised or pre-existing)

    def __str__(self) -> str:
        return (
            f"pending actions: {self.raised} raised, {self.resolved} resolved, "
            f"{self.open_total} open"
        )


@dataclass(frozen=True)
class PendingAction:
    """The unified read model the Action Center renders (ADR-0073 §2b).

    Comes from a persisted action-task (``origin="task"``), a synthesized Health item
    (``origin="health"``), a pending governance approval (``origin="approval"``), or the
    memory review queue (``origin="memory"``).
    They are merged only here, at the read layer — neither Health nor the approval
    queue is ever copied into the task store.
    """

    id: str  # task id, or a synthetic "health:..." / "approval:..." id
    origin: Literal["task", "health", "approval", "memory"]
    source_kind: str  # "finance-statements", "system-health", ...
    title: str
    description: str
    action: TaskAction
    created_at: datetime | None = None
    dedup_key: str | None = None  # the task's dedup key (stable per blocker); None for health
    # Self-describing surface-feedback token (issue 0028) so any channel can mark
    # the item "not useful" and suppress similar ones. Computed at read time by the
    # Action Center composition layer; None when the item has no stable key.
    feedback_ref: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "origin": self.origin,
            "source_kind": self.source_kind,
            "title": self.title,
            "description": self.description,
            "action": self.action.model_dump(mode="json"),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "feedback_ref": self.feedback_ref,
        }


def pending_action_feedback_key(pa: PendingAction) -> tuple[str, str, dict[str, str]] | None:
    """The ``(subsystem, surface_kind, dims)`` a "not useful" on this item suppresses.

    Keyed on *stable dimensions*, never the per-instance id, so feedback generalizes:
    a health alert keys on ``(kind, target)`` — matching ``health_alert_dims`` so the
    existing ``health_pending_actions`` consult honours it; a task-backed blocker keys
    on its ``dedup_key`` — matching what ``reconcile`` upserts on. Returns None when no
    stable key is available (the item then carries no feedback affordance).
    """
    if pa.origin == "approval":
        # Not suppressible, deliberately. Everything else on this list is something the
        # harness is telling the user it noticed; an approval is a run *waiting on them*,
        # and "don't show me these again" would silently strand runs.
        return None
    if pa.origin == "health":
        # id == "health:<kind>:<target>" (see health_pending_actions).
        parts = pa.id.split(":", 2)
        if len(parts) == 3 and parts[1] and parts[2]:
            return ("system", "health_alert", {"kind": parts[1], "target": parts[2]})
        return None
    if pa.dedup_key:
        return (pa.source_kind, "pending_action", {"key": pa.dedup_key})
    return None


def pending_action_from_task(task: Task) -> PendingAction:
    """Map a persisted action-task to the unified read model."""
    if task.action is None:
        raise ValueError("task has no action")
    return PendingAction(
        id=task.id,
        origin="task",
        source_kind=task.source_kind or "other",
        title=task.title,
        description=task.description,
        action=task.action,
        created_at=task.created_at,
        dedup_key=task.dedup_key,
    )


@runtime_checkable
class PendingActionProvider(Protocol):
    """A domain that surfaces actionable blockers as pending actions.

    ``source_kind`` is the slice of the task store this provider owns — the
    reconciler only ever raises/resolves tasks with this ``source_kind``, so
    providers never step on each other. ``invoke`` executes a ``safe`` action
    server-side (e.g. re-run extraction); display-only actions (``copy_command``)
    and client-navigation actions are not invoked here.
    """

    source_kind: SourceKind

    def desired_actions(self) -> list[DesiredAction]:
        """The pending actions current domain state calls for."""
        ...

    def invoke(self, task: Task) -> str:
        """Execute the task's safe action; return a human-readable result note."""
        ...


@runtime_checkable
class ChoiceActionProvider(PendingActionProvider, Protocol):
    """A provider that raises choice cards (``TaskAction.choices``) answers them here.

    Optional: a provider that never raises a choice card only implements ``invoke``.
    """

    def invoke_choice(self, task: Task, choice: str, option: str | None) -> str:
        """Apply the owner's answer to a choice card; return a result note."""
        ...


# --------------------------------------------------------------------- registry
# Providers register themselves here so the Action Center does not have to name
# them. Before this, `bootstrap.py` imported FileManagerOrganizeProvider directly
# and branched on four source_kind strings, which meant the core had to know every
# domain that might raise a blocker — the exact coupling the plugin split removes
# (OSS plan M2.6). Process-level, mirroring `health.service.register_check_provider`.
_PROVIDERS: dict[str, PendingActionProvider] = {}


def register_provider(provider: PendingActionProvider) -> None:
    """Register (or replace) the provider that owns ``provider.source_kind``."""
    _PROVIDERS[str(provider.source_kind)] = provider


def unregister_provider(source_kind: str) -> None:
    _PROVIDERS.pop(source_kind, None)


def providers() -> tuple[PendingActionProvider, ...]:
    """Every registered provider, in registration order."""
    return tuple(_PROVIDERS.values())


def provider_for(source_kind: str) -> PendingActionProvider | None:
    """The provider owning ``source_kind``, or ``None`` if nothing registered it."""
    return _PROVIDERS.get(source_kind)


def render_review(task: Task) -> str:
    """Render a ``review`` action. Surface logic, not domain logic.

    A review action only shows the description a domain already wrote; nothing is
    executed. Handling it here means an Action Center can answer one for ANY
    source kind, including a kind whose plugin is not mounted in this profile —
    previously each domain provider re-implemented the same three lines, so a
    review silently 404'd when its owner was absent.
    """
    details = task.description.strip()
    return details or f"{task.source_kind or 'This'} item {task.id} is ready for review."


def invoke_and_reconcile(
    provider: PendingActionProvider,
    task: Task,
    task_store: TaskStore,
    *,
    choice: str | None = None,
    option: str | None = None,
) -> str:
    """Execute ``task``'s safe action, then reconcile its provider's slice.

    An invoked action changes the domain state its task was derived from — a confirmed
    account is no longer a candidate, a trusted domain is no longer unknown — so the
    task set is stale the moment ``invoke`` returns. Reconciling here is what makes
    the answer leave the list. Before this, every surface that invoked an action left
    it (and its pair) open until the next background ingest tick, so "Yes, it's mine"
    reported success while the question stayed on screen.

    Every invoking surface goes through this, so none can miss the reconcile. A failed
    reconcile never loses the result of an action that already ran; the next sync
    catches up.

    A choice card is answered with ``choice`` (and ``option`` when that answer carries
    one); the answer is checked against the card before the provider sees it, and a
    ValueError says what is missing.
    """
    if task.action is None:
        raise ValueError("task has no action")
    task.action.check_answer(choice, option)
    if task.action.choices:
        if not isinstance(provider, ChoiceActionProvider):
            raise ValueError(f"{provider.source_kind} cannot answer a choice card")
        picked = next(c for c in task.action.choices if c.value == choice)
        default = task.action.options.default if task.action.options else None
        chosen_option = (option or default) if picked.needs_option else None
        note = provider.invoke_choice(task, str(choice), chosen_option)
    else:
        note = provider.invoke(task)
    try:
        reconcile(provider, task_store)
    except Exception:  # the action ran; the list catches up on the next sync
        logger.warning(
            "pending-action reconcile after invoke failed for %s",
            provider.source_kind,
            exc_info=True,
        )
    return note


def reconcile_all(task_store: TaskStore) -> dict[str, PendingActionsSummary]:
    """Reconcile every registered provider. One failing provider never stops the rest."""
    out: dict[str, PendingActionsSummary] = {}
    for provider in providers():
        try:
            out[str(provider.source_kind)] = reconcile(provider, task_store)
        except Exception:  # a blocker list is advisory, never fatal
            logger.debug(
                "pending-action reconcile failed for %s", provider.source_kind, exc_info=True
            )
    return out


def reconcile(provider: PendingActionProvider, task_store: TaskStore) -> PendingActionsSummary:
    """Recompute the provider's desired actions and reconcile the task store to
    them: complete cleared blockers, then upsert the desired set. Idempotent."""
    desired = provider.desired_actions()
    desired_keys = {d.dedup_key for d in desired}

    resolved = 0
    existing = task_store.list(source_kind=provider.source_kind, has_action=True, limit=1000)
    for t in existing:
        if t.status in _ACTIVE_STATUSES and t.dedup_key not in desired_keys:
            task_store.complete(t.id)
            resolved += 1

    raised = 0
    for d in desired:
        if task_store.get_by_dedup_key(d.dedup_key) is None:
            raised += 1
        task_store.upsert(
            dedup_key=d.dedup_key,
            title=d.title,
            description=d.description,
            source_kind=provider.source_kind,
            source_id=d.source_id,
            action=d.action,
        )

    return PendingActionsSummary(raised=raised, resolved=resolved, open_total=len(desired))


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_PROVIDERS")
