"""The Action Center: pending actions, activities and the owner's tasks (ADR-0073).

    GET    /actions
    GET    /activities
    GET    /tasks
    POST   /tasks
    PATCH  /tasks/{task_id}
    POST   /actions/{action_id}/invoke

Moved out of ``create_app`` unchanged (review item: split the god function); the route
table and OpenAPI schema are identical before and after. The write guard in ``main``
still gates the mutating routes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, cast, get_args

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from iris_harness.services.activities.models import ActivityStatus
from iris_harness.services.tasks import SourceKind, TaskStatus

# The Literal the store accepts, as a runtime set: a query string is not a Literal, and
# passing one through unchecked let an unknown ?status= return every row instead of 400.
_ACTIVITY_STATUSES = frozenset(get_args(ActivityStatus))


class ActionAnswerRequest(BaseModel):
    """Optional body for ``POST /actions/{id}/invoke``: the answer to a choice card."""

    choice: str | None = Field(default=None, min_length=1, max_length=40)
    option: str | None = Field(default=None, min_length=1, max_length=40)


class TaskCreateRequest(BaseModel):
    """Request body for ``POST /tasks``.

    Deliberately narrow. ``Task`` carries twenty fields — dedup keys, wait-for
    conditions, parent goals, calendar links, a remediation ``action`` — and
    every one of them is set by the subsystem that owns it. A person typing a
    todo supplies three, and ``source_kind`` is not one of them: it is forced
    to ``manual`` below, so this route cannot mint a task that claims to have
    come from the finance pipeline.
    """

    # Strip at the boundary, exactly as `Task` does. Without it a title of
    # "   " is three valid characters here, and `Task` — which strips — then
    # rejects the empty result inside the store, turning a bad request into a
    # 500. Validation belongs where the request arrives.
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(default="", max_length=2000)
    due_at: datetime | None = None


class TaskUpdateRequest(BaseModel):
    """Request body for ``PATCH /tasks/{task_id}``."""

    status: TaskStatus


def _pending_action_provider(source_kind: str | None) -> Any:
    """The pending-action provider that owns ``source_kind`` (ADR-0073), or None.

    Every kind comes from the shared registry, filled by the plugins the runtime
    mounted: file_organizer registers organize, photos-albums, rag-ingest and
    filemanager-quarantine, and finance_workflows registers finance-statements (OSS
    plan M2.6 / M4.2 / M6.1b). Naming any of them here would put back the coupling
    the registry removed — and would answer for a plugin this profile has not mounted."""
    from iris_harness.services.tasks.pending_actions import provider_for

    return provider_for(str(source_kind))


def install_action_center_routes(app: FastAPI, runtime: Callable[[], Any]) -> None:
    """Register these routes. ``runtime`` returns the live runtime or raises 503."""

    # ── Action Center (ADR-0073) ───────────────────────────────────────────────

    @app.get("/actions")
    def list_actions() -> dict[str, Any]:
        """Unified Action Center: persisted pending-action tasks unioned with
        Health-synthesized actionable items (ADR-0073 §2b). Pure read — Health is
        recomputed on read, never copied into the task store."""
        from iris_harness.runtime.action_center import collect_pending_actions
        from iris_harness.services.health.service import current_snapshot
        from iris_harness.services.tasks import TaskStore

        rt = runtime()
        ts = TaskStore(db_path=rt.data_dir / "tasks.db")
        ts.ensure_schema()
        items = collect_pending_actions(
            ts, current_snapshot(), memory_store=getattr(rt, "memory_store", None)
        )
        return {"count": len(items), "actions": [a.as_dict() for a in items]}

    @app.get("/activities")
    def list_activities(status: str | None = None, limit: int = 100) -> dict[str, Any]:
        """Async Activity feed: durable records of background system jobs
        (FileManager categorize/cleanup/organize, later heartbeats/routines).

        Pure read from ``activities.db`` — the ActivityRunner (in-process) is the
        sole writer. The web feed + badge poll this; ``running`` + unseen
        ``completed`` drive the badge count. See
        docs/architecture/async-activities-and-notifications.md."""
        from iris_harness.services.activities import ActivityStore

        rt = runtime()
        store = ActivityStore(db_path=rt.data_dir / "activities.db")
        store.ensure_schema()
        if status is not None and status not in _ACTIVITY_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown status {status!r}; expected one of "
                + ", ".join(sorted(_ACTIVITY_STATUSES)),
            )
        items = store.list(
            status=cast("ActivityStatus | None", status),
            limit=max(1, min(limit, 500)),
        )
        running = sum(1 for a in items if a.status == "running")
        return {
            "count": len(items),
            "running": running,
            "activities": [a.model_dump(mode="json") for a in items],
        }

    @app.get("/tasks")
    def list_tasks(
        status: TaskStatus | None = None,
        source_kind: SourceKind | None = None,
    ) -> dict[str, Any]:
        """All tasks (the per-agent console + a future todo view reuse this)."""
        from iris_harness.services.tasks import TaskStore

        rt = runtime()
        ts = TaskStore(db_path=rt.data_dir / "tasks.db")
        ts.ensure_schema()
        tasks = ts.list(status=status, source_kind=source_kind, limit=500)
        return {"count": len(tasks), "tasks": [t.model_dump(mode="json") for t in tasks]}

    @app.post("/tasks", status_code=201)
    def create_task(request: TaskCreateRequest) -> dict[str, Any]:
        """Add a todo (Track 2 PR 7, plan decision 34). Write-gated.

        The store already emits ``task.created``, so anything subscribed to
        that — the outstanding-items rollup, the Action Center count — picks
        this up without the route telling them.
        """
        from iris_harness.services.tasks import TaskStore

        rt = runtime()
        ts = TaskStore(db_path=rt.data_dir / "tasks.db")
        ts.ensure_schema()
        task = ts.create(
            title=request.title,
            description=request.description,
            due_at=request.due_at,
            source_kind="manual",
        )
        return {"task": task.model_dump(mode="json")}

    @app.patch("/tasks/{task_id}")
    def update_task(task_id: str, request: TaskUpdateRequest) -> dict[str, Any]:
        """Update task status from the Web UI (chat/task controls). Write-gated.

        ``done`` and ``dropped`` use dedicated store methods so task lifecycle
        events remain consistent; ``open``/``doing`` clear ``completed_at``.
        """
        from iris_harness.foundation.eventbus import get_default_bus
        from iris_harness.services.tasks import TaskStore

        rt = runtime()
        # The process bus, so a plugin can answer a completion (finance closes the
        # due a done task names) without this route naming the plugin.
        ts = TaskStore(db_path=rt.data_dir / "tasks.db", bus=get_default_bus())
        ts.ensure_schema()
        if ts.get(task_id) is None:
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
        try:
            if request.status == "done":
                updated = ts.complete(task_id)
            elif request.status == "dropped":
                updated = ts.drop(task_id)
            else:
                updated = ts.update(task_id, status=request.status, completed_at=None)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return updated.model_dump(mode="json")

    @app.post("/actions/{action_id}/invoke")
    def invoke_action(action_id: str, answer: ActionAnswerRequest | None = None) -> dict[str, Any]:
        """Execute a safe pending action (ADR-0073). Write-gated; dispatched to the
        owning provider by source_kind. Display-only actions (copy_command, and all
        Health-derived items) reject invocation — the user runs their command. A
        choice card is answered with ``{"choice": ..., "option": ...}`` (ADR-0121)."""
        from iris_harness.services.tasks import TaskStore

        rt = runtime()
        ts = TaskStore(db_path=rt.data_dir / "tasks.db")
        ts.ensure_schema()
        task = ts.get(action_id)
        if task is None or task.action is None:
            raise HTTPException(status_code=404, detail="no such pending action")
        if not task.action.safe:
            raise HTTPException(
                status_code=400,
                detail="this action is display-only; run its command locally",
            )
        # A review action just shows what the domain already wrote, so the Action
        # Center answers it itself — no provider, and no dependence on the owning
        # plugin being mounted in this profile (OSS plan M2.6).
        if task.action.kind == "review":
            from iris_harness.services.tasks.pending_actions import render_review

            return {"id": action_id, "result": render_review(task)}
        provider = _pending_action_provider(task.source_kind)
        if provider is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"no invoker for source_kind={task.source_kind}; the plugin that "
                    "owns it may not be mounted in this profile"
                ),
            )
        from iris_harness.services.tasks.pending_actions import invoke_and_reconcile

        try:
            note = invoke_and_reconcile(
                provider,
                task,
                ts,
                choice=answer.choice if answer else None,
                option=answer.option if answer else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"id": action_id, "result": note}
