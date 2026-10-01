"""Routines: list, author, approve, edit, run (the routine authoring API).

    GET    /routines
    GET    /routines/due
    POST   /routines
    PATCH  /routines/{routine_id}
    DELETE /routines
    DELETE /routines/{routine_id}
    POST   /routines/tick
    POST   /routines/{routine_id}/run
    POST   /routines/{routine_id}/preview

Moved out of ``create_app`` unchanged (review item: split the god function); the route
table and OpenAPI schema are identical before and after. The write guard in ``main``
still gates the mutating routes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from iris_harness.services.routines import RoutineApprovalStatus, RoutineSpec, create_routine_spec


class RoutineCreateRequest(BaseModel):
    """Request body for ``POST /routines``."""

    title: str = Field(..., min_length=1, max_length=200)
    goal: str | None = Field(default=None, min_length=1, max_length=1000)
    schedule: str = Field(..., min_length=1, max_length=128)
    template: str = Field(..., min_length=1, max_length=128)
    delivery_channel: str = Field(default="console", min_length=1, max_length=64)
    source_preferences: tuple[str, ...] = Field(default_factory=tuple)
    required_capabilities: tuple[str, ...] = Field(default_factory=tuple)
    approval_status: RoutineApprovalStatus = RoutineApprovalStatus.DRAFT
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutineUpdateRequest(BaseModel):
    """Request body for ``PATCH /routines/{routine_id}``."""

    title: str | None = Field(default=None, min_length=1, max_length=200)
    goal: str | None = Field(default=None, min_length=1, max_length=1000)
    schedule: str | None = Field(default=None, min_length=1, max_length=128)
    template: str | None = Field(default=None, min_length=1, max_length=128)
    delivery_channel: str | None = Field(default=None, min_length=1, max_length=64)
    source_preferences: tuple[str, ...] | None = None
    required_capabilities: tuple[str, ...] | None = None
    approval_status: RoutineApprovalStatus | None = None
    promotion_candidate: bool | None = None
    metadata: dict[str, Any] | None = None


def _routine_payload(spec: RoutineSpec) -> dict[str, Any]:
    return spec.model_dump(mode="json")


def install_routines_routes(app: FastAPI, runtime: Callable[[], Any]) -> None:
    """Register these routes. ``runtime`` returns the live runtime or raises 503."""

    @app.get("/routines")
    def list_routines(status: RoutineApprovalStatus | None = None) -> dict[str, Any]:
        rt = runtime()
        routines = (
            rt.routine_store.list_by_status(status)
            if status is not None
            else rt.routine_store.list_all()
        )
        return {
            "count": len(routines),
            "routines": [_routine_payload(item) for item in routines],
            "routine_store": str(rt.routine_store.db_path),
        }

    @app.get("/routines/due")
    def list_due_routines() -> dict[str, Any]:
        rt = runtime()
        try:
            routines = rt.routine_store.list_due()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "count": len(routines),
            "routines": [_routine_payload(item) for item in routines],
            "routine_store": str(rt.routine_store.db_path),
        }

    @app.post("/routines")
    def create_routine(request: RoutineCreateRequest) -> dict[str, Any]:
        rt = runtime()
        spec = create_routine_spec(
            title=request.title,
            goal=request.goal or request.title,
            schedule=request.schedule,
            template=request.template,
            delivery_channel=request.delivery_channel,
            source_preferences=request.source_preferences,
            required_capabilities=request.required_capabilities,
            approval_status=request.approval_status,
            metadata=request.metadata,
        )
        return {"routine": _routine_payload(rt.routine_store.save(spec))}

    @app.patch("/routines/{routine_id}")
    def update_routine(routine_id: str, request: RoutineUpdateRequest) -> dict[str, Any]:
        rt = runtime()
        spec = rt.routine_store.load(routine_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"routine '{routine_id}' not found")
        updates = request.model_dump(exclude_unset=True)
        if not updates:
            return {"routine": _routine_payload(spec)}
        updates["updated_at"] = datetime.now(UTC)
        updated = spec.model_copy(update=updates)
        return {"routine": _routine_payload(rt.routine_store.save(updated))}

    @app.delete("/routines")
    def clear_routines() -> dict[str, Any]:
        rt = runtime()
        routines = rt.routine_store.list_all()
        deleted_count = rt.routine_store.clear()
        return {
            "deleted_count": deleted_count,
            "count": deleted_count,
            "deleted_routines": [_routine_payload(item) for item in routines],
            "routine_store": str(rt.routine_store.db_path),
        }

    @app.delete("/routines/{routine_id}")
    def remove_routine(routine_id: str) -> dict[str, Any]:
        rt = runtime()
        spec = rt.routine_store.load(routine_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"routine '{routine_id}' not found")
        deleted = rt.routine_store.delete(routine_id)
        return {"deleted": deleted, "deleted_routine": _routine_payload(spec)}

    @app.post("/routines/tick")
    def tick_routines() -> dict[str, Any]:
        rt = runtime()
        run = rt.heartbeats.trigger_by_name("routine_tick")
        if run is None:
            raise HTTPException(status_code=404, detail="heartbeat 'routine_tick' not found")
        return {
            "name": run.name,
            "status": run.status.value,
            "output": run.output,
            "error": run.error,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        }

    @app.post("/routines/{routine_id}/run")
    def run_routine(routine_id: str) -> dict[str, Any]:
        """Execute a single routine now, regardless of its schedule."""
        rt = runtime()
        record = rt.run_routine(routine_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"routine '{routine_id}' not found")
        return {
            "routine_id": record.routine_id,
            "title": record.title,
            "template": record.template,
            "status": str(record.status),
            "detail": record.detail,
            "heartbeat_status": record.heartbeat_status,
        }

    @app.post("/routines/{routine_id}/preview")
    def preview_routine(routine_id: str) -> dict[str, Any]:
        """Render a routine's output for review — no delivery, no counters."""
        rt = runtime()
        spec = rt.routine_store.load(routine_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"routine '{routine_id}' not found")
        try:
            body = rt.preview_routine(routine_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"routine_id": routine_id, "title": spec.title, "body": body}
