"""Pydantic contracts for recurring IRIS routines."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from iris_harness.foundation.clock import utc_now


class RoutineApprovalStatus(StrEnum):
    """Lifecycle state for a routine specification."""

    DRAFT = "draft"
    CLARIFY = "clarify"
    TEMPLATE = "template"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    PAUSED = "paused"
    RETIRED = "retired"


class RoutineApprovalRequestStatus(StrEnum):
    """Lifecycle state for a user-facing routine approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class RoutineExecutionStatus(StrEnum):
    """Outcome for one routine execution attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


def _slugify(value: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug or "routine")[:max_length].strip("-") or "routine"


def new_routine_id(title: str, *, created_at: datetime | None = None) -> str:
    """Return a stable-looking routine id with a readable title slug."""

    stamp = (created_at or utc_now()).strftime("%Y%m%d%H%M%S")
    return f"routine-{stamp}-{_slugify(title)}-{uuid.uuid4().hex[:8]}"


def new_routine_approval_request_id(*, created_at: datetime | None = None) -> str:
    """Return a stable-looking approval request id."""

    stamp = (created_at or utc_now()).strftime("%Y%m%d%H%M%S")
    return f"routine-approval-{stamp}-{uuid.uuid4().hex[:8]}"


class RoutineFormatting(BaseModel):
    """Per-routine presentation knobs (roadmap "next slice", 2026-06-26).

    A typed view over the render keys the tick handler already forwards
    (``_routine_render_params`` -> ``render_brief_package``): section ``order``, a
    ``header``/``footer`` line, and ``content_lines_per_item``. ``tone`` is captured for
    future use. Round-trips to/from a ``RoutineSpec.metadata`` dict, so this model is the
    authoring-side schema for what was previously loose metadata keys.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    section_order: tuple[str, ...] = Field(default_factory=tuple)
    header: str | None = None
    footer: str | None = None
    content_lines_per_item: int | None = Field(default=None, ge=1, le=10)
    tone: str | None = None  # "brief" | "formal" | "casual" | …

    def is_empty(self) -> bool:
        return not (
            self.section_order
            or self.header
            or self.footer
            or self.content_lines_per_item is not None
            or self.tone
        )

    def to_metadata(self) -> dict[str, Any]:
        """Only the set keys, using the names ``_routine_render_params`` reads."""
        out: dict[str, Any] = {}
        if self.section_order:
            out["section_order"] = list(self.section_order)
        if self.header:
            out["header"] = self.header
        if self.footer:
            out["footer"] = self.footer
        if self.content_lines_per_item is not None:
            out["content_lines_per_item"] = self.content_lines_per_item
        if self.tone:
            out["tone"] = self.tone
        return out

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> RoutineFormatting:
        return cls(
            section_order=tuple(metadata.get("section_order") or ()),
            header=metadata.get("header"),
            footer=metadata.get("footer"),
            content_lines_per_item=metadata.get("content_lines_per_item"),
            tone=metadata.get("tone"),
        )


class RoutineSpec(BaseModel):
    """Durable description of when, why, and how IRIS should run a routine."""

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    schema_version: str = Field(default="routine-spec/v1", frozen=True)
    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    goal: str = Field(..., min_length=1)
    schedule: str = Field(..., min_length=1)
    template: str = Field(..., min_length=1)
    source_preferences: tuple[str, ...] = Field(default_factory=tuple)
    required_capabilities: tuple[str, ...] = Field(default_factory=tuple)
    delivery_channel: str = Field(default="console", min_length=1)
    approval_status: RoutineApprovalStatus = RoutineApprovalStatus.DRAFT
    run_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    last_run_at: datetime | None = None
    promotion_candidate: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_approved_for_execution(self) -> bool:
        return self.approval_status in {
            RoutineApprovalStatus.APPROVED,
            RoutineApprovalStatus.SCHEDULED,
        }

    def with_status(self, status: RoutineApprovalStatus) -> RoutineSpec:
        return self.model_copy(update={"approval_status": status, "updated_at": utc_now()})

    def record_run(self, *, success: bool, finished_at: datetime | None = None) -> RoutineSpec:
        finished = finished_at or utc_now()
        return self.model_copy(
            update={
                "run_count": self.run_count + 1,
                "success_count": self.success_count + (1 if success else 0),
                "failure_count": self.failure_count + (0 if success else 1),
                "last_run_at": finished,
                "updated_at": finished,
            }
        )


class RoutineExecutionRecord(BaseModel):
    """One routine attempted by a routine_tick heartbeat."""

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    routine_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    template: str = Field(..., min_length=1)
    status: RoutineExecutionStatus
    detail: str = ""
    heartbeat_status: str | None = None


class RoutineApprovalRequest(BaseModel):
    """Durable request asking a user to approve or cancel a draft routine."""

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    schema_version: str = Field(default="routine-approval-request/v1", frozen=True)
    id: str = Field(..., min_length=1)
    routine_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    status: RoutineApprovalRequestStatus = RoutineApprovalRequestStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def with_status(self, status: RoutineApprovalRequestStatus) -> RoutineApprovalRequest:
        updated_at = utc_now()
        return self.model_copy(
            update={
                "status": status,
                "updated_at": updated_at,
                "resolved_at": (
                    None if status == RoutineApprovalRequestStatus.PENDING else updated_at
                ),
            }
        )


class RoutineTickResult(BaseModel):
    """Summary returned by routine due processing."""

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    schema_version: str = Field(default="routine-tick/v1", frozen=True)
    checked_at: datetime
    due_count: int = Field(..., ge=0)
    executed_count: int = Field(..., ge=0)
    success_count: int = Field(..., ge=0)
    failure_count: int = Field(..., ge=0)
    skipped_count: int = Field(..., ge=0)
    executions: tuple[RoutineExecutionRecord, ...] = Field(default_factory=tuple)


def create_routine_spec(
    *,
    title: str,
    goal: str,
    schedule: str,
    template: str,
    delivery_channel: str = "console",
    source_preferences: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    approval_status: RoutineApprovalStatus = RoutineApprovalStatus.DRAFT,
    metadata: dict[str, Any] | None = None,
) -> RoutineSpec:
    """Create a new routine with generated id and Pydantic validation."""

    created_at = utc_now()
    return RoutineSpec(
        id=new_routine_id(title, created_at=created_at),
        title=title,
        goal=goal,
        schedule=schedule,
        template=template,
        delivery_channel=delivery_channel,
        source_preferences=source_preferences,
        required_capabilities=required_capabilities,
        approval_status=approval_status,
        created_at=created_at,
        updated_at=created_at,
        metadata=metadata or {},
    )


def create_routine_approval_request(
    *,
    routine_id: str,
    session_id: str,
    prompt: str,
    metadata: dict[str, Any] | None = None,
) -> RoutineApprovalRequest:
    """Create a pending approval request for a draft routine."""

    created_at = utc_now()
    return RoutineApprovalRequest(
        id=new_routine_approval_request_id(created_at=created_at),
        routine_id=routine_id,
        session_id=session_id,
        prompt=prompt,
        created_at=created_at,
        updated_at=created_at,
        metadata=metadata or {},
    )
