"""Core models for the IRIS governor kernel."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def governor_utc_now() -> datetime:
    """Return the current UTC timestamp for governor audit records."""
    return datetime.now(UTC)


class RateLimitPolicy(BaseModel):
    """Simple token-bucket policy for one governed route."""

    model_config = ConfigDict(frozen=True)

    requests: int = Field(..., ge=1)
    window_seconds: int = Field(..., ge=1)


class GovernorRoutePolicy(BaseModel):
    """Policy rule for one exact or wildcard route pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    route: str = Field(..., min_length=1)
    description: str = Field(default="", min_length=0)
    allowed_actions: tuple[str, ...] = Field(default_factory=tuple)
    requires_approval: bool = Field(default=False)
    rate_limit: RateLimitPolicy | None = Field(default=None)


class GovernorPolicyConfig(BaseModel):
    """Top-level configuration loaded from config/governor/policy.yaml."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    version: str = Field(default="1", min_length=1)
    routes: tuple[GovernorRoutePolicy, ...] = Field(default_factory=tuple)


class GovernorGuardRequest(BaseModel):
    """Validated request payload for an IRIS guard decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    route: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    cost_estimate_usd: float | None = Field(default=None, ge=0)


class GovernorGuardDecision(BaseModel):
    """Structured decision returned from the governor kernel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    route: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    allowed: bool = Field(default=False)
    reason: str = Field(default="", min_length=0)
    matched_policy: str | None = Field(default=None)
    requires_approval: bool = Field(default=False)
    retry_after_seconds: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=governor_utc_now)


class GovernorAuditEntry(BaseModel):
    """Append-only governor audit row."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    event_id: int = Field(..., ge=1)
    route: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    allowed: bool = Field(default=False)
    reason: str = Field(default="", min_length=0)
    matched_policy: str | None = Field(default=None)
    requires_approval: bool = Field(default=False)
    retry_after_seconds: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=governor_utc_now)


class GovernorHealthReport(BaseModel):
    """Health and configuration snapshot for the thin HTTP wrapper."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    status: str = Field(default="ok", min_length=1)
    policy_version: str = Field(default="1", min_length=1)
    route_count: int = Field(default=0, ge=0)
    routes: tuple[str, ...] = Field(default_factory=tuple)
    audit_db_path: str = Field(..., min_length=1)
