"""Embedded IRIS governor service used by the thin HTTP wrapper and local callers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .audit import GovernorAuditLogger
from .models import GovernorGuardDecision, GovernorGuardRequest, GovernorHealthReport
from .policy import GovernorPolicyEngine
from .rate_limiter import TokenBucketRateLimiter


class IRISGovernorService:
    """Route-aware governance service for IRIS-owned consumers."""

    def __init__(
        self,
        *,
        policy_engine: GovernorPolicyEngine,
        audit_logger: GovernorAuditLogger,
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        self.policy_engine = policy_engine
        self.audit_logger = audit_logger
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()

    @classmethod
    def from_repo_root(
        cls,
        repo_root: Path,
        *,
        policy_path: Path | None = None,
        audit_db_path: Path | None = None,
    ) -> IRISGovernorService:
        """Construct the embedded governor kernel from repository configuration."""
        return cls(
            policy_engine=GovernorPolicyEngine.from_repo_root(repo_root, policy_path=policy_path),
            audit_logger=GovernorAuditLogger.from_repo_root(repo_root, db_path=audit_db_path),
        )

    def guard(
        self,
        route: str,
        payload: Mapping[str, Any] | None = None,
        *,
        identity: Mapping[str, str | None] | None = None,
    ) -> GovernorGuardDecision:
        """Evaluate one route decision, then append it to the governor audit log.

        ``identity`` (``run_id`` / ``call_id`` / ``session_id``) is what the harness knows of
        the call the decision is for; it is stored beside the decision, never read from the
        request payload."""
        request = _coerce_guard_request(route, payload)
        matched_policy = self.policy_engine.match_route(request.route)
        decision = self.policy_engine.evaluate(request, matched_policy=matched_policy)

        if decision.allowed and matched_policy is not None:
            rate_limit = self.rate_limiter.evaluate(matched_policy.route, matched_policy.rate_limit)
            if not rate_limit.allowed:
                decision = decision.model_copy(
                    update={
                        "allowed": False,
                        "reason": f"rate limit exceeded for route '{request.route}'",
                        "retry_after_seconds": rate_limit.retry_after_seconds,
                    }
                )

        self.audit_logger.record_decision(decision, identity=identity)
        return decision

    def list_routes(self) -> tuple[str, ...]:
        """Return the configured route patterns."""
        return self.policy_engine.list_routes()

    def health_report(self) -> GovernorHealthReport:
        """Return a stable health snapshot for HTTP clients and tests."""
        routes = self.list_routes()
        return GovernorHealthReport(
            policy_version=self.policy_engine.config.version,
            route_count=len(routes),
            routes=routes,
            audit_db_path=str(self.audit_logger.db_path),
        )


def _coerce_guard_request(
    route: str,
    payload: Mapping[str, Any] | None,
) -> GovernorGuardRequest:
    """Normalize a guard request to the shared kernel model."""
    body = dict(payload or {})
    action_value = str(body.get("action") or route.replace("/", "_")).strip()
    if not action_value:
        action_value = route.replace("/", "_")

    if "metadata" in body:
        raw_metadata = body.get("metadata")
        if raw_metadata is None:
            metadata: dict[str, Any] = {}
        elif isinstance(raw_metadata, Mapping):
            metadata = dict(raw_metadata)
        else:
            raise ValueError("guard payload field 'metadata' must be a mapping")
    else:
        metadata = {
            key: value for key, value in body.items() if key not in {"action", "cost_estimate_usd"}
        }

    cost_estimate = body.get("cost_estimate_usd")
    return GovernorGuardRequest(
        route=route,
        action=action_value,
        metadata=metadata,
        cost_estimate_usd=float(cost_estimate) if cost_estimate is not None else None,
    )
