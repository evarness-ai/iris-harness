"""Policy loading and route evaluation for the IRIS governor."""

from __future__ import annotations

from collections.abc import Mapping
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

from .exceptions import GovernorPolicyError
from .models import (
    GovernorGuardDecision,
    GovernorGuardRequest,
    GovernorPolicyConfig,
    GovernorRoutePolicy,
)

DEFAULT_POLICY_PATH = Path("config/governor/policy.yaml")


def load_governor_policy(
    repo_root: Path,
    *,
    policy_path: Path | None = None,
) -> GovernorPolicyConfig:
    """Load the IRIS governor policy from disk."""
    resolved_policy_path = (policy_path or (repo_root / DEFAULT_POLICY_PATH)).resolve()
    if not resolved_policy_path.exists():
        raise GovernorPolicyError(f"governor policy file does not exist: {resolved_policy_path}")

    payload = yaml.safe_load(resolved_policy_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise GovernorPolicyError("governor policy must decode to a mapping")
    try:
        return GovernorPolicyConfig.model_validate(payload)
    except Exception as exc:  # pragma: no cover - pydantic error formatting
        raise GovernorPolicyError(f"invalid governor policy: {exc}") from exc


def route_matches(pattern: str, route: str) -> bool:
    """Return whether an exact or wildcard policy pattern matches a route."""
    if pattern == route:
        return True
    if "*" not in pattern:
        return False
    return fnmatchcase(route, pattern)


class GovernorPolicyEngine:
    """Policy engine for route matching and pre-rate-limit authorization."""

    def __init__(self, config: GovernorPolicyConfig) -> None:
        self.config = config

    @classmethod
    def from_repo_root(
        cls,
        repo_root: Path,
        *,
        policy_path: Path | None = None,
    ) -> GovernorPolicyEngine:
        """Construct a policy engine from repository configuration."""
        return cls(load_governor_policy(repo_root, policy_path=policy_path))

    def list_routes(self) -> tuple[str, ...]:
        """Return configured route patterns in stable order."""
        return tuple(route.route for route in self.config.routes)

    def match_route(self, route: str) -> GovernorRoutePolicy | None:
        """Return the first configured route policy that matches the request route."""
        for policy in self.config.routes:
            if route_matches(policy.route, route):
                return policy
        return None

    def evaluate(
        self,
        request: GovernorGuardRequest,
        *,
        matched_policy: GovernorRoutePolicy | None = None,
    ) -> GovernorGuardDecision:
        """Evaluate route registration, action constraints, and approval gates."""
        policy = matched_policy or self.match_route(request.route)
        if policy is None:
            return GovernorGuardDecision(
                route=request.route,
                action=request.action,
                allowed=False,
                reason=f"route '{request.route}' is not configured in the IRIS governor policy",
                metadata=request.metadata,
            )

        if policy.allowed_actions and request.action not in policy.allowed_actions:
            return GovernorGuardDecision(
                route=request.route,
                action=request.action,
                allowed=False,
                reason=(f"action '{request.action}' is not allowed for route '{request.route}'"),
                matched_policy=policy.route,
                requires_approval=policy.requires_approval,
                metadata=request.metadata,
            )

        if policy.requires_approval and not _has_approval_flag(request.metadata):
            return GovernorGuardDecision(
                route=request.route,
                action=request.action,
                allowed=False,
                reason=(f"route '{request.route}' requires metadata.approval_granted=true"),
                matched_policy=policy.route,
                requires_approval=True,
                metadata=request.metadata,
            )

        return GovernorGuardDecision(
            route=request.route,
            action=request.action,
            allowed=True,
            reason="Allowed",
            matched_policy=policy.route,
            requires_approval=policy.requires_approval,
            metadata=request.metadata,
        )


def _has_approval_flag(metadata: Mapping[str, Any]) -> bool:
    """Return whether the guard metadata carries an explicit approval flag."""
    return bool(metadata.get("approval_granted", False) or metadata.get("approved", False))
