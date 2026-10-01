"""IRIS-owned governance kernel for route authorization and audit."""

from .audit import DEFAULT_AUDIT_DB_PATH, GovernorAuditLogger
from .exceptions import (
    GovernorAuditError,
    GovernorError,
    GovernorPolicyError,
    GovernorRouteNotConfigured,
)
from .models import (
    GovernorAuditEntry,
    GovernorGuardDecision,
    GovernorGuardRequest,
    GovernorHealthReport,
    GovernorPolicyConfig,
    GovernorRoutePolicy,
    RateLimitPolicy,
)
from .policy import DEFAULT_POLICY_PATH, GovernorPolicyEngine, load_governor_policy, route_matches
from .rate_limiter import TokenBucketRateLimiter
from .service import IRISGovernorService

__all__ = [
    "DEFAULT_AUDIT_DB_PATH",
    "DEFAULT_POLICY_PATH",
    "GovernorAuditEntry",
    "GovernorAuditError",
    "GovernorAuditLogger",
    "GovernorError",
    "GovernorGuardDecision",
    "GovernorGuardRequest",
    "GovernorHealthReport",
    "GovernorPolicyConfig",
    "GovernorPolicyEngine",
    "GovernorPolicyError",
    "GovernorRouteNotConfigured",
    "GovernorRoutePolicy",
    "IRISGovernorService",
    "RateLimitPolicy",
    "TokenBucketRateLimiter",
    "load_governor_policy",
    "route_matches",
]
