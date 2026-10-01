"""IRIS governor exception hierarchy."""

from __future__ import annotations


class GovernorError(RuntimeError):
    """Base exception for IRIS governor failures."""


class GovernorPolicyError(GovernorError):
    """Raised when governor policy configuration is invalid."""


class GovernorRouteNotConfigured(GovernorPolicyError):
    """Raised when a guarded route is missing from policy."""


class GovernorAuditError(GovernorError):
    """Raised when the append-only audit store cannot be written."""
