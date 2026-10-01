"""Data model for the System Health capability (ADR-0069).

A ``HealthSnapshot`` is the full set of ``HealthCheck`` verdicts at a point in
time. Checks have one of four states; ``grey`` (not configured) is informational
and never draws attention. ``alerts()`` is a *projection* of the snapshot — the
red checks needing attention — not a stored record (ADR-0069 §3).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class HealthState(str, Enum):
    """Four-state verdict for a single Health Check (ADR-0069 §2)."""

    GREEN = "green"  # healthy
    YELLOW = "yellow"  # degraded but functioning
    RED = "red"  # failed
    GREY = "grey"  # not configured — informational, never alerts

    @property
    def rank(self) -> int:
        """Severity ordering; higher = worse. Grey is below green (ignorable)."""
        return {"grey": 0, "green": 1, "yellow": 2, "red": 3}[self.value]


class CheckKind(str, Enum):
    """The class of target a Health Check describes."""

    SERVICE = "service"
    CREDENTIAL = "credential"
    HARDWARE = "hardware"
    PLUGIN = "plugin"  # a mounted plugin: loaded / degraded / failed (OSS plan decision 8)


@dataclass(frozen=True)
class Reconnect:
    """How the console can reconnect the credential a row reports.

    Declared by the plugin that owns the credential; the core and the web console
    only carry it. The console renders a Reconnect button for any row that has one
    and POSTs ``{"provider": provider, "account": account}`` to ``route`` (an
    ``account`` of ``None`` connects a new one). ``setup_route`` answers a GET with
    whether the server is set up to reconnect at all. ``group`` / ``group_label``
    gather the rows of one identity provider into one card per account, so the
    console never has to name a provider itself.
    """

    route: str
    setup_route: str
    group: str
    group_label: str
    provider: str
    label: str
    account: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "route": self.route,
            "setup_route": self.setup_route,
            "group": self.group,
            "group_label": self.group_label,
            "provider": self.provider,
            "label": self.label,
            "account": self.account,
        }


@dataclass(frozen=True)
class HealthCheck:
    """One observable condition with a verdict and the target it describes.

    ``action``, when set, is the exact remediation command for a red check
    (e.g. ``iris auth gmail login``) that the agent can run under the governed
    path. ``endpoint`` is the probed URL for service checks (shown in the UI).
    ``subject`` names the one thing under ``target`` the row is about when a target
    has several (the account address of a ``Gmail`` row), so an incident about one
    account is not closed by another account's green row (ADR-0116).
    ``fix_url``, when set, is the console page that fixes the row (a relative path
    such as ``/settings#connections``); alerts carry it beside the command.
    ``reconnect``, when set, says how the console reconnects the credential.
    """

    target: str
    kind: CheckKind
    state: HealthState
    detail: str
    endpoint: str | None = None
    action: str | None = None
    subject: str | None = None
    fix_url: str | None = None
    reconnect: Reconnect | None = None

    @property
    def key(self) -> str:
        """Stable identity across snapshots: ``target`` or ``target:subject``."""
        return f"{self.target}:{self.subject}" if self.subject else self.target

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form for the endpoint / CLI renderers."""
        return {
            "target": self.target,
            "kind": self.kind.value,
            "state": self.state.value,
            "detail": self.detail,
            "endpoint": self.endpoint,
            "action": self.action,
            "subject": self.subject,
            "fix_url": self.fix_url,
            "reconnect": self.reconnect.as_dict() if self.reconnect else None,
        }


@dataclass(frozen=True)
class HealthSnapshot:
    """All current Health Check verdicts at ``sampled_at`` (ISO-8601, UTC)."""

    checks: tuple[HealthCheck, ...]
    sampled_at: str

    def worst(self) -> HealthState:
        """The most severe state across all checks (green if empty)."""
        if not self.checks:
            return HealthState.GREEN
        return max((c.state for c in self.checks), key=lambda s: s.rank)

    def summary(self) -> str:
        """One-line tally, e.g. ``5 green, 1 yellow, 1 red``."""
        counts: dict[str, int] = {}
        for check in self.checks:
            counts[check.state.value] = counts.get(check.state.value, 0) + 1
        order = ("red", "yellow", "green", "grey")
        parts = [f"{counts[s]} {s}" for s in order if counts.get(s)]
        return ", ".join(parts) if parts else "no checks"

    def as_dict(self) -> dict[str, object]:
        """JSON-ready form: overall state, tally, every check, and the alerts."""
        return {
            "state": self.worst().value,
            "summary": self.summary(),
            "sampled_at": self.sampled_at,
            "checks": [c.as_dict() for c in self.checks],
            "alerts": [c.as_dict() for c in alerts(self)],
        }


def alerts(snapshot: HealthSnapshot) -> list[HealthCheck]:
    """Projection of the snapshot needing human attention: the red checks.

    Not persisted; recomputed on every read. An alert exists exactly as long as
    its underlying red check does, and clears when the next probe goes green
    (ADR-0069 §3). The banner renders these; credential reds carry an ``action``
    command, others are surfaced for visibility (e.g. a down service).
    """
    return [c for c in snapshot.checks if c.state is HealthState.RED]


__all__ = [
    "CheckKind",
    "HealthCheck",
    "HealthSnapshot",
    "HealthState",
    "Reconnect",
    "alerts",
]
