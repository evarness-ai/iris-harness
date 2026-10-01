"""Credential Health Checks -- local-only (ADR-0069 slice 2).

The core's own credentials -- cloud-LLM API-key *presence* and the audit key -- plus
every credential check a plugin registers (``api.register_credential_check``), as
``HealthCheck``s. Zero egress by default: keys are checked for presence, never
validated with a paid call (ADR-0069 §4, the Q2 "hybrid" decision); a registered check
may live-probe only when it is handed ``net_probe=True`` (the opt-in slice-5 probe).

The core names no provider. An OAuth token's rows are the owning plugin's: each
registers one check, and the rows it returns are what System Health shows -- the
core does not reshape them. A check that raises is reported, not dropped: a yellow row
under its name, and (through the plugin fault boundary) a failure charged to the plugin.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from threading import Lock

from iris_harness.foundation.process_state import track_globals
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthState

logger = logging.getLogger(__name__)

# Cloud-LLM keys: presence-only. Any of the aliased env vars counts as present.
_CLOUD_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Anthropic", ("ANTHROPIC_API_KEY",)),
    ("OpenRouter", ("OPENROUTER_API_KEY",)),
    ("GitHub", ("GITHUB_TOKEN", "GITHUB_PAT_CODING_AGENT")),
)

#: ``check(net_probe) -> rows``: one plugin-owned credential's System Health rows.
CredentialCheck = Callable[[bool], list[HealthCheck]]

# Keyed, not appended, so a process that builds a second runtime (the playground does)
# replaces a plugin's check instead of reporting its rows twice. Registration order is
# the order the rows appear in, after the core's own.
_lock = Lock()
_registered: dict[str, CredentialCheck] = {}


def register_credential_check(key: str, check: CredentialCheck) -> None:
    """Install (or replace) the credential check ``key``; its rows join every snapshot."""
    with _lock:
        _registered[key] = check


def unregister_credential_check(key: str) -> None:
    with _lock:
        _registered.pop(key, None)


def clear_credential_checks() -> None:
    """Drop every registered check (tests)."""
    with _lock:
        _registered.clear()


def registered_credential_checks(*, net_probe: bool = False) -> list[HealthCheck]:
    """Every registered check's rows, in registration order.

    A check that raises gives one yellow row under its key instead of its rows: the
    owner sees that a credential could not be checked, rather than the row vanishing.
    """
    with _lock:
        checks = list(_registered.items())
    rows: list[HealthCheck] = []
    for key, check in checks:
        try:
            rows.extend(check(net_probe))
        except Exception as exc:
            logger.warning("credential check %r failed", key, exc_info=True)
            rows.append(
                HealthCheck(
                    key,
                    CheckKind.CREDENTIAL,
                    HealthState.YELLOW,
                    f"check failed: {type(exc).__name__}: {exc}",
                )
            )
    return rows


def cloud_key_checks(env: Mapping[str, str] | None = None) -> list[HealthCheck]:
    """Cloud-LLM API keys: present (green) or not set (grey). Never validated."""
    environ = env if env is not None else os.environ
    checks: list[HealthCheck] = []
    for label, keys in _CLOUD_KEYS:
        present = any(environ.get(k) for k in keys)
        checks.append(
            HealthCheck(
                label,
                CheckKind.CREDENTIAL,
                HealthState.GREEN if present else HealthState.GREY,
                "API key present" if present else "no key set",
            )
        )
    return checks


def audit_key_checks() -> list[HealthCheck]:
    """The audit key every governed call needs (``kernel/governance/audit/digest.py``).

    Reads where the key stands without resolving it -- a health check must not open the
    OS keyring (a Keychain dialog) on its own: red once a governed call found no master
    key (every tool call is refused until one is set), green once one was resolved, and no
    row before any governed call asked.
    """
    from iris_harness.kernel.governance.audit.digest import (
        NO_AUDIT_KEY_MESSAGE,
        audit_key_status,
    )

    state, detail = audit_key_status()
    if state == "unresolved":
        return []
    if state == "ready":
        return [HealthCheck("Audit key", CheckKind.CREDENTIAL, HealthState.GREEN, detail)]
    return [
        HealthCheck(
            "Audit key",
            CheckKind.CREDENTIAL,
            HealthState.RED,
            f"{NO_AUDIT_KEY_MESSAGE} ({detail})",
        )
    ]


def credential_checks(
    *,
    net_probe: bool = False,
    env: Mapping[str, str] | None = None,
) -> list[HealthCheck]:
    """All credential checks: the core's own, then every registered one. With
    ``net_probe``, the registered checks may live-probe for revocation (opt-in; the only
    piece that makes new outbound calls)."""
    checks: list[HealthCheck] = []
    checks.extend(cloud_key_checks(env))
    checks.extend(audit_key_checks())
    checks.extend(registered_credential_checks(net_probe=net_probe))
    return checks


__all__ = [
    "CredentialCheck",
    "audit_key_checks",
    "clear_credential_checks",
    "cloud_key_checks",
    "credential_checks",
    "register_credential_check",
    "registered_credential_checks",
    "unregister_credential_check",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_registered")
