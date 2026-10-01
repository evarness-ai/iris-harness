"""Remote evaluator client (story 12.gov-3.9).

Speaks the same JSON shape as ``src/iris_harness/server/evaluator/main.py``:
``POST /evaluate`` with a ``StepRecord`` body returns
``{signals: [SignalResult, ...]}``. The client exposes the same
``evaluate(step) -> tuple[SignalResult, ...]`` signature as the
in-process ``EvaluatorRegistry`` so the ``EvaluatorHook`` can be
mode-agnostic.

Posture on service unavailability (AC-3):

- caller's run was on ``tier_3`` (cloud)  → fail-closed (return a synthetic
  ``halt`` SignalResult so the kernel denies the step)
- caller's run was on ``tier_1`` / ``tier_2`` (local) → fail-open
  (return a synthetic ``warn`` so the kernel allows the step but
  records the gap in audit)

This matches §4.1's posture: cloud-route disruption is a security
event; local-route disruption is a reliability annoyance.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from iris_harness.foundation.auth import auth_headers
from iris_harness.foundation.observability.logging_setup import log_egress
from iris_harness.kernel.governance.evaluator.types import SignalResult, StepRecord
from iris_harness.kernel.governance.hooks.types import LLMTier

logger = logging.getLogger(__name__)


class RemoteEvaluatorClient:
    """HTTP client matching the in-process ``EvaluatorRegistry`` surface."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout_s,
            transport=transport,
            headers=auth_headers(),
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    def evaluate(self, step: StepRecord) -> tuple[SignalResult, ...]:
        """Call ``POST /evaluate``. On error returns a synthetic single result."""
        try:
            log_egress(
                destination=urlparse(self._base_url).netloc or "evaluator",
                method="POST",
                kind="service",
                purpose="evaluator",
            )
            response = self._client.post("/evaluate", json=step.model_dump(mode="json"))
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "remote evaluator unavailable at %s: %s; applying fail-%s for tier=%s",
                self._base_url,
                exc,
                "closed" if step.tier == "tier_3" else "open",
                step.tier,
            )
            return (_unavailable_result(step.tier, str(exc)),)

        raw_signals = body.get("signals") if isinstance(body, dict) else None
        if not isinstance(raw_signals, list):
            logger.warning(
                "remote evaluator returned malformed body: %r; failing %s",
                body,
                "closed" if step.tier == "tier_3" else "open",
            )
            return (_unavailable_result(step.tier, "malformed response"),)

        try:
            return tuple(SignalResult.model_validate(r) for r in raw_signals)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "remote evaluator response parse error: %s; failing %s",
                exc,
                "closed" if step.tier == "tier_3" else "open",
            )
            return (_unavailable_result(step.tier, str(exc)),)

    def reset_run_state(self, run_id: str) -> None:
        """Best-effort reset request; failures are logged but never raised."""
        try:
            self._client.post(f"/reset/{run_id}").raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("remote evaluator reset failed for run=%s: %s", run_id, exc)

    def signal_count(self) -> int:
        """Probe ``/healthz`` for diagnostics. Returns 0 on error."""
        try:
            log_egress(
                destination=urlparse(self._base_url).netloc or "evaluator",
                method="GET",
                kind="service",
                purpose="evaluator.health",
            )
            response = self._client.get("/healthz")
            response.raise_for_status()
            data = response.json()
            return int(data.get("signal_count", 0))
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning("remote evaluator healthz probe failed: %s", exc)
            return 0

    def close(self) -> None:
        self._client.close()


def _unavailable_result(tier: LLMTier | None, error: str) -> SignalResult:
    """Synthesize a SignalResult for the fail-closed/open posture."""
    if tier == "tier_3":
        return SignalResult(
            name="evaluator_unavailable",
            verdict="halt",
            reason=(
                f"remote evaluator unavailable on cloud route (tier_3); " f"fail-closed: {error}"
            ),
            severity="critical",
            audit_metadata={"error": error, "tier": tier, "posture": "fail-closed"},
        )
    return SignalResult(
        name="evaluator_unavailable",
        verdict="warn",
        reason=(
            f"remote evaluator unavailable on local route (tier={tier}); " f"fail-open: {error}"
        ),
        severity="warn",
        audit_metadata={"error": error, "tier": tier, "posture": "fail-open"},
    )


# Worst-wins selector matching ``EvaluatorRegistry.worst`` so callers
# get the same surface whether they hold a registry or a client.
_VERDICT_RANK: dict[str, int] = {
    "ok": 0,
    "warn": 1,
    "require_approval": 2,
    "halt": 3,
}


def worst(results: tuple[SignalResult, ...]) -> SignalResult | None:
    if not results:
        return None
    return max(results, key=lambda r: _VERDICT_RANK[r.verdict])


def isolation_db_open_readonly(db_path: str) -> Any:
    """Open the evaluator DB in URI read-only mode (defense in depth).

    Returns a ``sqlite3.Connection`` configured for read-only access
    via the ``mode=ro`` URI parameter. Even when the filesystem
    permissions are misconfigured, SQLite refuses INSERT/UPDATE on
    this connection. Used by the agent process when it needs to
    query the evaluator DB directly (read-only diagnostics).
    """
    import sqlite3

    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)
