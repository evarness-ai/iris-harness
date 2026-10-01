"""EvaluatorHook — adapts the evaluator (local or remote) to the kernel PostStep contract.

Reads the per-step bookkeeping out of ``ctx.payload`` and
``ctx.metadata``, builds a ``StepRecord``, runs the configured
backend (local ``EvaluatorRegistry`` OR a ``RemoteEvaluatorClient``),
and translates the worst-wins result into a ``HookDecision``.

Verdict mapping:

- ``ok``               → ``allow``
- ``warn``             → ``allow`` with ``severity='warn'`` (audit only)
- ``require_approval`` → ``require_approval``
- ``halt``             → ``deny``

Audit metadata for each signal result is preserved so ``iris run
inspect`` (12.gov-3.5) can render the per-signal trail.

Out-of-process mode (12.gov-3.9) is a transport swap: the hook calls
``client.evaluate(step)`` instead of ``registry.evaluate(step)`` and
``client.reset_run_state(run_id)`` on terminal verdicts. The verdict
mapping + audit shape are identical so ``iris run inspect`` works
the same.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from typing import TYPE_CHECKING, Any, Protocol

from iris_harness.kernel.governance.evaluator.registry import EvaluatorRegistry
from iris_harness.kernel.governance.evaluator.types import SignalResult, StepRecord
from iris_harness.kernel.governance.hooks.types import (
    HookContext,
    HookDecision,
    HookOutcome,
    HookPoint,
)

if TYPE_CHECKING:
    from iris_harness.kernel.governance.approvals.queue import ApprovalQueue
    from iris_harness.kernel.governance.approvals.router import ChannelRouter

logger = logging.getLogger(__name__)


class EvaluatorBackend(Protocol):
    """Protocol shared by the local registry and the remote HTTP client."""

    def evaluate(self, step: StepRecord) -> tuple[SignalResult, ...]: ...

    def reset_run_state(self, run_id: str) -> None: ...

    def signal_count(self) -> int: ...


class EvaluatorHook:
    """``PostStep`` hook firing either the in-process registry or a remote client."""

    name: str = "evaluator"
    hook_point: HookPoint = HookPoint.POST_STEP
    priority: int = 50  # leaf-most: signals look at the assembled context

    def __init__(
        self,
        *,
        registry: EvaluatorRegistry | None = None,
        backend: EvaluatorBackend | None = None,
        approval_queue: ApprovalQueue | None = None,
        channel_router: ChannelRouter | None = None,
    ) -> None:
        chosen = backend if backend is not None else registry
        if chosen is None:
            raise ValueError("EvaluatorHook requires either a registry or a backend")
        self._backend: EvaluatorBackend = chosen
        self._approval_queue = approval_queue
        self._channel_router = channel_router

    async def __call__(self, ctx: HookContext) -> HookDecision:
        if self._backend.signal_count() == 0:
            return HookDecision(
                outcome="allow",
                reason="evaluator: no signals registered",
            )

        step = _build_step_record(ctx)
        results = self._backend.evaluate(step)
        worst = _worst(results)
        if worst is None:
            return HookDecision(outcome="allow", reason="evaluator: no signal results")

        # Reset per-run state on any terminal verdict — a halted run
        # won't fire PostStep again, and require_approval pauses for
        # HITL which may resume from a fresh checkpoint.
        if worst.verdict in ("halt", "require_approval"):
            self._backend.reset_run_state(step.run_id)

        outcome = _OUTCOME_FOR_VERDICT[worst.verdict]
        severity = worst.severity
        audit_metadata = _collect_audit_metadata(results)

        # AC-6: critical-severity signals always hard-halt, regardless of signal verdict.
        # A classification_violation at critical severity must never be soft-paused.
        if outcome == "require_approval" and worst.severity == "critical":
            return HookDecision(
                outcome="deny",
                reason=f"evaluator: {worst.name} -> critical severity (hard halt): {worst.reason}",
                severity=severity,
                audit_metadata=audit_metadata,
            )

        # AC-1/AC-2/AC-3: enqueue approval request and notify channel if wired.
        approval_request_id = None
        if outcome == "require_approval" and self._approval_queue is not None:
            approval_id_str = self._approval_queue.enqueue(
                step.run_id,
                None,  # checkpoint_id — written by the caller (agentic_core) after the hook
                worst.name,
                f"Signal '{worst.name}' triggered: {worst.reason}",
                # Where the turn came from, so the router can deliver it back there. This
                # was hardcoded "cli" — which is why a halt raised by a web turn was
                # routed by `sys.stdin.isatty()` (false under the API server) to Telegram
                # or to the server's stderr, and never to the person in the browser.
                channel=_origin_channel(ctx),
                # Which conversation this halt belongs to, so a lapse can be announced
                # in it. `AgenticCore` puts it on the PostStep metadata beside the
                # channel; a caller that supplies neither simply gets no in-chat notice.
                session_id=_session_id(ctx),
            )
            row = self._approval_queue.get(approval_id_str)
            if row is not None and self._channel_router is not None:
                self._channel_router.notify(row)
            approval_request_id = _uuid.UUID(approval_id_str)
            audit_metadata = {**audit_metadata, "approval_id": approval_id_str}

        return HookDecision(
            outcome=outcome,
            reason=f"evaluator: {worst.name} -> {worst.verdict}: {worst.reason}",
            severity=severity,
            audit_metadata=audit_metadata,
            approval_request_id=approval_request_id,
        )


def _origin_channel(ctx: HookContext) -> str:
    """The gateway this step's turn came from; "cli" when nothing said.

    ``AgenticCore`` puts it on the PostStep context's metadata. The default keeps every
    caller that predates it — tests, the sandbox loop, anything driving the kernel
    directly — on exactly the behaviour they had.
    """
    raw = ctx.metadata.get("origin_channel")
    channel = str(raw).strip().lower() if raw else ""
    return channel or "cli"


def _session_id(ctx: HookContext) -> str | None:
    """The conversation this step's turn belongs to, or None when nothing said."""
    raw = ctx.metadata.get("session_id")
    session_id = str(raw).strip() if raw else ""
    return session_id or None


_OUTCOME_FOR_VERDICT: dict[str, HookOutcome] = {
    "ok": "allow",
    "warn": "allow",
    "require_approval": "require_approval",
    "halt": "deny",
}


_VERDICT_RANK: dict[str, int] = {
    "ok": 0,
    "warn": 1,
    "require_approval": 2,
    "halt": 3,
}


def _worst(results: tuple[SignalResult, ...]) -> SignalResult | None:
    """Worst-wins selector. Duplicates ``EvaluatorRegistry.worst`` so the
    hook works against either backend without holding a registry handle.
    """
    if not results:
        return None
    return max(results, key=lambda r: _VERDICT_RANK[r.verdict])


def _build_step_record(ctx: HookContext) -> StepRecord:
    payload = ctx.payload or {}
    metadata = ctx.metadata or {}

    def _pick(key: str) -> Any:
        return payload.get(key, metadata.get(key))

    return StepRecord(
        run_id=ctx.run_id,
        step_id=ctx.step_id if ctx.step_id is not None else 0,
        agent_type=ctx.agent_type,
        thought=_str_or_none(_pick("thought")),
        tool_name=_str_or_none(_pick("tool_name")),
        tool_args_hash=_str_or_none(_pick("tool_args_hash")),
        tool_args_text=_str_or_none(_pick("tool_args_text")),
        tool_error=_str_or_none(_pick("tool_error")),
        tier=ctx.tier,
        classification=ctx.classification,
        cost_usd=_float_or_none(_pick("cost_usd")),
        original_task=_str_or_none(_pick("original_task")),
        metadata={k: v for k, v in metadata.items() if k not in _STEP_KEYS},
    )


_STEP_KEYS = frozenset(
    {
        "thought",
        "tool_name",
        "tool_args_hash",
        "tool_args_text",
        "tool_error",
        "cost_usd",
        "original_task",
    }
)


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _collect_audit_metadata(results: tuple[SignalResult, ...]) -> dict[str, Any]:
    return {
        "signals": [
            {
                "name": r.name,
                "verdict": r.verdict,
                "severity": r.severity,
                "reason": r.reason,
                **r.audit_metadata,
            }
            for r in results
        ]
    }
