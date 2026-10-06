"""GovernanceKernel — central enforcement point.

Every LLM call, tool invocation, and cloud egress fires hooks through
this kernel. Hooks register at kernel init; after ``init_lock()`` the
registry is frozen and further registration raises (design §5.3: hook
plugin trust boundary).

Dispatch semantics:

- Hooks at a hook point fire in ``priority`` ascending (lower = earlier).
- ``transform`` rebinds ``ctx.payload`` for the next hook at the same
  point. ``set_classification`` / ``set_tier`` rebind those fields too,
  regardless of outcome (a hook can deny *and* annotate).
- First ``deny`` or ``require_approval`` short-circuits remaining hooks.
- ``fire()`` returns ``(decision, final_context)`` so callers can thread
  classification/tier annotations into the next hook point.
- Hook exceptions become an ``error``-severity deny (fail-closed at the
  kernel level). The per-route fail-open/fail-closed posture from §4.1
  layers on top once the policy engine is wired in.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.hooks.tool_payload import SIDE_EFFECT_ID
from iris_harness.kernel.governance.hooks.types import (
    Hook,
    HookContext,
    HookDecision,
    HookPoint,
)

logger = logging.getLogger(__name__)

# Context payload keys copied into the audit row payload. Deliberately a
# whitelist: prompts and tool args must never land in the audit store. The two
# context_* keys are ADR-0077 context-budget telemetry (counts only, no content).
_AUDITED_PAYLOAD_KEYS: tuple[str, ...] = (
    "model",
    "provider",
    "tool_name",
    "target_tier",
    "context_tokens",
    "evicted_tokens",
    # PRE_RESPONSE: whether a model wrote the answer, and which deterministic handler
    # did when none did (deterministic-path parity, step c).
    "deterministic",
    "handler",
    # A capability call (plugin-capabilities §4): who called which provider's method; and,
    # for every governed call, keyed fingerprints of what went in and came out -- never
    # the text itself.
    "caller",
    "capability",
    "method",
    "capability_provider",
    "args_digest",
    "result_digest",
    # Which keyed digest the two above are (``hmac-sha256/v1/<key-id>``): every governed
    # tool, capability and MCP call carries them (``kernel/governance/audit/digest.py``).
    "digest_alg",
    # PRE_RESPONSE: who reads the answer (``owner`` | ``other``, ADR-0125).
    "audience",
    "stream_item",
    "stream_end",
    "stream_partial",
)


def _current_trace_id_hex() -> str | None:
    """Return the active OpenTelemetry trace id, when a span is recording.

    Lets audit rows cross-reference traces in the OTLP backend. Best-effort: returns
    ``None`` when OTel is absent, no span is current, or the span is a
    non-recording placeholder.
    """
    try:
        from opentelemetry import trace as otel_trace

        span_context = otel_trace.get_current_span().get_span_context()
        if not span_context.is_valid:
            return None
        return format(span_context.trace_id, "032x")
    except Exception:  # noqa: BLE001 - audit enrichment must never raise
        return None


class HookRegistrationLockedError(RuntimeError):
    """Raised when ``register()`` is called after ``init_lock()``."""


class KernelNotInitializedError(RuntimeError):
    """Raised when ``fire()`` is called before ``init_lock()``."""


class GovernanceKernel:
    """Central dispatch for all governance hooks.

    Construct once at process start, register all hooks, call
    ``init_lock()``, then fire from AgenticCore stages and the coding
    pipeline. The kernel is single-instance per process; cross-service
    callers go through the HTTP wrapper at ``src/iris_harness/server/governor/``.

    ``audit_log`` is the Phase 3 hot-tier sink. When ``None`` the
    kernel skips persistence (test convenience); production callers
    construct one via ``build_default_kernel``.
    """

    def __init__(self, *, audit_log: AuditLog | None = None) -> None:
        self._hooks: dict[HookPoint, list[Hook]] = defaultdict(list)
        self._init_locked: bool = False
        self._audit_log = audit_log

    def register(self, hook: Hook, *, at: HookPoint | None = None) -> None:
        """Register a hook. Permitted only before ``init_lock()``.

        ``at`` registers the hook at a point other than its declared ``hook_point``.
        One screening hook can then serve two points with one instance and one
        configuration — the data classifier runs at ``PRE_TURN`` (what the user said)
        and at ``PRE_CLASSIFY`` (what enters a model).
        """
        if self._init_locked:
            raise HookRegistrationLockedError(
                f"Hook {hook.name!r} cannot be registered after init_lock(); "
                "runtime hook registration is forbidden (design §5.3 "
                "hook plugin trust boundary)."
            )
        point = at if at is not None else hook.hook_point
        self._hooks[point].append(hook)
        self._hooks[point].sort(key=lambda h: h.priority)
        logger.debug(
            "registered governance hook name=%s point=%s priority=%d",
            hook.name,
            point.value,
            hook.priority,
        )

    def init_lock(self) -> None:
        """Freeze the hook registry. Required before ``fire()``."""
        self._init_locked = True
        logger.info(
            "governance kernel locked: %d hooks across %d hook points",
            sum(len(hs) for hs in self._hooks.values()),
            len(self._hooks),
        )

    @property
    def is_locked(self) -> bool:
        return self._init_locked

    def hook_count(self, hook_point: HookPoint) -> int:
        """Number of hooks registered at ``hook_point`` (for diagnostics)."""
        return len(self._hooks[hook_point])

    def hook_names(self, hook_point: HookPoint) -> tuple[str, ...]:
        """Names of the hooks at ``hook_point``, in the order they will run.

        A caller asking "is X registered?" should ask that, not compare a total: a
        count says nothing about *which* hooks are there, and breaks the moment an
        unrelated one is added at the same point.
        """
        return tuple(hook.name for hook in self._hooks[hook_point])

    async def fire(
        self, hook_point: HookPoint, ctx: HookContext
    ) -> tuple[HookDecision, HookContext]:
        """Run hooks at ``hook_point``. Returns the final decision and context.

        The returned context reflects every ``transform`` /
        ``set_classification`` / ``set_tier`` applied by hooks in the
        chain. Callers thread it into subsequent hook points.
        """
        if not self._init_locked:
            raise KernelNotInitializedError(
                "GovernanceKernel.fire() called before init_lock(); "
                "call init_lock() after registering all hooks."
            )

        decision = HookDecision(outcome="allow", reason="no hooks registered")
        current_ctx = ctx

        for hook in self._hooks[hook_point]:
            try:
                decision = await hook(current_ctx)
            except Exception as exc:
                logger.exception("governance hook %s raised at %s", hook.name, hook_point.value)
                exc_decision = HookDecision(
                    outcome="deny",
                    reason=f"hook {hook.name!r} raised: {exc.__class__.__name__}",
                    severity="error",
                    audit_metadata={"hook": hook.name, "exception": str(exc)},
                    decided_by=hook.name,
                )
                self._audit(hook_point, hook.name, exc_decision, current_ctx)
                return exc_decision, current_ctx

            self._audit(hook_point, hook.name, decision, current_ctx)

            updates: dict[str, Any] = {}
            if decision.set_classification is not None:
                updates["classification"] = decision.set_classification
            if decision.set_tier is not None:
                updates["tier"] = decision.set_tier
            recorded = decision.audit_metadata.get(SIDE_EFFECT_ID)
            if isinstance(recorded, str) and recorded:
                # Sticky: ``fire`` hands back only the last hook's decision, so a hook that
                # runs after the pre-execution ledger hook would hide the row it wrote.
                # The runner reads it from the final context instead.
                updates["metadata"] = {**current_ctx.metadata, SIDE_EFFECT_ID: recorded}
            if decision.outcome == "transform" and decision.transformed_payload is not None:
                updates["payload"] = decision.transformed_payload
                # Who rewrote the payload, in order: a caller that finds the payload
                # inconsistent (a capability result whose text disagrees with its field
                # map) can refuse it and name the hook responsible.
                updates["metadata"] = {
                    **updates.get("metadata", current_ctx.metadata),
                    "transformed_by": [*current_ctx.metadata.get("transformed_by", ()), hook.name],
                }
            if updates:
                current_ctx = current_ctx.model_copy(update=updates)

            if decision.outcome in ("deny", "require_approval"):
                return decision.model_copy(update={"decided_by": hook.name}), current_ctx

        return decision, current_ctx

    def _audit(
        self,
        hook_point: HookPoint,
        plugin: str,
        decision: HookDecision,
        ctx: HookContext,
    ) -> None:
        """Persist one row per hook firing. Never raises."""
        if self._audit_log is None:
            return
        try:
            payload: dict[str, Any] = dict(decision.audit_metadata)
            for key in _AUDITED_PAYLOAD_KEYS:
                if key in ctx.payload and key not in payload:
                    payload[key] = ctx.payload[key]
            # Who made the call. A capability call carries it in its payload; a tool call
            # (a model's, a plugin's ``api.tools``, an MCP client's ``mcp:<client>``) carries
            # it in metadata, where only the hooks that copied it into ``audit_metadata``
            # recorded it -- so the other rows of the same call said nothing about who made it.
            caller = ctx.metadata.get("caller")
            if isinstance(caller, str) and caller:
                payload.setdefault("caller", caller)
            trace_id = _current_trace_id_hex()
            if trace_id is not None:
                payload.setdefault("trace_id", trace_id)
            # Stamp the active session id (set by session_scope around the turn and
            # re-entered around agent execution) so audit rows correlate to a chat
            # turn — every pipeline stage stamps its own run_id, so run_id alone
            # cannot tie a hook back to a session. Mirrors the trace_id stamp above.
            from iris_harness.foundation.observability.session_log import current_session_id

            session_id = current_session_id()
            if session_id is not None:
                payload.setdefault("session_id", session_id)
            self._audit_log.record(
                run_id=ctx.run_id,
                step_id=ctx.step_id,
                agent_type=ctx.agent_type,
                hook_point=hook_point.value,
                plugin=plugin,
                decision=decision.outcome,
                severity=decision.severity,
                reason=decision.reason,
                classification=ctx.classification,
                tier=ctx.tier,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001 - audit failure must not break enforcement
            logger.warning(
                "audit_log: write failed for %s/%s: %s",
                hook_point.value,
                plugin,
                exc,
            )

    def fire_sync(
        self, hook_point: HookPoint, ctx: HookContext
    ) -> tuple[HookDecision, HookContext]:
        """Synchronous wrapper around ``fire()``.

        AgenticCore and the existing tier_router are synchronous; this
        helper lets them call the kernel without an async refactor.
        Refuses to run inside an active event loop — async callers must
        use ``fire()`` directly.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.fire(hook_point, ctx))
        raise RuntimeError(
            "GovernanceKernel.fire_sync() cannot run inside an active "
            "event loop; use fire() from async code."
        )
