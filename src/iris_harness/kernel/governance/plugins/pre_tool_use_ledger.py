"""PreToolUseLedgerHook -- the write-ahead row of a high-risk tool call.

A provider can succeed and the process die before ``POST_TOOL_USE`` writes anything, which
leaves an irreversible action with no evidence at all. For the calls where that matters --
the high-risk class, the tools whose own declaration makes every call wait for the owner's
approval (``destructive_approval.pinned_by_declaration``: ``effect: destructive``, or a
write declared ``approval: pinned``) -- this hook writes a durable ``pending`` row to the
side-effect ledger *before* the call runs. ``PostToolUseLedgerHook`` then settles that same
row (``completed`` / ``error``) instead of inserting a fresh one. A row still ``pending``
after a crash is exactly the case ``iris run resume`` exists for: it runs the row's probe,
or asks the owner.

Reads and every other write -- ``confirm: once`` and ``confirm: never`` writes, a code
caller's ``confirm: once`` write included -- are unchanged: they are recorded after the
call by the post hook, and a failed write there never blocks anything. A call that declares
no effect (an external MCP server's tool) is never high-risk.

Fail closed. If the row cannot be written -- no ledger is configured, or the write raises --
the call is denied and nothing runs: an irreversible action with no durable record is the
very gap this closes. The hook runs last at ``PRE_TOOL_USE`` (priority 100, after the
credential broker at 90), so every hook that can deny or ask for approval has already
spoken: a call that is held or refused leaves no row. It skips, too, a tool call that
carries no approval (``approved_by``): the runner refuses such a call itself, and nothing
runs. This hook never reads the arguments, which the broker may by then have resolved to
a secret. It confirms the record to the runner in its decision's ``audit_metadata``
(``tool_payload.SIDE_EFFECT_ID``); a runner finding a high-risk call allowed without it
refuses to run it, so a kernel built without this hook cannot run one either.

The row's key is the post hook's (``<run_id>:<step_id>:<tool_call_id>``; the runner mints
the call id before ``PRE_TOOL_USE`` and hands the same id to ``POST_TOOL_USE``). It holds
the tool name and the effect -- never arguments, results or message text. Every row is
written with no probe: a row still ``pending`` is a call that never settled, which may or may
not have run, and a probe run against it could only guess (a declared probe would report
"not landed" for a call that did land). ``resume`` therefore asks the owner. The post hook
sets the declared probe, and the effect's own id for a ``TOOL_PROBE_MAP`` tool, once the call
has returned and the effect is known.

The row is inserted exclusively (``record(..., exclusive=True)``): a key that already holds a
row -- a reused call id, a second call under the same run and step with none -- denies the
call. Letting the insert be ignored would confirm to the runner a record that is another
call's, so this call would run with no row of its own.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iris_harness.kernel.governance.hooks.tool_payload import (
    SIDE_EFFECT_ID,
    TOOL_CALL_ID,
    TOOL_EFFECT,
    tool_name_of,
)
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.plugins.destructive_approval import pinned_by_declaration
from iris_harness.kernel.governance.plugins.post_tool_use_ledger import (
    PRE_RECORDED,
    side_effect_key,
)
from iris_harness.kernel.governance.side_effects.probes import NO_PROBE
from iris_harness.kernel.governance.side_effects.store import SideEffectKeyExists

if TYPE_CHECKING:
    from iris_harness.kernel.governance.side_effects import SideEffectLedger

logger = logging.getLogger(__name__)

#: Runs after every other ``PRE_TOOL_USE`` hook (the credential broker is 90).
PRIORITY = 100


class PreToolUseLedgerHook:
    """``PreToolUse`` hook that writes the pending row of a high-risk call before it runs."""

    name: str = "pre_tool_use_ledger"
    hook_point: HookPoint = HookPoint.PRE_TOOL_USE
    priority: int = PRIORITY

    def __init__(self, ledger: SideEffectLedger | None) -> None:
        self._ledger = ledger

    async def __call__(self, ctx: HookContext) -> HookDecision:
        tool = tool_name_of(ctx.payload)
        if not tool or not pinned_by_declaration(
            ctx.metadata.get(TOOL_EFFECT), ctx.metadata.get("tool_confirm")
        ):
            return HookDecision(outcome="allow", reason="pre_tool_use_ledger: not a high-risk call")
        if not ctx.metadata.get("approved_by") and "capability" not in ctx.payload:
            # A tool call nobody approved got this far only because no approval hook is
            # registered: the runner refuses it (fail closed) and nothing runs, so there is
            # nothing to record. A capability call carries no approval of its own; one that
            # is allowed this far runs.
            return HookDecision(
                outcome="allow", reason="pre_tool_use_ledger: the call is not approved to run"
            )
        if self._ledger is None:
            return self._deny(tool, "no side-effect ledger is configured")

        step_id = ctx.step_id or 0
        tool_call_id = ctx.metadata.get(TOOL_CALL_ID)
        key = side_effect_key(ctx.run_id, step_id, str(tool_call_id) if tool_call_id else None)
        try:
            self._ledger.record(
                side_effect_id=key,
                run_id=ctx.run_id,
                step_id=step_id,
                tool=tool,
                verification_probe=NO_PROBE,
                probe_metadata={
                    "subject": key,
                    "effect": ctx.metadata.get(TOOL_EFFECT),
                    PRE_RECORDED: True,
                },
                exclusive=True,
            )
        except SideEffectKeyExists:
            # The key already has a row: another call's. This hook cannot show it is this
            # same call (it never reads arguments), so the row it would have confirmed is
            # not provably this call's record, and the call would run on someone else's.
            logger.error(
                "pre_tool_use_ledger: the ledger key for run=%s tool=%s is already taken",
                ctx.run_id,
                tool,
            )
            return self._deny(tool, "its ledger key already holds a record of another call")
        except Exception as exc:  # noqa: BLE001
            # The class only: the message of a storage error can carry a path or a value.
            logger.error(
                "pre_tool_use_ledger: could not write the pre-execution row for run=%s "
                "tool=%s (%s)",
                ctx.run_id,
                tool,
                type(exc).__name__,
            )
            return self._deny(
                tool, f"the pre-execution record could not be written ({type(exc).__name__})"
            )
        return HookDecision(
            outcome="allow",
            reason=f"pre_tool_use_ledger: recorded {tool} before it runs",
            audit_metadata={SIDE_EFFECT_ID: key},
        )

    @staticmethod
    def _deny(tool: str, why: str) -> HookDecision:
        return HookDecision(
            outcome="deny",
            reason=(
                f"pre_tool_use_ledger: {tool!r} is irreversible and needs a durable record "
                f"before it runs, but {why}; nothing was run"
            ),
            severity="error",
            audit_metadata={"tool_name": tool},
        )


__all__ = ["PRIORITY", "PreToolUseLedgerHook"]
