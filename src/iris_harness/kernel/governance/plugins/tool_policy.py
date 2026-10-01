"""ToolPolicyHook — Phase 1 ``PreToolUse`` guard.

This is the small mandatory-passage policy hook for ordinary ReAct tool
calls. Phase 4 persona-specific controls (PersonaSurface, FSJail,
CommandSandbox, NetworkEgress) will add richer rules at the same hook
point; this hook provides a simple allow/deny surface today.
"""

from __future__ import annotations

from iris_harness.foundation.capabilities import CAPABILITY_TOOL_PREFIX
from iris_harness.kernel.governance.caller_policy import is_mcp_caller
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint


class ToolPolicyHook:
    """Allow or deny tool calls by tool name."""

    name: str = "tool_policy"
    hook_point: HookPoint = HookPoint.PRE_TOOL_USE
    priority: int = 20

    def __init__(
        self,
        *,
        allowed_tools: frozenset[str] | None = None,
        blocked_tools: frozenset[str] = frozenset(),
        confirm_once_tools: frozenset[str] = frozenset(),
    ) -> None:
        self._allowed_tools = allowed_tools
        self._blocked_tools = blocked_tools
        # Confirm once per run (multi-step loop plan, decision 8; ADR-0110). A tool
        # whose declaration says ``confirm: once`` — ``ctx.metadata["tool_confirm"]``,
        # carried from its manifest by the loop — may run only after the run proposed
        # this write and then asked the user (``ctx.metadata["asked_user"]``; an
        # earlier clarifying question does not count). A first write without that
        # evidence is turned back with the instruction to ask.
        # Not a halt: PreToolUse's ``require_approval`` becomes the observation the
        # model reads next. ``confirm_once_tools`` is the operator's override by
        # name (IRIS_GOVERNANCE_CONFIRM_ONCE_TOOLS); the core's default names no tool.
        self._confirm_once_tools = confirm_once_tools

    async def __call__(self, ctx: HookContext) -> HookDecision:
        tool_name = ctx.payload.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            return HookDecision(
                outcome="deny",
                reason="tool_policy: missing tool_name in PreToolUse context",
                severity="error",
                audit_metadata={"missing": "tool_name"},
            )

        declared_once = ctx.metadata.get("tool_confirm") == "once"
        # A code caller has no chat turn to ask in, so the harness escalates its
        # ``confirm: once`` write to a per-call approval the owner answers from the queue
        # (plugin-capabilities decision 1). The approval hook asks; this rule steps aside.
        queued_instead = (
            ctx.metadata.get("deferred_executor") is True
            and ctx.metadata.get("per_call_approval") is True
        )
        confirm_once = (declared_once and not queued_instead) or (
            tool_name in self._confirm_once_tools
        )
        if confirm_once and is_mcp_caller(ctx.metadata.get("caller")):
            # A client outside IRIS (``iris mcp serve``) has no IRIS chat in which the
            # owner could confirm, and its own word is not the owner's: refused, not asked.
            return HookDecision(
                outcome="deny",
                reason=(
                    f"tool_policy: {tool_name!r} writes on the owner's behalf and needs their "
                    "confirmation in an IRIS chat, which a client outside IRIS cannot give; "
                    "it was not run"
                ),
                severity="warn",
                audit_metadata={"tool_name": tool_name, "policy": "confirm_once_tools"},
            )
        if confirm_once and ctx.metadata.get("asked_user") is not True:
            return HookDecision(
                outcome="require_approval",
                reason=(
                    f"tool_policy: {tool_name!r} writes on the user's behalf and the user has "
                    "not confirmed it yet. Do NOT call it again now. First call ask_user with "
                    "ONE question that lists everything you intend to create (each item's "
                    "text, date and time); after the user confirms, call "
                    f"{tool_name!r} once per item."
                ),
                severity="info",
                audit_metadata={"tool_name": tool_name, "policy": "confirm_once_tools"},
            )
        if tool_name in self._blocked_tools:
            return HookDecision(
                outcome="deny",
                reason=f"tool_policy: tool {tool_name!r} is blocked",
                severity="warn",
                audit_metadata={"tool_name": tool_name, "policy": "blocked_tools"},
            )

        # The operator's allow-list names tools; a capability call
        # (``capability:<name>.<method>``) is governed by the permission contract instead
        # (plugin-capabilities §4), so the list does not apply to it. The block list does.
        if (
            self._allowed_tools is not None
            and tool_name not in self._allowed_tools
            and not tool_name.startswith(CAPABILITY_TOOL_PREFIX)
        ):
            return HookDecision(
                outcome="deny",
                reason=f"tool_policy: tool {tool_name!r} is not in the allowlist",
                severity="warn",
                audit_metadata={"tool_name": tool_name, "policy": "allowed_tools"},
            )

        return HookDecision(
            outcome="allow",
            reason=f"tool_policy: tool {tool_name!r} permitted",
            audit_metadata={"tool_name": tool_name},
        )
