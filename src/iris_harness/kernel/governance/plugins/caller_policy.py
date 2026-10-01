"""CallerPolicyHook — ``PreToolUse``: a plugin calls only the tools it is allowed.

The loop's calls are the model's (``model:<agent>``) and the core's are the harness's
(``core:<workflow>``); both keep the access they have. A ``plugin:<name>`` call — code in
one plugin calling a tool through ``api.tools`` — passes only when the permission contract
allows it: the tool is the plugin's own, or its manifest lists it under ``uses: tools``,
and the operator has not taken it away (docs/architecture/plugin-capabilities.md §4).

Priority 12: before the tool policy (20), so a call a plugin was never allowed is refused
for that reason, not for another. The policy itself is config the runtime compiles and
registers (``governance/caller_policy.py``); with none registered, plugin calls are denied.
"""

from __future__ import annotations

from iris_harness.kernel.governance.caller_policy import caller_policy
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint

_PLUGIN_PREFIX = "plugin:"


class CallerPolicyHook:
    """Deny a ``plugin:`` caller any tool the permission contract does not allow it."""

    name: str = "caller_policy"
    hook_point: HookPoint = HookPoint.PRE_TOOL_USE
    priority: int = 12

    async def __call__(self, ctx: HookContext) -> HookDecision:
        caller = str(ctx.metadata.get("caller") or "")
        tool = str(ctx.payload.get("tool_name") or "")
        if not caller.startswith(_PLUGIN_PREFIX):
            return HookDecision(outcome="allow", reason=f"caller_policy: {caller or 'model'}")
        policy = caller_policy()
        if policy is None:
            return HookDecision(
                outcome="deny",
                reason=f"caller_policy: no permission contract is loaded, so {caller} "
                f"may not call {tool!r}",
                severity="warn",
                audit_metadata={"caller": caller, "tool_name": tool},
            )
        denied = policy(caller, tool)
        if denied:
            return HookDecision(
                outcome="deny",
                reason=f"caller_policy: {denied}",
                severity="warn",
                audit_metadata={"caller": caller, "tool_name": tool},
            )
        return HookDecision(
            outcome="allow",
            reason=f"caller_policy: {caller} may call {tool!r}",
            audit_metadata={"caller": caller, "tool_name": tool},
        )


__all__ = ["CallerPolicyHook"]
