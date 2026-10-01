"""McpClientEgressHook -- ``PostToolUse``: private data does not leave for an MCP client.

A tool result served over MCP (``iris mcp serve``, caller ``mcp:<client>``) leaves IRIS's
governance for another program -- one that may hand it to a cloud model. So, by the
result's label after every other ``PostToolUse`` hook (the output classifier raises it;
the session's label is its floor):

* ``secret`` is withheld from every MCP client, a local one included. A secret is what
  IRIS keeps in its vault and never lets out of its own process; a local client is still
  a program IRIS does not govern, with its own model, logs and network.
* ``personal`` is withheld unless the owner declared the client local
  (``kernel/governance/mcp_clients.py``, from the owner's MCP serve config).
* anything else passes.

Priority 42: after the output classifier (10), so it judges the raised label, and after
the side-effect ledger (40), so a withheld call's side effects are still recorded; before
the retrieved-content guard (45), which stays last so its ``transform`` is the chain's
final decision. A deny here withholds the result (the runner enforces it); the call
itself has run.
"""

from __future__ import annotations

from iris_harness.kernel.governance.caller_policy import is_mcp_caller
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.mcp_clients import mcp_client_is_local


class McpClientEgressHook:
    """Withhold a private result from an MCP client that may not hold it."""

    name: str = "mcp_client_egress"
    hook_point: HookPoint = HookPoint.POST_TOOL_USE
    priority: int = 42

    async def __call__(self, ctx: HookContext) -> HookDecision:
        caller = str(ctx.metadata.get("caller") or "")
        if not is_mcp_caller(caller):
            return HookDecision(outcome="allow", reason="mcp_client_egress: not an MCP client")
        label = ctx.classification
        local = mcp_client_is_local(caller)
        meta = {"caller": caller, "classification": label, "local": local}
        if label == "secret":
            return HookDecision(
                outcome="deny",
                reason=(
                    f"mcp_client_egress: the result is classified secret, and a secret never "
                    f"leaves IRIS, not even for a local client; withheld from {caller}"
                ),
                severity="warn",
                audit_metadata=meta,
            )
        if label == "personal" and not local:
            return HookDecision(
                outcome="deny",
                reason=(
                    f"mcp_client_egress: the result holds the owner's personal data and "
                    f"{caller} is not declared local in the owner's mcp-serve.yaml; withheld"
                ),
                severity="warn",
                audit_metadata=meta,
            )
        return HookDecision(
            outcome="allow",
            reason=f"mcp_client_egress: {label or 'unlabelled'} result may go to {caller}",
            audit_metadata=meta,
        )


__all__ = ["McpClientEgressHook"]
