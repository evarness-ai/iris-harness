"""PluginEgressHook -- ``PreEgress``: a plugin contacts only the hosts it declared (#103).

A plugin's request through the SDK's governed HTTP client fires ``PRE_EGRESS`` with the
address it is about to contact (scheme, host, port, method) and the run's data class. This
hook asks the compiled policy (``governance/plugin_egress.py``, built from the manifests'
``egress`` blocks) and allows the request only when the plugin declared that host over that
scheme and port, for data of that class. Every other case is a deny, audited with the host:
an undeclared host, a plugin with no ``egress`` block, a run holding more than the host is
declared to receive, no policy registered. Fail closed.

``PluginEgressOutcomeHook`` writes the ``POST_EGRESS`` row: status, bytes each way, duration
and, when the request raised, the exception class. Neither hook sees, and no row carries,
the request's path, query, headers or body.

What this proves is a property of the governed client's calls only: an in-process plugin
can still open its own socket (docs/architecture/plugin-egress.md, "What this does not prove").
"""

from __future__ import annotations

from typing import Any

from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.plugin_egress import egress_policy


def _egress(ctx: HookContext) -> dict[str, Any]:
    raw = ctx.payload.get("egress")
    return dict(raw) if isinstance(raw, dict) else {}


class PluginEgressHook:
    """Deny a plugin's request to any host its manifest does not declare."""

    name: str = "plugin_egress"
    hook_point: HookPoint = HookPoint.PRE_EGRESS
    priority: int = 20

    async def __call__(self, ctx: HookContext) -> HookDecision:
        egress = _egress(ctx)
        plugin = str(egress.get("plugin") or "")
        host = str(egress.get("host") or "")
        where = f"{egress.get('scheme', '')}://{host}:{egress.get('port', '')}"
        if egress.get("malformed"):
            return HookDecision(
                outcome="deny",
                reason=f"plugin_egress: {egress['malformed']}",
                severity="warn",
                audit_metadata={"egress": {**egress, "allowed": False}},
            )
        policy = egress_policy()
        if policy is None:
            return HookDecision(
                outcome="deny",
                reason=f"plugin_egress: no egress policy is loaded, so {plugin!r} may not "
                f"contact {host!r}",
                severity="warn",
                audit_metadata={"egress": {**egress, "allowed": False}},
            )
        verdict = policy.decide(
            plugin,
            scheme=str(egress.get("scheme") or ""),
            host=host,
            port=int(egress.get("port") or 0),
            classification=ctx.classification,
        )
        if not verdict.allowed:
            return HookDecision(
                outcome="deny",
                reason=f"plugin_egress: {verdict.reason} ({where})",
                severity="warn",
                audit_metadata={"egress": {**egress, "allowed": False}},
            )
        return HookDecision(
            outcome="allow",
            reason=f"plugin_egress: {plugin} may contact {host} ({verdict.reason})",
            audit_metadata={
                "egress": {**egress, "allowed": True, "data": verdict.data, "rule": verdict.rule}
            },
        )


class PluginEgressOutcomeHook:
    """Record how a governed request ended. Always allows: the request already happened."""

    name: str = "plugin_egress_outcome"
    hook_point: HookPoint = HookPoint.POST_EGRESS
    priority: int = 20

    async def __call__(self, ctx: HookContext) -> HookDecision:
        egress = _egress(ctx)
        error = egress.get("error")
        ended = f"failed ({error})" if error else f"-> {egress.get('status')}"
        if egress.get("aborted"):
            ended += f", cut off: {egress['aborted']}"
        return HookDecision(
            outcome="allow",
            reason=(
                f"plugin_egress_outcome: {egress.get('method', '')} {egress.get('host', '')} "
                f"{ended}"
            ),
            severity="warn" if error else "info",
        )


__all__ = ["PluginEgressHook", "PluginEgressOutcomeHook"]
