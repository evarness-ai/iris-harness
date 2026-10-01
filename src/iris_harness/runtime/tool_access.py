"""The permission contract, compiled: who may call which tool (plugin-capabilities §4),
and which owner-identity kinds a capability consumer may see unmasked (ADR-0125).

Config, not code. Every mounted plugin's manifest says what it may use (``uses: tools``),
and it may always call its own tools; ``config/governance/tool-access.yaml`` lets the
operator take a tool away from a plugin caller. ``compile_caller_policy`` joins the two into
the one policy the kernel's ``CallerPolicyHook`` enforces, and ``build_runtime`` registers
it once plugins have mounted. The operator file can only narrow, never widen: a grant lives
in the plugin's manifest, where the dependency is visible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from iris_harness.foundation.capabilities import CAPABILITY_TOOL_PREFIX, split_capability_tool
from iris_harness.kernel.governance.caller_policy import CallerPolicy
from iris_harness.kernel.governance.unmask_grants import UnmaskPolicy

logger = logging.getLogger(__name__)

ACCESS_FILE = ("governance", "tool-access.yaml")


def load_operator_denials(config_dir: Path) -> dict[str, frozenset[str]]:
    """``{plugin name: tools taken away}`` from ``tool-access.yaml``; empty when absent.

    A malformed file is an error the operator must see, not a silent open door: it is
    logged at WARNING and, because a denial list that cannot be read cannot be honoured,
    the whole policy then denies every plugin call (see ``compile_caller_policy``).
    """
    path = config_dir.joinpath(*ACCESS_FILE)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    deny = raw.get("deny") or {}
    if not isinstance(deny, dict):
        raise ValueError(f"{path}: `deny` must map a plugin name to a list of tools")
    return {str(plugin): frozenset(str(t) for t in (tools or ())) for plugin, tools in deny.items()}


def compile_caller_policy(registry: Any, *, config_dir: Path) -> CallerPolicy:
    """The policy for ``CallerPolicyHook``: the manifests' grants minus the operator's."""
    try:
        denials = load_operator_denials(config_dir)
    except Exception as exc:  # noqa: BLE001 — unreadable denials: refuse, never widen
        logger.warning("tool-access.yaml unreadable (%s); every plugin tool call is denied", exc)
        reason = f"config/governance/tool-access.yaml is unreadable ({exc})"
        return lambda _caller, _tool: reason

    def policy(caller: str, tool: str) -> str | None:
        plugin = caller.removeprefix("plugin:")
        taken = denials.get(plugin, frozenset())
        capability_call = split_capability_tool(tool)
        # A capability can be taken away whole (`capability:mail.read`) or one method at a
        # time (`capability:mail.read.search`).
        whole = f"{CAPABILITY_TOOL_PREFIX}{capability_call[0]}" if capability_call else None
        if tool in taken or (whole is not None and whole in taken):
            return f"the operator took {tool!r} away from {caller} (tool-access.yaml)"
        denied: str | None = registry.caller_denial(caller, tool)
        return denied

    return policy


def compile_unmask_policy(registry: Any) -> UnmaskPolicy:
    """The policy for capability masking: each consumer's own manifest grants.

    ``caller`` is the harness's stamp (``plugin:<name>``), so a plugin's grants are read
    from its own manifest and no other. Anything that is not a mounted plugin with a
    manifest -- a ``core:`` caller included -- is granted nothing.
    """

    def policy(caller: str, capability: str) -> frozenset[str]:
        if not caller.startswith("plugin:"):
            return frozenset()
        record = registry.get(caller.removeprefix("plugin:"))
        manifest = getattr(record, "manifest", None)
        if manifest is None:
            return frozenset()
        return frozenset(manifest.capabilities.unmask.get(capability, ()))

    return policy


__all__ = ["compile_caller_policy", "compile_unmask_policy", "load_operator_denials"]
