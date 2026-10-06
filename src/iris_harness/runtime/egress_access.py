"""Which hosts each mounted plugin may contact: the manifests' ``egress`` blocks, compiled.

The permission contract for a plugin's network use (docs/architecture/plugin-egress.md,
issue #103). Every mounted plugin's manifest declares its hosts; ``compile_egress_policy``
joins them into the one :class:`PluginEgressPolicy` the kernel's ``plugin_egress`` hook
enforces, and ``build_runtime`` registers it once plugins have mounted -- the same seam as
``tool_access.compile_caller_policy``. A plugin that is not mounted, failed to mount or is
disabled has no entry, so it may contact nothing.
"""

from __future__ import annotations

from typing import Any

from iris_harness.kernel.governance.plugin_egress import PluginEgress, PluginEgressPolicy
from iris_harness.runtime.plugin_host.registry import PluginStatus

_MOUNTED = (PluginStatus.LOADED, PluginStatus.DEGRADED)


def compile_egress_policy(registry: Any) -> PluginEgressPolicy:
    """The manifests of every mounted plugin, by plugin name."""
    plugins: dict[str, PluginEgress] = {}
    for record in registry.plugins():
        if record.status in _MOUNTED and record.manifest is not None:
            plugins[record.name] = record.manifest.egress.compile()
    return PluginEgressPolicy(plugins)


__all__ = ["compile_egress_policy"]
