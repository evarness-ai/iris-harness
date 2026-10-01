"""The runtime plugin contract (OSS plan decisions 1, 3, 4, 8).

Everything outside the governed core is a plugin. A plugin is a Python package
that exposes ``setup(api: PluginAPI) -> None`` and registers capabilities of six
kinds — intercept, tool, intent handler, heartbeat handler, channel, confirmation
executor — through :class:`PluginAPI`. Registrations are recorded in the host's
plugin registry and every callable is wrapped by one fault boundary, so a plugin
failure degrades that capability and shows red in System Health instead of taking
the turn down.

Which plugins mount, and in what order, is declared by a profile
(``config/profiles/<name>.yaml``, overlaid by ``$IRIS_HOME/profile.yaml`` and env).

This package is the **author-facing** half of the seam. The host-facing half --
discovery, the registry, ``PluginAPI`` itself, the manifest schema, profile
resolution, ``--dump-config`` -- lives in ``iris_harness.runtime.plugin_host``,
because deciding which plugins mount and what happens when one misbehaves is
composition-root work, and because the loader *constructs* the ``PluginAPI`` it
hands to ``setup()``.

Only what a plugin author writes against is re-exported here. The loader, the
registry and its records, and profile resolution used to be re-exported too; no
plugin calls them (only host tests did), and a name in this package is a promise
to keep it stable, so they are imported from ``runtime.plugin_host`` instead
(core/SDK boundary plan, PR 1).
"""

from __future__ import annotations

from iris_harness.runtime.harness_services import HarnessServices
from iris_harness.runtime.plugin_host.api import PluginAPI
from iris_harness.runtime.plugin_host.manifest import PluginManifest, RegistrationKind

from .cli import PluginCLI, register_plugin_commands

__all__ = [
    "HarnessServices",
    "PluginAPI",
    "PluginCLI",
    "PluginManifest",
    "RegistrationKind",
    "register_plugin_commands",
]
