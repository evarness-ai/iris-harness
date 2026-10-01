"""The host side of the plugin seam (OSS plan M6.2 layer 7, decision 5).

Discovery, the fault-bounded registry, profile resolution and `--dump-config`.
These are composition-root work -- the runtime decides *which* plugins mount and
*what happens* when one misbehaves -- so they sit with the runtime, not in the
package a plugin author imports. ``PluginAPI`` and the manifest schema are here too,
against decision 5's list, for a reason the imports made plain: the loader
*constructs* a ``PluginAPI`` and the registry records manifest kinds, so leaving
either above the runtime puts the host on an upward import of the layer it serves.
``iris_harness.sdk`` re-exports every author-facing name -- a facade is what makes the SDK boundary strict for
plugin authors; it is not what decides where the machinery lives.
"""

from iris_harness.runtime.plugin_host.api import PluginAPI
from iris_harness.runtime.plugin_host.dump import dump_config
from iris_harness.runtime.plugin_host.loader import (
    BUILTIN_PACKAGE,
    ENTRY_POINT_GROUP,
    MANIFEST_FILENAME,
    InProcessPlugin,
    PluginSource,
    add_in_process,
    discover_plugin,
    in_process_plugin,
    load_plugins,
)
from iris_harness.runtime.plugin_host.manifest import PluginManifest, RegistrationKind
from iris_harness.runtime.plugin_host.profile import (
    DEFAULT_PLUGINS,
    EffectiveProfile,
    PluginRef,
    load_profile,
)
from iris_harness.runtime.plugin_host.registry import (
    PluginRecord,
    PluginRegistry,
    PluginStatus,
)

__all__ = [
    "BUILTIN_PACKAGE",
    "DEFAULT_PLUGINS",
    "ENTRY_POINT_GROUP",
    "MANIFEST_FILENAME",
    "EffectiveProfile",
    "InProcessPlugin",
    "PluginAPI",
    "PluginManifest",
    "PluginRecord",
    "PluginRef",
    "PluginRegistry",
    "PluginSource",
    "PluginStatus",
    "RegistrationKind",
    "add_in_process",
    "discover_plugin",
    "dump_config",
    "in_process_plugin",
    "load_plugins",
    "load_profile",
]
