"""Plugin-contributed ``iris`` subcommands, without booting the harness.

A command group is a *static* surface: ``iris files --help`` must not pay for
building a runtime, opening databases, or calling any plugin's ``setup``. So this
is a second, much smaller entry point than :mod:`iris_harness.runtime.plugin_host.loader` —
it reads the profile, reads each enabled plugin's manifest, imports only the one
module named by ``cli:``, and calls it with a :class:`PluginCLI`.

That means a CLI contribution gets **no** ``HarnessServices``: no tier router, no
agent executor, no event bus. A command body that needs the runtime should build
it itself, the way the core's own commands do.

Failures are contained. A plugin whose CLI module is missing, raises on import, or
throws while registering is logged and skipped — ``iris`` still starts, minus that
plugin's commands. Losing a subcommand is a smaller harm than an unusable CLI.

A command body prints through `console` (the one console the core CLI prints to)
and reports a failure with `print_error`, so plugin output reads like the core's.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iris_harness.foundation.console import console, print_error
from iris_harness.foundation.paths import config_dir as resolve_config_dir
from iris_harness.runtime.plugin_host.loader import (
    BUILTIN_PACKAGE,
    MANIFEST_FILENAME,
    PluginSource,
    discover_plugin,
)
from iris_harness.runtime.plugin_host.profile import load_profile

logger = logging.getLogger(__name__)


@dataclass
class PluginCLI:
    """What a plugin's ``cli:`` function receives.

    ``group("files")`` hands back the core's existing ``iris files`` Typer app, so a
    plugin adds commands *into* a group the harness already publishes rather than
    starting a rival one. Asking for a group the core does not define creates it and
    attaches it to the root app, which is how a plugin ships a whole new group.
    """

    root: Any
    _groups: dict[str, Any] = field(default_factory=dict)

    def group(self, name: str, help: str | None = None) -> Any:
        """The Typer app for ``iris <name>``, created and attached if it is new.

        ``help`` is the group's one-line description on ``iris --help``. It applies
        only when this call creates the group: a plugin that adds commands to a group
        the core (or an earlier plugin) already published does not get to relabel it.

        A dotted name addresses a nested group the core already published, e.g.
        ``group("files.photos")`` is ``iris files photos``. Typer gives no way to
        look a sub-app up by name after the fact, so the core registers the nested
        ones it wants plugins to extend; asking for an unregistered dotted name
        raises rather than silently attaching a second group at the top level.
        """
        if "." in name and name not in self._groups:
            raise KeyError(
                f"no such command group: {name!r} — the harness has not published it "
                "for plugins to extend"
            )
        existing = self._groups.get(name)
        if existing is not None:
            return existing
        import typer  # only the CLI path needs typer

        created = typer.Typer(name=name, help=help, no_args_is_help=True)
        self.root.add_typer(created, name=name)
        self._groups[name] = created
        return created

    def register_group(self, name: str, app: Any) -> None:
        """Record a group the core already built, so ``group(name)`` returns it."""
        self._groups[name] = app


def _import_cli(source: PluginSource) -> Any:
    """Import just the module named by ``cli:`` and return its function."""
    manifest = source.manifest
    module_name = manifest.cli_module
    function_name = manifest.cli_function
    if module_name is None or function_name is None:
        return None
    if source.kind == "builtin":
        module = importlib.import_module(f"{source.location}.{module_name}")
    elif source.entry_point is not None:
        # The entry point names a module inside the plugin's package; `cli:` names a
        # module inside that same package. So resolve the package the entry point sits
        # in -- itself when it points at a package, its parent when it points at a
        # module. Taking the top-level name instead (the pre-M6.1b rule) breaks the
        # moment a plugin is nested, e.g. iris_personal.plugins.finance_workflows.
        ep_module = source.entry_point.value.split(":", 1)[0].strip()
        try:
            spec = importlib.util.find_spec(ep_module)
        except (ImportError, ValueError):
            spec = None
        is_package = spec is not None and spec.submodule_search_locations is not None
        package = ep_module if is_package else ep_module.rpartition(".")[0] or ep_module
        module = importlib.import_module(f"{package}.{module_name}")
    else:
        assert source.directory is not None
        file_path = source.directory / f"{module_name.replace('.', '/')}.py"
        unique = f"iris_plugin_cli_{source.name.replace('-', '_')}_{module_name}"
        spec = importlib.util.spec_from_file_location(unique, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load plugin CLI module: {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique] = module
        spec.loader.exec_module(module)
    fn = getattr(module, function_name, None)
    if not callable(fn):
        raise ImportError(f"plugin {source.name!r}: {manifest.cli} is not callable")
    return fn


def register_plugin_commands(
    cli: PluginCLI,
    *,
    config_dir: Path | None = None,
    home_dir: Path | None = None,
    builtin_package: str = BUILTIN_PACKAGE,
) -> list[str]:
    """Add every enabled plugin's subcommands to ``cli``. Returns the names that ran.

    Never raises: this sits on the ``iris`` import path, so one broken plugin must
    not take the whole CLI down with it.
    """
    added: list[str] = []
    try:
        profile = load_profile(config_dir or resolve_config_dir(), home_dir=home_dir)
    except Exception:  # a bad profile must not break `iris --help`
        logger.debug("plugin CLI: profile unreadable; no plugin commands", exc_info=True)
        return added

    for ref in profile.plugins:
        if not ref.enabled:
            continue
        try:
            source = discover_plugin(ref.name, home_dir=home_dir, builtin_package=builtin_package)
            if source is None or source.manifest.cli is None:
                continue
            register = _import_cli(source)
            if register is None:
                continue
            register(cli)
            added.append(ref.name)
        except Exception:  # skip this plugin's commands, keep the CLI
            logger.warning(
                "plugin %r: CLI commands not registered (%s missing or failing); "
                "the rest of `iris` is unaffected",
                ref.name,
                MANIFEST_FILENAME,
                exc_info=True,
            )
    return added


__all__ = ["PluginCLI", "console", "print_error", "register_plugin_commands"]
