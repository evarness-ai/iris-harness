"""Mount a ``flavor: declarative`` plugin: its manifest is the whole plugin.

OSS plan decision 5: a skill is the declarative flavor of plugin. Each tool under
``tools:`` names the plain function it is bound to (``impl: package.module:function``),
what the model reads about it (``description``) and its typed ``args``. The loader builds
the plugin's ``setup`` here -- the plugin ships none -- and that ``setup`` registers each
tool through ``PluginAPI.register_tool``, exactly as a python plugin would: the
manifest's effect/content/confirm declaration applies, the call joins the governed tool
pool and runs through ``PRE_TOOL_USE``, the approval rules and ``POST_TOOL_USE``, inside
the plugin's fault boundary.

Arguments are checked against the declaration before every call (and before any
approval is queued, through the tool's ``validate``): an unknown argument, a missing
required one or a value of the wrong type comes back to the model as an observation
saying what is wrong, and the function never runs. The function returns a string (the
observation) or anything JSON can encode.

Binding happens at mount: an ``impl`` that does not import, is not callable, or cannot
take the declared arguments fails the plugin's ``setup`` -- the plugin is recorded
``FAILED`` with the reason, and the harness boots without it.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from .manifest import PluginManifest, ToolDeclaration, arg_value_problem

if TYPE_CHECKING:
    from .api import PluginAPI


def resolve_impl(impl: str) -> Callable[..., Any]:
    """The function ``package.module:function`` names; ``ImportError`` when it is not one."""
    module_name, _, attr = impl.partition(":")
    module = importlib.import_module(module_name)
    function = getattr(module, attr, None)
    if not callable(function):
        raise ImportError(f"impl {impl!r}: {module_name} has no callable {attr!r}")
    return function  # type: ignore[no-any-return]


def _check_signature(tool: str, decl: ToolDeclaration, function: Callable[..., Any]) -> None:
    """The function accepts every declared argument by keyword and needs no other."""
    try:
        inspect.signature(function).bind(**dict.fromkeys(decl.args))
    except TypeError as exc:
        raise TypeError(
            f"tool {tool!r}: {decl.impl} cannot take the declared args "
            f"({', '.join(decl.args) or 'none'}): {exc}"
        ) from exc


def check_args(decl: ToolDeclaration, args: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    """The call's keyword arguments with defaults filled, or why they cannot be used."""
    unknown = sorted(set(args) - set(decl.args))
    if unknown:
        return {}, f"unknown argument(s): {', '.join(unknown)}"
    values: dict[str, Any] = {}
    for name, arg in decl.args.items():
        if name not in args:
            if arg.required:
                return {}, f"missing argument {name!r}"
            if arg.default is not None:
                values[name] = arg.default
            continue
        problem = arg_value_problem(arg, args[name])
        if problem is not None:
            return {}, f"argument {name!r} {problem}"
        values[name] = args[name]
    return values, None


def describe(decl: ToolDeclaration) -> str:
    """What the model reads: the description, then each argument."""
    parts = []
    for name, arg in decl.args.items():
        kind = f"one of {'|'.join(arg.options)}" if arg.options else arg.type
        if not arg.required:
            kind += ", optional" + (f", default {arg.default}" if arg.default is not None else "")
        parts.append(f'"{name}": {kind}')
    return f"{decl.description.strip()} Args: {{{', '.join(parts)}}}."


def bind_tool(
    decl: ToolDeclaration, function: Callable[..., Any]
) -> Callable[[dict[str, Any]], str]:
    """The tool call: checked arguments in, the function's result as observation text out."""

    def call(args: dict[str, Any]) -> str:
        values, problem = check_args(decl, args)
        if problem is not None:
            return f"error: {problem}"
        result = function(**values)
        return result if isinstance(result, str) else json.dumps(result)

    return call


def declarative_setup(manifest: PluginManifest) -> Callable[[PluginAPI], None]:
    """The ``setup`` a declarative plugin does not ship: bind every tool, then register it.

    Binding (import, signature) runs when this is called -- at mount, inside the loader's
    ``setup`` boundary -- so a bad ``impl`` fails the plugin before anything registers.
    """
    bound: list[tuple[str, ToolDeclaration, Callable[..., Any]]] = []
    for name, decl in manifest.tools.items():
        if decl.impl is None:  # the manifest's validation refuses this; never bind nothing
            raise ImportError(f"tool {name!r} declares no impl")
        function = resolve_impl(decl.impl)
        _check_signature(name, decl, function)
        bound.append((name, decl, function))

    def setup(api: PluginAPI) -> None:
        for name, decl, function in bound:

            def validate(args: dict[str, Any], decl: ToolDeclaration = decl) -> str | None:
                return check_args(decl, args)[1]

            api.register_tool(name, describe(decl), bind_tool(decl, function), validate=validate)

    setup.__qualname__ = f"declarative_setup[{manifest.name}]"
    return setup


__all__ = ["bind_tool", "check_args", "declarative_setup", "describe", "resolve_impl"]
