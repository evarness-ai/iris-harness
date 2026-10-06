"""Plugin registry: what each plugin registered, how it is doing, and the fault boundary.

The registry is the runtime's single view of plugin state. The runtime reads
intercepts, tools and confirmation executors from it at dispatch time; intent
handlers, heartbeats and channels are pushed into their subsystems at load
time but still recorded here so the plugin inventory (``GET /plugins``,
``iris plugins``), the drift panel and System Health see one list.

Fault boundary (OSS plan decision 8): every callable a plugin hands over is
wrapped by :meth:`PluginRegistry.guard`. An exception inside a plugin is
recorded against that plugin (count + last error) and the call degrades in the
way that is safe for its kind:

* intercept       → ``None`` (the chain falls through to the next intercept)
* tool            → an error observation string (the model sees it and moves on)
* intent handler  → an error answer string (the turn completes with an apology)
* heartbeat       → a ``FAILED`` :class:`HeartbeatRun` (the tick is reported, not lost)
* channel / confirmation executor → recorded, then re-raised — silently dropping
  a delivery or an approval would be worse than the exception.

Event-bus subscriptions are not a registration kind — a plugin that subscribes is
consuming a harness service, not providing a capability — but they go through the
same boundary (:meth:`PluginRegistry.add_subscription`, degrading to ``None``) so a
plugin whose subscriber raises still shows up as degraded.

The keyed core seams a plugin fills through :class:`PluginAPI` (API routers, public
callbacks, agent panels, learned sources) are not registration kinds either: each is
a core registry that stays keyed, so a second runtime replaces rather than stacks.
:meth:`PluginRegistry.add_seam` records them (:meth:`PluginRegistry.seams`) and guards
their callables, recording a failure against the plugin and re-raising, because each
consumer already contains a failure (the API skips a router that fails to build, the
digest footer skips a failing source).

Declared capabilities (docs/architecture/plugin-capabilities.md §2) are not a
registration kind either: a capability is a typed interface in the SDK's catalogue, not a
callable the core dispatches. :meth:`PluginRegistry.provide_capability` records a provider
against the manifest's ``capabilities: provides`` (refusing anything undeclared, the closure
rule) and :meth:`PluginRegistry.resolve_capability` hands a consumer ONE implementation --
the provider's, or the spec's fan-out over several. Every call into a provider goes through
the guard, so a failure there is recorded against the provider and re-raised to the
consumer, which owns its degraded path.

One :class:`HealthCheck` per plugin (kind ``plugin``) projects this state into
System Health, so a broken plugin is a red row with the error, not a stack trace
in the chat.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.agent.tool_runner import CapabilityCall, GovernedToolRunner, ToolUnavailable
from iris_harness.foundation.capabilities import (
    CapabilitySpec,
    CapabilityUnavailable,
    published_capability,
    split_capability_tool,
)
from iris_harness.foundation.observability.session_log import current_session_id
from iris_harness.kernel.governance.turn_label import current_turn_label
from iris_harness.runtime.intercepts import InterceptSpec
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthState
from iris_harness.services.heartbeat.models import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatStatus,
)

from .manifest import PluginManifest, RegistrationKind

logger = logging.getLogger(__name__)


class PluginStatus(StrEnum):
    LOADED = "loaded"  # setup ran; no failures since
    DEGRADED = "degraded"  # setup ran; at least one guarded call failed
    FAILED = "failed"  # setup raised, requirements unmet, or not found
    DISABLED = "disabled"  # in the profile but enabled: false
    UNSUPPORTED = "unsupported"  # e.g. trust: mcp before the MCP host lands


@dataclass(frozen=True)
class Registration:
    """One thing a plugin registered."""

    plugin: str
    kind: RegistrationKind
    name: str
    detail: str = ""


@dataclass
class PluginRecord:
    """Everything the registry knows about one plugin."""

    name: str
    source: str  # "builtin:system", "entry_point:foo", "home:/path", "profile" (not found)
    status: PluginStatus
    manifest: PluginManifest | None = None
    trust: str = "in-process"
    registrations: list[Registration] = field(default_factory=list)
    failure_count: int = 0
    last_error: str | None = None
    last_failure_at: str | None = None
    load_error: str | None = None
    # Where the manifest was read from; ``None`` when the plugin was never found
    # (or is disabled, so discovery did not run). Read by the plugin inventory.
    directory: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "status": self.status.value,
            "version": self.manifest.version if self.manifest else None,
            "trust": self.trust,
            "registrations": [
                {"kind": r.kind.value, "name": r.name, "detail": r.detail}
                for r in self.registrations
            ],
            "failure_count": self.failure_count,
            "last_error": self.last_error,
            "load_error": self.load_error,
        }


@dataclass(frozen=True)
class InterceptRegistration:
    spec: InterceptSpec
    handler: Callable[..., Any]
    # Optional: status text to stream *before* this intercept runs, for one that
    # scans a folder or embeds files and would otherwise leave the stream silent
    # for seconds. ``hint(message) -> str | None``; ``None`` means "not mine".
    activity_hint: Callable[[str], str | None] | None = None


MOUNTED = (PluginStatus.LOADED, PluginStatus.DEGRADED)


def _capability_facade(guard: _ProviderGuard, caller: str) -> object:
    """The object a consumer holds: the Protocol's methods, bound to ``caller``.

    Built as a slotted class with one method per Protocol member and no instance state, so
    there is nothing else to reach: any other attribute -- the provider's own methods and
    data, the guard, ``__dict__`` -- is an ``AttributeError``. The caller is the harness's
    stamp (``plugin:<consumer>`` or ``core:<module>``), fixed when the facade is built.
    """
    name = guard.spec.name

    def method(attr: str) -> Callable[..., Any]:
        def call(_self: object, *args: Any, **kwargs: Any) -> Any:
            return guard.invoke(caller, attr, args, kwargs)

        call.__name__ = call.__qualname__ = attr
        return call

    def refuse(_self: object, attr: str) -> Any:
        raise AttributeError(
            f"capability {name!r} has no method {attr!r} (its Protocol does not declare it)"
        )

    namespace: dict[str, Any] = {m: method(m) for m in guard.spec.members()}
    namespace.update(
        __slots__=(), __getattr__=refuse, __repr__=lambda _self: f"{guard!r} for {caller}"
    )
    return type(f"Capability[{name}]", (), namespace)()


class _ProviderGuard:
    """One provider's implementation of a capability: governed, and behind the fault boundary.

    Never handed to a consumer (it gets :func:`_capability_facade`). A call:

    1. checks the provider is still mounted -- a facade held from before its provider
       failed raises :class:`CapabilityUnavailable` instead of calling into it;
    2. binds the arguments to the Protocol's signature;
    3. runs through the governed runner (``GovernedToolRunner.execute_call`` and its async
       twins) as ``capability:<name>.<method>``: ``PRE_TOOL_USE`` before the provider runs,
       ``POST_TOOL_USE`` over what it returned, the consumer getting the redacted copy. With
       no kernel bound the call fails closed;
    4. reaches the provider only through the guard, including what it returns lazily -- an
       awaitable, an async stream, a stream -- so an exception inside the provider is
       recorded against the provider plugin (``capability:<name>.<method>``) and re-raised.
    """

    def __init__(
        self, registry: PluginRegistry, plugin: str, spec: CapabilitySpec, impl: object
    ) -> None:
        self._registry = registry
        self._plugin = plugin
        self.spec = spec
        self._impl = impl

    def invoke(self, caller: str, attr: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self._check_mounted()
        bound = self.spec.signature(attr).bind(None, *args, **kwargs)
        arguments = dict(bound.arguments)
        arguments.pop(next(iter(arguments)))  # the Protocol's ``self``
        declared = self.spec.methods[attr]
        shape = self.spec.shapes[attr]
        call = CapabilityCall(
            caller=caller,
            provider=self._plugin,
            capability=self.spec.name,
            method=attr,
            effect=declared.effect,
            confirm=declared.confirm_mode,
            fields=declared.fields,
            shape=shape,
            value_type=self.spec.value_types[attr],
            content=declared.content,
            # The turn's label, read now, when the consumer makes the call (an async one
            # runs later): the harness stamps it; the consumer has no way to pass one.
            classification=current_turn_label(),
        )
        runner = GovernedToolRunner(
            kernel=self._registry.kernel(),
            agent_type=caller,
            session_id=current_session_id(),
            resumable=False,
        )
        provider_call = self._guarded(attr, shape)
        if shape == "async":
            return runner.aexecute_call(call, provider_call, arguments)
        if shape == "astream":
            return runner.aexecute_stream(call, provider_call, arguments)
        return runner.execute_call(call, provider_call, arguments)

    def _guarded(self, attr: str, shape: str) -> Callable[..., Any]:
        """The provider's method inside the fault boundary, lazily returned values too."""
        where = f"capability:{self.spec.name}.{attr}"
        method = self._registry.guard_where(
            self._plugin, where, getattr(self._impl, attr), degrade=None
        )

        def call(**kwargs: Any) -> Any:
            result = method(**kwargs)
            if shape == "astream":
                return self._guard_agen(where, result)
            if shape == "stream":
                return self._guard_gen(where, iter(result))
            if shape == "async":
                return self._guard_awaitable(where, result)
            return result

        return call

    def _check_mounted(self) -> None:
        record = self._registry.get(self._plugin)
        if record is None or record.status not in MOUNTED:
            status = record.status.value if record is not None else "not registered"
            raise CapabilityUnavailable(
                f"capability {self.spec.name!r}: provider plugin {self._plugin!r} is no "
                f"longer mounted ({status})"
            )

    def _fail(self, where: str, exc: Exception) -> None:
        self._registry.record_failure(self._plugin, where=where, exc=exc)

    async def _guard_awaitable(self, where: str, awaitable: Awaitable[Any]) -> Any:
        try:
            return await awaitable
        except Exception as exc:  # the boundary exists to catch anything
            self._fail(where, exc)
            raise

    async def _guard_agen(self, where: str, agen: AsyncIterator[Any]) -> AsyncIterator[Any]:
        try:
            async for item in agen:
                yield item
        except Exception as exc:
            self._fail(where, exc)
            raise

    def _guard_gen(self, where: str, gen: Iterator[Any]) -> Iterator[Any]:
        try:
            yield from gen
        except Exception as exc:
            self._fail(where, exc)
            raise

    def __repr__(self) -> str:
        return f"<capability {self.spec.name} provided by plugin {self._plugin!r}>"


class PluginRegistry:
    """Registered capabilities by kind + per-plugin health, with the fault boundary."""

    def __init__(self) -> None:
        self._plugins: dict[str, PluginRecord] = {}
        self._intercepts: dict[str, InterceptRegistration] = {}
        self._tools: list[ToolSpec] = []
        self._confirmation_executors: dict[str, Callable[..., Any]] = {}
        self._intent_handlers: dict[str, tuple[Callable[..., Any], Callable[..., Any] | None]] = {}
        # intent -> its guarded degrade path, for intents a plugin puts on the governed loop
        self._loop_intents: dict[str, Callable[..., Any]] = {}
        self._heartbeats: list[tuple[str, Callable[..., Any], HeartbeatDefinition | None]] = []
        self._channels: list[Any] = []
        self._subscriptions: list[tuple[str, str, str]] = []  # (plugin, topic, bus scope)
        self._seams: list[tuple[str, str, str]] = []  # (plugin, seam, key)
        # capability name -> [(provider plugin, its guard)], in mount order
        self._capabilities: dict[str, list[tuple[str, _ProviderGuard]]] = {}
        # The kernel capability calls are governed by; bound by the runtime once it has one.
        # Unbound (None), every capability call fails closed.
        self._kernel: Callable[[], Any] = lambda: None

    # ------------------------------------------------------------------ plugins
    def add_plugin(self, record: PluginRecord) -> PluginRecord:
        self._plugins[record.name] = record
        return record

    def get(self, name: str) -> PluginRecord | None:
        return self._plugins.get(name)

    def plugins(self) -> list[PluginRecord]:
        return list(self._plugins.values())

    def _record(self, plugin: str, kind: RegistrationKind, name: str, detail: str = "") -> None:
        record = self._plugins.get(plugin)
        if record is None:
            record = self.add_plugin(
                PluginRecord(name=plugin, source="unknown", status=PluginStatus.LOADED)
            )
        record.registrations.append(Registration(plugin, kind, name, detail))

    def record_failure(self, plugin: str, *, where: str, exc: BaseException) -> None:
        record = self._plugins.get(plugin)
        if record is None:
            record = self.add_plugin(
                PluginRecord(name=plugin, source="unknown", status=PluginStatus.LOADED)
            )
        record.failure_count += 1
        record.last_error = f"{where}: {type(exc).__name__}: {exc}"
        record.last_failure_at = datetime.now(UTC).isoformat()
        if record.status is PluginStatus.LOADED:
            record.status = PluginStatus.DEGRADED
        logger.exception("plugin %r failed in %s", plugin, where)

    # ---------------------------------------------------------- fault boundary
    def guard(
        self,
        plugin: str,
        kind: RegistrationKind,
        name: str,
        fn: Callable[..., Any],
        *,
        degrade: Callable[[BaseException], Any] | None,
    ) -> Callable[..., Any]:
        """Wrap ``fn`` so an exception is recorded against ``plugin`` and degrades.

        ``degrade`` maps the exception to the kind's safe return value; ``None``
        means record-then-re-raise (channels, confirmation executors).
        """
        return self.guard_where(plugin, f"{kind.value}:{name}", fn, degrade=degrade)

    def guard_where(
        self,
        plugin: str,
        where: str,
        fn: Callable[..., Any],
        *,
        degrade: Callable[[BaseException], Any] | None,
    ) -> Callable[..., Any]:
        """:meth:`guard` addressed by a free-form ``where`` label.

        Used for callables that are not one of the six registration kinds — today
        only event-bus subscriptions, which are a *service* a plugin consumes
        rather than a capability it provides.
        """

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # the boundary exists to catch anything
                self.record_failure(plugin, where=where, exc=exc)
                if degrade is None:
                    raise
                return degrade(exc)

        wrapped.__name__ = getattr(fn, "__name__", "plugin_callable")
        wrapped.__doc__ = getattr(fn, "__doc__", None)
        # So the trace view can name the plugin's code, not this wrapper.
        wrapped.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapped

    def guard_stream(
        self, plugin: str, kind: RegistrationKind, name: str, fn: Callable[..., Iterator[Any]]
    ) -> Callable[..., Iterator[Any]]:
        """Generator flavour of :meth:`guard`: record, then re-raise mid-stream."""
        where = f"{kind.value}:{name}"

        def wrapped(*args: Any, **kwargs: Any) -> Iterator[Any]:
            try:
                yield from fn(*args, **kwargs)
            except Exception as exc:
                self.record_failure(plugin, where=where, exc=exc)
                raise

        wrapped.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapped

    # -------------------------------------------------------------- intercepts
    def add_intercept(
        self,
        plugin: str,
        spec: InterceptSpec,
        handler: Callable[..., Any],
        activity_hint: Callable[[str], str | None] | None = None,
    ) -> None:
        if spec.name in self._intercepts:
            owner = next(
                (
                    r.plugin
                    for p in self._plugins.values()
                    for r in p.registrations
                    if r.kind is RegistrationKind.INTERCEPT and r.name == spec.name
                ),
                "?",
            )
            raise ValueError(f"intercept {spec.name!r} already registered by plugin {owner!r}")
        guarded = self.guard(
            plugin, RegistrationKind.INTERCEPT, spec.name, handler, degrade=lambda _exc: None
        )
        guarded_hint = (
            self.guard_where(
                plugin, f"intercept_hint:{spec.name}", activity_hint, degrade=lambda _exc: None
            )
            if activity_hint is not None
            else None
        )
        self._intercepts[spec.name] = InterceptRegistration(
            spec=spec, handler=guarded, activity_hint=guarded_hint
        )
        self._record(plugin, RegistrationKind.INTERCEPT, spec.name, spec.trace_text or "")

    def intercept(self, name: str) -> InterceptRegistration | None:
        return self._intercepts.get(name)

    def intercepts(self) -> list[InterceptRegistration]:
        return list(self._intercepts.values())

    # ------------------------------------------------------------------- tools
    def add_tool(self, plugin: str, tool: ToolSpec) -> None:
        if any(t.name == tool.name for t in self._tools):
            raise ValueError(f"tool {tool.name!r} already registered by a plugin")

        def unavailable(exc: BaseException) -> Any:
            # Raised past the boundary (the failure is already recorded against the
            # plugin), so the runner sees a failed call rather than a result: the caller
            # is told this sentence, but the call does not count as one that ran.
            raise ToolUnavailable(
                f"{tool.name} is unavailable (plugin {plugin!r} raised "
                f"{type(exc).__name__}: {exc}). Answer without it."
            ) from exc

        guarded = self.guard(
            plugin, RegistrationKind.TOOL, tool.name, tool.call, degrade=unavailable
        )
        # `_replace`, not a field-by-field copy: a copy silently drops every field
        # added to ToolSpec later (ADR-0118's describe/undo would have been lost here).
        self._tools.append(tool._replace(call=guarded, plugin=plugin))
        self._record(plugin, RegistrationKind.TOOL, tool.name)

    def tools(self) -> list[ToolSpec]:
        return list(self._tools)

    # ------------------------------------------------------- permission contract
    def caller_denial(self, caller: str, tool: str) -> str | None:
        """Why ``caller`` (``plugin:<name>``) may not call ``tool``, or None if it may.

        A plugin may call its own tools and the ones its manifest lists under
        ``uses: tools`` (docs/architecture/plugin-capabilities.md §4), and the methods of
        each capability it lists under ``capabilities: uses`` or ``requires``
        (``capability:<name>.<method>``). Everything else is denied, including a caller
        that is not a mounted plugin.
        """
        plugin = caller.removeprefix("plugin:")
        record = self._plugins.get(plugin)
        if record is None:
            return f"{caller} is not a mounted plugin"
        capability_call = split_capability_tool(tool)
        if capability_call is not None:
            # A capability method: granted by the manifest's `capabilities: uses/requires`.
            capability = capability_call[0]
            consumes = record.manifest.capabilities.consumes if record.manifest else ()
            if capability in consumes:
                return None
            return (
                f"{caller} may not call {tool!r}: its manifest does not list {capability!r} "
                "under `capabilities: uses` or `requires`"
            )
        own = {r.name for r in record.registrations if r.kind is RegistrationKind.TOOL}
        if tool in own:
            return None
        allowed = set(record.manifest.uses.tools) if record.manifest is not None else set()
        if tool in allowed:
            return None
        return (
            f"{caller} may not call {tool!r}: it is not the plugin's own tool and its "
            "manifest does not list it under `uses: tools`"
        )

    def uses_tools(self) -> dict[str, tuple[str, ...]]:
        """Each mounted plugin's ``uses: tools`` allow-list (for drift and the graph)."""
        return {
            name: record.manifest.uses.tools
            for name, record in self._plugins.items()
            if record.manifest is not None and record.manifest.uses.tools
        }

    # ------------------------------------------------------------- capabilities
    def _refuse(self, plugin: str, name: str, reason: str) -> None:
        self.record_failure(plugin, where=f"capability:{name}", exc=ValueError(reason))

    def provide_capability(self, plugin: str, name: str, impl: object) -> bool:
        """Register ``plugin``'s implementation of ``name``; False (and a failure) if refused.

        Refused, and recorded as the plugin's failure, when the manifest does not declare
        it under ``capabilities: provides`` (declared == registered), when the SDK publishes
        no such capability, when ``impl`` lacks part of the interface, when the plugin
        provides it twice, or when the capability has no fan-out and another plugin already
        provides it.
        """
        record = self._plugins.get(plugin)
        manifest = record.manifest if record is not None else None
        if manifest is None or name not in manifest.capabilities.provides:
            self._refuse(
                plugin,
                name,
                f"plugin {plugin!r} provides capability {name!r} but its manifest does not "
                "declare it under 'capabilities: provides'",
            )
            return False
        spec = published_capability(name)
        if spec is None:  # no Protocol to agree on: nothing a consumer could code against
            self._refuse(
                plugin,
                name,
                f"capability {name!r} is not published by this SDK "
                "(iris_harness.sdk.capabilities)",
            )
            return False
        missing = spec.missing_members(impl)
        if missing:
            self._refuse(
                plugin,
                name,
                f"plugin {plugin!r}'s {name!r} implementation lacks {', '.join(missing)}",
            )
            return False
        providers = self._capabilities.setdefault(name, [])
        if any(owner == plugin for owner, _impl in providers):
            self._refuse(plugin, name, f"plugin {plugin!r} already provides {name!r}")
            return False
        if providers and spec.fan_out is None:
            self._refuse(
                plugin,
                name,
                f"capability {name!r} has one provider (plugin {providers[0][0]!r}); "
                f"plugin {plugin!r} cannot provide it too",
            )
            return False
        providers.append((plugin, _ProviderGuard(self, plugin, spec, impl)))
        return True

    def bind_kernel(self, kernel: Callable[[], Any]) -> None:
        """Govern capability calls by ``kernel()`` (read per call; None fails closed)."""
        self._kernel = kernel

    def kernel(self) -> Any:
        """The governance kernel capability calls run through, or None."""
        return self._kernel()

    def _mounted_providers(self, name: str) -> list[tuple[str, _ProviderGuard]]:
        # A provider whose setup failed after it provided is not mounted: it provides nothing.
        return [
            (owner, impl)
            for owner, impl in self._capabilities.get(name, [])
            if (rec := self._plugins.get(owner)) is not None and rec.status in MOUNTED
        ]

    def capability_providers(self, name: str) -> tuple[str, ...]:
        """The mounted plugins providing ``name``, in mount order."""
        return tuple(owner for owner, _impl in self._mounted_providers(name))

    def resolve_capability(self, plugin: str, name: str) -> Any:
        """ONE implementation of ``name`` for ``plugin``, or None when nothing provides it.

        ``plugin`` must declare ``name`` under ``capabilities: uses`` or ``requires``;
        undeclared use is refused (a failure on the plugin, and None). Several providers
        are combined by the capability's fan-out (decision 3), so the consumer never
        iterates them.
        """
        record = self._plugins.get(plugin)
        manifest = record.manifest if record is not None else None
        if manifest is None or name not in manifest.capabilities.consumes:
            self._refuse(
                plugin,
                name,
                f"plugin {plugin!r} asked for capability {name!r} but its manifest does not "
                "declare it under 'capabilities: uses' or 'requires'",
            )
            return None
        return self._facade_for(f"plugin:{plugin}", name)

    def capability_for_core(self, module: str, name: str) -> Any:
        """ONE implementation of ``name`` for core code, bound to ``core:<module>``.

        The core has no manifest; its capability calls are governed and audited like a
        plugin's, and the caller policy gives ``core:`` callers the access they have today.
        """
        return self._facade_for(f"core:{module}", name)

    def _facade_for(self, caller: str, name: str) -> Any:
        """Each mounted provider's facade bound to ``caller``, fanned out into one."""
        impls = [
            _capability_facade(guard, caller) for _owner, guard in self._mounted_providers(name)
        ]
        if not impls:
            return None
        spec = published_capability(name)
        if len(impls) == 1 or spec is None or spec.fan_out is None:
            return impls[0]
        return spec.fan_out(impls)

    def provided_capabilities(self) -> dict[str, tuple[str, ...]]:
        """``{capability: providers}`` for every capability a mounted plugin provides."""
        provided = {name: self.capability_providers(name) for name in self._capabilities}
        return {name: owners for name, owners in provided.items() if owners}

    def declared_capabilities(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Each mounted plugin's ``capabilities:`` block (for drift and the graph)."""
        return {
            name: {
                "provides": record.manifest.capabilities.provides,
                "uses": record.manifest.capabilities.uses,
                "requires": record.manifest.capabilities.requires,
            }
            for name, record in self._plugins.items()
            if record.manifest is not None and record.status in MOUNTED
        }

    def read_first_intents(self) -> frozenset[str]:
        """Every intent a mounted plugin declares ``read_first_intents`` for."""
        return frozenset(
            intent
            for record in self._plugins.values()
            if record.manifest is not None
            and record.status in (PluginStatus.LOADED, PluginStatus.DEGRADED)
            for intent in record.manifest.read_first_intents
        )

    def tools_serving(self, intents: Iterable[str]) -> frozenset[str]:
        """The tools of every mounted plugin that serves one of ``intents``.

        A plugin serves an intent (or the agent name it routes to) when it registered
        the intent handler for it, put it on the governed loop, or lists it under
        ``read_first_intents``. The tool shortlist reads this as the turn's own domain
        when it has no embedder to rank by (ADR-0077 addendum, 2026-10-01): what the
        router already decided about the turn, said by the plugins' own declarations.
        """
        wanted = {intent for intent in intents if intent}
        if not wanted:
            return frozenset()
        on_loop = {p for p, seam, key in self._seams if seam == "loop_intent" and key in wanted}
        names: set[str] = set()
        for record in self._plugins.values():
            if record.status not in MOUNTED:
                continue
            serves = (
                record.name in on_loop
                or any(
                    r.kind is RegistrationKind.INTENT_HANDLER and r.name in wanted
                    for r in record.registrations
                )
                or (
                    record.manifest is not None
                    and bool(wanted & set(record.manifest.read_first_intents))
                )
            )
            if serves:
                names.update(
                    r.name for r in record.registrations if r.kind is RegistrationKind.TOOL
                )
        return frozenset(names)

    # ---------------------------------------------------------- intent handlers
    def add_intent_handler(
        self,
        plugin: str,
        agent_type: str,
        handler: Callable[..., Any],
        stream_handler: Callable[..., Iterator[Any]] | None,
    ) -> tuple[Callable[..., Any], Callable[..., Iterator[Any]] | None]:
        guarded = self.guard(
            plugin,
            RegistrationKind.INTENT_HANDLER,
            agent_type,
            handler,
            degrade=lambda exc: (
                f"The {agent_type} capability is unavailable right now "
                f"(plugin {plugin!r} failed: {type(exc).__name__})."
            ),
        )
        guarded_stream = (
            self.guard_stream(plugin, RegistrationKind.INTENT_HANDLER, agent_type, stream_handler)
            if stream_handler is not None
            else None
        )
        self._intent_handlers[agent_type] = (guarded, guarded_stream)
        self._record(
            plugin,
            RegistrationKind.INTENT_HANDLER,
            agent_type,
            "sync+stream" if stream_handler else "sync",
        )
        return guarded, guarded_stream

    def intent_handlers(self) -> dict[str, tuple[Callable[..., Any], Callable[..., Any] | None]]:
        return dict(self._intent_handlers)

    def add_loop_intent(
        self, plugin: str, intent: str, fallback: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Record that ``plugin`` puts ``intent`` on the governed loop; guard its fallback.

        The fallback is guarded like an intent handler (a failure is charged to the
        plugin and answers with an apology), because it is what the owner sees when
        the loop could not answer. The entry is recorded as the ``loop_intent`` seam.
        """
        guarded = self.guard(
            plugin,
            RegistrationKind.INTENT_HANDLER,
            intent,
            fallback,
            degrade=lambda exc: (
                f"The {intent} capability is unavailable right now "
                f"(plugin {plugin!r} failed: {type(exc).__name__})."
            ),
        )
        self._loop_intents[intent] = guarded
        self.declare_seam(plugin, "loop_intent", intent)
        return guarded

    def loop_intents(self) -> dict[str, Callable[..., Any]]:
        """``{intent: guarded fallback}`` for every intent a plugin put on the loop."""
        return dict(self._loop_intents)

    # --------------------------------------------------------------- heartbeats
    def add_heartbeat(
        self,
        plugin: str,
        name: str,
        handler: Callable[..., Any],
        definition: HeartbeatDefinition | None,
    ) -> Callable[..., Any]:
        def degrade(exc: BaseException) -> HeartbeatRun:
            return HeartbeatRun(
                name=name,
                status=HeartbeatStatus.FAILED,
                finished_at=datetime.now(UTC),
                error=f"plugin {plugin!r}: {type(exc).__name__}: {exc}",
            )

        guarded = self.guard(plugin, RegistrationKind.HEARTBEAT, name, handler, degrade=degrade)
        self._heartbeats.append((name, guarded, definition))
        self._record(
            plugin,
            RegistrationKind.HEARTBEAT,
            name,
            definition.schedule if definition is not None else "handler only",
        )
        return guarded

    def heartbeats(self) -> list[tuple[str, Callable[..., Any], HeartbeatDefinition | None]]:
        return list(self._heartbeats)

    # ----------------------------------------------------------------- channels
    def add_channel(self, plugin: str, connector: Any) -> Any:
        name = str(getattr(connector, "name", type(connector).__name__))
        original_send = connector.send
        connector.send = self.guard(  # record delivery failures; still raise to the caller
            plugin, RegistrationKind.CHANNEL, name, original_send, degrade=None
        )
        self._channels.append(connector)
        self._record(plugin, RegistrationKind.CHANNEL, name)
        return connector

    def channels(self) -> list[Any]:
        return list(self._channels)

    # --------------------------------------------------- confirmation executors
    def add_confirmation_executor(
        self, plugin: str, kind: str, executor: Callable[..., Any]
    ) -> None:
        if kind in self._confirmation_executors:
            raise ValueError(f"confirmation executor {kind!r} already registered")
        self._confirmation_executors[kind] = self.guard(
            plugin, RegistrationKind.CONFIRMATION_EXECUTOR, kind, executor, degrade=None
        )
        self._record(plugin, RegistrationKind.CONFIRMATION_EXECUTOR, kind)

    def confirmation_executors(self) -> dict[str, Callable[..., Any]]:
        return dict(self._confirmation_executors)

    # ------------------------------------------------------------- subscriptions
    def add_subscription(
        self, plugin: str, topic: str, handler: Callable[..., Any], *, scope: str = "runtime"
    ) -> Callable[..., Any]:
        """Guard a bus subscriber and record it against ``plugin``.

        Not a registration kind: subscribing is consuming a harness service, the
        way a plugin consumes ``tier_router``. It is recorded so the plugin inventory
        (``GET /plugins/{name}``, ``iris plugins show``) and System Health can
        attribute a failing subscriber, and guarded so the
        failure lands on the plugin's record rather than only in the bus's log.
        Degrades to ``None`` — one plugin's bad subscriber must not stop the other
        subscribers on that topic, which is also ``EventBus``'s own contract.

        ``scope`` is recorded with the topic because a subscription on the wrong
        bus fails *silently* (the handler just never fires), so the inventory has to
        show which bus each one landed on (OSS plan M3.2).
        """
        guarded = self.guard_where(
            plugin, f"subscription:{topic}", handler, degrade=lambda _exc: None
        )
        self._subscriptions.append((plugin, topic, scope))
        return guarded

    def subscriptions(self) -> list[tuple[str, str, str]]:
        """``(plugin, topic, scope)`` for every subscription made through the API."""
        return list(self._subscriptions)

    # -------------------------------------------------------------------- seams
    def add_seam(
        self, plugin: str, seam: str, key: str, fn: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Record that ``plugin`` filled ``key`` of the core seam ``seam``; guard ``fn``.

        The guard records a failure against the plugin (a yellow row in System Health)
        and re-raises: the seam's consumer already skips a failing entry, so this adds
        attribution without changing what the owner sees.
        """
        self.declare_seam(plugin, seam, key)
        return self.guard_where(plugin, f"{seam}:{key}", fn, degrade=None)

    def declare_seam(self, plugin: str, seam: str, key: str) -> None:
        """Record a seam entry with no callable (a public callback path)."""
        self._seams.append((plugin, seam, key))

    def seams(self) -> list[tuple[str, str, str]]:
        """``(plugin, seam, key)`` for every core seam filled through the API."""
        return list(self._seams)

    # ------------------------------------------------------------------ health
    def health_checks(self) -> list[HealthCheck]:
        """One check per plugin (kind ``plugin``) for the System Health snapshot."""
        checks: list[HealthCheck] = []
        for rec in self._plugins.values():
            target = f"plugin:{rec.name}"
            if rec.status is PluginStatus.LOADED:
                state, detail = HealthState.GREEN, (
                    f"loaded from {rec.source}; {len(rec.registrations)} registration(s)"
                )
            elif rec.status is PluginStatus.DEGRADED:
                state = HealthState.YELLOW
                detail = f"{rec.failure_count} failure(s); last: {rec.last_error}"
            elif rec.status is PluginStatus.FAILED:
                state, detail = HealthState.RED, rec.load_error or "failed to load"
            else:  # disabled / unsupported — informational
                state, detail = HealthState.GREY, rec.load_error or rec.status.value
            checks.append(
                HealthCheck(
                    target=target,
                    kind=CheckKind.PLUGIN,
                    state=state,
                    detail=detail,
                    # `--dump-config` only reads manifests, so it cannot show why
                    # setup() failed; the live inventory carries the load error.
                    action=(f"iris plugins show {rec.name}" if state is HealthState.RED else None),
                )
            )
        return checks

    # ---------------------------------------------------------------- describe
    def describe(self) -> dict[str, Any]:
        """JSON-ready tree of the whole registry (tests, debugging).

        The owner-facing view is per plugin: ``inventory.py`` behind ``GET /plugins``
        and ``iris plugins``. ``--dump-config`` never boots, so it never sees this.
        """
        return {
            "plugins": [rec.as_dict() for rec in self._plugins.values()],
            "intercepts": [r.spec.name for r in self._intercepts.values()],
            "tools": [t.name for t in self._tools],
            "intent_handlers": sorted(self._intent_handlers),
            "heartbeats": [name for name, _h, _d in self._heartbeats],
            "channels": [str(getattr(c, "name", "?")) for c in self._channels],
            "confirmation_executors": sorted(self._confirmation_executors),
            "subscriptions": [
                f"{plugin}:{topic}@{scope}" for plugin, topic, scope in self._subscriptions
            ],
            "seams": [f"{plugin}:{seam}:{key}" for plugin, seam, key in self._seams],
            "capabilities": {
                name: list(owners) for name, owners in self.provided_capabilities().items()
            },
        }


__all__ = [
    "InterceptRegistration",
    "PluginRecord",
    "PluginRegistry",
    "PluginStatus",
    "Registration",
]
