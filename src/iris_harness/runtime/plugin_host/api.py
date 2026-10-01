"""The surface a plugin's ``setup(api)`` sees.

A plugin never imports runtime internals: it receives one :class:`PluginAPI`
bound to its name, calls ``register_*`` for what it provides, and reads the
few harness services it may need from :attr:`PluginAPI.services`. Every
registration is recorded in the :class:`PluginRegistry` and wrapped by the
fault boundary there; the kinds that live in a subsystem (intent handlers,
heartbeats, channels) are also pushed into that subsystem immediately, through
the same entry points the core uses, so a plugin-registered capability is
kernel-governed by construction — it runs through the same ``AgentExecutor``,
``HeartbeatScheduler``, ``ChannelGateway`` and ReAct tool loop as a built-in.

**Who is calling is the harness's stamp.** The registry is private to the API, and the
``services`` a plugin reads carry a tool *catalogue* (``describe`` only), not the tool
service: the only bound entries are ``api.tools`` (``plugin:<this plugin>``),
``api.capability`` and ``api.register_owner_identity_source`` (the same). A plugin has no public way to act as the core
(``core:<module>``) or as another plugin.

**The in-process trust limit.** A plugin mounted ``trust: in-process`` runs in the harness's
own interpreter, so this is a contract, not a sandbox: reaching into private attributes
(``api._registry``, ``_service``) or importing runtime internals is unsupported and can
impersonate anyone; the import-linter contracts forbid the second for the code in this
repository. The real boundary is ``trust: mcp`` -- an out-of-process plugin that holds no
Python reference to the harness at all.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from iris_harness.agent.agentic_core import ToolDescription, ToolSpec
from iris_harness.kernel.governance.identity_redaction import register_owner_identity_source
from iris_harness.runtime.harness_services import HarnessServices
from iris_harness.runtime.intercepts import InterceptSpec
from iris_harness.runtime.plugin_host.harness_topics import harness_bus, refuse_harness_topic
from iris_harness.runtime.tool_service import ToolCatalogue, ToolService
from iris_harness.services.heartbeat.models import HeartbeatDefinition
from iris_harness.services.research.providers import (
    SearchHit,
    SearchProvider,
    check_hits,
    register_search_provider,
)

from .manifest import RegistrationKind
from .registry import MOUNTED, PluginRegistry

if TYPE_CHECKING:
    from iris_harness.kernel.governance.owner_identity import IdentityKind
    from iris_harness.runtime.tool_service import BoundTools
    from iris_harness.services.health.models import HealthCheck

# The seam name a plugin's identity source is recorded under (inventory, System Health).
OWNER_IDENTITY_SEAM = "owner_identity"
# The seam a plugin's search provider is recorded under.
SEARCH_PROVIDER_SEAM = "search_provider"


@dataclasses.dataclass(frozen=True)
class _GuardedSearchProvider:
    """A plugin's search provider with both calls inside the plugin's fault boundary."""

    is_available: Callable[[], bool]
    search: Callable[..., Any]


def _checked_search(name: str, provider: SearchProvider) -> Callable[..., list[SearchHit]]:
    """``provider.search``, its return value held to the contract (``check_hits``)."""

    # ``wraps``: the trace view names the plugin's own method, not this check.
    @functools.wraps(provider.search)
    def search(*args: Any, **kwargs: Any) -> list[SearchHit]:
        return check_hits(name, provider.search(*args, **kwargs))

    return search


class PluginAPI:
    """Registration surface bound to one plugin.

    ``services`` is the plugin's view of the harness: the same objects the core uses, except
    that ``services.tools`` is a :class:`ToolCatalogue` (what exists and what it declares)
    rather than the tool service, which could bind any caller. The registry is private.
    """

    def __init__(self, *, plugin: str, services: HarnessServices, registry: PluginRegistry) -> None:
        self.plugin = plugin
        self._registry = registry
        service = services.tools
        self._tool_service = service if isinstance(service, ToolService) else None
        self.services = dataclasses.replace(
            services,
            tools=ToolCatalogue(self._tool_service) if self._tool_service is not None else None,
        )

    # -- tools: another plugin's tool, from code, through governance ----------------
    @property
    def tools(self) -> BoundTools | None:
        """Registered tools, bound to this plugin as the caller (``plugin:<name>``).

        ``api.tools.call(name, args)`` runs the tool through the same governed runner the
        model's calls take: ``PRE_TOOL_USE``, the approval rules, ``POST_TOOL_USE`` and an
        audit row naming this plugin. The caller is the harness's stamp, not the plugin's
        claim. None when no runtime is wired (a bare API in a test).
        """
        service = self._tool_service
        return service.for_caller(f"plugin:{self.plugin}") if service is not None else None

    def on_approved_call(self, handler: Callable[[Any], Any]) -> None:
        """Run ``handler(payload)`` when one of THIS plugin's queued calls is settled.

        A call ``api.tools`` held for the owner's approval (its ``ToolResult`` carries the
        ``approval_id``) runs later, once they approve it — long after the calling code
        returned. This is how the outcome gets back: an
        ``ApprovalCallCompletedPayload`` (``approval_id``, ``caller``, ``tool``,
        ``status``: ran / failed / denied / rejected / expired, masked ``summary``) on
        ``approval.call_completed``, filtered to this plugin's own calls. A subscription
        like any other: recorded on the registry, inside the fault boundary.
        """
        from iris_harness.kernel.governance.approvals.events import (
            APPROVAL_CALL_COMPLETED,
            ApprovalCallCompletedPayload,
        )

        caller = f"plugin:{self.plugin}"

        def mine(payload: Any) -> Any:
            if isinstance(payload, ApprovalCallCompletedPayload) and payload.caller == caller:
                return handler(payload)
            return None

        # The one sanctioned subscription to the topic, filtered to this plugin.
        self._subscribe(APPROVAL_CALL_COMPLETED, mine, scope="runtime")

    # -- capabilities: typed service interfaces, provided and consumed in code ---------
    def provide(self, name: str, impl: object) -> None:
        """Provide capability ``name`` (``domain.verb``) with ``impl``, in ``setup``.

        ``impl`` implements the capability's Protocol from ``iris_harness.sdk.capabilities``.
        Refused -- recorded as this plugin's failure, a yellow row in System Health, and not
        provided -- when the manifest does not list ``name`` under ``capabilities: provides``,
        the SDK publishes no such capability, or ``impl`` lacks part of its interface
        (docs/architecture/plugin-capabilities.md §2). Every call into ``impl`` then goes
        through the fault boundary, so a failure there is attributed to this plugin.
        """
        self._registry.provide_capability(self.plugin, name, impl)

    def capability(self, name: str) -> Any:
        """The implementation of capability ``name``, or None when no mounted plugin provides it.

        ``name`` must be listed under ``capabilities: uses`` (None means: take your degraded
        path) or ``requires`` (you are not loaded without it). Asking for one you did not
        declare is refused: recorded as this plugin's failure, and None. With several
        providers you still get one implementation, which fans out to all of them.

        Every method call on it is governed like a tool call, as ``plugin:<this plugin>``
        (``capability:<name>.<method>``): the caller policy, the method's declared effect,
        an audit row, and the owner's identity masked out of a *copy* of the result. A call
        governance stops raises ``CapabilityDenied`` (a ``CapabilityUnavailable``).
        """
        return self._registry.resolve_capability(self.plugin, name)

    # -- intercept: deterministic short-circuit before the agent loop -------------
    def register_intercept(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        trace_text: str | None = None,
        trace_fields: tuple[str, ...] = (),
        passes_channel: bool = False,
        activity_hint: Callable[[str], str | None] | None = None,
        guard_output: bool = False,
    ) -> None:
        """``handler(message, *, session_id, span[, channel]) -> ChatResult | None``.

        Return ``None`` to fall through. The chain position comes from the
        profile / ``config/intercepts.yaml``; unlisted plugin intercepts run after
        the declared ones, in registration order.

        ``activity_hint(message) -> str | None`` is optional and belongs to an
        intercept that takes seconds — it scans a folder, embeds files, runs a
        vision pass. The harness streams the string it returns *before* running the
        chain, so the caller sees a "working" line instead of silence; return
        ``None`` for a message this intercept would not claim. It is a property of
        the intercept, not a separate registration.

        ``guard_output=True`` declares that the answer repeats text someone else wrote
        (an email subject, a sender, statement text). Every answer passes the model-free
        response check; when the model-based output guard is enabled it also runs on
        this intercept's answers. ``config/intercepts.yaml`` can set it too — either side
        saying yes is enough.
        """
        spec = InterceptSpec(
            name=name,
            handler=f"plugin:{self.plugin}",
            passes_channel=passes_channel,
            trace_text=trace_text,
            trace_fields=tuple(trace_fields),
            guard_output=guard_output,
        )
        self._registry.add_intercept(self.plugin, spec, handler, activity_hint)

    # -- tool: joins the governed ReAct tool pool --------------------------------
    def register_tool(
        self,
        name: str,
        description: str,
        call: Callable[[dict[str, Any]], str],
        *,
        describe: Callable[[dict[str, Any]], ToolDescription] | None = None,
        validate: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> None:
        """Register a ReAct tool, with the effect its manifest declares (ADR-0110).

        A plugin that has a manifest must declare every tool it registers under
        ``tools:``; an undeclared one is refused and recorded as the plugin's failure
        (a yellow row in System Health), because an undeclared write would bypass the
        confirm-once rule. A plugin mounted without a manifest (tests, probes) keeps
        the defaults: a read tool.

        ``describe`` is for a destructive tool (ADR-0118): given the call's arguments it
        returns the approval card's title and one line per item, looked up in the
        plugin's own data. Without it the card shows the raw call. ``validate`` checks
        the arguments against the same data before any approval is queued and returns
        why the call cannot run (the model sees it), or None.
        """
        spec = self.declare_tool(
            ToolSpec(
                name=name,
                description=description,
                call=call,
                describe=describe,
                validate=validate,
            )
        )
        if spec is not None:
            self._registry.add_tool(self.plugin, spec)

    def declare_tool(self, tool: ToolSpec) -> ToolSpec | None:
        """Return *tool* carrying its manifest declaration, or None when undeclared.

        ``register_tool`` uses this for the shared pool. A plugin agent that runs its
        own loop over its own tools (the email agent) passes them through here too, so
        that loop honours the same declaration — effect, confirm, guidance, pinned,
        answers_directly — rather than the ToolSpec defaults. An undeclared tool is
        recorded as the plugin's failure, exactly as registration records it.
        """
        record = self._registry.get(self.plugin)
        manifest = record.manifest if record is not None else None
        if manifest is None:
            return tool
        declared = manifest.tools.get(tool.name)
        if declared is None:
            self._registry.record_failure(
                self.plugin,
                where=f"tool:{tool.name}",
                exc=ValueError(
                    f"tool {tool.name!r} is registered by plugin {self.plugin!r} but not "
                    "declared under 'tools:' in its manifest (ADR-0110)"
                ),
            )
            return None
        return tool._replace(
            effect=declared.effect,
            confirm=declared.confirm_mode,
            guidance=declared.guidance,
            pinned=declared.pinned,
            answers_directly=declared.answers_directly,
            undo=declared.undo,
            undo_window_days=declared.undo_window_days,
            content=declared.content,
            verify=declared.verify,
            sends_to=declared.sends_to,
            executes_code=declared.executes_code,
        )

    # -- intent handler: a per-intent agent on the executor -----------------------
    def register_intent_handler(
        self,
        agent_type: str,
        handler: Callable[..., Any],
        *,
        stream_handler: Callable[..., Iterator[Any]] | None = None,
    ) -> None:
        guarded, guarded_stream = self._registry.add_intent_handler(
            self.plugin, agent_type, handler, stream_handler
        )
        self.services.agent_executor.register(agent_type, guarded)
        if guarded_stream is not None:
            self.services.agent_executor.register_stream(agent_type, guarded_stream)

    def register_loop_intent(self, intent: str, *, fallback: Callable[..., Any]) -> None:
        """Have the harness's governed loop answer ``intent``, with ``fallback`` as its degrade path.

        When the loop is on, the harness answers the intent on the one loop it builds
        and governs (PRE_LLM_CALL, the tool runner, approvals and resume), over the
        tools every plugin registered, and calls ``fallback(task)`` when a turn of this
        intent errors, answers nothing or answers without reading. ``fallback`` has the
        intent-handler shape (``task -> str | (str, dict)``) and should be deterministic:
        it is the floor under the loop. With the loop off this registers nothing; an
        intent's lane then is whatever :meth:`register_intent_handler` put there.
        """
        self._registry.add_loop_intent(self.plugin, intent, fallback)

    # -- heartbeat: background tick ----------------------------------------------
    def register_heartbeat(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        schedule: str | None = None,
        description: str = "",
        enabled: bool = True,
    ) -> None:
        """Register a handler under ``name``; with ``schedule`` also register the definition.

        Without ``schedule`` the handler only backs a definition declared in
        ``config/heartbeats.yaml`` (the core's own pattern).
        """
        definition = (
            HeartbeatDefinition(
                name=name,
                handler=name,
                schedule=schedule,
                enabled=enabled,
                description=description,
            )
            if schedule is not None
            else None
        )
        guarded = self._registry.add_heartbeat(self.plugin, name, handler, definition)
        self.services.heartbeats.register_handler(name, guarded)
        if definition is not None:
            self.services.heartbeats.register(definition)

    # -- channel: a delivery connector on the gateway -----------------------------
    def register_channel(self, connector: Any) -> None:
        guarded = self._registry.add_channel(self.plugin, connector)
        self.services.channels.register(guarded)

    # -- event bus: a subsystem *service*, not a registration kind ---------------
    def _bus_for(self, scope: str, topic: str, verb: str) -> Any:
        """The bus ``scope`` names, or a clear error.

        IRIS runs **two** buses, and which one a topic lives on is a property of
        the producer, not a detail a plugin may guess wrong (OSS plan M3.2):

        ``"runtime"`` (default)
            ``HarnessServices.events`` — this runtime's private bus. Activities
            publish here so completion subscribers fire only for THIS runtime and
            do not cross-talk between runtimes in tests.

        ``"process"``
            ``eventbus.get_default_bus()`` — the process-global singleton. The
            domain chains (``email.new_arrived`` → triage → ``email.classified``
            → wiki/followup) publish here, because ``iris email recategorize``
            and ``iris email reingest-wiki`` emit on them with **no runtime
            built at all**. A CLI-driven chain cannot use a runtime-scoped bus.

        Subscribing on the wrong bus is silent — the handler simply never fires —
        so the scope is explicit and wrong values raise.
        """
        if scope == "runtime":
            # The raw bus: PluginAPI is harness code and applies the harness-topic
            # refusal itself; the plugin's own ``services.events`` is the guarded view.
            bus = harness_bus(self.services.events)
            if bus is None:
                raise RuntimeError(
                    f"plugin {self.plugin!r} {verb} {topic!r} but the harness "
                    "exposes no event bus"
                )
            return bus
        if scope == "process":
            from iris_harness.foundation.eventbus import get_default_bus

            return get_default_bus()
        raise ValueError(
            f"plugin {self.plugin!r} {verb} {topic!r} with unknown bus scope "
            f"{scope!r} — expected 'runtime' or 'process'"
        )

    def subscribe(
        self, topic: str, handler: Callable[[Any], Any], *, scope: str = "runtime"
    ) -> None:
        """Run ``handler(payload)`` whenever ``topic`` is emitted on the ``scope`` bus.

        Async work that finishes outside a chat turn — an Activity completing, a
        reminder firing, a mail sweep landing — reaches a plugin here rather than
        through a seventh registration kind. The producer already publishes typed
        payloads on named topics (e.g. ``iris_harness.services.activities.events``), so a
        plugin subscribes to the same contract the core does.

        See :meth:`_bus_for` for choosing ``scope``. Either way the handler is
        wrapped by the registry's fault boundary and degrades to ``None``: a
        raising subscriber is recorded against this plugin and the remaining
        subscribers on the topic still run.

        ``approval.call_completed`` is refused here: it carries every plugin's approved
        calls and their summaries, so a plugin hears its own through
        :meth:`on_approved_call`, which filters by caller, and never anyone else's.
        """
        refuse_harness_topic(self.plugin, topic, "subscribe to")
        self._subscribe(topic, handler, scope=scope)

    def _subscribe(self, topic: str, handler: Callable[[Any], Any], *, scope: str) -> None:
        bus = self._bus_for(scope, topic, "subscribed to")
        bus.on(topic, self._registry.add_subscription(self.plugin, topic, handler, scope=scope))

    def publish(self, topic: str, payload: Any = None, *, scope: str = "runtime") -> None:
        """Emit ``payload`` on ``topic`` from synchronous code.

        A harness-owned topic (``approval.call_completed``) is refused: only the harness
        says how an approved call ended, so no plugin can forge another's outcome.
        """
        refuse_harness_topic(self.plugin, topic, "publish")
        self._bus_for(scope, topic, "published").emit_sync(topic, payload)

    # -- keyed core seams: surfaces the core serves, filled by a plugin ----------
    # Not registration kinds (see the registry's module docstring): each forwards to
    # a keyed core registry, recorded on the plugin registry and inside the fault boundary.
    def register_api_router(self, key: str, factory: Callable[[], Any]) -> None:
        """Mount ``factory()`` (a ``fastapi.APIRouter``) on the API service under ``key``.

        The owner's rule is that every capability has an API; the API service mounts
        this after its own routes and never imports the plugin. Keyed: registering
        ``key`` again replaces it. The factory runs once per app build, with no
        arguments, so close over what ``setup()`` was given.
        """
        from iris_harness.runtime.api_routes import register_api_router

        register_api_router(key, self._registry.add_seam(self.plugin, "api_router", key, factory))

    def register_public_callback(self, path: str) -> None:
        """Let browsers reach ``GET path`` (under ``/api/v1/``) with no token or cookie.

        Only for a redirect back from a third party (an OAuth consent page), and only
        because the route authenticates the request itself with a one-time secret it
        issued earlier: declaring the path hands the plugin that duty.
        """
        from iris_harness.runtime.api_routes import register_public_callback

        register_public_callback(path)
        self._registry.declare_seam(self.plugin, "public_callback", path)

    def register_agent_panel(self, agent: str, build: Callable[[], dict[str, Any]]) -> None:
        """Show ``build()`` (a JSON body) as ``agent``'s panel on ``GET /agents/{agent}``.

        Runs per dashboard request. Keyed by agent: registering again replaces it.
        """
        from iris_harness.runtime.agent_panels import register_agent_panel

        register_agent_panel(
            agent, self._registry.add_seam(self.plugin, "agent_panel", agent, build)
        )

    def register_learned_source(
        self, name: str, source: Callable[[datetime, datetime], list[str]]
    ) -> None:
        """Add ``source(start, end) -> phrases`` to the digest's "learned yesterday" line.

        ``start`` / ``end`` bound the owner's previous local day (tz-aware); return
        short phrases in order ("hid 2 senders"). A failing source is skipped, never
        blanking the line. Keyed by ``name``: registering again replaces it.
        """
        from iris_harness.services.digest.learned import (
            register_learned_source,
        )

        register_learned_source(
            name, self._registry.add_seam(self.plugin, "learned_source", name, source)
        )

    def register_footer_line(
        self, name: str, line: Callable[[datetime, datetime], str | None]
    ) -> None:
        """Add one more line to the digest footer, after "learned yesterday".

        ``line(start, end)`` gets the owner's previous local day (tz-aware) and returns
        one short line, or None to stay silent ("Email jobs: 3/3 judged"). Its words are
        yours, from your config. A failing line is skipped, never taking the digest down.
        Keyed by ``name``: registering again replaces it.
        """
        from iris_harness.services.digest.footer import register_footer_line

        register_footer_line(name, self._registry.add_seam(self.plugin, "footer_line", name, line))

    def register_search_provider(
        self, name: str, provider: SearchProvider, *, priority: int | None = None
    ) -> None:
        """Add ``provider`` to the ``research`` tool's search-provider chain as ``name``.

        ``provider`` is an ``iris_harness.sdk.research.SearchProvider``. It serves research
        calls through the same chain as the built-in providers, so the egress guards, the
        cache, the rerank and the audit apply to it unchanged. Its place is
        ``config/search_providers.yaml``'s priority for ``name``, else ``priority``, else
        the file's default (lower runs first). Its calls run inside this plugin's fault
        boundary -- one that raises, or returns anything but a list of ``SearchHit``, is
        charged to the plugin, and the chain moves on -- and it leaves the chain when the
        plugin is no longer mounted. A name the manifest does not declare under
        ``search_providers:``, a name another mounted plugin holds, or an object that is not
        a provider, is refused and recorded as this plugin's failure.
        """
        where = f"{SEARCH_PROVIDER_SEAM}:{name}"
        record = self._registry.get(self.plugin)
        manifest = record.manifest if record is not None else None
        # Without a manifest (a test, a probe) nothing is declared and nothing is held to
        # it -- the same default an undeclared tool gets (ADR-0110).
        if manifest is not None and name not in manifest.search_providers:
            self._registry.record_failure(
                self.plugin,
                where=where,
                exc=ValueError(
                    f"search provider {name!r} is registered by plugin {self.plugin!r} but "
                    "not declared under 'search_providers:' in its manifest"
                ),
            )
            return
        if not isinstance(provider, SearchProvider):
            self._registry.record_failure(
                self.plugin,
                where=where,
                exc=TypeError(
                    f"plugin {self.plugin!r} registered search provider {name!r}, which is "
                    "not a SearchProvider (is_available() and search(query, *, max_results))"
                ),
            )
            return
        guarded = _GuardedSearchProvider(
            is_available=self._registry.guard_where(
                self.plugin, f"{where}.is_available", provider.is_available, degrade=None
            ),
            # The contract check inside the boundary: a wrong return type is the plugin's
            # failure (degraded, a yellow row), not a silent empty answer.
            search=self._registry.guard_where(
                self.plugin, f"{where}.search", _checked_search(name, provider), degrade=None
            ),
        )
        registry, plugin = self._registry, self.plugin

        def mounted() -> bool:
            record = registry.get(plugin)
            return record is not None and record.status in MOUNTED

        try:
            register_search_provider(
                name, guarded, owner=f"plugin:{plugin}", priority=priority, live=mounted
            )
        except ValueError as exc:
            self._registry.record_failure(self.plugin, where=where, exc=exc)
            return
        self._registry.declare_seam(self.plugin, SEARCH_PROVIDER_SEAM, name)

    def register_credential_check(
        self, name: str, check: Callable[[bool], list[HealthCheck]]
    ) -> None:
        """Add ``check(net_probe)``'s rows to System Health's credentials, after the core's.

        For a credential this plugin owns (an OAuth token, an API key): return one
        ``HealthCheck`` of kind ``credential`` per account, in your words. ``net_probe``
        is the owner's opt-in to a live check (a token refresh); without it, stay
        local. A check that raises is shown as a yellow row under ``name`` and charged to
        this plugin, never dropped. Keyed by ``name``: registering again replaces it.
        """
        from iris_harness.services.health.credentials import register_credential_check

        register_credential_check(
            name, self._registry.add_seam(self.plugin, "credential_check", name, check)
        )

    # -- owner identity: what the guards protect, bound to the manifest -----------
    def register_owner_identity_source(
        self,
        provider: Callable[[], Mapping[str, Iterable[str]]],
        *,
        fingerprint: Callable[[], Hashable] | None = None,
    ) -> None:
        """Hand the guards owner identity this plugin knows (ADR-0125): ``{kind: literals}``.

        An account address the plugin signs in to, say ``{"email": ["owner@..."]}``. The
        source is ``plugin:<this plugin>`` -- one per plugin, registering again replaces
        it -- and it may return only the kinds its manifest declares under
        ``identity: provides``; anything else is dropped and charged to the plugin. A
        plugin that declares no kinds is refused, as an undeclared tool is.
        ``fingerprint``, when given, is a cheap probe that changes when the literals do;
        without one the literals are read once, then again after any identity change.
        """
        record = self._registry.get(self.plugin)
        manifest = record.manifest if record is not None else None
        declared = manifest.identity.provides if manifest is not None else ()
        if not declared:
            self._registry.record_failure(
                self.plugin,
                where="owner_identity",
                exc=ValueError(
                    f"plugin {self.plugin!r} registers an owner-identity source but its "
                    "manifest declares no kinds under 'identity: provides' (ADR-0125)"
                ),
            )
            return
        name = f"{OWNER_IDENTITY_SEAM}:{self.plugin}"

        def undeclared(kinds: frozenset[str]) -> None:
            self._registry.record_failure(
                self.plugin,
                where=name,
                exc=ValueError(
                    f"plugin {self.plugin!r}'s owner-identity source returned "
                    f"{', '.join(sorted(kinds))}, not declared under 'identity: provides'; "
                    "dropped"
                ),
            )

        register_owner_identity_source(
            f"plugin:{self.plugin}",
            self._registry.add_seam(self.plugin, OWNER_IDENTITY_SEAM, self.plugin, provider),
            fingerprint=(
                self._registry.guard_where(
                    self.plugin, f"{name}:fingerprint", fingerprint, degrade=None
                )
                if fingerprint is not None
                else None
            ),
            kinds=cast("tuple[IdentityKind, ...]", declared),
            on_undeclared=undeclared,
        )

    # -- confirmation executor: resolves a pending approval of ``kind`` ----------
    def register_confirmation_executor(self, kind: str, executor: Callable[..., Any]) -> None:
        self._registry.add_confirmation_executor(self.plugin, kind, executor)

    # -- introspection ------------------------------------------------------------
    def kinds(self) -> tuple[RegistrationKind, ...]:
        rec = self._registry.get(self.plugin)
        if rec is None:
            return ()
        return tuple(dict.fromkeys(r.kind for r in rec.registrations))


__all__ = ["HarnessServices", "PluginAPI"]
