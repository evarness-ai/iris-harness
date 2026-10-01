"""Plugin manifest (``manifest.yaml`` beside ``plugin.py``).

Mirrors the skill manifest style (Pydantic v2, ``extra="forbid"``) so a plugin
author who has written a skill recognises the shape. ``provides`` is advisory:
it feeds the drift panel ("declared provides X, registered nothing of kind X")
and the ``--dump-config`` tree; the registry is the source of truth.

``flavor: declarative`` (OSS plan decision 5) is a plugin that is only its manifest: each
tool under ``tools:`` carries its ``description``, the plain function it is bound to
(``impl: package.module:function``) and its typed ``args``, and the loader mounts it with
no ``setup()`` of the plugin's (``declarative.py``).
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iris_harness.foundation.capabilities import is_capability_name
from iris_harness.foundation.settings.catalog import SettingDeclaration
from iris_harness.kernel.governance.hooks.tool_payload import ToolContent, ToolSendsTo
from iris_harness.kernel.governance.owner_identity import OWNER_PII_KINDS
from iris_harness.kernel.governance.side_effects.probes import probe_names
from iris_harness.services.digest.expiry import (
    ExpiryKindDeclaration,
    check_expiry_kind_name,
)


class RegistrationKind(StrEnum):
    """The six v1 registration kinds (OSS plan decision 3)."""

    INTERCEPT = "intercept"
    TOOL = "tool"
    INTENT_HANDLER = "intent_handler"
    HEARTBEAT = "heartbeat"
    CHANNEL = "channel"
    CONFIRMATION_EXECUTOR = "confirmation_executor"


TrustLevel = Literal["in-process", "mcp"]
PluginFlavor = Literal["python", "declarative"]


class PluginUses(BaseModel):
    """What this plugin may use of other plugins' — the permission contract's allow-list.

    A plugin may always call its own tools. To call another plugin's tool from code
    (``api.tools.call``) it lists the tool here; the kernel's caller policy denies any
    ``plugin:<name>`` call that is neither its own nor listed
    (docs/architecture/plugin-capabilities.md §4). Config, not code: the harness compiles
    every mounted manifest into one policy, and an operator can narrow it in
    ``config/governance/tool-access.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    tools: tuple[str, ...] = Field(default_factory=tuple)


# The kinds a manifest may unmask: owner_identity.OWNER_PII_KINDS, as a type (a test keeps
# the two equal). Never ``secret``, and a ``link`` is never masked in a result.
GrantableKind = Literal["name", "email", "phone", "address", "handle"]


class PluginCapabilities(BaseModel):
    """The capabilities this plugin provides, uses and requires (plugin-capabilities §2).

    ``provides``: service interfaces the plugin registers with ``api.provide``. ``uses``:
    interfaces it works without — mounted with none, ``api.capability`` returns None and the
    plugin takes its degraded path. ``requires``: interfaces it cannot run without — with no
    provider mounted, the plugin is not loaded, with the reason in System Health. Each name
    is a ``domain.verb`` capability whose Protocol ``iris_harness.sdk.capabilities``
    publishes. The shape is checked here; what the name resolves to is checked where it is
    used, so a name nothing provides is never silent: ``api.provide`` refuses a capability
    the SDK does not publish, a ``requires`` nobody provides keeps the plugin unloaded, and a
    ``uses`` nobody provides is drift (the closure rule, §5).

    Separate from the top-level ``provides:`` (registration kinds): a capability is an
    interface other plugins consume, not a kind the core dispatches.

    ``unmask`` (ADR-0125): a capability's results reach the consumer with the owner's
    personal identifiers as pseudonyms (``[owner:email#1]``). A ``uses`` or ``requires``
    entry may grant kinds back, per capability::

        capabilities:
          uses:
            - mail.read:
                unmask: [name, address]

    The entry is read into ``unmask`` (``{capability: kinds}``); a bare name grants
    nothing. Only ``name``, ``email``, ``phone``, ``address`` and ``handle`` can be
    granted -- never ``secret``. The harness reads the grant for the caller it stamps on
    the call, so a plugin cannot use another plugin's grant.
    """

    model_config = ConfigDict(extra="forbid")

    provides: tuple[str, ...] = Field(default_factory=tuple)
    uses: tuple[str, ...] = Field(default_factory=tuple)
    requires: tuple[str, ...] = Field(default_factory=tuple)
    unmask: dict[str, tuple[GrantableKind, ...]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _entries_with_grants(cls, raw: Any) -> Any:
        """Split ``{name: {unmask: [...]}}`` entries of uses/requires into name + grant."""
        if not isinstance(raw, dict):
            return raw
        out = dict(raw)
        unmask: dict[str, list[Any]] = {
            str(k): list(v or ()) for k, v in dict(out.get("unmask") or {}).items()
        }
        for role in ("uses", "requires"):
            entries = out.get(role)
            if not isinstance(entries, list | tuple):
                continue
            names: list[Any] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    names.append(entry)
                    continue
                if len(entry) != 1:
                    raise ValueError(
                        f"capabilities: a {role} entry names one capability, got {sorted(entry)}"
                    )
                ((name, spec),) = entry.items()
                spec = spec or {}
                if not isinstance(spec, dict) or set(spec) - {"unmask"}:
                    raise ValueError(
                        f"capabilities: {role} entry {name!r} takes only 'unmask', got {spec!r}"
                    )
                names.append(name)
                unmask.setdefault(str(name), []).extend(spec.get("unmask") or ())
            out[role] = names
        if unmask:
            out["unmask"] = unmask
        return out

    @field_validator("provides", "uses", "requires")
    @classmethod
    def _capability_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for name in value:
            if not is_capability_name(name):
                raise ValueError(f"capability {name!r} is not a domain.verb name")
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def _one_role_each(self) -> PluginCapabilities:
        # A plugin consumes another plugin's interface, not its own; and a capability it
        # needs is either optional or required, never both.
        for a, b in (("provides", "uses"), ("provides", "requires"), ("uses", "requires")):
            both = sorted(set(getattr(self, a)) & set(getattr(self, b)))
            if both:
                raise ValueError(f"capabilities: {', '.join(both)} listed under both {a} and {b}")
        return self

    @model_validator(mode="after")
    def _grants_for_consumed(self) -> PluginCapabilities:
        # A grant unmasks results the plugin receives; it means nothing for a capability
        # the plugin does not consume (or provides itself).
        stray = sorted(set(self.unmask) - set(self.consumes))
        if stray:
            raise ValueError(
                f"capabilities: unmask for {', '.join(stray)}, not listed under uses or requires"
            )
        self.unmask = {
            name: tuple(dict.fromkeys(kinds)) for name, kinds in self.unmask.items() if kinds
        }
        return self

    @property
    def consumes(self) -> tuple[str, ...]:
        """Every capability the plugin may ask for with ``api.capability``."""
        return self.uses + self.requires


class PluginIdentity(BaseModel):
    """The owner-identity kinds this plugin supplies (ADR-0125).

    A plugin that knows an address of the owner's -- the account it signs in to -- hands
    it to the guards with ``api.register_owner_identity_source``. It must declare here
    which kinds that source returns: a registration without a declaration is refused, and
    a kind it returns but did not declare is dropped and charged to the plugin. Only the
    owner's personal identifiers (``OWNER_PII_KINDS``) are providable: never ``secret`` (a
    secret is recognised by its shape, never declared), and not ``link`` -- the owner's
    confirmed ``blog``/``website`` facts are the only declared links (ADR-0125).
    """

    model_config = ConfigDict(extra="forbid")

    provides: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("provides")
    @classmethod
    def _declarable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for kind in value:
            if kind not in OWNER_PII_KINDS:
                raise ValueError(
                    f"identity: {kind!r} is not a kind a plugin may provide "
                    f"({', '.join(OWNER_PII_KINDS)})"
                )
        return tuple(dict.fromkeys(value))


class PluginRequirements(BaseModel):
    """Preconditions checked before ``setup`` runs (missing → plugin not loaded)."""

    model_config = ConfigDict(extra="forbid")

    python: str = Field(default=">=3.12")
    packages: tuple[str, ...] = Field(default_factory=tuple)
    env_vars: tuple[str, ...] = Field(default_factory=tuple)


ToolEffect = Literal["read", "write", "destructive"]
ToolConfirm = Literal["once", "never"]
# ADR-0118 amendment: a write that is not data loss but must still be approved per call
# on the pinned card (sending an email). The only form today is ``pinned``.
ToolApproval = Literal["pinned"]
# The gate the loop applies: a plugin declares ``confirm`` for a write, but a destructive
# tool is never confirmed, it is approved per call (ADR-0118), so its gate is its own word.
ToolGate = Literal["once", "never", "approval"]
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
# A search provider's name in the chain (and in config/search_providers.yaml): the
# registry's own rule (services/research/providers.py).
_SEARCH_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_ARG_NAME = r"^[a-z_][a-z0-9_]*$"
# ``package.module:function`` -- an importable module path, absolute.
_IMPL = r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$"

ToolArgType = Literal["string", "number", "integer", "boolean", "enum"]
# What a JSON value of each declared type is in Python. ``bool`` is an ``int`` subclass,
# so it is refused for every type but ``boolean`` where the value is checked.
ARG_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "enum": (str,),
}


def arg_value_problem(arg: ToolArg, value: Any) -> str | None:
    """Why ``value`` is not a value of ``arg``'s declared type, or None."""
    wrong_bool = isinstance(value, bool) and arg.type != "boolean"
    if wrong_bool or not isinstance(value, ARG_TYPES[arg.type]):
        article = "an" if arg.type[0] in "aeiou" else "a"
        return f"must be {article} {arg.type}"
    if arg.options and value not in arg.options:
        return f"must be one of {', '.join(arg.options)}"
    return None


class ToolArg(BaseModel):
    """One argument of a declarative tool (``flavor: declarative``).

    ``type`` is ``string``, ``number``, ``integer``, ``boolean`` or ``enum`` (with
    ``options``). An argument is required unless it says ``required: false``; an optional
    one may give a ``default`` (passed when the call leaves it out; without one, the
    function's own default applies).
    """

    model_config = ConfigDict(extra="forbid")

    type: ToolArgType = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    options: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _consistent(self) -> ToolArg:
        if self.type == "enum" and not self.options:
            raise ValueError("an enum argument lists its 'options'")
        if self.type != "enum" and self.options:
            raise ValueError("'options' only applies to an enum argument")
        if self.default is not None:
            if self.required:
                raise ValueError("a 'default' is for an argument with required: false")
            problem = arg_value_problem(self, self.default)
            if problem is not None:
                raise ValueError(f"'default' {self.default!r} {problem}")
        return self


class ToolDeclaration(BaseModel):
    """One tool's contract with the loop (ADR-0110).

    ``effect``: ``read`` leaves the user's world as it was; ``write`` changes something
    on their behalf; ``destructive`` removes or overwrites the user's data (ADR-0118:
    deleting or trashing an email, deleting an event, overwriting a file). ``confirm`` is
    only meaningful for a write: ``once`` (the default) has the loop ask the user once
    per run before the first such write; ``never`` is for a write the user's own
    sentence already authorised. A destructive tool takes no ``confirm`` — every call
    waits for the owner's itemised approval — and names its ``undo`` tool when the
    service has a reversible form. A ``write`` may declare ``approval: pinned`` instead
    of ``confirm``: every call then waits for the same per-call approval a destructive
    call does (the card pins the exact call; approving runs it, rejecting runs nothing),
    without being called data loss (ADR-0118 amendment: ``send_email``). ``guidance`` is prose the core shows in the prompt
    while this tool is on the loop — routing advice the plugin owns, in the plugin's
    words.
    """

    model_config = ConfigDict(extra="forbid")

    effect: ToolEffect = "read"
    confirm: ToolConfirm | None = None
    guidance: str = ""
    # Always on the prompt's menu, whatever the query sounds like (the shortlist
    # keeps it). For a tool other tools' guidance sends the model to — search_inbox
    # after a finance miss, research for anything current. Use sparingly: every
    # pinned tool takes a menu slot on every turn.
    pinned: bool = False
    # The tool's output is already a finished answer to the user (a narrated digest,
    # not raw rows). When it is the run's first tool and it succeeds, the loop hands
    # that output back as the Final Answer instead of paying another model call to
    # restate it — and a repeat of it returns the result in hand. Leave it off for a
    # tool whose output needs interpreting, or that is usually one step of several.
    # Reads, and writes with ``confirm: never`` whose output is the full account.
    answers_directly: bool = False
    # ADR-0118 decision 3: the tool that reverses this one (``trash_email`` ->
    # ``restore_email``). Destructive tools only; it must be declared in the same
    # manifest as a ``write`` with ``confirm: never``, checked by the manifest.
    undo: str | None = None
    # How long ``undo`` stays possible, in days (Gmail's Trash keeps mail 30). Shown on
    # the approval card; a destructive tool with no undo reads "cannot be undone".
    undo_window_days: int | None = Field(default=None, ge=1)
    # ADR-0118 amendment: route this write through the destructive tools' approval path
    # (queue, pinned card, resume that runs exactly the pinned call). Writes only; a
    # destructive tool is always approved this way and does not declare it.
    approval: ToolApproval | None = None
    # Whether the output is text a third party wrote (a web page, an email, a retrieved
    # document). ``external`` has the retrieved-content injection guard scan every result
    # at POST_TOOL_USE; ``internal`` (the owner's or IRIS's own data) is not scanned.
    content: ToolContent = "internal"
    # Where the call's arguments go, when that is a destination the owner-PII guards
    # treat on its own: ``search_engine`` for a tool that hands them to a web search
    # provider (ADR-0125's web-search column). Undeclared: no such destination.
    sends_to: ToolSendsTo | None = None
    # Whether a call runs code the model wrote (a sandboxed shell, an interpreter). Not an
    # effect on the owner's data, so not ``effect``; what ``iris mcp serve`` reads to keep
    # such a tool off what it serves by default (it is served only when named).
    executes_code: bool = False
    # The side-effect probe that can tell, at ``iris run resume``, whether a call of this
    # non-read tool landed (a name ``kernel/governance/side_effects/probes.py`` registers).
    # Without one, resume cannot tell, and the owner approves before the call is retried.
    verify: str | None = None
    # ``flavor: declarative`` only (plan decision 5): the tool is the manifest. What the
    # model reads about it, the plain function it is bound to (``package.module:function``,
    # importable) and its typed arguments, checked before every call. A ``python`` plugin
    # says all three in code, through ``register_tool``.
    description: str = ""
    impl: str | None = Field(default=None, pattern=_IMPL)
    args: dict[str, ToolArg] = Field(default_factory=dict)

    @field_validator("args")
    @classmethod
    def _arg_names(cls, value: dict[str, ToolArg]) -> dict[str, ToolArg]:
        for name in value:
            if not re.match(_ARG_NAME, name):
                raise ValueError(f"args: {name!r} is not an argument name (lowercase, digits, _)")
        return value

    @property
    def declares_binding(self) -> bool:
        """Whether the declaration carries a declarative tool's binding fields."""
        return bool(self.description or self.impl or self.args)

    @field_validator("verify")
    @classmethod
    def _verify_names_a_probe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        known = probe_names()
        if value not in known:
            raise ValueError(
                f"'verify' names no registered side-effect probe: {value!r} "
                f"(registered: {sorted(known)})"
            )
        return value

    @model_validator(mode="after")
    def _confirm_only_for_writes(self) -> ToolDeclaration:
        if self.effect != "write" and self.confirm is not None:
            # A destructive tool is approved per call, never confirmed once — so there
            # is no way to declare ``confirm: never`` on a delete (ADR-0118 decision 1).
            raise ValueError("'confirm' only applies to a tool with effect: write")
        if self.approval is not None and self.effect != "write":
            raise ValueError("'approval' only applies to a tool with effect: write")
        if self.approval is not None and self.confirm is not None:
            # Approved per call on the card, so there is nothing to confirm once.
            raise ValueError("'approval' and 'confirm' cannot both be declared")
        if self.undo is not None and self.effect != "destructive":
            raise ValueError("'undo' only applies to a tool with effect: destructive")
        if self.undo_window_days is not None and self.undo is None:
            raise ValueError("'undo_window_days' needs an 'undo' tool to be undone with")
        if self.verify is not None and self.effect == "read":
            # A read has no side effect to verify; the ledger never records one.
            raise ValueError("'verify' only applies to a tool with effect: write or destructive")
        if self.answers_directly and self.confirm_mode != "never":
            # A write that asks first, or waits on a card, is not done when it returns:
            # the run goes on to the confirmation or the approval. A write with
            # ``confirm: never`` is done, and when its output already says what was done
            # (count, what is left, how to undo) the model's retelling only loses facts
            # (ADR-0110 amendment, 2026-09-22 probe: "deleted" for "moved to Trash").
            raise ValueError(
                "'answers_directly' only applies to a read tool or a write with confirm: never"
            )
        return self

    @property
    def confirm_mode(self) -> ToolGate:
        """The effective gate: writes ask once unless told never; reads never ask;
        destructive calls, and writes declared ``approval: pinned``, wait for an
        approval (ADR-0118)."""
        if self.effect == "read":
            return "never"
        if self.effect == "destructive" or self.approval == "pinned":
            return "approval"
        return self.confirm or "once"


_SCREEN_ID = r"^[a-z][a-z0-9_-]*$"
# One path segment: the console matches a screen by its first segment, so a screen's
# sub-routes (``/agents/:name``) belong to it without being declared.
_SCREEN_ROUTE = r"^/[a-z0-9][a-z0-9-]*$"
# A lucide icon name in kebab case (``chart-line``); the console maps the names it
# bundles and draws a generic icon for any other.
_SCREEN_ICON = r"^[a-z][a-z0-9-]*$"


class WebScreen(BaseModel):
    """One screen of the web console, owned by the plugin that declares it (OSS plan R17).

    The console's navigation is the core's screens plus those of every mounted plugin:
    a plugin that is not mounted takes its screens with it, and a direct link to one
    shows that the plugin is not installed. ``group`` names one of the core's nav groups
    (``config/webui/nav.yaml``); ``order`` places the screen inside it. ``nav: false``
    keeps the route owned and gated without a menu entry (a page a notification opens).
    The screen's component ships in the console bundle; the manifest only says it is
    this plugin's.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=_SCREEN_ID)
    label: str = Field(..., min_length=1, max_length=40)
    route: str = Field(..., pattern=_SCREEN_ROUTE)
    # The page header; the label when unset.
    title: str | None = Field(default=None, min_length=1, max_length=80)
    subtitle: str = Field(default="", max_length=200)
    icon: str = Field(default="puzzle", pattern=_SCREEN_ICON)
    group: str = Field(default="apps", pattern=_SCREEN_ID)
    order: int = Field(default=100, ge=0, le=10_000)
    nav: bool = True


class PluginWebUI(BaseModel):
    """What a plugin adds to the web console: its screens."""

    model_config = ConfigDict(extra="forbid")

    screens: tuple[WebScreen, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _screens_distinct(self) -> PluginWebUI:
        for attr in ("id", "route"):
            seen: set[str] = set()
            for screen in self.screens:
                value = getattr(screen, attr)
                if value in seen:
                    raise ValueError(f"webui: two screens declare {attr} {value!r}")
                seen.add(value)
        return self


class PluginManifest(BaseModel):
    """Declared identity + surface of one plugin."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, pattern=r"^[a-z][a-z0-9_-]*$")
    version: str = Field(default="0.0.0", min_length=1)
    description: str = Field(default="")
    # ``module:function`` relative to the plugin directory / package.
    entrypoint: str = Field(default="plugin:setup", pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    # Optional ``module:function`` that adds this plugin's ``iris`` subcommands.
    # Loaded by the CLI at start-up WITHOUT building a runtime and without calling
    # ``setup`` — a command group is a static surface, so it must not cost the price
    # of booting the harness just to print ``--help``. See ``plugins/cli.py``.
    cli: str | None = Field(default=None, pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    flavor: PluginFlavor = "python"
    # Default trust: in-process behind the fault boundary (decision 8). ``mcp``
    # runs the plugin out of process over the signed MCP bridge (M2+).
    trust: TrustLevel = "in-process"
    provides: tuple[RegistrationKind, ...] = Field(default_factory=tuple)
    requires: PluginRequirements = Field(default_factory=PluginRequirements)
    uses: PluginUses = Field(default_factory=PluginUses)
    capabilities: PluginCapabilities = Field(default_factory=PluginCapabilities)
    # ADR-0125: the owner-identity kinds `api.register_owner_identity_source` may return.
    identity: PluginIdentity = Field(default_factory=PluginIdentity)
    # The search providers `api.register_search_provider` may add to the research
    # tool's chain, by name. A name registered but not declared here is refused and
    # charged to the plugin, as an undeclared tool is: the provider receives the
    # owner's web queries, so the manifest says which ones before any code runs.
    search_providers: tuple[str, ...] = Field(default_factory=tuple)
    # ADR-0110: every tool the plugin registers, with its effect. ``register_tool``
    # consults this — a registered tool that is not declared here is refused, a
    # declared tool never registered shows in the drift report. The core derives the
    # loop mechanics (confirm once before a fan-out of writes, the prompt's write
    # marking, the on-loop guidance) from these two words; the plugin never sees them.
    tools: dict[str, ToolDeclaration] = Field(default_factory=dict)
    # Intents (or the agent names they route to) whose answers must come from this
    # plugin's data: the ReAct loop accepts a final answer only after a read tool has
    # returned, turns the first one without back, and degrades the second to the
    # intent's deterministic handler. Without it a model answered "what is in my
    # calendar this week?" with invented events and no tool call (2026-09-29).
    read_first_intents: tuple[str, ...] = Field(default_factory=tuple)
    # ADR-0115 / memris PR 10: a directory (relative to the plugin) holding a memory
    # vocabulary fragment — ontology.yaml, and optionally shapes.yaml and mappings.yaml.
    # Memory reads it for every INSTALLED plugin (iris_harness.memory.ontology), mounted
    # or not, so the loader itself does nothing with it.
    ontology: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_][\w./-]*$")
    # ADR-0120: every IRIS_* setting this plugin reads, described. The core catalog lists
    # only the core's own; the settings catalog merges these in, and a completeness test
    # fails when the plugin reads a name it does not declare here.
    settings: dict[str, SettingDeclaration] = Field(default_factory=dict)
    # Core/SDK boundary plan PR 5 (email slice step 4): the kinds of dated item this
    # plugin ages out by local days, each with its default and range. The owner tunes
    # them in digest.yaml's ``expiry:`` by the same key; the plugin reads the value with
    # ``sdk.digest.expiry_days(key)``. Read for every INSTALLED plugin, mounted or not
    # (``services.digest.expiry.declared_expiry_kinds``), so the loader does nothing
    # with it beyond validating it here.
    expiry: dict[str, ExpiryKindDeclaration] = Field(default_factory=dict)
    # ADR-0120: why the app may not turn this plugin off (``IRIS_PLUGINS_DISABLE``), when
    # it may not — the plugin the app itself runs through, the harness's own. The plugin
    # says so, so the core names no plugin. Operators can still do it in server.env.
    locked: str | None = Field(default=None, min_length=1, max_length=200)
    # OSS plan R17: the web-console screens this plugin owns. Shown in the nav only
    # while the plugin is mounted (``plugin_host.nav``).
    webui: PluginWebUI = Field(default_factory=PluginWebUI)

    @field_validator("provides")
    @classmethod
    def _dedupe(cls, value: tuple[RegistrationKind, ...]) -> tuple[RegistrationKind, ...]:
        seen: list[RegistrationKind] = []
        for kind in value:
            if kind not in seen:
                seen.append(kind)
        return tuple(seen)

    @field_validator("tools")
    @classmethod
    def _tool_names(cls, value: dict[str, ToolDeclaration]) -> dict[str, ToolDeclaration]:
        for name in value:
            if not _TOOL_NAME.match(name):
                raise ValueError(f"tools: {name!r} is not a tool name (lowercase, digits, _)")
        return value

    @field_validator("search_providers")
    @classmethod
    def _search_provider_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for name in value:
            if not _SEARCH_PROVIDER_NAME.match(name):
                raise ValueError(
                    f"search_providers: {name!r} is not a provider name ([a-z][a-z0-9_-]*)"
                )
        return tuple(dict.fromkeys(value))

    @field_validator("expiry")
    @classmethod
    def _expiry_names(
        cls, value: dict[str, ExpiryKindDeclaration]
    ) -> dict[str, ExpiryKindDeclaration]:
        for name in value:
            check_expiry_kind_name(name)
        return value

    @field_validator("settings")
    @classmethod
    def _setting_names(cls, value: dict[str, SettingDeclaration]) -> dict[str, SettingDeclaration]:
        for name in value:
            if not name.startswith("IRIS_"):
                raise ValueError(f"settings: {name!r} is not an IRIS_* name")
        return value

    @model_validator(mode="after")
    def _tools_need_the_kind(self) -> PluginManifest:
        if self.tools and RegistrationKind.TOOL not in self.provides:
            raise ValueError("tools are declared but 'provides' does not list 'tool'")
        return self

    @model_validator(mode="after")
    def _flavor_shape(self) -> PluginManifest:
        """A declarative plugin is its manifest; a python plugin's tools are its code's."""
        if self.flavor == "python":
            bound = sorted(name for name, decl in self.tools.items() if decl.declares_binding)
            if bound:
                raise ValueError(
                    f"tools: {', '.join(bound)} declare description/impl/args, which only a "
                    "'flavor: declarative' plugin does (a python plugin registers them in code)"
                )
            return self
        # Declarative (plan decision 5): no code of its own runs at mount, so nothing but
        # declared tools can be provided, and there is no setup() to point at.
        for key in ("entrypoint", "cli"):
            if key in self.model_fields_set:
                raise ValueError(f"a 'flavor: declarative' plugin has no '{key}'")
        if self.trust != "in-process":
            raise ValueError("a 'flavor: declarative' plugin is mounted in-process")
        others = [kind.value for kind in self.provides if kind is not RegistrationKind.TOOL]
        if self.search_providers:
            others.append("search_providers")
        if others:
            raise ValueError(
                f"a 'flavor: declarative' plugin provides only tools, not {', '.join(others)}"
            )
        if not self.tools:
            raise ValueError("a 'flavor: declarative' plugin declares at least one tool")
        for name, decl in self.tools.items():
            if decl.impl is None or not decl.description.strip():
                raise ValueError(
                    f"tools: {name!r} needs 'description' and 'impl' (package.module:function) "
                    "in a 'flavor: declarative' plugin"
                )
        return self

    @model_validator(mode="after")
    def _undo_names_a_restoring_tool(self) -> PluginManifest:
        # ADR-0118 decision 3: restoring what the owner just lost is what their own
        # sentence ("undo that") authorises, so the undo tool is a write that does not
        # ask — and it must exist here, where the loop will find it.
        for name, decl in self.tools.items():
            if decl.undo is None:
                continue
            target = self.tools.get(decl.undo)
            if target is None:
                raise ValueError(f"tools: {name!r} names undo {decl.undo!r}, which is not declared")
            if target.effect != "write" or target.confirm_mode != "never":
                raise ValueError(
                    f"tools: {name!r} names undo {decl.undo!r}, which must be "
                    "effect: write with confirm: never"
                )
        return self

    @property
    def entry_module(self) -> str:
        return self.entrypoint.split(":", 1)[0]

    @property
    def entry_function(self) -> str:
        return self.entrypoint.split(":", 1)[1]

    @property
    def cli_module(self) -> str | None:
        return self.cli.split(":", 1)[0] if self.cli else None

    @property
    def cli_function(self) -> str | None:
        return self.cli.split(":", 1)[1] if self.cli else None


def load_manifest(path: Path) -> PluginManifest:
    """Parse ``manifest.yaml``; raises ``ValueError`` with the file named on any problem."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"plugin manifest unreadable: {path} ({exc})") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"plugin manifest must be a mapping: {path}")
    try:
        return PluginManifest.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError — keep the path in the message
        raise ValueError(f"plugin manifest invalid: {path}: {exc}") from exc


__all__ = [
    "ARG_TYPES",
    "GrantableKind",
    "PluginCapabilities",
    "PluginFlavor",
    "PluginIdentity",
    "PluginManifest",
    "PluginRequirements",
    "PluginUses",
    "PluginWebUI",
    "RegistrationKind",
    "ToolArg",
    "ToolArgType",
    "ToolConfirm",
    "ToolDeclaration",
    "ToolEffect",
    "TrustLevel",
    "WebScreen",
    "arg_value_problem",
    "load_manifest",
]
