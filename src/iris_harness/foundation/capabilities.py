"""Declared capabilities: the catalogue of typed service interfaces plugins share in code.

A capability is a ``domain.verb`` name (``mail.read``) plus a ``Protocol``: a typed service
interface one plugin provides and others consume in code, with no model involved
(docs/architecture/plugin-capabilities.md §2, decisions 2 and 3).

The catalogue is defined here, at the bottom of the layers, and is static: every Protocol,
its :class:`CapabilitySpec` and the ``CAPABILITIES`` map (read-only at runtime too).
``iris_harness.sdk.capabilities`` re-exports the same objects and is the stable import path
for authors. Here and not in the SDK because every layer that checks or consumes a
capability must import it -- the plugin host (runtime) and the core's own consumers
(services) -- and none may import the SDK above them. The catalogue is closed: a new
capability is a change here, shipped in an SDK release.

**Every method is governed like a tool** (§4): its call is ``capability:<name>.<method>``
on ``PRE_TOOL_USE`` / ``POST_TOOL_USE``, so each :class:`MethodSpec` declares what a tool's
manifest declares -- its ``effect`` (and a write's ``confirm``) -- plus the text-bearing
``fields`` of what it returns, which ``POST_TOOL_USE`` may transform (identity masking)
before the consumer sees them (``capability_fields``). Protocol methods return plain data
(dataclasses, pydantic models, TypedDicts, sequences, scalars), never live objects: a
handle the harness cannot see into is refused when the spec is defined.

``CapabilitySpec.fan_out`` is decision 3: a capability several plugins may provide (Gmail
and IMAP both provide ``mail.read``) says how their implementations combine into ONE, so a
consumer never iterates providers. Without it the capability has one provider, and a second
is refused.

The catalogue is closed for 0.x: it holds ``weather.forecast`` so far, and the rest arrive
with rollout step 4.
"""

from __future__ import annotations

import collections.abc as cabc
import inspect
import re
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

from iris_harness.foundation.capability_fields import UndeclarableType, text_paths

# decision 2: ``domain.verb``, both halves lowercase words.
CAPABILITY_NAME = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
#: What a capability call is called on the tool hooks (``capability:mail.read.search``).
CAPABILITY_TOOL_PREFIX = "capability:"

CapabilityEffect = Literal["read", "write"]
CapabilityConfirm = Literal["once", "never"]
#: Whether a result is text a third party wrote (``external``), which the retrieved-content
#: injection guard scans, or the owner's / IRIS's own data (``internal``).
CapabilityContent = Literal["internal", "external"]
#: How a method hands back its result. ``value``: a plain return. ``async``: ``async def``.
#: ``stream`` / ``astream``: a method returning ``Iterator[X]`` / ``AsyncIterator[X]``, whose
#: items are governed. There is no "sync method returning an awaitable": its ``PRE_TOOL_USE``
#: would have to run synchronously, which the kernel refuses inside the event loop the
#: awaitable is meant for -- declare such a method ``async def``.
MethodShape = Literal["value", "async", "stream", "astream"]

_STREAMS = (cabc.Iterator, cabc.Iterable, cabc.Generator)
_ASTREAMS = (cabc.AsyncIterator, cabc.AsyncIterable, cabc.AsyncGenerator)
_AWAITABLES = (cabc.Awaitable, cabc.Coroutine)


def is_capability_name(name: str) -> bool:
    """True when ``name`` has the ``domain.verb`` shape."""
    return bool(CAPABILITY_NAME.match(name))


def capability_tool_name(capability: str, method: str) -> str:
    """The name a capability call carries on the tool hooks: ``capability:<name>.<method>``."""
    return f"{CAPABILITY_TOOL_PREFIX}{capability}.{method}"


def split_capability_tool(tool: str) -> tuple[str, str] | None:
    """``(capability, method)`` for a ``capability:<name>.<method>`` tool name, else None."""
    if not tool.startswith(CAPABILITY_TOOL_PREFIX):
        return None
    capability, _, method = tool.removeprefix(CAPABILITY_TOOL_PREFIX).rpartition(".")
    if not is_capability_name(capability) or not method:
        return None
    return capability, method


class CapabilityUnavailable(RuntimeError):
    """A capability call cannot reach its provider. Catch it to take your degraded path."""


class CapabilityDenied(CapabilityUnavailable):
    """Governance stopped a capability call, or withheld its result.

    ``outcome`` is the kernel's: ``deny``, or ``require_approval`` for a call held for the
    owner (a ``confirm: once`` write from code, Decision 1).
    """

    def __init__(self, reason: str, *, outcome: str = "deny") -> None:
        super().__init__(reason)
        self.reason = reason
        self.outcome = outcome


@dataclass(frozen=True)
class MethodSpec:
    """One Protocol method's contract with governance, as a tool's manifest entry is.

    ``effect``: ``read`` leaves the owner's world as it was; ``write`` changes it, and
    ``confirm`` says whether it asks first (``once``, the default for a write) or not
    (``never``). ``fields``: the text-bearing paths of the return value (per item for a
    stream), exactly the ``str`` leaves of its type (``capability_fields``). ``content``:
    ``external`` when those fields carry text a third party wrote (a web page, an email),
    which the retrieved-content injection guard scans at ``POST_TOOL_USE``.
    """

    effect: CapabilityEffect = "read"
    confirm: CapabilityConfirm | None = None
    fields: tuple[str, ...] = ()
    content: CapabilityContent = "internal"

    def __post_init__(self) -> None:
        if self.effect != "write" and self.confirm is not None:
            raise ValueError("'confirm' only applies to a method with effect: write")

    @property
    def confirm_mode(self) -> str:
        """The gate the tool policy applies: reads never ask, writes ask once by default."""
        return "never" if self.effect == "read" else (self.confirm or "once")


def _unwrap(tp: Any, kinds: tuple[Any, ...]) -> Any | None:
    """The item / awaited type when ``tp`` is one of ``kinds`` (``Iterator[X]`` -> X)."""
    origin = typing.get_origin(tp)
    if origin in kinds:
        args = typing.get_args(tp)
        return args[-1] if kinds is _AWAITABLES else (args[0] if args else Any)
    return None


def _shape_and_value(fn: Any, returns: Any) -> tuple[MethodShape, Any]:
    if inspect.iscoroutinefunction(fn):
        if _unwrap(returns, _ASTREAMS + _STREAMS + _AWAITABLES) is not None:
            raise ValueError("an async method returns a value; declare a stream with a sync def")
        return "async", returns
    if _unwrap(returns, _AWAITABLES) is not None:
        raise ValueError("a method returning an awaitable must be declared async def")
    for kinds, shape in ((_ASTREAMS, "astream"), (_STREAMS, "stream")):
        inner = _unwrap(returns, kinds)
        if inner is not None:
            return typing.cast(MethodShape, shape), inner
    return "value", returns


@dataclass(frozen=True)
class CapabilitySpec:
    """One capability in the catalogue.

    ``protocol`` is the interface a provider implements and a consumer codes against. Its
    members must all be methods with a declared return type and plain parameters (no
    ``*args``, ``**kwargs`` or positional-only), and ``methods`` must declare each one
    (:class:`MethodSpec`). ``fan_out`` receives every provider's implementation, in mount
    order, and returns the one a consumer gets; ``None`` means one provider only.
    """

    name: str
    protocol: type[Any]
    methods: Mapping[str, MethodSpec]
    description: str = ""
    fan_out: Callable[[Sequence[Any]], Any] | None = None
    shapes: Mapping[str, MethodShape] = field(init=False, repr=False, compare=False)
    # Each method's declared value type (a stream's item type): what a result is walked as.
    value_types: Mapping[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not is_capability_name(self.name):
            raise ValueError(f"capability {self.name!r} is not a domain.verb name")
        members = self.members()
        not_methods = [
            m
            for m in members
            if not inspect.isfunction(inspect.getattr_static(self.protocol, m, None))
        ]
        if not_methods:
            raise ValueError(
                f"capability {self.name!r}: Protocol members must be methods, not "
                f"{', '.join(not_methods)}"
            )
        if set(members) != set(self.methods):
            missing = sorted(set(members) - set(self.methods))
            extra = sorted(set(self.methods) - set(members))
            raise ValueError(
                f"capability {self.name!r}: every Protocol method needs one MethodSpec "
                f"(undeclared: {missing or '-'}; not in the Protocol: {extra or '-'})"
            )
        shapes: dict[str, MethodShape] = {}
        value_types: dict[str, Any] = {}
        for member in members:
            shapes[member], value_types[member] = self._check_method(member)
        object.__setattr__(self, "shapes", types.MappingProxyType(shapes))
        object.__setattr__(self, "value_types", types.MappingProxyType(value_types))

    def _check_method(self, member: str) -> tuple[MethodShape, Any]:
        where = f"capability {self.name!r}: {member}()"
        fn = inspect.getattr_static(self.protocol, member)
        params = list(inspect.signature(fn).parameters.values())[1:]  # drop self
        odd = [
            p.name
            for p in params
            if p.kind
            in (p.VAR_POSITIONAL, p.VAR_KEYWORD, p.POSITIONAL_ONLY)  # not bindable by name
        ]
        if odd:
            raise ValueError(f"{where}: parameters must be plain named ones, not {odd}")
        hints = typing.get_type_hints(fn)
        if "return" not in hints:
            raise ValueError(f"{where}: declare the return type")
        try:
            shape, value_type = _shape_and_value(fn, hints["return"])
            declarable = text_paths(value_type)
        except (UndeclarableType, ValueError) as exc:
            raise ValueError(f"{where}: {exc}") from exc
        declared = set(self.methods[member].fields)
        if declared != declarable:
            raise ValueError(
                f"{where}: 'fields' must name every text field of the return type, and only "
                f"those (undeclared: {sorted(declarable - declared) or '-'}; not text fields: "
                f"{sorted(declared - declarable) or '-'})"
            )
        return shape, value_type

    def members(self) -> tuple[str, ...]:
        """The methods a provider must supply (the Protocol's declared members)."""
        declared = getattr(self.protocol, "__protocol_attrs__", None)
        if declared is None:  # not a typing.Protocol: its public attributes
            declared = {n for n in dir(self.protocol) if not n.startswith("_")}
        return tuple(sorted(m for m in declared if not m.startswith("_")))

    def signature(self, member: str) -> inspect.Signature:
        """The Protocol method's signature (``self`` included), for binding a call."""
        return inspect.signature(inspect.getattr_static(self.protocol, member))

    def missing_members(self, impl: object) -> tuple[str, ...]:
        """The methods ``impl`` lacks, cannot call, or implements with the wrong shape."""
        wrong: list[str] = []
        for member in self.members():
            method = getattr(impl, member, None)
            if not callable(method):
                wrong.append(member)
            elif inspect.iscoroutinefunction(method) != (self.shapes[member] == "async"):
                wrong.append(f"{member} (async in one, sync in the other)")
        return tuple(wrong)


@dataclass(frozen=True)
class ForecastPeriod:
    """One stretch of a forecast: a span of time and the conditions expected in it."""

    start: datetime
    end: datetime
    temperature_c: float
    # 0.0 to 1.0; None when the source does not say.
    precipitation_probability: float | None
    wind_speed_kph: float | None
    # The source's own words for the conditions ("Light rain"): third-party text.
    summary: str


@dataclass(frozen=True)
class Forecast:
    """A weather forecast for one place, as the provider's source issued it."""

    # The place the source resolved the request to ("Lisbon, Portugal"): third-party text.
    location: str
    issued_at: datetime
    periods: tuple[ForecastPeriod, ...]


class WeatherForecast(Protocol):
    """``weather.forecast``: what the weather is expected to be at a place.

    One provider. ``location`` is a place name or address as the owner would say it;
    ``days`` is how many days ahead from now (the provider may return fewer). A provider
    that cannot answer raises ``CapabilityUnavailable`` rather than returning an empty
    forecast. The core ships no implementation: a plugin provides it.
    """

    async def forecast(self, location: str, days: int = 3) -> Forecast: ...


WEATHER_FORECAST = CapabilitySpec(
    name="weather.forecast",
    protocol=WeatherForecast,
    methods={
        # A read: it leaves the owner's world as it was. The text it returns came from a
        # weather service, so the injection guard scans it (``external``).
        "forecast": MethodSpec(
            effect="read",
            fields=("location", "periods.[].summary"),
            content="external",
        ),
    },
    description="The expected weather at a place, for the days ahead.",
)

# Every capability the SDK publishes, by name -- read-only, so it stays closed at runtime.
# Step 4 adds one Protocol and spec per capability above this line.
CAPABILITIES: Mapping[str, CapabilitySpec] = types.MappingProxyType(
    {WEATHER_FORECAST.name: WEATHER_FORECAST}
)


def published_capability(name: str) -> CapabilitySpec | None:
    """The catalogue's spec for ``name``, or None when there is no such capability."""
    return CAPABILITIES.get(name)


__all__ = [
    "CAPABILITIES",
    "CAPABILITY_NAME",
    "CAPABILITY_TOOL_PREFIX",
    "CapabilityConfirm",
    "CapabilityContent",
    "CapabilityDenied",
    "CapabilityEffect",
    "CapabilitySpec",
    "CapabilityUnavailable",
    "Forecast",
    "ForecastPeriod",
    "MethodShape",
    "MethodSpec",
    "WEATHER_FORECAST",
    "WeatherForecast",
    "capability_tool_name",
    "is_capability_name",
    "published_capability",
    "split_capability_tool",
]
