"""Which hosts a plugin may contact: the egress declaration, compiled (issue #103).

A plugin's manifest says which hosts its code talks to (``egress:``). The plugin host, which
sits above the kernel and is what knows the manifests, compiles every mounted manifest into
one :class:`PluginEgressPolicy` and registers it once plugins have mounted; the kernel's
``plugin_egress`` hook reads it on every ``PRE_EGRESS`` (a call a plugin makes through the
SDK's governed HTTP client). The same seam as ``caller_policy.py``.

Fail closed, and say so: with no policy registered, or for a plugin the policy does not
know, every host is denied with a reason naming what is missing -- never allowed because
nobody was asked. An empty declaration is a closed door.

What a decision proves is bounded by what asks: it covers calls made through the governed
client. An in-process plugin can still open its own socket (docs/architecture/plugin-egress.md).
"""

from __future__ import annotations

import ipaddress
import logging
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Final

from iris_harness.foundation.process_state import track_globals
from iris_harness.kernel.governance.public_suffix import is_public_suffix

logger = logging.getLogger(__name__)

#: The data classes a host may be declared to receive. Never ``secret``: a secret does not
#: leave the owner's machines to a plugin's host (``config/governance/egress.yaml``).
EGRESS_DATA_CLASSES: Final[tuple[str, ...]] = ("public", "internal", "personal")
_CLASS_RANK: Final[dict[str, int]] = {"public": 0, "internal": 1, "personal": 2, "secret": 3}

#: The most DECODED body bytes a governed response may have, unless the plugin's manifest
#: declares another (``egress.max_response_bytes``, issue #175)...
DEFAULT_RESPONSE_BYTES: Final = 10 * 1024 * 1024
#: ...and the most any manifest may declare: a larger value is refused at mount.
MAX_RESPONSE_BYTES_CEILING: Final = 64 * 1024 * 1024

DEFAULT_DATA_CLASS: Final = "internal"
DEFAULT_SCHEMES: Final[tuple[str, ...]] = ("https",)
_DEFAULT_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}
SCHEMES: Final[tuple[str, ...]] = ("http", "https")

_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


MAX_LABEL_LENGTH: Final = 63
MAX_HOST_LENGTH: Final = 253
MAX_HOSTS_PER_PLUGIN: Final = 256

# ASCII only, matched with ``fullmatch`` (``$`` would accept a trailing newline).
_DECLARED_LABEL = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?", re.ASCII)
_HEX_OR_OCTAL_NUMBER = re.compile(r"0x[0-9a-f]*|[0-9]+", re.ASCII)


def _refuse(value: str, why: str) -> ValueError:
    return ValueError(f"{value!r}: {why}")


def normalize_host(value: str) -> str:
    """A request's host as compared: the same rules as a declared host, minus the wildcard.

    ASCII only, no whitespace or control character, lower-case, at most one trailing dot
    (dropped), no empty label (so no repeated dots), labels and name within the DNS limits.
    Raises ``ValueError`` for anything else, including an IPv6 literal (it has no DNS-label
    form). An IPv4 literal passes here; :meth:`PluginEgressPolicy.decide` refuses it.
    """
    if not value or not value.isascii() or any(c.isspace() or not c.isprintable() for c in value):
        raise _refuse(value, "is not a host (ASCII only, no whitespace or control characters)")
    host = value.lower()
    if host.endswith("."):
        host = host[:-1]
    if not host or len(host) > MAX_HOST_LENGTH:
        raise _refuse(value, f"is empty or longer than {MAX_HOST_LENGTH} characters")
    for label in host.split("."):
        if len(label) > MAX_LABEL_LENGTH or not _DECLARED_LABEL.fullmatch(label):
            raise _refuse(value, "is not a host name (letters, digits and hyphens per label)")
    return host


def is_ip_literal(host: str) -> bool:
    """Is ``host`` an address in any spelling a resolver or client might read as one?"""
    last = host.rsplit(".", 1)[-1]
    if _HEX_OR_OCTAL_NUMBER.fullmatch(last):
        return True
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


#: Names that mean "this machine or this network" by convention (RFC 6761, RFC 6762, common
#: resolver defaults), never a public service. Refused as a request's target even under
#: ``open_web``, and as a declared host.
INTERNAL_NAME_SUFFIXES: Final[tuple[str, ...]] = ("localhost", "local", "internal", "localdomain")


def is_internal_name(host: str) -> bool:
    """Is ``host`` (already normalised: lower-case, no trailing dot) a local-by-convention name?"""
    return any(host == s or host.endswith("." + s) for s in INTERNAL_NAME_SUFFIXES)


def normalize_host_pattern(value: str) -> str:
    """A declared host: an exact DNS name, or ``*.suffix`` for any subdomain of ``suffix``.

    Strict, because this is what an owner reads as the plugin's allowlist: ASCII letters,
    digits and hyphens only (an internationalised name is declared in its ``xn--`` form, which
    must decode), at most one trailing dot (dropped), labels of at most 63 and a name of at
    most 253 characters. A bare ``*`` is refused: "any host" is ``open_web: true``, which says
    so out loud. Refused outright, fail closed: IP literals in any spelling (dotted, short,
    hex, octal, IPv6), ``localhost`` and the local-network suffixes, and numeric last
    labels, which a resolver may read as an address; and a wildcard whose base is a single
    label (``*.com``). A wildcard over a public suffix (``*.co.uk``, ``*.com.au``) is refused too, from the
    vendored ICANN section of the Public Suffix List (``public_suffix.py``).
    """
    if not value or not value.isascii() or any(c.isspace() or not c.isprintable() for c in value):
        raise _refuse(value, "is not a host (ASCII only, no whitespace or control characters)")
    host = value.lower()
    if host.endswith("."):
        host = host[:-1]
    wildcard = host.startswith("*.")
    if host == "*" or (host.startswith("*") and not wildcard):
        raise _refuse(
            value, "a wildcard is only `*.<domain>`; to allow any host declare `open_web: true`"
        )
    body = host[2:] if wildcard else host
    if not body or len(body) > MAX_HOST_LENGTH:
        raise _refuse(value, f"is empty or longer than {MAX_HOST_LENGTH} characters")
    labels = body.split(".")
    for label in labels:
        if len(label) > MAX_LABEL_LENGTH or not _DECLARED_LABEL.fullmatch(label):
            raise _refuse(
                value,
                "is not a host (labels of letters, digits and hyphens, at most "
                f"{MAX_LABEL_LENGTH} characters; no scheme, port, path or userinfo)",
            )
        if label.startswith("xn--"):
            try:
                label.encode("ascii").decode("idna")
            except UnicodeError:
                raise _refuse(value, f"label {label!r} is not valid punycode") from None
    if is_internal_name(body):
        raise _refuse(
            value,
            "localhost and the local-network names (.local, .internal, .localdomain) are not "
            "hosts a plugin may declare",
        )
    if _HEX_OR_OCTAL_NUMBER.fullmatch(labels[-1]):
        raise _refuse(value, "an IP address (in any spelling) is not a declarable host")
    try:
        ipaddress.ip_address(body)
    except ValueError:
        pass
    else:
        raise _refuse(value, "an IP address is not a declarable host")
    if wildcard and len(labels) < 2:
        raise _refuse(value, "a wildcard needs a registrable domain (`*.example.org`)")
    if wildcard and is_public_suffix(body):
        raise _refuse(
            value,
            f"`{body}` is a public suffix, so the wildcard would cover every registrant under "
            f"it; declare the domain you mean (`*.example.{body}`)",
        )
    return host


@dataclass(frozen=True)
class HostRule:
    """One declared host: where, over what, and the highest data class it receives."""

    host: str
    schemes: tuple[str, ...] = DEFAULT_SCHEMES
    #: ``()`` means the default port of each declared scheme.
    ports: tuple[int, ...] = ()
    data: str = DEFAULT_DATA_CLASS

    def matches_host(self, host: str) -> bool:
        if self.host.startswith("*."):
            return host.endswith(self.host[1:]) and len(host) > len(self.host) - 1
        return host == self.host

    def allows_port(self, scheme: str, port: int) -> bool:
        return port in self.ports if self.ports else port == _DEFAULT_PORTS.get(scheme)


@dataclass(frozen=True)
class PluginEgress:
    """What one plugin declared."""

    hosts: tuple[HostRule, ...] = ()
    open_web: bool = False
    #: ``egress.max_response_bytes``; ``None`` means :data:`DEFAULT_RESPONSE_BYTES`.
    max_response_bytes: int | None = None

    @property
    def declared(self) -> bool:
        return bool(self.hosts) or self.open_web


@dataclass(frozen=True)
class EgressVerdict:
    """The answer for one request: allowed or not, why, and what the manifest said."""

    allowed: bool
    reason: str
    #: The data class the matched declaration allows the host to receive (None: denied).
    data: str | None = None
    #: The declared pattern that matched (``*`` for ``open_web``), None when denied.
    rule: str | None = None


@dataclass(frozen=True)
class PluginEgressPolicy:
    """Every mounted plugin's declaration, by plugin name."""

    plugins: Mapping[str, PluginEgress] = field(default_factory=dict)

    def response_cap(self, plugin: str) -> int:
        """The most decoded body bytes a response to ``plugin`` may have: its declared
        ``max_response_bytes`` (never above :data:`MAX_RESPONSE_BYTES_CEILING`, whatever a
        hand-built declaration says), else :data:`DEFAULT_RESPONSE_BYTES`."""
        declaration = self.plugins.get(plugin)
        declared = declaration.max_response_bytes if declaration is not None else None
        if declared is None:
            return DEFAULT_RESPONSE_BYTES
        return max(1, min(declared, MAX_RESPONSE_BYTES_CEILING))

    def decide(
        self,
        plugin: str,
        *,
        scheme: str,
        host: str,
        port: int,
        classification: str | None = None,
    ) -> EgressVerdict:
        """May ``plugin`` send ``classification`` data to ``scheme://host:port``?"""
        scheme = scheme.lower()
        try:
            host = normalize_host(host)
        except ValueError:
            return EgressVerdict(False, f"{host!r} is not a valid host name")
        if is_ip_literal(host):
            return EgressVerdict(
                False, f"{host} is an IP address; a plugin may contact only a declared host name"
            )
        if is_internal_name(host):
            # Even under open_web: these name this machine or the local network, whatever
            # the declaration says (a name an attacker can aim at a service on the host).
            return EgressVerdict(
                False, f"{host} is a local-network name; a plugin may not contact it"
            )
        declaration = self.plugins.get(plugin)
        if declaration is None:
            return EgressVerdict(False, f"plugin {plugin!r} has no mounted manifest")
        if not declaration.declared:
            return EgressVerdict(
                False, f"plugin {plugin!r} declares no egress, so it may contact no host"
            )
        if declaration.open_web:
            # Any host, on the schemes a page fetcher uses; the data class is the loosest
            # a declaration can state, and the call is still recorded.
            if scheme not in SCHEMES:
                return EgressVerdict(False, f"scheme {scheme!r} is not an HTTP scheme")
            return EgressVerdict(True, "open_web", data="personal", rule="*")
        for rule in declaration.hosts:
            if not rule.matches_host(host):
                continue
            if scheme not in rule.schemes:
                return EgressVerdict(
                    False,
                    f"{host} is declared for {'/'.join(rule.schemes)} only, not {scheme}",
                )
            if not rule.allows_port(scheme, port):
                return EgressVerdict(False, f"{host} is not declared for port {port}")
            if classification is not None and _rank(classification) > _rank(rule.data):
                return EgressVerdict(
                    False,
                    f"this run holds {classification} data and {host} is declared to receive "
                    f"{rule.data}; declare `data: {classification}` only if the plugin means "
                    "to send it",
                )
            return EgressVerdict(True, "declared", data=rule.data, rule=rule.host)
        return EgressVerdict(False, f"{host} is not in plugin {plugin!r}'s egress.hosts")


def _rank(data_class: str) -> int:
    # An unknown label is treated as the most sensitive one: refuse rather than guess.
    return _CLASS_RANK.get(data_class, max(_CLASS_RANK.values()))


# -- the registered policy --------------------------------------------------------------

_lock = threading.Lock()
_policy: PluginEgressPolicy | None = None
# Requests that completed whose ``post_egress`` ledger row could not be written (issue #175).
_unrecorded = 0


def register_egress_policy(policy: PluginEgressPolicy | None) -> None:
    """Install (or, with ``None``, remove) the policy the egress hook enforces."""
    global _policy
    with _lock:
        _policy = policy


def egress_policy() -> PluginEgressPolicy | None:
    """The registered policy, or ``None`` when no runtime has registered one."""
    with _lock:
        return _policy


_kernel_getter: Callable[[], Any] | None = None


def bind_egress_kernel(getter: Callable[[], Any] | None) -> None:
    """Govern governed-client requests by ``getter()``'s kernel (read per call)."""
    global _kernel_getter
    with _lock:
        _kernel_getter = getter


def egress_kernel() -> Any:
    """The kernel the governed client fires ``PRE/POST_EGRESS`` on, or ``None`` (fail closed)."""
    with _lock:
        getter = _kernel_getter
    return getter() if getter is not None else None


# -- the call in progress ---------------------------------------------------------------


@dataclass(frozen=True)
class EgressScope:
    """The governed call a plugin's HTTP request is made inside, stamped by the harness.

    Set around the tool (or capability provider) invocation by the tool runner, so the
    egress rows carry the turn's run id, step, tool, caller and data class. Never the
    plugin's claim: the plugin never constructs one.
    """

    run_id: str
    agent_type: str
    tool: str
    #: The plugin that owns the tool being run (the client is authorised by its OWN plugin,
    #: bound by the harness, not by this).
    tool_plugin: str | None = None
    caller: str | None = None
    step_id: int | None = None
    classification: str | None = None
    #: The governed call's own id (the runner's ``call_id``, a ULID, #134): the parent of
    #: every request made inside it, for a tool call and a capability call alike.
    tool_call_id: str | None = None
    #: Requests seen inside this call, by (method, host, port), for ``attempt`` / ``replay_of``.
    attempts: dict[Any, list[Any]] = field(default_factory=dict, compare=False, repr=False)


_scope: ContextVar[EgressScope | None] = ContextVar("iris_egress_scope", default=None)


@contextmanager
def egress_scope(scope: EgressScope) -> Iterator[None]:
    """Mark the code run inside the block as part of ``scope``'s governed call."""
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


def note_unrecorded_outcome(host: str, call_id: str | None) -> None:
    """Count a governed request that completed but whose outcome row never reached the ledger.

    A failed PRE row means the request is not sent; by the time the POST row is written the
    request has happened and the response is in hand, so the response is still returned (an
    error here would report a POST that took effect as failed, and a retry would repeat it).
    The loss is made visible instead: this counter, an error in the log naming the call, and
    a System Health row. The same ledger being down fails the next PRE write, so every later
    request is refused; at most the outcomes in flight when it went down are lost.
    """
    global _unrecorded
    with _lock:
        _unrecorded += 1
    logger.error(
        "egress outcome not recorded: call %s to %s completed but its post_egress ledger row "
        "could not be written",
        call_id or "(no call id)",
        host or "(unknown host)",
    )


def unrecorded_outcomes() -> int:
    """How many governed requests have completed with no ``post_egress`` row, this process."""
    return _unrecorded


def current_egress_scope() -> EgressScope | None:
    """The governed call in progress on this thread/task, or ``None`` outside one."""
    return _scope.get()


__all__ = [
    "DEFAULT_DATA_CLASS",
    "DEFAULT_SCHEMES",
    "EGRESS_DATA_CLASSES",
    "SCHEMES",
    "EgressScope",
    "EgressVerdict",
    "HostRule",
    "PluginEgress",
    "PluginEgressPolicy",
    "DEFAULT_RESPONSE_BYTES",
    "MAX_RESPONSE_BYTES_CEILING",
    "current_egress_scope",
    "note_unrecorded_outcome",
    "unrecorded_outcomes",
    "egress_policy",
    "egress_scope",
    "INTERNAL_NAME_SUFFIXES",
    "is_internal_name",
    "is_ip_literal",
    "normalize_host",
    "normalize_host_pattern",
    "register_egress_policy",
    "bind_egress_kernel",
    "egress_kernel",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_policy", "_kernel_getter", "_unrecorded")
