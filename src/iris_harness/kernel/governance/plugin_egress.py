"""Which hosts a plugin may contact: the egress declaration, compiled (issue #103).

A plugin's manifest says which hosts its code talks to (``egress:``). The plugin host, which
sits above the kernel and is what knows the manifests, compiles every mounted manifest into
one :class:`PluginEgressPolicy` and registers it once plugins have mounted. The same seam as
``caller_policy.py``.

Declared only, NOT ENFORCED until #103b: nothing in the kernel reads the registered policy
yet, so it changes no plugin's behaviour.

The policy itself fails closed, and says so: asked about a host with no policy registered,
or for a plugin it does not know, it denies with a reason naming what is missing. Once the
governed client (#103b) consults it, an empty declaration will be a closed door; today it is
not one.
"""

from __future__ import annotations

import ipaddress
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from iris_harness.foundation.process_state import track_globals

#: The data classes a host may be declared to receive. Never ``secret``: a secret does not
#: leave the owner's machines to a plugin's host (``config/governance/egress.yaml``).
EGRESS_DATA_CLASSES: Final[tuple[str, ...]] = ("public", "internal", "personal")
_CLASS_RANK: Final[dict[str, int]] = {"public": 0, "internal": 1, "personal": 2, "secret": 3}

DEFAULT_DATA_CLASS: Final = "internal"
DEFAULT_SCHEMES: Final[tuple[str, ...]] = ("https",)
_DEFAULT_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}
SCHEMES: Final[tuple[str, ...]] = ("http", "https")

_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


def normalize_host(value: str) -> str:
    """A host as compared: lower-case, no trailing dot. Raises ``ValueError`` if it is not one.

    A host is a DNS name or an IP literal, with no scheme, port, path, userinfo or
    whitespace: those are separate declarations (``schemes``, ``ports``).
    """
    host = value.strip().lower().rstrip(".")
    if not host or any(c in host for c in "/:@?#[] \t"):
        raise ValueError(
            f"{value!r} is not a host (no scheme, port, path or userinfo; use `ports` and "
            "`schemes` to say those)"
        )
    return host


MAX_LABEL_LENGTH: Final = 63
MAX_HOST_LENGTH: Final = 253
MAX_HOSTS_PER_PLUGIN: Final = 256

# ASCII only, matched with ``fullmatch`` (``$`` would accept a trailing newline).
_DECLARED_LABEL = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?", re.ASCII)
_HEX_OR_OCTAL_NUMBER = re.compile(r"0x[0-9a-f]*|[0-9]+", re.ASCII)


def _refuse(value: str, why: str) -> ValueError:
    return ValueError(f"{value!r}: {why}")


def normalize_host_pattern(value: str) -> str:
    """A declared host: an exact DNS name, or ``*.suffix`` for any subdomain of ``suffix``.

    Strict, because this is what an owner reads as the plugin's allowlist: ASCII letters,
    digits and hyphens only (an internationalised name is declared in its ``xn--`` form, which
    must decode), at most one trailing dot (dropped), labels of at most 63 and a name of at
    most 253 characters. A bare ``*`` is refused: "any host" is ``open_web: true``, which says
    so out loud. Refused outright, fail closed: IP literals in any spelling (dotted, short,
    hex, octal, IPv6), ``localhost``, and numeric last labels, which a resolver may read as an
    address; and a wildcard whose base is a single label (``*.com``). A wildcard over a
    multi-label public suffix (``*.co.uk``) is not caught: the repo ships no public-suffix
    list (docs/architecture/plugin-egress.md).
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
    if body == "localhost" or body.endswith(".localhost"):
        raise _refuse(value, "localhost is not a host a plugin may declare")
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


def register_egress_policy(policy: PluginEgressPolicy | None) -> None:
    """Install (or, with ``None``, remove) the policy the egress hook enforces."""
    global _policy
    with _lock:
        _policy = policy


def egress_policy() -> PluginEgressPolicy | None:
    """The registered policy, or ``None`` when no runtime has registered one."""
    with _lock:
        return _policy


__all__ = [
    "DEFAULT_DATA_CLASS",
    "DEFAULT_SCHEMES",
    "EGRESS_DATA_CLASSES",
    "SCHEMES",
    "EgressVerdict",
    "HostRule",
    "PluginEgress",
    "PluginEgressPolicy",
    "egress_policy",
    "normalize_host",
    "normalize_host_pattern",
    "register_egress_policy",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_policy")
