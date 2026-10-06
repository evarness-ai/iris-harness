"""Which hosts a plugin may contact: the egress declaration, compiled (issue #103).

A plugin's manifest says which hosts its code talks to (``egress:``). The plugin host, which
sits above the kernel and is what knows the manifests, compiles every mounted manifest into
one :class:`PluginEgressPolicy` and registers it once plugins have mounted. The same seam as
``caller_policy.py``.

Declared, not yet enforced: nothing in the kernel reads the registered policy yet, so it
changes no plugin's behaviour; the enforcement point lands separately (issue #103).

Fail closed, and say so: with no policy registered, or for a plugin the policy does not
know, every host is denied with a reason naming what is missing -- never allowed because
nobody was asked. An empty declaration is a closed door.
"""

from __future__ import annotations

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


def normalize_host_pattern(value: str) -> str:
    """A declared host: an exact host, or ``*.suffix`` for any subdomain of ``suffix``.

    A bare ``*`` is refused: "any host" is ``open_web: true``, which says so out loud.
    """
    host = normalize_host(value)
    if host == "*" or host.startswith("*") and not host.startswith("*."):
        raise ValueError(
            f"{value!r}: a wildcard is only `*.<domain>`; to allow any host declare `open_web: true`"
        )
    body = host[2:] if host.startswith("*.") else host
    labels = body.split(".")
    ip_like = all(label.isdigit() for label in labels) and len(labels) == 4
    if not ip_like and not all(_LABEL.match(label) for label in labels):
        raise ValueError(f"{value!r} is not a valid host name")
    if host.startswith("*.") and len(labels) < 2:
        raise ValueError(f"{value!r}: a wildcard needs a registrable domain (`*.example.org`)")
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
