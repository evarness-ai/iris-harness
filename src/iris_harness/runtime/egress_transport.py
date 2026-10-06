"""The network side of the governed HTTP client: resolve once, check, connect to that address.

A host *name* that passed the egress policy can still resolve to an address inside the
owner's machine or network (an attacker-chosen name that points at ``127.0.0.1`` or the
cloud metadata address, or a name that answers with a public address first and an internal one
the second time). This module closes that gap at the one place it can be closed: the
connection. Each connect resolves the name ONCE, refuses the connection when any resolved
address is loopback, private, link-local, shared, unspecified, multicast, reserved or
embeds such an IPv4 (v4-mapped, NAT64, 6to4), and connects to the checked address itself, so
a second lookup never happens. TLS server-name indication, certificate verification and the
``Host`` header still use the host name: httpcore passes the origin's name to ``start_tls``
and to the request, only the TCP connect target is replaced.

It also enforces the request's total wall-clock deadline on every socket operation, which
httpx's own timeouts do not (each of those is per operation, so a server that answers one
byte per second never trips them).

Not covered: the name lookup itself (``getaddrinfo`` has no timeout, so the deadline starts
counting at the connect), and an in-process plugin that opens its own socket
(docs/architecture/plugin-egress.md).
"""

from __future__ import annotations

import ipaddress
import socket
import ssl
import time
import typing

import httpcore
import httpx

_BLOCKED = "the host name resolves to an address inside this machine or network"
_NO_ADDRESS = "the host name did not resolve"

_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_SIX_TO_FOUR = ipaddress.ip_network("2002::/16")
# IPv6 blocks that embed or alias an IPv4 address, or are reserved: refused outright.
_RESERVED_V6 = tuple(
    ipaddress.ip_network(n)
    for n in (
        "::/96",  # IPv4-compatible (::127.0.0.1 and the like), deprecated
        "::ffff:0:0/96",  # v4-mapped (unwrapped first)
        "::ffff:0:0:0/96",  # SIIT (::ffff:0:a.b.c.d)
        "64:ff9b:1::/48",  # local-use NAT64
        "5f00::/16",  # SRv6 segment identifiers
    )
)

# The resolver is a seam for tests (no real DNS): ``socket.getaddrinfo`` in production.
_resolve: typing.Callable[..., list[tuple[typing.Any, ...]]] = socket.getaddrinfo


class BlockedAddress(Exception):
    """A connect was refused because of where the name resolved. Not an ``OSError``: httpcore
    maps those to ``ConnectError`` and the reason would be lost."""


class DeadlineExceeded(Exception):
    """The request's total time ran out inside a socket operation (not an ``OSError`` either)."""


def blocked_address(address: str) -> bool:
    """Is ``address`` somewhere a plugin's request must never be sent?"""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return True  # not an address at all: refuse rather than guess
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return blocked_address(str(ip.ipv4_mapped))
        if ip in _NAT64:
            return blocked_address(str(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)))
        if any(ip in net for net in _RESERVED_V6):
            return True
        if ip in _SIX_TO_FOUR:
            return blocked_address(str(ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)))
    # ``is_global`` is False for loopback, private (incl. fc00::/7), link-local (incl.
    # 169.254.0.0/16, the metadata address), shared 100.64.0.0/10, 0.0.0.0/8, ::,
    # multicast and the reserved blocks.
    return not ip.is_global or ip.is_multicast


def resolve_checked(host: str, port: int) -> list[str]:
    """Resolve ``host`` once; every address must be allowed. Raises :class:`BlockedAddress`."""
    try:
        infos = _resolve(host, port, type=socket.SOCK_STREAM)
    except Exception:  # noqa: BLE001 - any resolver failure is "did not resolve"
        raise BlockedAddress(_NO_ADDRESS) from None
    addresses: list[str] = []
    for info in infos:
        address = str(info[4][0])
        if blocked_address(address):
            raise BlockedAddress(_BLOCKED)
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise BlockedAddress(_NO_ADDRESS)
    return addresses


class _Deadline:
    def __init__(self, seconds: float) -> None:
        self.at = time.monotonic() + seconds

    def remaining(self) -> float:
        left = self.at - time.monotonic()
        if left <= 0:
            raise DeadlineExceeded
        return left

    def clamp(self, timeout: float | None) -> float:
        left = self.remaining()
        return left if timeout is None else min(timeout, left)


class _DeadlineStream(httpcore.NetworkStream):
    """A socket stream whose every operation is bounded by what is left of the deadline."""

    def __init__(self, inner: httpcore.NetworkStream, deadline: _Deadline) -> None:
        self._inner = inner
        self._deadline = deadline

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._inner.read(max_bytes, self._deadline.clamp(timeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._inner.write(buffer, self._deadline.clamp(timeout))

    def close(self) -> None:
        self._inner.close()

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        tls = self._inner.start_tls(ssl_context, server_hostname, self._deadline.clamp(timeout))
        return _DeadlineStream(tls, self._deadline)

    def get_extra_info(self, info: str) -> typing.Any:
        return self._inner.get_extra_info(info)


class PinnedBackend(httpcore.SyncBackend):
    """Resolve once, check every address, connect to a checked one, under a deadline."""

    def __init__(self, deadline_seconds: float) -> None:
        super().__init__()
        self._deadline = _Deadline(deadline_seconds)
        #: Why a connect was refused (a fixed sentence), for the client to report.
        self.refused: str | None = None

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[typing.Any] | None = None,
    ) -> httpcore.NetworkStream:
        try:
            addresses = resolve_checked(host, port)
        except BlockedAddress as exc:
            self.refused = str(exc)
            raise
        last: Exception | None = None
        for address in addresses:
            try:
                stream = super().connect_tcp(
                    address, port, self._deadline.clamp(timeout), local_address, socket_options
                )
            except httpcore.ConnectError as exc:
                last = exc
                continue
            return _DeadlineStream(stream, self._deadline)
        raise last or httpcore.ConnectError(_NO_ADDRESS)

    def connect_unix_socket(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise BlockedAddress("a unix socket is not a governed destination")


class PinnedTransport(httpx.HTTPTransport):
    """``httpx.HTTPTransport`` connecting through :class:`PinnedBackend`.

    The pool is built here with the backend rather than patched after the fact, but httpx
    gives no public hook for it, so the swap touches ``_pool``; it refuses to run (fails
    closed) if a future httpx no longer has the pool it expects.
    """

    def __init__(self, backend: PinnedBackend) -> None:
        super().__init__(trust_env=False, proxy=None, retries=0)
        if type(self._pool) is not httpcore.ConnectionPool:
            raise RuntimeError("the installed httpx has no connection pool to pin")
        self._pool.close()
        self._pool = httpcore.ConnectionPool(
            ssl_context=httpx.create_ssl_context(trust_env=False),
            max_connections=1,
            max_keepalive_connections=0,
            http1=True,
            http2=False,
            network_backend=backend,
        )


__all__ = [
    "BlockedAddress",
    "DeadlineExceeded",
    "PinnedBackend",
    "PinnedTransport",
    "blocked_address",
    "resolve_checked",
]
