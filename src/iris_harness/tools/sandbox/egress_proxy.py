"""Standalone egress-allowlist proxy for the code_exec sandbox (exp-006 GAP-16).

Runs in a bare ``python:3.12-slim`` sidecar container (mounted read-only) — **stdlib
only, no iris imports** — so it cannot share code with ``egress.py`` (the host-side
matcher is duplicated there intentionally; keep the two ``_host_allowed`` in sync).

The sandbox container sits on an ``--internal`` Docker network (no direct route out)
with ``HTTP(S)_PROXY`` pointed here, so this proxy is the *only* egress path. It permits
``CONNECT`` (HTTPS) and absolute-form HTTP only to allowlisted hosts (exact or subdomain
suffix match); everything else → ``403``. IP-literal targets never match a domain
allowlist, so they are denied too.

Config via env:
  IRIS_EGRESS_ALLOWLIST  comma-separated host suffixes (e.g. "pypi.org,github.com")
  IRIS_EGRESS_PORT       listen port (default 8888)
"""

from __future__ import annotations

import os
import select
import socket
import sys
import threading

_ALLOWLIST: tuple[str, ...] = tuple(
    h.strip().lower() for h in os.environ.get("IRIS_EGRESS_ALLOWLIST", "").split(",") if h.strip()
)
_PORT = int(os.environ.get("IRIS_EGRESS_PORT", "8888"))
_BUFSIZE = 65536
_CONNECT_TIMEOUT = 10


def _host_allowed(host: str, allowlist: tuple[str, ...]) -> bool:
    """True if ``host`` equals or is a subdomain of any allowlist entry."""
    host = host.strip().lower().rstrip(".")
    if ":" in host:
        host = host.split(":", 1)[0]
    return bool(host) and any(host == e or host.endswith("." + e) for e in allowlist)


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _tunnel(a: socket.socket, b: socket.socket) -> None:
    """Bidirectionally pipe two sockets until either closes."""
    pair = [a, b]
    try:
        while True:
            readable, _, errored = select.select(pair, [], pair, 60)
            if errored or not readable:
                break
            for src in readable:
                data = src.recv(_BUFSIZE)
                if not data:
                    return
                (b if src is a else a).sendall(data)
    except OSError:
        return


def _deny(client: socket.socket, target: str, kind: str) -> None:
    _log(f"DENY {kind} {target}")
    try:
        client.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
    except OSError:
        pass


def _handle(client: socket.socket) -> None:
    try:
        client.settimeout(_CONNECT_TIMEOUT)
        head = b""
        while b"\r\n" not in head:
            chunk = client.recv(_BUFSIZE)
            if not chunk:
                return
            head += chunk
            if len(head) > _BUFSIZE:
                return
        line, _, rest = head.partition(b"\r\n")
        parts = line.decode("latin-1", "replace").split()
        if len(parts) < 2:
            return
        method, target = parts[0].upper(), parts[1]
        proto = parts[2] if len(parts) > 2 else "HTTP/1.1"

        if method == "CONNECT":
            hostname = target.split(":", 1)[0]
            port = int(target.split(":", 1)[1]) if ":" in target else 443
            if not _host_allowed(hostname, _ALLOWLIST):
                _deny(client, target, "CONNECT")
                return
            try:
                upstream = socket.create_connection((hostname, port), timeout=_CONNECT_TIMEOUT)
            except OSError:
                client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                return
            _log(f"ALLOW CONNECT {target}")
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            client.settimeout(None)
            upstream.settimeout(None)
            _tunnel(client, upstream)
            upstream.close()
            return

        if target.lower().startswith("http://"):
            hostport, _, path = target[len("http://") :].partition("/")
            hostname = hostport.split(":", 1)[0]
            port = int(hostport.split(":", 1)[1]) if ":" in hostport else 80
            if not _host_allowed(hostname, _ALLOWLIST):
                _deny(client, hostport, "HTTP")
                return
            try:
                upstream = socket.create_connection((hostname, port), timeout=_CONNECT_TIMEOUT)
            except OSError:
                client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                return
            _log(f"ALLOW HTTP {hostport}")
            upstream.sendall(f"{method} /{path} {proto}\r\n".encode("latin-1") + rest)
            client.settimeout(None)
            upstream.settimeout(None)
            _tunnel(client, upstream)
            upstream.close()
            return

        _deny(client, target, method)
    except Exception as exc:  # noqa: BLE001 - a proxy must never crash the loop on one conn
        _log(f"ERR {exc!r}")
    finally:
        try:
            client.close()
        except OSError:
            pass


def main() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", _PORT))  # noqa: S104 - only reachable on the internal docker net
    server.listen(128)
    _log(f"egress-proxy listening on :{_PORT} allowlist={list(_ALLOWLIST)}")
    while True:
        try:
            client, _ = server.accept()
        except OSError:
            continue
        threading.Thread(target=_handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()
