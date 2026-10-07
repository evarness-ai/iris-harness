"""Egress-allowlist enforcement for the code_exec Docker sandbox (exp-006 GAP-16).

A network-LAYER allowlist for untrusted sandbox code (an ``HTTP_PROXY`` env alone is
advisory — raw sockets bypass it). The sandbox container runs on an ``--internal`` Docker
network (no direct route out) with ``HTTP(S)_PROXY`` pointed at a small CONNECT-proxy
sidecar (``egress_proxy.py``) that permits only allowlisted hosts. Raw / non-proxy egress
is physically blocked by the internal network; the proxy is the only way out and it
enforces the allowlist.

This module owns the host-side allowlist config + idempotent Docker lifecycle (networks +
proxy container). It shells out to the ``docker`` CLI, mirroring ``DockerSandbox``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path

# This module's purpose is to drive the `docker` CLI as a subprocess.
# ruff: noqa: S603

logger = logging.getLogger(__name__)

INTERNAL_NETWORK = "iris-egress-internal"
EXTERNAL_NETWORK = "iris-egress-external"
PROXY_CONTAINER = "iris-egress-proxy"
PROXY_IMAGE = "python:3.12-slim"
PROXY_PORT = 8888
_PROXY_SCRIPT = Path(__file__).resolve().parent / "egress_proxy.py"
_ALLOWLIST_LABEL = "iris.egress.allowlist"

# Conservative defaults: Python package installs (pip) + source fetches (GitHub). Override
# at the call site via the IRIS_SANDBOX_EGRESS_ALLOWLIST env (comma-separated host suffixes).
DEFAULT_ALLOWLIST: tuple[str, ...] = (
    "pypi.org",
    "files.pythonhosted.org",
    "pythonhosted.org",
    "github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
    "codeload.github.com",
    # Live market quotes for the stock_quote tool (key-less, read-only GET).
    "query1.finance.yahoo.com",
)

_ALLOWLIST_ENV = "IRIS_SANDBOX_EGRESS_ALLOWLIST"


def configured_allowlist() -> tuple[str, ...]:
    """The effective allowlist: ``IRIS_SANDBOX_EGRESS_ALLOWLIST`` (csv) or the default."""
    raw = os.environ.get(_ALLOWLIST_ENV, "").strip()
    if not raw:
        return DEFAULT_ALLOWLIST
    return tuple(h.strip().lower() for h in raw.split(",") if h.strip())


def host_allowed(host: str, allowlist: tuple[str, ...]) -> bool:
    """True if ``host`` equals or is a subdomain of any allowlist entry.

    Kept in sync with ``egress_proxy._host_allowed`` (which cannot import this module —
    it runs stdlib-only in a bare sidecar container).
    """
    host = host.strip().lower().rstrip(".")
    if ":" in host:
        host = host.split(":", 1)[0]
    return bool(host) and any(host == e or host.endswith("." + e) for e in allowlist)


def proxy_url() -> str:
    return f"http://{PROXY_CONTAINER}:{PROXY_PORT}"


def proxy_env() -> dict[str, str]:
    """Env pointing a container's HTTP clients (curl, pip, requests) at the proxy."""
    url = proxy_url()
    return {
        "HTTP_PROXY": url,
        "HTTPS_PROXY": url,
        "http_proxy": url,
        "https_proxy": url,
        "PIP_PROXY": url,
        "no_proxy": "",
        "NO_PROXY": "",
    }


def _allowlist_hash(allowlist: tuple[str, ...]) -> str:
    return hashlib.sha256(",".join(sorted(allowlist)).encode()).hexdigest()[:12]


def _run(args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def _network_exists(name: str) -> bool:
    return _run(["docker", "network", "inspect", name]).returncode == 0


def _ensure_network(name: str, *, internal: bool) -> None:
    if _network_exists(name):
        return
    args = ["docker", "network", "create"]
    if internal:
        args.append("--internal")
    args.append(name)
    proc = _run(args)
    if proc.returncode != 0 and not _network_exists(name):  # tolerate create races
        raise RuntimeError(f"failed to create docker network {name}: {proc.stderr.strip()}")


def ensure_egress_infra(allowlist: tuple[str, ...]) -> None:
    """Idempotently ensure the internal + external networks and a current allowlist proxy.

    Reuses a running proxy whose allowlist matches; recreates it when stale (not running or
    allowlist changed). Raises ``RuntimeError`` if the infra cannot be brought up — callers
    should then fail CLOSED (run the sandbox with no network), never fall back to open egress.
    """
    want_hash = _allowlist_hash(allowlist)
    _ensure_network(INTERNAL_NETWORK, internal=True)
    _ensure_network(EXTERNAL_NETWORK, internal=False)

    inspect = _run(
        [
            "docker",
            "inspect",
            "-f",
            '{{.State.Running}}|{{index .Config.Labels "' + _ALLOWLIST_LABEL + '"}}',
            PROXY_CONTAINER,
        ]
    )
    if inspect.returncode == 0:
        running, _, have_hash = inspect.stdout.strip().partition("|")
        if running == "true" and have_hash == want_hash:
            return  # already current
        _run(["docker", "rm", "-f", PROXY_CONTAINER])  # stale → recreate

    if not _PROXY_SCRIPT.is_file():
        raise RuntimeError(f"egress proxy script missing: {_PROXY_SCRIPT}")

    started = _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            PROXY_CONTAINER,
            "--network",
            INTERNAL_NETWORK,
            "--restart",
            "unless-stopped",
            "--label",
            f"{_ALLOWLIST_LABEL}={want_hash}",
            "--memory",
            "128m",
            "--cpus",
            "0.5",
            "--pids-limit",
            "64",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "-v",
            f"{_PROXY_SCRIPT}:/egress_proxy.py:ro",
            "-e",
            f"IRIS_EGRESS_ALLOWLIST={','.join(allowlist)}",
            "-e",
            f"IRIS_EGRESS_PORT={PROXY_PORT}",
            # The sandbox is a sibling container: it reaches the proxy over the internal
            # network, so the sidecar must listen beyond its own loopback.
            "-e",
            "IRIS_EGRESS_BIND=0.0.0.0",
            PROXY_IMAGE,
            "python",
            "/egress_proxy.py",
        ]
    )
    if started.returncode != 0:
        raise RuntimeError(f"failed to start egress proxy: {started.stderr.strip()}")

    # Give the proxy outbound reachability (sandbox stays internal-only).
    connect = _run(["docker", "network", "connect", EXTERNAL_NETWORK, PROXY_CONTAINER])
    if connect.returncode != 0 and "already exists" not in connect.stderr.lower():
        logger.warning("egress proxy external-network connect failed: %s", connect.stderr.strip())


def teardown_egress_infra() -> None:
    """Best-effort removal of the proxy container + networks (cleanup / tests)."""
    _run(["docker", "rm", "-f", PROXY_CONTAINER])
    _run(["docker", "network", "rm", INTERNAL_NETWORK])
    _run(["docker", "network", "rm", EXTERNAL_NETWORK])
