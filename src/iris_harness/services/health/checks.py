"""Health Check probes — services + hardware (ADR-0069 slice 1).

Cheap, near-zero-egress checks: probe the IRIS service ports + the Ollama
backend (local by default; ``OLLAMA_BASE_URL`` moves it to another box, which is
how the cloud harness reaches the Mac), and read host pressure from the existing tier-governor sampler
(``iris_harness.services.system.status.host_status`` — no re-sampling). Credential checks and the
opt-in network refresh-probe land in slice 2; ``build_snapshot`` already accepts
``net_probe`` so the signature is stable.

IO is injected (``service_prober``, ``host_status_fn``) so the snapshot builds
deterministically under test without a live stack.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from iris_harness.foundation.clock import utc_now_iso
from iris_harness.foundation.observability.logging_setup import log_egress
from iris_harness.foundation.paths import data_dir
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthSnapshot, HealthState
from iris_harness.services.heartbeat.diagnostics import HeartbeatDiagnostic

# Free-RAM floor below which the host is "degraded" — mirrors the tier
# governor's pressure threshold (iris_harness.llm.arbiter), kept as a local constant to
# avoid coupling to a private module value.
_RAM_FREE_GB_YELLOW = 4.0

# Disk thresholds. The trial VM is a B2pts_v2 whose disk is shared by the image,
# the logs, the audit trail and every store; 2026-09-19's first housekeeping pass
# reclaimed 109 MB from one log file alone. Yellow leaves room to notice, red
# leaves room to act.
_DISK_YELLOW_PCT = 85.0
_DISK_RED_PCT = 95.0

# Below this, the process is newly started. A deliberate deploy clears it within
# the window; a container that OOMs and restarts sits here permanently, which on
# a small shared VM is the failure worth catching.
_UPTIME_YELLOW_SECONDS = 10 * 60

# Import time is close enough to process start for this purpose, and needs no
# psutil: the module is imported once, during startup.
_PROCESS_STARTED_AT = time.time()

# Per-probe timeout. Services are local, so this stays tight to keep the
# synchronous tool call responsive (slice 3 caches the snapshot via heartbeat).
_PROBE_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class ServiceTarget:
    """A locally-probeable HTTP service. Host/port come from env (start_iris.sh
    overrides) with the same defaults the launcher uses.

    ``enabled_when`` marks an opt-in service: it returns True only when the
    current configuration asked for the service. An opt-in service that is not
    asked for and not running is GREY (informational), never RED — the harness
    works without it. ``start_action`` is the command that brings the service up;
    it rides on every non-green service check so the alert is actionable.

    ``base_url_env`` names an env var that moves the service off this host
    entirely (the cloud harness reaches the Mac's Ollama over the tailnet). When
    it is set, the probe follows it instead of ``host``/``port``, so a remote
    backend is not reported down because nothing answers on our own loopback.
    """

    name: str
    port_env: str
    default_port: int
    path: str
    enabled_when: Callable[[], bool] | None = None
    start_action: str | None = None
    base_url_env: str | None = None

    def base_url(self) -> str | None:
        """The configured remote base URL, or ``None`` when the service is local.

        A trailing ``/v1`` is dropped: the same variable feeds the OpenAI-compat
        client (``llm.tier_router``), but the health path lives off the root.
        """
        if self.base_url_env is None:
            return None
        raw = os.environ.get(self.base_url_env, "").strip().rstrip("/")
        if not raw:
            return None
        return raw[: -len("/v1")] if raw.endswith("/v1") else raw

    def url(self, host: str) -> str:
        base = self.base_url()
        if base is not None:
            return f"{base}{self.path}"
        port = os.environ.get(self.port_env, str(self.default_port))
        return f"http://{host}:{port}{self.path}"

    def location(self) -> str | None:
        """``host:port`` when the service lives on another box, else ``None``.

        It rides on the *detail*, never on ``start_action``: the Action Center
        offers that action as a ``copy_command``, so it has to stay pasteable.
        """
        base = self.base_url()
        return None if base is None else urlparse(base).netloc

    def is_enabled(self) -> bool:
        """Required services are always enabled; opt-in ones ask their predicate."""
        return True if self.enabled_when is None else self.enabled_when()


def _remote_evaluator_requested() -> bool:
    """The evaluator sidecar serves the kernel only in ``remote`` mode
    (``governance.wiring._remote_evaluator_backend_from_env``); the default is
    in-process."""
    return os.environ.get("IRIS_GOVERNANCE_EVALUATOR_MODE", "local").strip().lower() == "remote"


_LLM_PROXY_PORT_ENV = "IRIS_PROXY_PORT"
_LLM_PROXY_DEFAULT_PORT = 4000


def _llm_proxy_requested() -> bool:
    """True when this deployment routes models through the failover proxy.

    The proxy and LM Studio share ``LM_STUDIO_BASE_URL``, because the harness
    talks to both with the same OpenAI-compatible client. They are told apart by
    port: a deployment with the proxy points it at the proxy's port
    (``IRIS_PROXY_PORT``, 4000), a Mac points it at LM Studio's 1234, and only the
    former serves ``/health``.
    """
    raw = os.environ.get("LM_STUDIO_BASE_URL", "").strip()
    if not raw:
        return False
    try:
        expected = int(os.environ.get(_LLM_PROXY_PORT_ENV, str(_LLM_PROXY_DEFAULT_PORT)))
        return urlparse(raw).port == expected
    except ValueError:  # an unparseable port in either place is not a request
        return False


def _channel_gateway_requested() -> bool:
    """The gateway (``src/iris_harness/server/channel_gateway``) runs the Telegram long-poller
    and the WebSocket bridge; nothing asks for it until a bot token is set."""
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN", "").strip())


# Backend services + the local model brain. The Web UI (vite) is the renderer,
# not a probed dependency; cloud LLM providers are presence-only (slice 2).
_SERVICES: tuple[ServiceTarget, ...] = (
    ServiceTarget(
        "governor",
        "IRIS_GOVERNOR_PORT",
        8080,
        "/healthz",
        start_action="uvicorn iris_harness.server.governor.main:app --port 8080",
    ),
    ServiceTarget(
        "evaluator",
        "IRIS_EVALUATOR_PORT",
        8090,
        "/healthz",
        enabled_when=_remote_evaluator_requested,
        start_action="iris evaluator start",
    ),
    ServiceTarget(
        "iris_api",
        "IRIS_API_PORT",
        8003,
        "/healthz",
        start_action="iris serve",
    ),
    ServiceTarget(
        "channel_gateway",
        "IRIS_CHANNEL_GATEWAY_PORT",
        8006,
        "/health",
        enabled_when=_channel_gateway_requested,
        start_action="uvicorn iris_harness.server.channel_gateway.main:app --port 8006",
    ),
    ServiceTarget(
        # The LLM failover proxy: Mac first, Azure for open-data tiers. It and
        # the profile-gated `litellm` are alternatives on the same port, so one
        # probe covers whichever is serving.
        #
        # Deliberately NOT following LM_STUDIO_BASE_URL the way ollama follows
        # OLLAMA_BASE_URL: the proxy shares the governor's network namespace, so
        # it is always a loopback neighbour. Following the variable would make a
        # Mac probe LM Studio's /health, get a 404, and report YELLOW — and the
        # not-enabled downgrade below only rescues RED.
        "llm_proxy",
        _LLM_PROXY_PORT_ENV,
        _LLM_PROXY_DEFAULT_PORT,
        "/health",
        enabled_when=_llm_proxy_requested,
        start_action="iris-compose up -d proxy",
    ),
    ServiceTarget(
        "ollama",
        "IRIS_OLLAMA_PORT",
        11434,
        "/api/tags",
        start_action="ollama serve",
        # The cloud harness runs no models: OLLAMA_BASE_URL points at the Mac
        # over the tailnet, and that is the box this probe must ask.
        base_url_env="OLLAMA_BASE_URL",
    ),
)


def _classify_status(code: int | None) -> tuple[HealthState, str]:
    """Map an HTTP result to a (state, detail). ``None`` = unreachable."""
    if code is None:
        return HealthState.RED, "unreachable"
    if 200 <= code < 300:
        return HealthState.GREEN, f"HTTP {code}"
    return HealthState.YELLOW, f"degraded (HTTP {code})"


def _probe_one(url: str) -> int | None:
    """Return the HTTP status code, or ``None`` if the service is unreachable."""
    import httpx

    try:
        log_egress(
            destination=urlparse(url).netloc,
            method="GET",
            kind="service",
            purpose="health-probe",
        )
        resp = httpx.get(url, timeout=_PROBE_TIMEOUT_S)
        return resp.status_code
    except Exception:  # noqa: BLE001 — any transport error means "down"
        return None


def service_checks(
    *,
    host: str | None = None,
    prober: Callable[[str], int | None] = _probe_one,
) -> list[HealthCheck]:
    """Probe every service target and classify each."""
    resolved_host = host or os.environ.get("IRIS_API_HOST", "127.0.0.1")
    checks: list[HealthCheck] = []
    for target in _SERVICES:
        url = target.url(resolved_host)
        state, detail = _classify_status(prober(url))
        where = target.location()
        if where is not None and state is not HealthState.GREEN:
            # Without this, a remote backend's "unreachable" reads as advice to
            # fix this machine — the cloud harness's Ollama runs on the Mac.
            detail = f"{detail} at {where}"
        if state is HealthState.RED and not target.is_enabled():
            # Not asked for and not running: nothing is wrong, so nothing alerts.
            state = HealthState.GREY
            detail = "not running (optional; not enabled in this configuration)"
        checks.append(
            HealthCheck(
                target=target.name,
                kind=CheckKind.SERVICE,
                state=state,
                detail=detail,
                endpoint=url,
                action=None if state is HealthState.GREEN else target.start_action,
            )
        )
    return checks


def hardware_check(host_status_fn: Callable[[], object] | None = None) -> HealthCheck:
    """Read host pressure (reusing the tier governor's sampler) into one check."""
    if host_status_fn is None:
        from iris_harness.services.system.status import host_status

        host_status_fn = host_status
    host = host_status_fn()

    if getattr(host, "thermal_throttled", False):
        state = HealthState.YELLOW
        why = "thermal-throttled"
    elif getattr(host, "ram_free_gb", 1e9) < _RAM_FREE_GB_YELLOW:
        state = HealthState.YELLOW
        why = "low free RAM"
    else:
        state = HealthState.GREEN
        why = "nominal"

    detail = getattr(host, "summary", lambda: why)()
    if state is HealthState.YELLOW:
        detail = f"{detail} — {why}"
    return HealthCheck(target="host", kind=CheckKind.HARDWARE, state=state, detail=detail)


def _fmt_gb(n_bytes: float) -> str:
    return f"{n_bytes / 1_000_000_000:.1f} GB"


def _fmt_duration(seconds: float) -> str:
    """Two units, largest first: 4d 2h / 3h 14m / 7m."""
    m, h, d = int(seconds // 60), int(seconds // 3600), int(seconds // 86400)
    if d:
        return f"{d}d {h - d * 24}h"
    if h:
        return f"{h}h {m - h * 60}m"
    return f"{max(m, 0)}m"


def disk_check(
    path: Path | None = None,
    usage_fn: Callable[[str], object] = shutil.disk_usage,
) -> HealthCheck:
    """Free space on the volume holding IRIS's data (plan decision 29).

    Nothing watched the disk before this, and on the trial VM it is the likely
    failure: the data directory is a bind mount of the VM's only volume, so
    "the volume IRIS writes to" and "the VM's disk" are the same thing.
    """
    target_path = path or data_dir()
    # A relative data dir may not exist yet on a fresh checkout; the volume it
    # would land on is what we are measuring, so walk up to something real.
    probe = target_path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    try:
        usage = usage_fn(str(probe))
    except OSError as exc:  # unreadable mount, or a path that vanished mid-check
        return HealthCheck(
            target="disk",
            kind=CheckKind.HARDWARE,
            state=HealthState.GREY,
            detail=f"could not read {probe}: {exc}",
        )

    total = float(getattr(usage, "total", 0) or 0)
    used = float(getattr(usage, "used", 0) or 0)
    if total <= 0:
        return HealthCheck(
            target="disk",
            kind=CheckKind.HARDWARE,
            state=HealthState.GREY,
            detail=f"no usable size reported for {probe}",
        )

    pct = used / total * 100
    detail = f"{_fmt_gb(used)}/{_fmt_gb(total)} used ({pct:.0f}%) on {probe}"
    if pct >= _DISK_RED_PCT:
        return HealthCheck(
            target="disk",
            kind=CheckKind.HARDWARE,
            state=HealthState.RED,
            detail=f"{detail} — almost full",
            action="iris housekeeping run",
        )
    if pct >= _DISK_YELLOW_PCT:
        return HealthCheck(
            target="disk",
            kind=CheckKind.HARDWARE,
            state=HealthState.YELLOW,
            detail=f"{detail} — filling up",
            action="iris housekeeping run",
        )
    return HealthCheck(
        target="disk", kind=CheckKind.HARDWARE, state=HealthState.GREEN, detail=detail
    )


def uptime_check(
    started_at: float | None = None,
    now_fn: Callable[[], float] = time.time,
) -> HealthCheck:
    """How long this process has been up (plan decision 29).

    Answers the question a deploy leaves behind — did it come back, and has it
    stayed up? A restart loop shows as a permanently short uptime rather than as
    a service that happens to answer between crashes.
    """
    began = _PROCESS_STARTED_AT if started_at is None else started_at
    seconds = max(now_fn() - began, 0.0)
    if seconds < _UPTIME_YELLOW_SECONDS:
        # "restarted 0m ago" is a worse sentence than the thing it describes.
        when = "just now" if seconds < 60 else f"{_fmt_duration(seconds)} ago"
        return HealthCheck(
            target="uptime",
            kind=CheckKind.HARDWARE,
            state=HealthState.YELLOW,
            detail=f"restarted {when}",
        )
    return HealthCheck(
        target="uptime",
        kind=CheckKind.HARDWARE,
        state=HealthState.GREEN,
        detail=f"up {_fmt_duration(seconds)}",
    )


def build_snapshot(
    *,
    net_probe: bool = False,
    host: str | None = None,
    service_prober: Callable[[str], int | None] = _probe_one,
    host_status_fn: Callable[[], object] | None = None,
    disk_check_fn: Callable[[], HealthCheck] = disk_check,
    uptime_check_fn: Callable[[], HealthCheck] = uptime_check,
    credential_checker: Callable[[], list[HealthCheck]] | None = None,
    heartbeat_diagnostics: list[HeartbeatDiagnostic] | None = None,
) -> HealthSnapshot:
    """Build the current HealthSnapshot: services + hardware + credentials.

    ``net_probe`` enables the opt-in credential revocation refresh-probe (wired in
    slice 5; inert until then). IO callables are injectable for deterministic
    tests — pass ``credential_checker=lambda: []`` to skip the account-DB read,
    and ``disk_check_fn``/``uptime_check_fn`` to skip the volume read and clock.
    """
    checks: list[HealthCheck] = []
    checks.extend(service_checks(host=host, prober=service_prober))
    checks.append(hardware_check(host_status_fn=host_status_fn))
    # Injectable like every other IO here: disk_check reads a real volume and
    # uptime_check is a clock, so a test that did not pass its own would depend
    # on the machine it runs on and on how long the process had been alive.
    checks.append(disk_check_fn())
    checks.append(uptime_check_fn())
    if credential_checker is None:
        from iris_harness.services.health.credentials import credential_checks

        checks.extend(credential_checks(net_probe=net_probe))
    else:
        checks.extend(credential_checker())
    if heartbeat_diagnostics:
        checks.extend(_heartbeat_checks(heartbeat_diagnostics))
    return HealthSnapshot(checks=tuple(checks), sampled_at=utc_now_iso())


def _heartbeat_checks(diagnostics: list[HeartbeatDiagnostic]) -> list[HealthCheck]:
    checks: list[HealthCheck] = []
    for item in diagnostics:
        checks.append(
            HealthCheck(
                target=f"heartbeat:{item.name}",
                kind=CheckKind.SERVICE,
                state=HealthState.RED,
                detail=item.detail,
                endpoint="/heartbeat",
                action=item.action,
            )
        )
    return checks


__all__ = [
    "ServiceTarget",
    "build_snapshot",
    "disk_check",
    "hardware_check",
    "service_checks",
    "uptime_check",
]
