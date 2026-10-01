# ruff: noqa: S603
"""Pluggable sandbox-runtime seam (Phase 6, sub-phase 6c.1).

The contract the ``code_exec`` / ``run_command`` call sites depend on, plus a
factory that selects the configured backend with availability detection and a
no-silent-downgrade fallback. 6c.1 ships the seam with **Docker only** (default,
no behavior change); ``GVisorSandbox`` lands in 6c.2 behind the same Protocol.
See ``docs/architecture/sandbox-hardening.md``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from iris_harness.tools.sandbox.config import SandboxConfig, SandboxRuntimeName
from iris_harness.tools.sandbox.docker_sandbox import DockerSandbox, GVisorSandbox
from iris_harness.tools.sandbox.models import ExecResult
from iris_harness.tools.sandbox.workspace import SessionWorkspace

logger = logging.getLogger(__name__)


@runtime_checkable
class SandboxRuntime(Protocol):
    """The swappable contract every sandbox backend satisfies."""

    name: str

    def ensure_ready(self, *, auto_build_image: bool = False) -> None: ...

    def run_shell(self, cmd: str, *, timeout: int = 30) -> ExecResult: ...


def _docker_available() -> bool:
    docker_exe = shutil.which("docker")
    if docker_exe is None:
        return False
    try:
        proc = subprocess.run(
            [docker_exe, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return False


def _gvisor_available() -> bool:
    """gVisor needs Docker + the ``runsc`` runtime registered (Linux-only)."""
    if not _docker_available():
        return False
    docker_exe = shutil.which("docker")
    if docker_exe is not None:
        try:
            proc = subprocess.run(
                [docker_exe, "info", "--format", "{{.Runtimes}}"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0 and "runsc" in (proc.stdout or ""):
                return True
        except Exception:  # noqa: BLE001,S110 - advisory probe; fall through to which()
            pass
    return shutil.which("runsc") is not None


def is_runtime_available(runtime: str) -> bool:
    """Whether ``runtime`` can actually run on this host."""
    if runtime == "docker":
        return _docker_available()
    if runtime == "gvisor":
        return _gvisor_available()
    # firecracker (and unknown) — not implemented yet (plan §6).
    return False


def instantiate_runtime(
    name: SandboxRuntimeName | str,
    workspace: SessionWorkspace,
    *,
    config: SandboxConfig,
    network: bool = True,
    egress_allowlist: tuple[str, ...] | None = None,
) -> SandboxRuntime:
    """Construct the runtime for ``name`` (no availability check — caller's job).

    Unknown names fall back to Docker, so a misconfiguration never yields an
    unsandboxed run."""
    limits = config.limits
    kwargs = {
        "memory": limits.memory,
        "cpus": limits.cpus,
        "pids_limit": str(limits.pids_limit),
        "network": network,
        "egress_allowlist": egress_allowlist,
    }
    if name == "gvisor":
        return GVisorSandbox(workspace, **kwargs)  # type: ignore[arg-type]
    return DockerSandbox(workspace, **kwargs)  # type: ignore[arg-type]


def resolve_runtime_name(
    config: SandboxConfig,
    *,
    available: Callable[[str], bool] | None = None,
) -> SandboxRuntimeName | None:
    """Pick the effective runtime name, applying availability + fallback.

    Returns the configured runtime when available; otherwise ``on_unavailable``
    decides: ``fallback`` → ``"docker"`` (logged — no silent downgrade), or
    ``disable`` → ``None``. ``None`` also when nothing is available.
    """
    check = available or is_runtime_available
    requested = config.resolved_runtime()
    if check(requested):
        return requested
    if config.on_unavailable == "disable":
        logger.warning(
            "sandbox runtime %r unavailable and on_unavailable=disable; code_exec disabled",
            requested,
        )
        return None
    if requested != "docker" and check("docker"):
        logger.warning(
            "sandbox runtime %r unavailable; falling back to docker (isolation downgrade)",
            requested,
        )
        return "docker"
    logger.warning("no sandbox runtime available (requested %r); code_exec disabled", requested)
    return None


def build_sandbox_runtime(
    workspace: SessionWorkspace,
    *,
    config: SandboxConfig | None = None,
    network: bool = True,
    egress_allowlist: tuple[str, ...] | None = None,
    available: Callable[[str], bool] | None = None,
) -> tuple[SandboxRuntime | None, str]:
    """Select + build the sandbox runtime. Returns ``(runtime_or_None, name)``.

    Never returns a less-isolated runtime than requested without a warning, and
    never an unsandboxed one (``None`` → caller disables code_exec)."""
    cfg = config or SandboxConfig.default()
    name = resolve_runtime_name(cfg, available=available)
    if name is None:
        return None, cfg.resolved_runtime()
    runtime = instantiate_runtime(
        name, workspace, config=cfg, network=network, egress_allowlist=egress_allowlist
    )
    return runtime, name
