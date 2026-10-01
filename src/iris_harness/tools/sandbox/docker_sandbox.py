"""Docker-backed code execution sandbox.

Runs one container per ``run_shell`` call: ephemeral, network-on, scratch
workspace mounted read-write. Designed as the LLM's *execution* surface — it
should be paired with prompts that constrain what the model puts inside ``cmd``.

Hardening (in order of importance):
  - ``--rm``                  → container removed after exit
  - ``--memory=1g``           → cap RAM
  - ``--cpus=2``              → cap CPU
  - ``--pids-limit=256``      → cap fork bombs
  - ``--cap-drop=ALL``        → no Linux capabilities
  - ``--security-opt=no-new-privileges``
  - tmpfs on ``/tmp``
  - workspace mount is the *only* host path visible
  - no Docker socket, no host network, no privileged

Network: pass ``egress_allowlist=(...)`` to restrict outbound traffic to an allowlist,
enforced at the network layer via an internal Docker network + a CONNECT-proxy sidecar
(see ``egress.py``); ``network=False`` fully isolates; the bare default leaves egress open
(``pip install`` works). The production code_exec path (``SandboxToolHost``) sets the
allowlist (exp-006 GAP-16).
"""

# This module's whole purpose is to invoke `docker` as a subprocess.
# ruff: noqa: S603, S607

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from . import egress
from .exceptions import (
    DockerUnavailableError,
    SandboxImageMissingError,
)
from .models import ExecResult
from .workspace import SessionWorkspace

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "iris-sandbox:latest"
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300
DEFAULT_MEMORY = "1g"
DEFAULT_CPUS = "2"
DEFAULT_PIDS = "256"


class DockerSandbox:
    """Per-call Docker sandbox bound to a session workspace.

    Implements the ``iris_harness.tools.sandbox.runtime.SandboxRuntime`` protocol. Subclasses
    (e.g. ``GVisorSandbox``) override :meth:`_runtime_args` to inject extra
    ``docker run`` flags; everything else (hardening, limits, egress) is shared.
    """

    name: str = "docker"

    def __init__(
        self,
        workspace: SessionWorkspace,
        *,
        image: str = DEFAULT_IMAGE,
        memory: str = DEFAULT_MEMORY,
        cpus: str = DEFAULT_CPUS,
        pids_limit: str = DEFAULT_PIDS,
        network: bool = True,
        egress_allowlist: tuple[str, ...] | None = None,
    ) -> None:
        self.workspace = workspace
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.network = network
        self.egress_allowlist = egress_allowlist

    def run_shell(self, cmd: str, *, timeout: int = DEFAULT_TIMEOUT) -> ExecResult:
        """Run ``cmd`` in a fresh container; return stdout/stderr/exit_code/artifacts."""
        if not cmd or not cmd.strip():
            return ExecResult(
                stdout="",
                stderr="empty command",
                exit_code=2,
                duration_ms=0.0,
            )

        timeout = max(1, min(int(timeout), MAX_TIMEOUT))
        self.ensure_ready(auto_build_image=False)

        egress_ready = True
        if self.network and self.egress_allowlist is not None:
            try:
                egress.ensure_egress_infra(self.egress_allowlist)
            except Exception:  # degrade CLOSED on any infra failure
                logger.warning(
                    "egress allowlist infra unavailable; running sandbox with NO network",
                    exc_info=True,
                )
                egress_ready = False

        container_name = f"iris-sandbox-{uuid.uuid4().hex[:12]}"
        argv = self._build_argv(cmd, container_name=container_name, egress_ready=egress_ready)
        logger.debug("sandbox run: %s", " ".join(argv))

        t0 = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout + 5,
                check=False,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (
                (exc.stdout or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes | bytearray)
                else (exc.stdout or "")
            )
            stderr = (
                (exc.stderr or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes | bytearray)
                else (exc.stderr or "")
            )
            stderr = (stderr + f"\n[sandbox killed: exceeded {timeout}s timeout]").lstrip()
            exit_code = 124
            timed_out = True
            self._force_remove_container(container_name)

        duration_ms = (time.monotonic() - t0) * 1000
        artifacts = self.workspace.snapshot_artifacts()

        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
            artifacts=artifacts,
            timed_out=timed_out,
        )

    def ensure_ready(self, *, auto_build_image: bool = False) -> None:
        """Ensure Docker daemon and sandbox image are available.

        When ``auto_build_image`` is true and the image is missing, build it once
        from the sandbox Dockerfile directory.
        """
        self._ensure_docker_available()
        try:
            self._ensure_image_available()
        except SandboxImageMissingError:
            if not auto_build_image:
                raise
            self._build_image()
            self._ensure_image_available()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _runtime_args(self) -> list[str]:
        """Extra ``docker run`` flags selecting the OCI runtime.

        Docker default: none. ``GVisorSandbox`` (6c.2) returns
        ``["--runtime=runsc"]``. Spliced right after ``docker run`` so the rest
        of the hardened argv is identical across runtimes."""
        return []

    def _build_argv(self, cmd: str, *, container_name: str, egress_ready: bool = True) -> list[str]:
        workspace_mount = f"{self.workspace.path}:/workspace:rw"
        argv: list[str] = [
            "docker",
            "run",
            *self._runtime_args(),
            "--rm",
            "--name",
            container_name,
            "-i",
            f"--memory={self.memory}",
            f"--cpus={self.cpus}",
            f"--pids-limit={self.pids_limit}",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--tmpfs=/tmp:rw,nosuid,nodev,size=128m",
            "-w",
            "/workspace",
            "-v",
            workspace_mount,
        ]
        if not self.network:
            argv.append("--network=none")
        elif self.egress_allowlist is not None:
            if egress_ready:
                # Internal network (no direct route out) + HTTP(S)_PROXY pointed at the
                # allowlist proxy. Raw / non-proxy egress is physically blocked.
                argv += ["--network", egress.INTERNAL_NETWORK]
                for key, value in egress.proxy_env().items():
                    argv += ["-e", f"{key}={value}"]
            else:
                argv.append("--network=none")  # fail CLOSED if the proxy is unavailable
        argv.extend([self.image, "bash", "-lc", cmd])
        return argv

    def _force_remove_container(self, name: str) -> None:
        """Best-effort kill+rm for an orphaned container after a timeout.

        `--rm` only fires when the docker CLI sees the container exit. If the
        CLI was killed by ``subprocess.run``'s timeout, the daemon-side
        container can keep running; this guarantees we still take it down.
        """
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            logger.warning("sandbox cleanup: failed to remove container %s", name, exc_info=True)

    def _ensure_docker_available(self) -> None:
        if shutil.which("docker") is None:
            raise DockerUnavailableError(
                "docker CLI not found on PATH. Install Docker Desktop and retry."
            )
        try:
            proc = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DockerUnavailableError(
                "docker daemon not responding (timeout). Is Docker Desktop running?"
            ) from exc
        if proc.returncode != 0:
            raise DockerUnavailableError(
                "docker daemon unreachable. Is Docker Desktop running?\n"
                + (proc.stderr or "").strip()
            )

    def _ensure_image_available(self) -> None:
        proc = subprocess.run(
            ["docker", "image", "inspect", self.image],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            raise SandboxImageMissingError(
                f"sandbox image {self.image!r} not found. Build it with:\n"
                f"  docker build -t {self.image} {Path(__file__).parent}"
            )

    def _build_image(self) -> None:
        """Build the sandbox image from ``src/iris_harness/sandbox``."""
        build_context = str(Path(__file__).parent)
        proc = subprocess.run(
            ["docker", "build", "-t", self.image, build_context],
            capture_output=True,
            text=True,
            timeout=1200,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "docker build failed").strip()
            raise SandboxImageMissingError(
                f"sandbox image {self.image!r} not found and auto-build failed:\n"
                f"  docker build -t {self.image} {build_context}\n"
                f"{detail}"
            )


class GVisorSandbox(DockerSandbox):
    """gVisor-isolated sandbox (Phase 6, 6c.2).

    Identical hardened argv to ``DockerSandbox`` plus ``--runtime=runsc``, so
    syscalls are intercepted by gVisor's userspace kernel instead of hitting the
    host kernel directly — shrinking the container-breakout surface. Linux-only
    (needs the ``runsc`` runtime registered with Docker); the factory falls back
    to Docker where it's unavailable.
    """

    name: str = "gvisor"

    def _runtime_args(self) -> list[str]:
        return ["--runtime=runsc"]
