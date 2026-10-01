"""Host-side driver for the Docker/gVisor eval sandbox (ADR-0070, IRIS_EVAL_SANDBOX=docker).

Runs the in-container entrypoint (:mod:`iris_harness.services.learning.eval_replay_cli`) inside the
``iris-eval`` image under a hardened ``docker run`` (cap-drop, no-new-privileges,
gVisor ``runsc`` when available, resource limits), pointing the container's model
client at the host's Ollama. The container has no host credentials mounted, so a
replayed tool can't reach the user's real external accounts even with network on.

The actual ``docker`` invocation is injected (``runner``) so the argv construction
and stdout→verdict parsing are unit-testable without Docker. The end-to-end run is
validated on a Docker host (build the image first — see ``sandbox/Dockerfile.eval``).
"""

# This module's purpose is to invoke `docker` as a subprocess with a fixed argv.
# ruff: noqa: S603, S607

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from iris_harness.services.learning.eval_harness import EvalQuery
from iris_harness.services.learning.eval_replay_cli import verdict_from_stdout, workload_to_json
from iris_harness.services.learning.preflight import PreflightVerdict, SandboxUnavailable

_CONTAINER_WORKLOAD_PATH = "/eval/workload.json"

logger = logging.getLogger(__name__)

EVAL_IMAGE = "iris-eval:latest"
# From a container, the host's Ollama is reachable via host.docker.internal.
_DEFAULT_OLLAMA = "http://host.docker.internal:11434"


@dataclass(frozen=True)
class RunResult:
    """Minimal result of running the eval container (decoupled from the sandbox)."""

    stdout: str
    stderr: str
    exit_code: int


def build_eval_docker_argv(
    intent: str,
    *,
    image: str = EVAL_IMAGE,
    repeats: int = 1,
    max_items: int = 8,
    min_queries: int = 3,
    min_rate: float = 0.8,
    ollama_base_url: str = _DEFAULT_OLLAMA,
    gvisor: bool = True,
    workload_host_path: str | None = None,
) -> list[str]:
    """Hardened ``docker run`` argv that runs the in-container eval entrypoint.

    ``workload_host_path`` (the host-frozen workload JSON) is mounted read-only and
    passed as ``--workload`` so the isolated container replays the host's queries
    rather than its own (empty) trace store.
    """
    argv: list[str] = ["docker", "run", "--rm"]
    if gvisor:
        argv.append("--runtime=runsc")  # gVisor userspace kernel; falls back if absent
    argv += [
        "--memory=2g",
        "--cpus=2",
        "--pids-limit=512",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs=/tmp:rw,nosuid,nodev,size=256m",
        "--add-host=host.docker.internal:host-gateway",
        "-e",
        f"OLLAMA_BASE_URL={ollama_base_url}",
        "-e",
        "IRIS_DISABLE_WARMUP=1",
        "-e",
        "IRIS_DISABLE_EXTERNAL_WRITES=1",
        "-e",
        f"IRIS_AUTH_SECRET={os.getenv('IRIS_AUTH_SECRET', 'eval-sandbox')}",
    ]
    if workload_host_path is not None:
        argv += ["-v", f"{workload_host_path}:{_CONTAINER_WORKLOAD_PATH}:ro"]
    argv += [
        image,
        "python",
        "-m",
        "iris_harness.services.learning.eval_replay_cli",
        "--intent",
        intent,
        "--max-items",
        str(max_items),
        "--repeats",
        str(repeats),
        "--min-queries",
        str(min_queries),
        "--min-rate",
        str(min_rate),
    ]
    if workload_host_path is not None:
        argv += ["--workload", _CONTAINER_WORKLOAD_PATH]
    return argv


def runsc_available() -> bool:
    """True when the gVisor ``runsc`` runtime is registered with Docker.

    Docker Desktop (macOS/Windows) doesn't ship runsc, so we fall back to plain
    runc isolation there rather than failing with 'unknown runtime runsc'.
    """
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return False
    return "runsc" in (proc.stdout or "")


def _default_runner(argv: list[str], *, timeout: int) -> RunResult:
    if shutil.which("docker") is None:
        raise SandboxUnavailable(
            "docker not found on PATH — install Docker (and the gVisor runsc runtime) "
            "to use IRIS_EVAL_SANDBOX=docker, or use IRIS_EVAL_SANDBOX=effect."
        )
    # Fixed argv (docker + our own args), no shell.
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return RunResult(stdout=proc.stdout or "", stderr=proc.stderr or "", exit_code=proc.returncode)


def run_eval_in_docker(
    intent: str,
    workload: Sequence[EvalQuery],
    *,
    image: str = EVAL_IMAGE,
    repeats: int = 1,
    max_items: int = 8,
    min_queries: int = 3,
    min_rate: float = 0.8,
    timeout: int = 900,
    gvisor: bool | None = None,
    runner: Callable[..., RunResult] | None = None,
) -> PreflightVerdict:
    """Run the replay-eval for ``intent`` over a host-frozen ``workload`` inside the
    gVisor/Docker eval image and return the parsed verdict. ``runner`` is injectable.

    ``gvisor=None`` auto-detects the runsc runtime (falls back to runc when absent).
    """
    use_gvisor = runsc_available() if gvisor is None else gvisor
    tmp = Path(tempfile.mkdtemp(prefix="iris-eval-wl-"))
    workload_file = tmp / "workload.json"
    workload_file.write_text(workload_to_json(list(workload)), encoding="utf-8")
    try:
        argv = build_eval_docker_argv(
            intent,
            image=image,
            repeats=repeats,
            max_items=max_items,
            min_queries=min_queries,
            min_rate=min_rate,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL_FOR_EVAL", _DEFAULT_OLLAMA),
            gvisor=use_gvisor,
            workload_host_path=str(workload_file),
        )
        run = runner or _default_runner
        result = run(argv, timeout=timeout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    verdict = verdict_from_stdout(result.stdout)
    if verdict is not None:
        return verdict
    raise SandboxUnavailable(
        f"eval container produced no verdict (exit={result.exit_code}). "
        f"Is the '{image}' image built? stderr: {result.stderr.strip()[:300]}"
    )


__all__ = ["build_eval_docker_argv", "run_eval_in_docker", "RunResult", "EVAL_IMAGE"]
