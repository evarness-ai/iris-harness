"""Docker-backed code execution sandbox for the IRIS chat agent.

Public surface:
  - ``DockerSandbox``       — runs shell commands in an isolated container
  - ``SessionWorkspace``    — per-session host directory mounted into containers
  - ``ExecResult``          — outcome of a run (stdout/stderr/exit_code/artifacts)
  - exception types         — Docker missing, image missing, timeout
"""

from __future__ import annotations

from .config import SandboxConfig, SandboxLimits
from .docker_sandbox import (
    DEFAULT_IMAGE,
    DEFAULT_TIMEOUT,
    MAX_TIMEOUT,
    DockerSandbox,
    GVisorSandbox,
)
from .exceptions import (
    DockerUnavailableError,
    SandboxError,
    SandboxImageMissingError,
    SandboxTimeoutError,
)
from .models import ExecResult
from .runtime import (
    SandboxRuntime,
    build_sandbox_runtime,
    instantiate_runtime,
    is_runtime_available,
    resolve_runtime_name,
)
from .workspace import SessionWorkspace

__all__ = [
    "DEFAULT_IMAGE",
    "DEFAULT_TIMEOUT",
    "MAX_TIMEOUT",
    "DockerSandbox",
    "DockerUnavailableError",
    "ExecResult",
    "GVisorSandbox",
    "SandboxConfig",
    "SandboxError",
    "SandboxImageMissingError",
    "SandboxLimits",
    "SandboxRuntime",
    "SandboxTimeoutError",
    "SessionWorkspace",
    "build_sandbox_runtime",
    "instantiate_runtime",
    "is_runtime_available",
    "resolve_runtime_name",
]
