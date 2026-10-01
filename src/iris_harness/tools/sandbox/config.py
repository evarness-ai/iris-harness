"""Sandbox runtime selection config (Phase 6, sub-phase 6c.1).

Loaded from ``config/governance/sandbox.yaml``. Default ``runtime: docker`` →
today's behavior byte-for-byte. ``IRIS_SANDBOX_RUNTIME`` overrides ``runtime``
for quick ops switches. See ``docs/architecture/sandbox-hardening.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

#: Sandbox isolation backend. ``docker`` ships today; ``gvisor`` lands in 6c.2;
#: ``firecracker`` is evaluated/deferred (plan §6).
SandboxRuntimeName = Literal["docker", "gvisor", "firecracker"]
#: What to do when the configured runtime isn't available on this host.
OnUnavailable = Literal["fallback", "disable"]


class SandboxLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory: str = "1g"
    cpus: str = "2"
    pids_limit: int = 256
    timeout_seconds: int = 30


class SandboxConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    runtime: SandboxRuntimeName = "docker"
    on_unavailable: OnUnavailable = "fallback"
    limits: SandboxLimits = Field(default_factory=SandboxLimits)

    def resolved_runtime(self) -> SandboxRuntimeName:
        """``runtime`` with the ``IRIS_SANDBOX_RUNTIME`` env override applied."""
        override = os.environ.get("IRIS_SANDBOX_RUNTIME", "").strip().lower()
        if override in ("docker", "gvisor", "firecracker"):
            return override  # type: ignore[return-value]
        return self.runtime

    @classmethod
    def default(cls) -> SandboxConfig:
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> SandboxConfig:
        """Load policy. Absent file → defaults (docker); malformed → ``ValueError``."""
        if not path.exists():
            return cls.default()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"sandbox config must decode to a mapping: {path}")
        return cls.model_validate(raw)
