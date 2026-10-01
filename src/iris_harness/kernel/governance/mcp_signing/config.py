"""MCP-signing policy config (Phase 6, sub-phase 6b.1).

Loaded from ``config/governance/mcp-signing.yaml``. Shadow-first, mirroring the
threat-detection rollout: ship ``mode: shadow`` (verify + audit, never block),
flip to ``enforce`` once servers are signed and the audit is clean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

#: shadow = log-don't-block; enforce = honor ``unsigned_policy``.
SigningMode = Literal["shadow", "enforce"]
#: Action under enforce for an unsigned / untrusted / invalid server.
UnsignedPolicy = Literal["allow", "warn", "deny"]


class MCPSigningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    enabled: bool = True
    mode: SigningMode = "shadow"
    unsigned_policy: UnsignedPolicy = "warn"
    trust_store: str = "config/governance/mcp-trust.yaml"

    @classmethod
    def disabled(cls) -> MCPSigningConfig:
        return cls(enabled=False)

    @classmethod
    def from_yaml(cls, path: Path) -> MCPSigningConfig:
        """Load policy. Absent file → ``disabled()`` (no verification); a
        malformed present file raises ``ValueError``."""
        if not path.exists():
            return cls.disabled()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"mcp-signing config must decode to a mapping: {path}")
        return cls.model_validate(raw)
