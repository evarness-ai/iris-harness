"""Canonical signable spec for an MCP server (Phase 6, sub-phase 6b.1).

The signed payload is a deterministic projection of the security-relevant config
— command/args/url/declared env-key *names* and (for local-path stdio) a hash of
the binary. Secret ``env`` *values* are never included (they live in the vault).
See ``docs/architecture/mcp-server-signing.md`` §1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerSpec:
    """The security-relevant fields of an MCP server that get signed."""

    name: str
    transport: str
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    env_keys: tuple[str, ...] = ()
    package_sha256: str | None = None

    def canonical_bytes(self) -> bytes:
        """Deterministic UTF-8 serialization signed/verified by Ed25519.

        Stable across runs: fixed key set, sorted keys, sorted env-key names,
        compact separators. Any change to a signed field flips the signature.
        """
        payload = {
            "name": self.name,
            "transport": self.transport,
            "command": self.command,
            "args": list(self.args),
            "url": self.url,
            "env_keys": sorted(self.env_keys),
            "package_sha256": self.package_sha256,
        }
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")


def file_sha256(path: Path) -> str | None:
    """Return ``sha256:<hex>`` for a readable local file, else ``None``.

    Used for local-path ``stdio`` binaries. Commands resolved at launch by a
    package manager (e.g. ``npx <pkg>``) are not files and return ``None`` — see
    the §0 scope boundary on upstream-package pinning.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return "sha256:" + hashlib.sha256(data).hexdigest()
