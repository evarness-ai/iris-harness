"""Trust store of MCP-signing public keys (Phase 6, sub-phase 6b.1).

Operator-managed ``config/governance/mcp-trust.yaml`` mapping ``key_id`` to a
public key, plus revocation lists. The agent process has no write access (the
root of trust lives outside the protected component — design §14 / §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrustedKey:
    key_id: str
    public_key: str
    comment: str = ""


@dataclass(frozen=True)
class TrustStore:
    keys: dict[str, TrustedKey] = field(default_factory=dict)
    revoked_key_ids: frozenset[str] = frozenset()
    revoked_signatures: frozenset[str] = frozenset()

    def public_key_for(self, key_id: str) -> str | None:
        """Public key for ``key_id``, or ``None`` if unknown or revoked."""
        if key_id in self.revoked_key_ids:
            return None
        entry = self.keys.get(key_id)
        return entry.public_key if entry else None

    def is_revoked(self, *, key_id: str, signature: str) -> bool:
        return key_id in self.revoked_key_ids or signature in self.revoked_signatures

    @classmethod
    def empty(cls) -> TrustStore:
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> TrustStore:
        """Load the trust store. An absent file is an empty store (no trusted
        keys → every signed server is ``untrusted``); a malformed present file
        raises ``ValueError``."""
        if not path.exists():
            return cls.empty()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"mcp-trust file must decode to a mapping: {path}")
        keys: dict[str, TrustedKey] = {}
        for entry in raw.get("keys", []) or []:
            if not isinstance(entry, dict):
                raise ValueError(f"mcp-trust key entry must be a mapping: {entry!r}")
            key_id = str(entry["key_id"])
            if key_id in keys:
                raise ValueError(f"duplicate trust-store key_id: {key_id}")
            keys[key_id] = TrustedKey(
                key_id=key_id,
                public_key=str(entry["public_key"]),
                comment=str(entry.get("comment", "")),
            )
        return cls(
            keys=keys,
            revoked_key_ids=frozenset(str(k) for k in raw.get("revoked_key_ids", []) or []),
            revoked_signatures=frozenset(str(s) for s in raw.get("revoked_signatures", []) or []),
        )
