"""Signature verdict + verification + policy resolution (Phase 6, 6b.1).

``verify_server_signature`` is the real replacement for the always-true
``mcp_allowlist.verify_mcp_signature`` stub (wired into the bridge in 6b.2).
``resolve_signing_action`` maps a verdict + policy to an allow/warn/deny action;
in ``mode: shadow`` everything downgrades to ``warn`` (log-don't-block).

Note: command/args/url/binary tampering all surface as ``invalid`` (the signature
no longer matches the recomputed spec) — a single cryptographic check, rather
than separate per-field statuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from iris_harness.kernel.governance.mcp_signing import keys
from iris_harness.kernel.governance.mcp_signing.config import MCPSigningConfig
from iris_harness.kernel.governance.mcp_signing.spec import ServerSpec
from iris_harness.kernel.governance.mcp_signing.trust import TrustStore

#: verified = good; unsigned = no signature; untrusted = unknown/revoked signer;
#: invalid = signature does not match the (recomputed) spec.
SignatureStatus = Literal["verified", "unsigned", "untrusted", "invalid"]

SigningAction = Literal["allow", "warn", "deny"]


@dataclass(frozen=True)
class SignatureVerdict:
    status: SignatureStatus
    reason: str
    key_id: str | None = None

    @property
    def is_trusted(self) -> bool:
        return self.status == "verified"


def verify_server_signature(
    *,
    spec: ServerSpec,
    signature: str | None,
    signed_by: str | None,
    trust_store: TrustStore,
) -> SignatureVerdict:
    """Verify one server's signature against the trust store."""
    if not signature or not signed_by:
        return SignatureVerdict("unsigned", "no signature on server entry")
    if trust_store.is_revoked(key_id=signed_by, signature=signature):
        return SignatureVerdict("untrusted", f"signer {signed_by!r} is revoked", key_id=signed_by)
    public_key = trust_store.public_key_for(signed_by)
    if public_key is None:
        return SignatureVerdict("untrusted", f"unknown signer {signed_by!r}", key_id=signed_by)
    if keys.verify(public_key, spec.canonical_bytes(), signature):
        return SignatureVerdict("verified", "signature verified", key_id=signed_by)
    return SignatureVerdict(
        "invalid",
        "signature does not match server spec (config or binary changed)",
        key_id=signed_by,
    )


def resolve_signing_action(verdict: SignatureVerdict, config: MCPSigningConfig) -> SigningAction:
    """Map a verdict + policy to an action.

    - signing disabled or verdict trusted → ``allow``
    - ``mode: shadow`` → ``warn`` (log-don't-block, never refuse a launch)
    - ``mode: enforce`` → the configured ``unsigned_policy``
    """
    if not config.enabled or verdict.is_trusted:
        return "allow"
    if config.mode == "shadow":
        return "warn"
    return config.unsigned_policy
