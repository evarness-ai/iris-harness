"""MCP server signing (Governance Phase 6 supply-chain workstream, 6b.1).

Ed25519 signature verification of an MCP server's canonical launch spec. 6b.1
ships the signing core (spec/keys/trust/config/verify) only — bridge wiring +
CLI + enforcement land in 6b.2-6b.4. See
``docs/architecture/mcp-server-signing.md``.
"""

from __future__ import annotations

from iris_harness.kernel.governance.mcp_signing.config import (
    MCPSigningConfig,
    SigningMode,
    UnsignedPolicy,
)
from iris_harness.kernel.governance.mcp_signing.keys import (
    generate_keypair,
    public_key_for,
    sign,
    verify,
)
from iris_harness.kernel.governance.mcp_signing.spec import ServerSpec, file_sha256
from iris_harness.kernel.governance.mcp_signing.trust import TrustedKey, TrustStore
from iris_harness.kernel.governance.mcp_signing.verifier import (
    SignatureVerdict,
    SigningAction,
    resolve_signing_action,
    verify_server_signature,
)

__all__ = [
    "MCPSigningConfig",
    "ServerSpec",
    "SignatureVerdict",
    "SigningAction",
    "SigningMode",
    "TrustStore",
    "TrustedKey",
    "UnsignedPolicy",
    "file_sha256",
    "generate_keypair",
    "public_key_for",
    "resolve_signing_action",
    "sign",
    "verify",
    "verify_server_signature",
]
