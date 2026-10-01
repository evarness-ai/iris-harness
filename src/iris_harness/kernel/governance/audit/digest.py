"""Keyed audit digests: what an audit row carries in place of a call's arguments and result.

An audit row never holds argument or result text (``kernel._AUDITED_PAYLOAD_KEYS``), only
fingerprints of it. An unkeyed hash of a short value -- an email address, a phone number --
is reversed by hashing guesses, so the fingerprint is an HMAC-SHA256 under a key only this
install holds: derived by HKDF-SHA256 from the vault master key
(``vault/keys.resolve_master_key``) with its own ``info``, so the audit key is never the
encryption key and knowing one says nothing about the other.

Every row that carries a digest also carries ``digest_alg`` (``hmac-sha256/v1/<key-id>``).
The key id is a short one-way fingerprint of the derived key, so after a master-key
rotation old rows stay unambiguous: they name the key they were taken under.

Resolution is lazy and once per process. Nothing resolves the key when the kernel is built
(``kernel_from_env`` / ``build_default_kernel``): on macOS the keyring is the login
Keychain, and a Keychain dialog at process start hangs the process. The first governed
call resolves it; the result is cached, thread-safe. A failure is not cached -- the next
call retries, so an owner who fixes the environment is not stuck with a dead process. A
keyring that has no backend fails fast (``keyring`` raises); a Keychain that shows a
dialog and waits for an answer is the operating system's wait, not a retry loop here.

With no key, no governed call runs (the owner's decision, 2026-09-30): the runner refuses
a tool call before ``PRE_TOOL_USE`` (``agent/tool_runner.py``), a capability call raises
``CapabilityDenied``, the MCP bridge raises. An unkeyed or empty digest is never written
in a key's place.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from iris_harness.foundation.capability_fields import canonical_json
from iris_harness.foundation.process_state import track_globals
from iris_harness.kernel.governance.hooks.tool_payload import ARGS_DIGEST, DIGEST_ALG, RESULT_DIGEST
from iris_harness.kernel.governance.vault import keys as vault_keys

logger = logging.getLogger(__name__)

#: The HKDF ``info`` of the audit key. Distinct from anything the vault derives, so the
#: audit key and the encryption key are independent. A new version is a new key.
AUDIT_DIGEST_INFO = b"iris/audit-digest/v1"
_KEY_ID_INFO = b"iris/audit-digest/key-id"
_ALG_PREFIX = "hmac-sha256/v1"
# 128 bits of the HMAC: RFC 2104 allows truncation to half the hash length.
_DIGEST_HEX = 32
_KEY_ID_HEX = 12

#: What a caller is told when a governed call is refused for want of a key.
NO_AUDIT_KEY_MESSAGE = (
    "IRIS can't audit this call: no vault master key. Set IRIS_VAULT_MASTER_KEY or "
    "configure an OS keyring; `iris doctor` checks this."
)


class AuditKeyUnavailable(RuntimeError):
    """No audit key: the vault master key could not be resolved (or is not a Fernet key)."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(f"{NO_AUDIT_KEY_MESSAGE} ({detail})" if detail else NO_AUDIT_KEY_MESSAGE)
        self.detail = detail


@dataclass(frozen=True)
class AuditDigester:
    """HMAC-SHA256 under the derived audit key. Holds the derived key, never the master key."""

    _key: bytes
    key_id: str

    @property
    def alg(self) -> str:
        """``digest_alg`` for the rows this digester fingerprints."""
        return f"{_ALG_PREFIX}/{self.key_id}"

    def digest(self, value: Any) -> str:
        """The keyed fingerprint of ``value`` (its canonical JSON)."""
        text = canonical_json(value).encode("utf-8")
        return hmac.new(self._key, text, hashlib.sha256).hexdigest()[:_DIGEST_HEX]

    def args_fields(self, args: Any) -> dict[str, str]:
        """The ``PRE_TOOL_USE`` payload's fingerprint of ``args`` and the algorithm."""
        return {ARGS_DIGEST: self.digest(args), DIGEST_ALG: self.alg}

    def result_fields(self, result: Any) -> dict[str, str]:
        """The ``POST_TOOL_USE`` payload's fingerprint of ``result`` and the algorithm."""
        return {RESULT_DIGEST: self.digest(result), DIGEST_ALG: self.alg}

    def __repr__(self) -> str:  # never print the key
        return f"AuditDigester(alg={self.alg!r})"


def derive_audit_key(master_key: bytes, *, info: bytes = AUDIT_DIGEST_INFO) -> bytes:
    """The audit key: HKDF-SHA256 over the raw bytes of the Fernet master key.

    Raises :class:`AuditKeyUnavailable` when ``master_key`` is not a Fernet key (32 bytes,
    url-safe base64): an unusable key is no key.
    """
    try:
        raw = base64.urlsafe_b64decode(master_key)
    except (binascii.Error, ValueError) as exc:
        raise AuditKeyUnavailable("the master key is not a Fernet key") from exc
    if len(raw) != 32:
        raise AuditKeyUnavailable("the master key is not a Fernet key")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(raw)


def digester_for(master_key: bytes, *, info: bytes = AUDIT_DIGEST_INFO) -> AuditDigester:
    """A digester for ``master_key`` (no caching; :func:`audit_digester` is the cached one)."""
    key = derive_audit_key(master_key, info=info)
    key_id = hmac.new(key, _KEY_ID_INFO, hashlib.sha256).hexdigest()[:_KEY_ID_HEX]
    return AuditDigester(key, key_id)


_lock = threading.Lock()
_digester: AuditDigester | None = None
_last_failure: str | None = None


def audit_digester() -> AuditDigester:
    """This process's digester, resolved on first use and cached.

    Raises :class:`AuditKeyUnavailable` when no master key can be resolved; the failure is
    not cached, so the next call tries again.
    """
    global _digester, _last_failure
    cached = _digester
    if cached is not None:
        return cached
    with _lock:
        if _digester is not None:
            return _digester
        try:
            master = vault_keys.resolve_master_key()
        except vault_keys.VaultMasterKeyUnavailableError as exc:
            _last_failure = "no master key in IRIS_VAULT_MASTER_KEY or the OS keyring"
            logger.warning("audit digests: %s; governed calls are refused", _last_failure)
            raise AuditKeyUnavailable(_last_failure) from exc
        try:
            _digester = digester_for(master)
        except AuditKeyUnavailable as exc:
            _last_failure = exc.detail
            logger.warning("audit digests: %s; governed calls are refused", _last_failure)
            raise
        _last_failure = None
        return _digester


AuditKeyState = Literal["ready", "unavailable", "unresolved"]


def audit_key_status() -> tuple[AuditKeyState, str]:
    """Where the audit key stands, WITHOUT resolving it (a health check must not open the
    Keychain): ``ready`` (the key id), ``unavailable`` (why the last attempt failed), or
    ``unresolved`` (no governed call has asked for it yet)."""
    with _lock:
        if _digester is not None:
            return "ready", _digester.alg
        if _last_failure is not None:
            return "unavailable", _last_failure
        return "unresolved", ""


def _reset_for_tests() -> None:
    """Forget the cached digester and the last failure (the suite's per-test fixture)."""
    global _digester, _last_failure
    with _lock:
        _digester = None
        _last_failure = None


__all__ = [
    "AUDIT_DIGEST_INFO",
    "NO_AUDIT_KEY_MESSAGE",
    "AuditDigester",
    "AuditKeyState",
    "AuditKeyUnavailable",
    "audit_digester",
    "audit_key_status",
    "derive_audit_key",
    "digester_for",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_digester", "_last_failure")
