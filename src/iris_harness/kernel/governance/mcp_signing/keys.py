"""Ed25519 keypair + sign/verify primitives (Phase 6, sub-phase 6b.1).

Thin wrappers over ``cryptography`` (already a dep). Keys and signatures are
base64 over the raw 32-byte key / 64-byte signature so they live cleanly inside
YAML. ``verify`` never raises — a malformed key/signature is a ``False``, so a
tampered entry fails closed rather than crashing the launch path.
"""

from __future__ import annotations

import base64

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519


def generate_keypair() -> tuple[str, str]:
    """Return ``(private_key_b64, public_key_b64)`` for a fresh Ed25519 key."""
    private = ed25519.Ed25519PrivateKey.generate()
    private_b64 = base64.b64encode(private.private_bytes_raw()).decode("ascii")
    public_b64 = base64.b64encode(private.public_key().public_bytes_raw()).decode("ascii")
    return private_b64, public_b64


def public_key_for(private_key_b64: str) -> str:
    """Derive the base64 public key from a base64 private key."""
    private = ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    return base64.b64encode(private.public_key().public_bytes_raw()).decode("ascii")


def sign(private_key_b64: str, payload: bytes) -> str:
    """Sign ``payload`` with a base64 Ed25519 private key; return base64 signature."""
    private = ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    return base64.b64encode(private.sign(payload)).decode("ascii")


def verify(public_key_b64: str, payload: bytes, signature_b64: str) -> bool:
    """Return whether ``signature_b64`` is a valid Ed25519 signature of ``payload``.

    Fails closed (``False``) on any malformed input — bad base64, wrong key/sig
    length, or a non-matching signature.
    """
    try:
        public = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
        public.verify(base64.b64decode(signature_b64), payload)
        return True
    except (InvalidSignature, ValueError):
        return False
