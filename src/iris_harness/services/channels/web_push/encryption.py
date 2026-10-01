"""Web Push payload encryption — RFC 8291, ``aes128gcm``.

Hand-rolled rather than taken from ``pywebpush``, and that needs justifying:
the two primitives this needs (``cryptography``, ``pyjwt``) are already
dependencies, the algorithm is eighty lines of fully specified steps, and the
RFC publishes a worked example — so correctness is *provable offline* rather
than trusted. ``tests/.../test_web_push_encryption.py`` reproduces that vector
byte for byte, including the final message body. A library would have added
three packages to a 1 GB VM for code that a published test vector can pin.

The scheme, once: the browser's subscription hands over a P-256 public key
(``p256dh``) and a 16-byte ``auth`` secret. The server makes a throwaway
keypair, does ECDH, mixes in the auth secret, and derives a content key and
nonce from a random salt. The push service forwards the ciphertext without
being able to read it — which is the point, since the payload carries whatever
IRIS is telling the owner.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

#: Record size written into the header. One record is enough for a
#: notification; anything larger than this would have to be split.
RECORD_SIZE = 4096

#: The largest payload a push service is required to accept (RFC 8030 §7.2 is
#: silent, but 4 KB is what every implementation guarantees). Leaves room for
#: the 86-byte header, the padding delimiter and the 16-byte GCM tag.
MAX_PAYLOAD_BYTES = RECORD_SIZE - 103


def b64url_decode(value: str) -> bytes:
    """Decode unpadded base64url, which is what subscriptions carry."""
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _hkdf(*, salt: bytes, ikm: bytes, info: bytes, length: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


@dataclass(frozen=True)
class EncryptedPayload:
    """The request body, and the headers that describe it."""

    body: bytes

    @property
    def headers(self) -> dict[str, str]:
        return {"Content-Encoding": "aes128gcm", "Content-Type": "application/octet-stream"}


def encrypt(
    *,
    plaintext: bytes,
    ua_public_key: bytes,
    auth_secret: bytes,
    salt: bytes | None = None,
    as_private_key: ec.EllipticCurvePrivateKey | None = None,
) -> EncryptedPayload:
    """Encrypt one push message for one subscription.

    ``salt`` and ``as_private_key`` exist so the RFC's worked example can be
    reproduced exactly; in production both are freshly random per message, and
    reusing either across messages would leak the plaintext relationship.
    """
    if len(plaintext) > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"push payload is {len(plaintext)} bytes; the limit is {MAX_PAYLOAD_BYTES}"
        )
    if len(auth_secret) != 16:
        raise ValueError(f"auth secret must be 16 bytes, got {len(auth_secret)}")

    salt = os.urandom(16) if salt is None else salt
    if len(salt) != 16:
        raise ValueError(f"salt must be 16 bytes, got {len(salt)}")

    as_private = as_private_key or ec.generate_private_key(ec.SECP256R1())
    as_public = as_private.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    ua_public = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ua_public_key)

    shared = as_private.exchange(ec.ECDH(), ua_public)
    # The order of the two public keys in key_info is fixed by the RFC: user
    # agent first. Swapping them derives a key the browser cannot reproduce,
    # and the only symptom is a notification that never arrives.
    ikm = _hkdf(
        salt=auth_secret,
        ikm=shared,
        info=b"WebPush: info\x00" + ua_public_key + as_public,
        length=32,
    )
    content_key = _hkdf(salt=salt, ikm=ikm, info=b"Content-Encoding: aes128gcm\x00", length=16)
    nonce = _hkdf(salt=salt, ikm=ikm, info=b"Content-Encoding: nonce\x00", length=12)

    # 0x02 marks the last record. 0x01 would mean "more follow", and a browser
    # given 0x01 on a single record waits for a continuation that never comes.
    ciphertext = AESGCM(content_key).encrypt(nonce, plaintext + b"\x02", None)

    header = salt + RECORD_SIZE.to_bytes(4, "big") + len(as_public).to_bytes(1, "big") + as_public
    return EncryptedPayload(body=header + ciphertext)


__all__ = [
    "MAX_PAYLOAD_BYTES",
    "RECORD_SIZE",
    "EncryptedPayload",
    "b64url_decode",
    "b64url_encode",
    "encrypt",
]
