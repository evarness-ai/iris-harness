"""The server's VAPID keypair: generated once, kept, never rotated casually.

Rotating it silently breaks every existing subscription — a browser pins the
public key it subscribed with, and a push signed by a different key is
refused. So the key is generated on first use and persisted, and the only way
to replace it is to delete the file, which also means re-subscribing every
device. The docstring is the warning; there is deliberately no rotate command.

``IRIS_WEB_PUSH_VAPID_PRIVATE_KEY`` overrides the file for a deployment that
would rather inject the secret than have it written to disk.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from iris_harness.foundation.paths import governance_data_dir
from iris_harness.services.channels.web_push.encryption import b64url_encode

logger = logging.getLogger(__name__)

PRIVATE_KEY_ENV = "IRIS_WEB_PUSH_VAPID_PRIVATE_KEY"
SUBJECT_ENV = "IRIS_WEB_PUSH_SUBJECT"

#: A push service operator's contact of last resort. Not a real address here —
#: the harness is single-owner and self-hosted — but the field is required.
DEFAULT_SUBJECT = "https://github.com/evarness-ai/iris-harness"

KEY_FILENAME = "vapid-private-key.pem"


def key_path() -> Path:
    return governance_data_dir() / KEY_FILENAME


def subject() -> str:
    return (os.environ.get(SUBJECT_ENV, "") or DEFAULT_SUBJECT).strip() or DEFAULT_SUBJECT


def _from_env() -> ec.EllipticCurvePrivateKey | None:
    raw = os.environ.get(PRIVATE_KEY_ENV, "").strip()
    if not raw:
        return None
    key = serialization.load_pem_private_key(raw.encode("utf-8"), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError(f"{PRIVATE_KEY_ENV} is not an EC private key")
    return key


def load_or_create(path: Path | None = None) -> ec.EllipticCurvePrivateKey:
    """The harness's VAPID private key, making one the first time."""
    from_env = _from_env()
    if from_env is not None:
        return from_env

    target = path or key_path()
    if target.is_file():
        key = serialization.load_pem_private_key(target.read_bytes(), password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise ValueError(f"{target} is not an EC private key")
        return key

    key = ec.generate_private_key(ec.SECP256R1())
    target.parent.mkdir(parents=True, exist_ok=True)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    # Written 0600 before any content reaches it: this key is what stops a
    # stranger who learns an endpoint from pushing to the owner's phone. Same
    # defence-in-depth the audit and cost ledgers use.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "wb") as handle:
        handle.write(pem)
    logger.info("web push: generated a VAPID keypair at %s", target)
    return key


def public_key_b64(path: Path | None = None) -> str:
    """The base64url public key the browser subscribes with."""
    public = load_or_create(path).public_key()
    return b64url_encode(
        public.public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )
    )


__all__ = [
    "DEFAULT_SUBJECT",
    "KEY_FILENAME",
    "PRIVATE_KEY_ENV",
    "SUBJECT_ENV",
    "key_path",
    "load_or_create",
    "public_key_b64",
    "subject",
]
