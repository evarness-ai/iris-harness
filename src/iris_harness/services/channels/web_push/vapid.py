"""VAPID — RFC 8292. How a push service knows the sender is this harness.

Every push request carries a short-lived ES256 JWT signed with the server's
private key, plus the matching public key in the clear. The browser pinned
that same public key when it subscribed, so a push service will only forward
messages signed by whoever owns the subscription's key — which is what stops
anyone who learns an endpoint URL from pushing to the owner's phone.
"""

from __future__ import annotations

import time
from urllib.parse import urlparse

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from iris_harness.services.channels.web_push.encryption import b64url_encode

#: Twelve hours. The RFC's ceiling is 24; half of it leaves room for a clock
#: that has drifted without making the token long-lived if it leaks.
TOKEN_LIFETIME_SECONDS = 12 * 60 * 60


def audience_for(endpoint: str) -> str:
    """The ``aud`` claim: the push service's origin, never the full endpoint.

    Mozilla and Apple both reject a token whose audience carries the path —
    the endpoint contains the subscription id, and it is not the audience.
    """
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"push endpoint is not an absolute URL: {endpoint!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def authorization_header(
    *,
    endpoint: str,
    private_key: ec.EllipticCurvePrivateKey,
    subject: str,
    now: int | None = None,
) -> str:
    """The ``Authorization`` value for one push request.

    ``subject`` is a contact URL (``mailto:`` or ``https:``) a push service
    operator could use to reach whoever is sending. Apple rejects a token
    without one.
    """
    if not subject.startswith(("mailto:", "https://")):
        raise ValueError(f"VAPID subject must be a mailto: or https: URL, got {subject!r}")

    issued = int(time.time()) if now is None else now
    token = jwt.encode(
        {"aud": audience_for(endpoint), "exp": issued + TOKEN_LIFETIME_SECONDS, "sub": subject},
        private_key,
        algorithm="ES256",
    )
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return f"vapid t={token}, k={b64url_encode(public_key)}"


__all__ = ["TOKEN_LIFETIME_SECONDS", "audience_for", "authorization_header"]
