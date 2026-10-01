"""The three routes a browser needs to receive notifications (track 2b PR 9).

    GET    /api/v1/push/key         the VAPID public key to subscribe with
    POST   /api/v1/push/subscribe   remember this browser
    DELETE /api/v1/push/subscribe   forget it

Versioned under ``/api/v1`` with the pairing routes, because these are the
surface a native client would call too (plan decision 7).

Who may call: any authenticated principal, control scope or not. Subscribing
is not a governed write — it asks to *receive* what IRIS already decided to
say, and a read-only phone that cannot be told its calendar broke is a
read-only phone nobody looks at. The write guard is for changing the
harness's state; this changes where its voice carries.

The endpoint URL is a bearer capability: whoever holds it can ask the push
service to deliver to that browser. So it is stored, never logged, and never
returned in a listing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from iris_harness.foundation.auth import Principal
from iris_harness.services.channels.web_push import PushSubscription, PushSubscriptionStore

logger = logging.getLogger(__name__)

KEY_PATH = "/api/v1/push/key"
SUBSCRIBE_PATH = "/api/v1/push/subscribe"

#: A P-256 point in base64url is 87 characters; the auth secret is 22. Bounded
#: so a malformed body is a 422 rather than a row of junk that fails at send.
_P256DH_LEN = 87
_AUTH_LEN = 22


class PushSubscribeRequest(BaseModel):
    """What ``PushSubscription.toJSON()`` gives the browser, plus a label."""

    endpoint: str = Field(..., min_length=8, max_length=2048)
    p256dh: str = Field(..., min_length=_P256DH_LEN, max_length=_P256DH_LEN)
    auth: str = Field(..., min_length=_AUTH_LEN, max_length=_AUTH_LEN)
    label: str = Field(default="", max_length=120)


class PushUnsubscribeRequest(BaseModel):
    endpoint: str = Field(..., min_length=8, max_length=2048)


def _principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if not isinstance(principal, Principal):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")
    return principal


def install_push_routes(app: FastAPI, store: Callable[[], PushSubscriptionStore]) -> None:
    """Register the push routes. ``store`` returns the shared subscription store."""

    @app.get(KEY_PATH)
    def push_key() -> dict[str, Any]:
        """The public key, and whether anything is subscribed yet.

        Generating the keypair on first read rather than at startup keeps a
        harness nobody notifies from writing a secret it never uses.
        """
        from iris_harness.services.channels.web_push import keys

        return {
            "public_key": keys.public_key_b64(),
            "subject": keys.subject(),
            "subscriptions": len(store().list()),
        }

    @app.post(SUBSCRIBE_PATH, status_code=201)
    def push_subscribe(http_request: Request, request: PushSubscribeRequest) -> dict[str, Any]:
        principal = _principal(http_request)
        # Tie it to the paired device, so revoking a lost phone in Devices
        # takes its notifications with it rather than leaving a subscription
        # nobody can see delivering to a phone nobody has.
        saved = store().save(
            PushSubscription(
                endpoint=request.endpoint,
                p256dh=request.p256dh,
                auth=request.auth,
                device_id=principal.device_id,
                label=request.label,
            )
        )
        logger.info(
            "web push: subscribed %s", saved.label or saved.device_id or "an unnamed browser"
        )
        # The endpoint is deliberately absent from the response: it is a
        # capability, and the caller already has it.
        return {"subscribed": True, "label": saved.label, "device_id": saved.device_id}

    @app.delete(SUBSCRIBE_PATH)
    def push_unsubscribe(request: PushUnsubscribeRequest) -> dict[str, Any]:
        """Forget a browser. Idempotent: unsubscribing twice is not an error.

        Unauthenticated-safe by construction — knowing the endpoint is the
        proof, and it is the browser's own.
        """
        return {"removed": store().delete(request.endpoint)}


__all__ = [
    "KEY_PATH",
    "SUBSCRIBE_PATH",
    "PushSubscribeRequest",
    "PushUnsubscribeRequest",
    "install_push_routes",
]
