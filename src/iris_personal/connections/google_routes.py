"""The Google reconnect API: the console (and any other client) calls these.

``GET  /api/v1/connections/google/client``    set up? which redirect URI? (no secret)
``PUT  /api/v1/connections/google/client``    store the Web client JSON (write-gated)
``POST /api/v1/connections/google/start``     the consent URL for one reconnect (write-gated)
``GET  /api/v1/connections/google/callback``  Google's redirect back (see below)

The two writes are gated by the API service's write guard like every other control
write: a paired control device, or the service secret with writes enabled. Nothing here
opens them.

The callback is the one route without a bearer check. Google sends the browser back
from its own site: a bearer header is never sent, and the ``SameSite=Strict`` device
cookie is not sent on a navigation that started cross-site. The route authenticates
the request by the ``state`` instead -- 256 random bits issued by ``start`` to an
authenticated, write-allowed caller, valid once, for ten minutes, and bound to the
provider and account that caller asked for. Without a live ``state`` the callback
does nothing but redirect to "link expired". It is declared through the core's
generic public-callback registry, so the core names no Google path.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from iris_harness.sdk.types import Principal

from . import google


class StartRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=40)
    account: str | None = Field(default=None, max_length=320)


class ClientUpload(BaseModel):
    client_json: str = Field(min_length=2, max_length=20_000)


def _actor(request: Request) -> str:
    """Who asked, as the audit ledger records it."""
    principal = getattr(request.state, "principal", None)
    if isinstance(principal, Principal) and principal.kind == "device":
        return f"device:{principal.device_id}"
    return "service"


def build_router() -> APIRouter:
    router = APIRouter(tags=["connections"])

    @router.get(google.CLIENT_PATH)
    def client_status() -> dict[str, Any]:
        """Whether the server can reconnect, and the redirect URI Google must list."""
        return google.setup_status()

    @router.put(google.CLIENT_PATH)
    def upload_client(body: ClientUpload) -> dict[str, Any]:
        """Store the Web application client. The secret is never returned."""
        try:
            client = google.save_client(body.client_json)
        except google.ClientConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        status = google.setup_status()
        target = status["redirect_uri"]
        # Google refuses a redirect the client does not list; say so now rather than
        # after the owner has gone through the consent page.
        status["redirect_uri_listed"] = (target in client.redirect_uris) if target else None
        return status

    @router.post(google.START_PATH)
    def start(body: StartRequest, request: Request) -> dict[str, Any]:
        """The Google consent URL for one reconnect; the browser goes there next."""
        try:
            url = google.start(body.provider, body.account, actor=_actor(request))
        except google.StartRefused as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
        return {"auth_url": url, "expires_in": int(google.STATE_TTL.total_seconds())}

    @router.get(google.CALLBACK_PATH)
    def callback(request: Request) -> RedirectResponse:
        """Google's redirect back: finish, then land on Settings > Connections."""
        params = {
            key: request.query_params[key]
            for key in ("state", "code", "error")
            if key in request.query_params
        }
        outcome = google.complete(params)
        response = RedirectResponse(outcome.redirect_url(), status_code=303)
        # The page this lands on must not be cached, and must not tell anyone which URL
        # (with its one-time code) brought the browser here.
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    return router


__all__ = ["build_router"]
