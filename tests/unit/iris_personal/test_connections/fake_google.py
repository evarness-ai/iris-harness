"""A fake Google for the reconnect tests: the consent page, the token endpoint and the
three "who am I" calls, over ``httpx.MockTransport``. No network.

It checks what real Google checks and a broken client would get wrong: the code is used
once, the PKCE verifier hashes to the challenge from the consent URL, the redirect URI
and the client secret match. Tokens are made-up strings that look like Google's, so a
test can search every response and log line for them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlsplit

import httpx

CLIENT_ID = "fake-web-client.apps.googleusercontent.example"
CLIENT_SECRET = "FAKE-CLIENT-SECRET-do-not-log"  # a fake
WEB_CLIENT_JSON = json.dumps(
    {
        "web": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["https://iris-vm.example.ts.net/api/v1/connections/google/callback"],
        }
    }
)
DESKTOP_CLIENT_JSON = json.dumps(
    {"installed": {"client_id": "desktop.example", "client_secret": "desktop-secret"}}
)


@dataclass
class FakeGoogle:
    #: Scopes Google reports as granted; None grants exactly what was asked.
    grant_scopes: str | None = None
    give_refresh_token: bool = True
    codes: dict[str, dict[str, str]] = field(default_factory=dict)
    issued: list[str] = field(default_factory=list)  # every token and code it handed out
    access_owner: dict[str, str] = field(default_factory=dict)
    token_calls: int = 0

    # ── the consent page ──
    def consent(self, auth_url: str, *, as_email: str) -> dict[str, str]:
        """The owner picks ``as_email`` and taps Allow; the callback's query params."""
        query = {k: v[0] for k, v in parse_qs(urlsplit(auth_url).query).items()}
        code = f"4/FAKE-CODE-{secrets.token_hex(8)}"
        self.issued.append(code)
        self.codes[code] = {
            "email": as_email,
            "challenge": query["code_challenge"],
            "scope": query["scope"],
            "redirect_uri": query["redirect_uri"],
            "client_id": query["client_id"],
        }
        return {"state": query["state"], "code": code}

    @staticmethod
    def cancel(auth_url: str) -> dict[str, str]:
        query = {k: v[0] for k, v in parse_qs(urlsplit(auth_url).query).items()}
        return {"state": query["state"], "error": "access_denied"}

    # ── the endpoints ──
    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return self._token(request)
        owner = self.access_owner.get(request.headers.get("authorization", "")[7:])
        if owner is None:
            return httpx.Response(401, json={"error": "invalid_token"})
        path = request.url.path
        if path == "/gmail/v1/users/me/profile":
            return httpx.Response(200, json={"emailAddress": owner})
        if path == "/calendar/v3/users/me/calendarList/primary":
            return httpx.Response(200, json={"id": owner})
        if path == "/drive/v3/about":
            return httpx.Response(200, json={"user": {"emailAddress": owner}})
        return httpx.Response(404)

    def _token(self, request: httpx.Request) -> httpx.Response:
        self.token_calls += 1
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        grant = self.codes.pop(form.get("code", ""), None)  # single use, as Google's
        if grant is None:
            return httpx.Response(400, json={"error": "invalid_grant"})
        digest = hashlib.sha256(form.get("code_verifier", "").encode()).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if (
            challenge != grant["challenge"]
            or form.get("redirect_uri") != grant["redirect_uri"]
            or form.get("client_id") != grant["client_id"]
            or form.get("client_secret") != CLIENT_SECRET
            or form.get("grant_type") != "authorization_code"
        ):
            return httpx.Response(400, json={"error": "invalid_grant"})
        access = f"ya29.FAKE-ACCESS-{secrets.token_hex(12)}"
        refresh = f"1//FAKE-REFRESH-{secrets.token_hex(12)}"
        self.issued += [access, refresh]
        self.access_owner[access] = grant["email"]
        body = {
            "access_token": access,
            "expires_in": 3599,
            "token_type": "Bearer",
            "scope": self.grant_scopes if self.grant_scopes is not None else grant["scope"],
        }
        if self.give_refresh_token:
            body["refresh_token"] = refresh
        return httpx.Response(200, json=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))
