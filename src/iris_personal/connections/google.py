"""Reconnect a Google account from the web console: the server runs the OAuth flow.

Until this, a revoked Gmail / Calendar / Drive token meant ``iris auth <x> login`` on
the Mac (the installed-app flow needs a browser on the machine that stores the token)
and ``keyring_bridge.py`` to copy the result to the VM. Here the server itself is the
OAuth client, with a Google "Web application" client whose redirect URI is the
console's public address, so the owner reconnects from the phone the alert arrived on.
The Mac login and the bridge are unchanged; this is a second way in, not a replacement.

The flow, and why each piece is there:

* ``start`` (write-gated: a paired control device, or the service secret with writes
  on) issues a random ``state`` and a PKCE verifier, bound to the provider and account
  asked for, single use, for ten minutes. The ``state`` is the only credential the
  callback has: Google sends the browser back without our cookie or bearer header.
* The consent URL asks for exactly the scopes the Mac login asks for (read from each
  plugin's OAuth module, never restated), offline access and a forced consent prompt
  (so Google returns a refresh token), and no ``include_granted_scopes`` (so one Web
  client shared by Gmail, Calendar and Drive never hands one of them another's scopes).
* ``complete`` (the callback) takes the ``state`` once, trades the code for tokens
  with the PKCE verifier, and asks the provider's own API which account the token
  belongs to -- the same lookup the Mac login does, which needs no scope beyond the
  ones above (an ID token would need ``openid email``). A different account than the
  one asked for saves nothing. The lookup is also the live check the result reports.
* Tokens go only to the keyring, in the ``Credentials.to_json()`` shape every loader
  already reads. No response, redirect, log line or exception message carries one.
* Every start and every outcome is a row in the governance audit ledger: who, which
  provider and account, what happened.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx

from iris_harness.sdk import vault as _credentials
from iris_harness.sdk.config import PUBLIC_URL_ENV, public_base_url
from iris_harness.sdk.health import CheckKind, HealthCheck, HealthState, Reconnect
from iris_harness.sdk.process_state import track_globals
from iris_personal.email.accounts import EmailAccountStore

if TYPE_CHECKING:
    from iris_harness.sdk import PluginAPI

logger = logging.getLogger(__name__)

GROUP = "google"
GROUP_LABEL = "Google"
BASE_PATH = "/api/v1/connections/google"
START_PATH = f"{BASE_PATH}/start"
CALLBACK_PATH = f"{BASE_PATH}/callback"
CLIENT_PATH = f"{BASE_PATH}/client"
#: Where the console shows the connections, and where the callback lands the browser.
SETTINGS_PATH = "/settings"
SETTINGS_FIX_URL = f"{SETTINGS_PATH}#connections"

STATE_TTL = timedelta(minutes=10)
#: A start that is never finished stays until it expires; this bounds how many a
#: control device can leave open at once (the oldest goes first).
MAX_PENDING = 32

#: The Web client's id and secret live in the keyring, beside the tokens they mint.
CLIENT_KEYRING_PROVIDER = "google-web-client"
CLIENT_KEYRING_ACCOUNT = "web"

GOOGLE_AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"  # noqa: S105 - a URL, not a secret
#: Tests and the local demo only: send every Google call to a fake on this machine. A
#: loopback address is required, so the setting cannot send a code or a token off-box.
TEST_BASE_ENV = "IRIS_GOOGLE_OAUTH_TEST_BASE"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_EMAIL = re.compile(r"^[^@\s/?#]+@[^@\s/?#]+\.[^@\s/?#]+$")
_ERROR_CODE = re.compile(r"[^a-z0-9_]")
_HTTP_TIMEOUT = 15.0


# ─── Providers ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GoogleProvider:
    """One Google API a plugin reads, as this flow needs to know it.

    ``key`` is both the keyring provider and the accounts-table provider the plugin's
    OAuth module uses (``gmail``, ``gcalendar``, ``gdrive``), so a token saved here is
    the one that module loads. ``scopes`` are that module's constants. ``identity_url``
    and ``identity_field`` are the provider's own "who am I" call -- the one its Mac
    login makes through the client library -- and ``order`` sorts the rows.
    """

    key: str
    label: str
    scopes: tuple[str, ...]
    identity_url: str
    identity_field: tuple[str, ...]
    order: int = 0


_providers_lock = Lock()
_providers: dict[str, GoogleProvider] = {}


def register_provider(api: PluginAPI, provider: GoogleProvider) -> None:
    """Make ``provider`` reconnectable, and mount the shared routes (once, keyed).

    Called from each plugin's ``setup`` with that plugin's ``api``: a profile without
    the gmail plugin offers no Gmail reconnect, and a core-only install mounts nothing
    at all. The routes are keyed, so the second and third plugin replace, not stack.
    """
    from .google_routes import build_router

    with _providers_lock:
        _providers[provider.key] = provider
    api.register_api_router("google_connections", build_router)
    api.register_public_callback(CALLBACK_PATH)


def providers() -> list[GoogleProvider]:
    with _providers_lock:
        return sorted(_providers.values(), key=lambda p: (p.order, p.key))


def get_provider(key: str) -> GoogleProvider | None:
    with _providers_lock:
        return _providers.get(key)


def clear_providers() -> None:
    """Drop every provider (tests)."""
    with _providers_lock:
        _providers.clear()


def with_reconnect(checks: Iterable[HealthCheck], provider_key: str) -> list[HealthCheck]:
    """Stamp a plugin's credential rows with how the console reconnects them.

    The rows stay the plugin's own (:func:`google_credential_checks`); this adds the
    ``reconnect`` the console renders a button from, and, on a row that needs the
    owner, the page that fixes it -- which the health alerts link to.
    """
    provider = get_provider(provider_key)
    out: list[HealthCheck] = []
    for check in checks:
        if provider is None or check.kind is not CheckKind.CREDENTIAL:
            out.append(check)
            continue
        needs_owner = check.state in (HealthState.RED, HealthState.YELLOW)
        out.append(
            replace(
                check,
                fix_url=SETTINGS_FIX_URL if needs_owner else check.fix_url,
                reconnect=Reconnect(
                    route=START_PATH,
                    setup_route=CLIENT_PATH,
                    group=GROUP,
                    group_label=GROUP_LABEL,
                    provider=provider.key,
                    label=provider.label,
                    account=check.subject,
                ),
            )
        )
    return out


# ─── Credential health ────────────────────────────────────────────────
#
# The System Health rows of a Google OAuth credential (ADR-0069 slice 2), local-only:
# expiry is read from the stored token blob, never validated with a paid call. A token
# with a refresh token self-heals on next use, so an expired *access* token stays
# green. A *revoked* token (``invalid_grant``) needs a live refresh: the opt-in
# network probe (slice 5). These lived in the core (``health.credentials``) until the
# email slice moved to the SDK (core/SDK boundary plan, email slice step 4): the rows
# are Google's and the account shape is the email account store's, so the core keeps
# only the seam each plugin registers its check on (``api.register_credential_check``).

# A configured account whose (non-refreshable) token expires within this window is
# yellow -- a heads-up to re-auth before it lapses.
_EXPIRY_YELLOW = timedelta(days=7)

#: One OAuth module's status, as ``(label, cli-verb, status()[, load_credentials])``.
#: The CLI verb forms the remediation ``iris auth <verb> login [--user <addr>]``; the
#: optional ``load_credentials`` is the live refresh probe, used only with net_probe.
StatusSource = tuple[Any, ...]

# Opt-in revocation probe (ADR-0069 slice 5). ``load_credentials()`` raises
# ``CredentialRevokedError`` when a refresh fails (revoked / refresh-token expired; it
# returns None only when nothing is stored, which the local check already reports) and
# only hits the network when the access token is already expired -- a natural rate
# limiter. Each per-account verdict is also cached for an hour so an enabled probe
# cannot hammer the provider on every 60s health_tick.
#
# A cached "revoked" verdict is honoured WITHOUT net_probe too (no egress: a dict
# read), so a revocation found by the health watch's one-off diagnosis stays red on the
# local ticks that follow (ADR-0116). It is tied to the token it was taken on: a
# re-login writes a new token (new expiry), and the stale verdict stops applying.
_PROBE_INTERVAL = timedelta(hours=1)
_probe_cache: dict[str, tuple[datetime, bool, str]] = {}


def reset_probe_cache() -> None:
    """Clear the per-account probe cache (used by tests)."""
    _probe_cache.clear()


def _probe_credential(
    key: str,
    loader: Callable[[str], object | None],
    address: str,
    *,
    now: datetime,
    token: str = "",
) -> bool | None:
    """Live-probe one credential. True = refresh ok, False = definitively failed
    (revoked), None = could not tell (transient/network) -- the caller must not
    override the local verdict on None. Definitive verdicts are cached an hour."""
    cached = _cached_verdict(key, now=now, token=token)
    if cached is not None:
        return cached
    try:
        ok = loader(address) is not None
    except _credentials.CredentialRevokedError:
        ok = False  # the provider refused the refresh: definitively revoked
    except Exception:  # noqa: BLE001 -- a network blip is "unknown", not "revoked"
        return None
    _probe_cache[key] = (now, ok, token)
    return ok


def _cached_verdict(key: str, *, now: datetime, token: str) -> bool | None:
    """A verdict from the last hour on this same token, else None. No egress."""
    cached = _probe_cache.get(key)
    if cached is None or now - cached[0] >= _PROBE_INTERVAL or cached[2] != token:
        return None
    return cached[1]


def _ensure_aware(dt: datetime) -> datetime:
    """Treat a naive expiry as UTC so comparisons never raise."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _login_cmd(cli: str, address: str | None) -> str:
    return f"iris auth {cli} login" + (f" --user {address}" if address else "")


def _classify_account(
    account: Any,
    *,
    cli: str,
    now: datetime,
    net_probe: bool = False,
    loader: Callable[[str], object | None] | None = None,
) -> tuple[HealthState, str, str | None]:
    """Map one provider account status to (state, detail, remediation)."""
    address = getattr(getattr(account, "email_account", None), "address", "account")

    if not getattr(account, "has_keychain_token", False):
        return HealthState.RED, f"{address}: no stored token", _login_cmd(cli, address)

    # Opt-in live probe: a token that looks fine locally may be revoked. Only a
    # *definitive* failure (verdict False) overrides to red; unknown leaves the local
    # verdict intact. Without net_probe, a recent verdict on this same token still
    # counts (see _probe_cache).
    token = str(getattr(account, "token_expiry", None))
    if loader is not None:
        key = f"{cli}:{address}"
        verdict = (
            _probe_credential(key, loader, address, now=now, token=token)
            if net_probe
            else _cached_verdict(key, now=now, token=token)
        )
        if verdict is False:
            return (
                HealthState.RED,
                f"{address}: token revoked — re-authenticate",
                _login_cmd(cli, address),
            )

    expiry = getattr(account, "token_expiry", None)

    if getattr(account, "refresh_token_present", False):
        # Self-heals on next use; revocation is caught by the slice-5 net probe.
        if expiry and _ensure_aware(expiry) < now:
            return HealthState.GREEN, f"{address}: connected (auto-refresh)", None
        return HealthState.GREEN, f"{address}: connected", None

    # No refresh token: once it lapses, manual re-auth is required.
    if expiry is None:
        return (
            HealthState.YELLOW,
            f"{address}: no refresh token, expiry unknown",
            _login_cmd(cli, address),
        )
    expiry = _ensure_aware(expiry)
    if expiry < now:
        return HealthState.RED, f"{address}: expired, no refresh token", _login_cmd(cli, address)
    if expiry - now < _EXPIRY_YELLOW:
        return (
            HealthState.YELLOW,
            f"{address}: expires {expiry.date()}, no refresh token",
            _login_cmd(cli, address),
        )
    return HealthState.GREEN, f"{address}: connected (expires {expiry.date()})", None


def google_credential_checks(
    sources: Iterable[StatusSource],
    *,
    now: datetime | None = None,
    net_probe: bool = False,
) -> list[HealthCheck]:
    """One check per configured account of each source; grey when a source has none.

    With ``net_probe`` on, each token-bearing account is live-probed (via the source's
    ``load_credentials``, its 4th element) to catch a revoked token that still looks
    valid locally."""
    when = now or datetime.now(UTC)
    checks: list[HealthCheck] = []
    for entry in sources:
        label, cli, status_fn = entry[0], entry[1], entry[2]
        loader = entry[3] if len(entry) > 3 else None
        try:
            accounts = status_fn()
        except Exception as exc:  # noqa: BLE001 -- introspection must never break health
            checks.append(
                HealthCheck(
                    label, CheckKind.CREDENTIAL, HealthState.YELLOW, f"status unavailable: {exc}"
                )
            )
            continue
        if not accounts:
            # Not configured: grey/informational, never alerts; offers a connect command.
            checks.append(
                HealthCheck(
                    label,
                    CheckKind.CREDENTIAL,
                    HealthState.GREY,
                    "not connected",
                    action=_login_cmd(cli, None),
                )
            )
            continue
        for account in accounts:
            state, detail, action = _classify_account(
                account, cli=cli, now=when, net_probe=net_probe, loader=loader
            )
            address = getattr(getattr(account, "email_account", None), "address", None)
            checks.append(
                HealthCheck(
                    label, CheckKind.CREDENTIAL, state, detail, action=action, subject=address
                )
            )
    return checks


# ─── Endpoints ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Endpoints:
    """Where Google's authorize page, token endpoint and APIs are."""

    auth_uri: str = GOOGLE_AUTH_URI
    token_uri: str = GOOGLE_TOKEN_URI
    api_origin: str | None = None  # None: each provider's real API host

    def api_url(self, url: str) -> str:
        if self.api_origin is None:
            return url
        origin = urlsplit(self.api_origin)
        parts = urlsplit(url)
        return urlunsplit((origin.scheme, origin.netloc, parts.path, parts.query, ""))


def endpoints() -> Endpoints:
    """Google's endpoints, or a loopback fake's when ``IRIS_GOOGLE_OAUTH_TEST_BASE`` says so."""
    raw = (os.environ.get(TEST_BASE_ENV) or "").strip().rstrip("/")
    if not raw:
        return Endpoints()
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or parts.hostname not in _LOOPBACK_HOSTS:
        logger.warning("%s must be a loopback http(s) URL; using Google", TEST_BASE_ENV)
        return Endpoints()
    return Endpoints(auth_uri=f"{raw}/o/oauth2/v2/auth", token_uri=f"{raw}/token", api_origin=raw)


def redirect_uri() -> str | None:
    """The callback on the console's public address, or None without ``IRIS_PUBLIC_URL``."""
    base = public_base_url()
    return f"{base}{CALLBACK_PATH}" if base else None


# ─── The Web client ───────────────────────────────────────────────────


class ClientConfigError(ValueError):
    """The uploaded client JSON is not a usable Google Web application client."""


@dataclass(frozen=True)
class WebClient:
    client_id: str
    client_secret: str = field(repr=False)
    redirect_uris: tuple[str, ...] = ()


def parse_client_json(raw: str) -> WebClient:
    """The Web client in a Google Cloud Console client JSON download.

    Refuses a Desktop ("installed") client by name: it is the file the Mac login
    already uses, and Google will not redirect a Desktop client to the server.
    """
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ClientConfigError("That file is not JSON.") from exc
    if not isinstance(data, dict):
        raise ClientConfigError("That file is not a Google client JSON.")
    if "web" not in data:
        if "installed" in data:
            raise ClientConfigError(
                "That is a Desktop client (the one the Mac login uses). Create a "
                "'Web application' client and upload its JSON."
            )
        raise ClientConfigError("That file has no 'web' client in it.")
    web = data["web"]
    if not isinstance(web, dict):
        raise ClientConfigError("The 'web' client in that file is malformed.")
    client_id = web.get("client_id")
    client_secret = web.get("client_secret")
    if not isinstance(client_id, str) or not client_id.strip():
        raise ClientConfigError("The 'web' client has no client_id.")
    if not isinstance(client_secret, str) or not client_secret.strip():
        raise ClientConfigError("The 'web' client has no client_secret.")
    uris = web.get("redirect_uris")
    redirect_uris = tuple(u for u in uris if isinstance(u, str)) if isinstance(uris, list) else ()
    return WebClient(client_id.strip(), client_secret.strip(), redirect_uris)


def save_client(raw: str) -> WebClient:
    """Validate and store the Web client. Only its id and secret are kept."""
    client = parse_client_json(raw)
    blob = json.dumps({"client_id": client.client_id, "client_secret": client.client_secret})
    _credentials.save_token(CLIENT_KEYRING_PROVIDER, CLIENT_KEYRING_ACCOUNT, blob)
    return client


def load_client() -> WebClient | None:
    raw = _credentials.load_token(CLIENT_KEYRING_PROVIDER, CLIENT_KEYRING_ACCOUNT)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return WebClient(str(data["client_id"]), str(data["client_secret"]))
    except (ValueError, KeyError, TypeError):
        logger.warning("google connect: the stored Web client is unreadable; upload it again")
        return None


def setup_status() -> dict[str, Any]:
    """What the Connections tab needs to know before it offers a button. No secret."""
    target = redirect_uri()
    listed = providers()
    return {
        "configured": load_client() is not None,
        "public_url_set": target is not None,
        "redirect_uri": target,
        "group": GROUP,
        "group_label": GROUP_LABEL,
        "providers": [{"provider": p.key, "label": p.label} for p in listed],
        "add_provider": listed[0].key if listed else None,
    }


# ─── Pending starts ───────────────────────────────────────────────────


@dataclass(frozen=True)
class _Pending:
    provider: str
    account: str | None
    verifier: str = field(repr=False)
    created: datetime
    actor: str
    attempt: str


class PendingStarts:
    """The ``state`` -> request map. Process memory on purpose: a PKCE verifier is a
    secret for ten minutes, and it never needs to outlive the process that issued it."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._by_state: dict[str, _Pending] = {}

    def issue(self, pending: _Pending, *, now: datetime) -> str:
        state = secrets.token_urlsafe(32)
        with self._lock:
            self._drop_expired(now)
            while len(self._by_state) >= MAX_PENDING:
                oldest = min(self._by_state, key=lambda s: self._by_state[s].created)
                del self._by_state[oldest]
            self._by_state[state] = pending
        return state

    def take(self, state: str | None, *, now: datetime) -> _Pending | None:
        """The request ``state`` was issued for, once; None if unknown, used or expired."""
        if not state:
            return None
        with self._lock:
            pending = self._by_state.pop(state, None)
        if pending is None or now - pending.created > STATE_TTL:
            return None
        return pending

    def _drop_expired(self, now: datetime) -> None:
        for key in [k for k, p in self._by_state.items() if now - p.created > STATE_TTL]:
            del self._by_state[key]

    def clear(self) -> None:
        with self._lock:
            self._by_state.clear()


pending_starts = PendingStarts()


# ─── start ────────────────────────────────────────────────────────────


class StartRefused(Exception):
    """A start the server will not issue; ``status`` is the HTTP answer."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _now() -> datetime:
    return datetime.now(UTC)


def _normalize_account(account: str | None) -> str | None:
    if account is None:
        return None
    addr = account.strip().lower()
    if not _EMAIL.match(addr):
        raise StartRefused(422, "account must be an email address")
    return addr


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def start(
    provider_key: str,
    account: str | None,
    *,
    actor: str,
    now: datetime | None = None,
) -> str:
    """The Google consent URL for reconnecting ``account`` (``None``: add one)."""
    when = now or _now()
    provider = get_provider(provider_key)
    if provider is None:
        raise StartRefused(404, f"no reconnectable Google provider {provider_key!r}")
    addr = _normalize_account(account)
    target = redirect_uri()
    if target is None:
        raise StartRefused(
            409,
            f"{PUBLIC_URL_ENV} is not set, so there is no address Google could send you "
            "back to. Set it (set_server_env.sh --public-url) and restart.",
        )
    client = load_client()
    if client is None:
        raise StartRefused(
            409,
            "This server has no Google Web application client yet. Add its JSON in "
            "Settings > Connections.",
        )
    verifier, challenge = _pkce_pair()
    attempt = secrets.token_hex(6)
    state = pending_starts.issue(
        _Pending(provider.key, addr, verifier, when, actor, attempt), now=when
    )
    params = {
        "client_id": client.client_id,
        "redirect_uri": target,
        "response_type": "code",
        "scope": " ".join(provider.scopes),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }
    if addr:
        params["login_hint"] = addr
    _audit("start", actor=actor, provider=provider.key, account=addr, attempt=attempt)
    return f"{endpoints().auth_uri}?{urlencode(params)}"


# ─── complete (the callback) ──────────────────────────────────────────


@dataclass(frozen=True)
class Outcome:
    """What the callback did, as the console's result screen shows it.

    ``result`` is one of ``connected``, ``cancelled``, ``wrong_account``, ``expired``,
    ``failed``. ``reason`` is a short machine word for ``failed``; ``approved_as`` is
    the account Google approved when it was not the one asked for.
    """

    result: str
    provider: str | None = None
    account: str | None = None
    approved_as: str | None = None
    reason: str | None = None

    def redirect_url(self) -> str:
        """The console page the browser lands on. Carries no token and no code."""
        query = {"connect": self.result}
        for key, value in (
            ("provider", self.provider),
            ("account", self.account),
            ("approved", self.approved_as),
            ("reason", self.reason),
        ):
            if value:
                query[key] = value
        return f"{SETTINGS_PATH}?{urlencode(query)}#connections"


HttpClientFactory = Callable[[], httpx.Client]


def _default_http() -> httpx.Client:
    return httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=False)


#: Replaced by tests with a client over ``httpx.MockTransport`` -- the fake Google.
http_client: HttpClientFactory = _default_http


def complete(params: Mapping[str, str], *, now: datetime | None = None) -> Outcome:
    """Finish a reconnect from Google's redirect. Never raises; every path is an Outcome."""
    when = now or _now()
    pending = pending_starts.take(params.get("state"), now=when)
    if pending is None:
        outcome = Outcome("expired")
        _audit("expired", actor="unknown", provider=None, account=None, attempt=None)
        return outcome
    outcome = _complete(pending, params)
    _audit(
        outcome.result,
        actor=pending.actor,
        provider=pending.provider,
        account=outcome.account or pending.account,
        attempt=pending.attempt,
        approved_as=outcome.approved_as,
        reason=outcome.reason,
    )
    return outcome


def _failed(pending: _Pending, reason: str) -> Outcome:
    return Outcome("failed", provider=pending.provider, account=pending.account, reason=reason)


def _complete(pending: _Pending, params: Mapping[str, str]) -> Outcome:
    provider = get_provider(pending.provider)
    error = params.get("error")
    if error:
        if error == "access_denied":
            return Outcome("cancelled", provider=pending.provider, account=pending.account)
        return _failed(pending, _ERROR_CODE.sub("_", error.lower())[:40] or "error")
    code = params.get("code")
    if not code:
        return _failed(pending, "no_code")
    if provider is None:
        return _failed(pending, "provider_gone")
    client = load_client()
    target = redirect_uri()
    if client is None or target is None:
        return _failed(pending, "not_set_up")
    ends = endpoints()
    try:
        with http_client() as http:
            tokens = _exchange(http, ends, client, code, target, pending.verifier)
            if isinstance(tokens, str):
                return _failed(pending, tokens)
            granted = set(str(tokens.get("scope") or "").split())
            if not set(provider.scopes) <= granted:
                return _failed(pending, "scopes_not_granted")
            if not tokens.get("refresh_token"):
                return _failed(pending, "no_refresh_token")
            email = _identity(http, ends, provider, str(tokens["access_token"]))
    except httpx.HTTPError as exc:
        # The type only: an httpx message can carry a URL, and nothing here may risk
        # carrying more than that.
        logger.warning("google connect: %s call failed (%s)", pending.provider, type(exc).__name__)
        return _failed(pending, "network")
    if not email:
        return _failed(pending, "identity_check")
    if pending.account is not None and email != pending.account:
        return Outcome(
            "wrong_account",
            provider=pending.provider,
            account=pending.account,
            approved_as=email,
        )
    _store(provider, email, tokens, client, ends)
    return Outcome("connected", provider=pending.provider, account=email)


def _exchange(
    http: httpx.Client,
    ends: Endpoints,
    client: WebClient,
    code: str,
    target: str,
    verifier: str,
) -> dict[str, Any] | str:
    """Trade the code for tokens; the token dict, or a failure reason word."""
    response = http.post(
        ends.token_uri,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client.client_id,
            "client_secret": client.client_secret,
            "redirect_uri": target,
            "code_verifier": verifier,
        },
        headers={"accept": "application/json"},
    )
    if response.status_code != 200:
        # Google's error body names the error ("invalid_grant") and nothing secret,
        # but only the status and that word are logged.
        word = ""
        try:
            word = str(response.json().get("error", ""))
        except ValueError:
            pass
        logger.warning(
            "google connect: token exchange refused (HTTP %s %s)",
            response.status_code,
            _ERROR_CODE.sub("_", word.lower())[:40],
        )
        return "token_exchange"
    try:
        tokens = response.json()
    except ValueError:
        return "token_exchange"
    if not isinstance(tokens, dict) or not tokens.get("access_token"):
        return "token_exchange"
    return tokens


def _identity(http: httpx.Client, ends: Endpoints, provider: GoogleProvider, access: str) -> str:
    """The account the token belongs to, from the provider's own API ("" if unknown)."""
    response = http.get(
        ends.api_url(provider.identity_url),
        headers={"authorization": f"Bearer {access}", "accept": "application/json"},
    )
    if response.status_code != 200:
        logger.warning(
            "google connect: %s identity check refused (HTTP %s)",
            provider.key,
            response.status_code,
        )
        return ""
    value: Any = response.json()
    for key in provider.identity_field:
        value = value.get(key) if isinstance(value, dict) else None
    return value.strip().lower() if isinstance(value, str) else ""


def _store(
    provider: GoogleProvider,
    email: str,
    tokens: Mapping[str, Any],
    client: WebClient,
    ends: Endpoints,
) -> None:
    """Keyring + accounts table, exactly as the plugin's Mac login leaves them."""
    from google.oauth2.credentials import Credentials

    expires_in = tokens.get("expires_in")
    expiry = None
    if isinstance(expires_in, int | float):
        # google-auth compares expiry as naive UTC.
        expiry = (_now() + timedelta(seconds=float(expires_in))).replace(tzinfo=None)
    creds = Credentials(  # type: ignore[no-untyped-call]
        token=str(tokens["access_token"]),
        refresh_token=str(tokens["refresh_token"]),
        token_uri=ends.token_uri,
        client_id=client.client_id,
        client_secret=client.client_secret,
        scopes=list(provider.scopes),
        expiry=expiry,
    )
    _credentials.save_token(provider.key, email, creds.to_json())  # type: ignore[no-untyped-call]
    store = EmailAccountStore()
    store.ensure_schema()
    if store.get_by_address(provider.key, email) is None:
        store.add(provider=provider.key, address=email)
    logger.info("google connect: %s reconnected for %s", provider.key, email)


# ─── Audit ────────────────────────────────────────────────────────────


def _audit(
    event: str,
    *,
    actor: str,
    provider: str | None,
    account: str | None,
    attempt: str | None,
    approved_as: str | None = None,
    reason: str | None = None,
) -> None:
    """One governance ledger row per start and per outcome. Never a token or a code.

    The row's free-text reason names the event and the provider only ("google
    connected: gmail"): a reason is copied verbatim wherever the ledger is shown, so it
    never carries the account's address or who acted. Both are in the payload, which
    the proof bundle never copies and pseudonymises where it reads an account."""
    from iris_harness.sdk.audit import AuditLog, audit_db_path

    ok = event in {"start", "connected"}
    payload = {
        "actor": actor,
        "provider": provider,
        "account": account,
        "outcome": event,
        "approved_as": approved_as,
        "reason": reason,
    }
    try:
        AuditLog(db_path=audit_db_path()).record(
            run_id=f"google-connect-{attempt or 'none'}",
            step_id=None,
            agent_type="api",
            hook_point="connection",
            plugin="google_connections",
            decision="allow" if ok else "deny",
            severity="info" if ok else "warn",
            reason=f"google {event}: {provider or '-'}",
            payload={k: v for k, v in payload.items() if v is not None},
        )
    except Exception:  # a ledger hiccup must not strand the owner mid-flow
        logger.warning("google connect: audit write failed for %s", event, exc_info=True)


__all__ = [
    "CALLBACK_PATH",
    "CLIENT_PATH",
    "ClientConfigError",
    "Endpoints",
    "GoogleProvider",
    "Outcome",
    "SETTINGS_FIX_URL",
    "START_PATH",
    "StartRefused",
    "StatusSource",
    "clear_providers",
    "complete",
    "get_provider",
    "google_credential_checks",
    "load_client",
    "parse_client_json",
    "pending_starts",
    "providers",
    "redirect_uri",
    "register_provider",
    "reset_probe_cache",
    "save_client",
    "setup_status",
    "start",
    "with_reconnect",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_providers", "_probe_cache")
