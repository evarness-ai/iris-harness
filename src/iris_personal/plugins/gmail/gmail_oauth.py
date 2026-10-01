"""Gmail OAuth (installed-app flow) — Phase 1 Track 1A.

Implements ``iris auth gmail {login,status,logout}`` per ADR-0003
(Keychain credentials) and ADR-0016 (email accounts registry).

OAuth tokens are stored as a single JSON blob per address in the
macOS Keychain via ``iris_harness.kernel.governance.vault.credentials``. The blob shape
matches ``google.oauth2.credentials.Credentials.to_json()`` so callers
(e.g. the Phase 1 Track 1C ``gmail-inbox`` skill) can rehydrate via
``Credentials.from_authorized_user_info(json.loads(blob))`` and let
the Google client library handle automatic refresh.

The ``client_secret.json`` (OAuth 2.0 Desktop-app client downloaded
from Google Cloud Console) lives at
``$IRIS_HOME/workspace/credentials/google_oauth_client.json`` by default;
``--client-secrets PATH`` overrides. See
``docs/usage-guides/gmail-auth.md`` for the one-time Google Cloud
setup walkthrough.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]
from googleapiclient.discovery import build

from iris_harness.sdk import vault as _credentials
from iris_harness.sdk.config import workspace_dir
from iris_personal.email.accounts import EmailAccount, EmailAccountStore

logger = logging.getLogger(__name__)


def default_client_secrets_path() -> Path:
    """``$IRIS_HOME/workspace/credentials/google_oauth_client.json`` (``~/.iris`` when
    ``IRIS_HOME`` is unset): resolved per call, so a relocated home is honoured."""
    return workspace_dir() / "credentials" / "google_oauth_client.json"


# gmail.modify: read, plus move to Trash and back (ADR-0118 step 5, owner's decision
# 2026-09-21). It cannot delete permanently -- that needs full mail.google.com access,
# which IRIS never asks for. A token granted before this keeps its own scopes and keeps
# reading; only trashing asks for a new consent.
DEFAULT_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
KEYRING_PROVIDER = "gmail"


# ─── Public types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class GmailAccountStatus:
    """Snapshot of one configured Gmail account's auth state."""

    email_account: EmailAccount
    has_keychain_token: bool
    token_expiry: datetime | None  # None if unknown or no token
    refresh_token_present: bool
    scopes: tuple[str, ...]  # from token blob; empty if no token


# ─── login / status / logout ──────────────────────────────────────────


def login(
    email: str,
    *,
    client_secrets_path: Path | None = None,
    scopes: list[str] | None = None,
) -> EmailAccount:
    """Run the installed-app OAuth flow and persist tokens.

    Opens the user's default browser to Google's auth screen, listens
    on a random localhost port for the callback, captures the code,
    exchanges it for tokens, and stores everything.

    Raises:
        FileNotFoundError: if the client-secrets file is missing.
        ValueError: if the authorized email doesn't match ``email``
            (defends against the user picking a different account in
            the browser than they typed in the CLI).
    """
    secrets = client_secrets_path or default_client_secrets_path()
    if not secrets.is_file():
        raise FileNotFoundError(
            f"OAuth client secrets not found at {secrets}. "
            "See docs/usage-guides/gmail-auth.md for one-time setup."
        )
    active_scopes = list(scopes) if scopes else list(DEFAULT_SCOPES)

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), active_scopes)
    creds = flow.run_local_server(port=0)

    resolved = _resolve_authorized_email(creds)
    if not resolved:
        raise RuntimeError("OAuth completed but Gmail profile lookup returned no email address.")
    if resolved.lower() != email.lower():
        raise ValueError(
            f"OAuth authorized {resolved!r} but CLI requested {email!r}. "
            "Re-run with the address matching the Google account you used in the browser."
        )

    _credentials.save_token(KEYRING_PROVIDER, resolved.lower(), creds.to_json())
    logger.debug("gmail oauth: stored token for %s", resolved.lower())

    store = EmailAccountStore()
    store.ensure_schema()
    existing = store.get_by_address("gmail", resolved.lower())
    if existing is None:
        return store.add(provider="gmail", address=resolved.lower())
    return existing


def status() -> list[GmailAccountStatus]:
    """Return a snapshot of all Gmail accounts known to IRIS."""
    store = EmailAccountStore()
    store.ensure_schema()
    out: list[GmailAccountStatus] = []
    for account in store.list(active_only=False):
        if account.provider != "gmail":
            continue
        token_blob = _credentials.load_token(KEYRING_PROVIDER, account.address)
        if token_blob is None:
            out.append(
                GmailAccountStatus(
                    email_account=account,
                    has_keychain_token=False,
                    token_expiry=None,
                    refresh_token_present=False,
                    scopes=(),
                )
            )
            continue
        info = json.loads(token_blob)
        expiry = _parse_google_expiry(info.get("expiry"))
        scopes = tuple(info.get("scopes") or ())
        out.append(
            GmailAccountStatus(
                email_account=account,
                has_keychain_token=True,
                token_expiry=expiry,
                refresh_token_present=bool(info.get("refresh_token")),
                scopes=scopes,
            )
        )
    return out


def logout(email: str) -> bool:
    """Remove tokens from Keychain and deactivate the email_account row.

    Returns True if an account row was found and deactivated; False if
    no such account existed (logout is still idempotent on the keychain
    side — the token is removed if present).
    """
    addr = email.strip().lower()
    _credentials.delete_token(KEYRING_PROVIDER, addr)
    logger.debug("gmail oauth: removed token for %s", addr)

    store = EmailAccountStore()
    store.ensure_schema()
    account = store.get_by_address("gmail", addr)
    if account is None:
        return False
    store.deactivate(account.id)
    return True


# ─── Public helper for downstream skills ──────────────────────────────


def force_refresh(email: str) -> bool | None:
    """Refresh the stored token now, whatever its expiry (the health watch's repair).

    True = refreshed and saved; False = Google refused (revoked / refresh token
    expired — only ``iris auth gmail login`` fixes that); None = nothing stored that
    could refresh. A network error propagates: it is neither verdict.
    """
    addr = email.strip().lower()
    raw = _credentials.load_token(KEYRING_PROVIDER, addr)
    if raw is None:
        return None
    creds = Credentials.from_authorized_user_info(json.loads(raw))  # type: ignore[no-untyped-call]
    if not creds.refresh_token:
        return None
    try:
        creds.refresh(Request())
    except RefreshError:
        return False
    _credentials.save_token(KEYRING_PROVIDER, addr, creds.to_json())
    return True


def load_credentials(email: str) -> Credentials | None:
    """Load and (if expired) refresh credentials for an account.

    Returns None if no token is stored. Refreshes transparently when
    the access token has expired and a refresh token is available,
    persisting the refreshed bundle back to Keychain. Raises
    ``CredentialRevokedError`` when Google refuses that refresh.
    """
    addr = email.strip().lower()
    raw = _credentials.load_token(KEYRING_PROVIDER, addr)
    if raw is None:
        return None
    info = json.loads(raw)
    creds = Credentials.from_authorized_user_info(info)  # type: ignore[no-untyped-call]
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            # Refresh token expired or revoked (an OAuth app in "testing" mode rotates
            # refresh tokens ~weekly). Say so, typed: callers already turn a missing
            # credential into a RuntimeError, and this is one — with the reason and the
            # exact re-login command, instead of a generic "no credentials".
            raise _credentials.CredentialRevokedError(
                "Gmail", addr, f"iris auth gmail login --user {addr}"
            ) from exc
        _credentials.save_token(KEYRING_PROVIDER, addr, creds.to_json())
        logger.debug("gmail oauth: refreshed token for %s", addr)
    return cast("Credentials | None", creds)


# ─── Internals ────────────────────────────────────────────────────────


def _resolve_authorized_email(creds: Credentials) -> str:
    """Call Gmail API's getProfile to learn which address was authorized."""
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    profile = service.users().getProfile(userId="me").execute()
    return cast("str", profile.get("emailAddress", ""))


def _parse_google_expiry(value: str | None) -> datetime | None:
    """Parse ``Credentials.to_json()``'s ``expiry`` field (RFC 3339 / ISO 8601)."""
    if not value:
        return None
    # google's to_json emits e.g. "2026-05-25T11:23:45.000Z" — handle Z suffix.
    try:
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        return datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        return None
