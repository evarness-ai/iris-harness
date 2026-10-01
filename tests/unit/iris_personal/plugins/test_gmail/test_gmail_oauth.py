"""Tests for the Gmail OAuth bootstrap (Track 1A).

OAuth itself is mocked — we never hit Google's servers from unit tests.
``InstalledAppFlow.from_client_secrets_file`` and the Gmail API call are
both patched to return canned objects so the persistence + registration
logic is the real test surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.auth.exceptions import RefreshError

from iris_harness.kernel.governance.vault import credentials
from iris_personal.email.accounts import EmailAccountStore
from iris_personal.plugins.gmail import gmail_oauth

# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def tmp_iris(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect both credentials.CREDENTIALS_DIR and EmailAccountStore.db_path."""
    cred_dir = tmp_path / "credentials"
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(
        "iris_personal.email.accounts.Path.home", lambda: tmp_path
    )  # benign — db_path is overridden per-call below
    return tmp_path


@pytest.fixture
def mock_keyring() -> dict[tuple[str, str], str]:
    """In-memory shim for keyring used by credentials.py."""
    store: dict[tuple[str, str], str] = {}

    def fake_set(service: str, account: str, value: str) -> None:
        store[(service, account)] = value

    def fake_get(service: str, account: str) -> str | None:
        return store.get((service, account))

    def fake_delete(service: str, account: str) -> None:
        store.pop((service, account), None)

    with (
        patch.object(credentials.keyring, "set_password", fake_set),
        patch.object(credentials.keyring, "get_password", fake_get),
        patch.object(credentials.keyring, "delete_password", fake_delete),
    ):
        yield store


@pytest.fixture
def stub_email_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmailAccountStore:
    """Patch EmailAccountStore default db_path to a tmp file."""
    db_path = tmp_path / "iris.db"

    def fake_init(self, db_path=db_path):  # type: ignore[no-untyped-def]
        self.db_path = db_path

    monkeypatch.setattr(EmailAccountStore, "__init__", fake_init)
    return EmailAccountStore()


@pytest.fixture
def stub_client_secrets(tmp_path: Path) -> Path:
    """Write a tiny client_secrets.json placeholder (contents don't matter — mocked)."""
    p = tmp_path / "google_oauth_client.json"
    p.write_text('{"installed":{"client_id":"x","client_secret":"y"}}')
    return p


def _make_fake_credentials(
    *,
    refresh_token: str = "rt-123",  # noqa: S107 — mock OAuth token, not a real credential
    scopes: list[str] | None = None,
    expiry: str = "2026-12-31T23:59:00Z",
) -> MagicMock:
    """A MagicMock that mimics google.oauth2.credentials.Credentials."""
    fake = MagicMock(name="Credentials")
    payload = {
        "token": "at-456",
        "refresh_token": refresh_token,
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "x",
        "client_secret": "y",
        "scopes": scopes or list(gmail_oauth.DEFAULT_SCOPES),
        "expiry": expiry,
    }
    fake.to_json.return_value = json.dumps(payload)
    fake.expired = False
    fake.refresh_token = refresh_token
    return fake


# ─── login() ──────────────────────────────────────────────────────────


def test_login_missing_client_secrets_raises(
    tmp_iris: Path, mock_keyring, stub_email_store
) -> None:
    missing = tmp_iris / "no_such_file.json"
    with pytest.raises(FileNotFoundError):
        gmail_oauth.login("user@example.com", client_secrets_path=missing)


def test_login_authorized_email_mismatch_raises(
    tmp_iris: Path, mock_keyring, stub_email_store, stub_client_secrets: Path
) -> None:
    fake_creds = _make_fake_credentials()

    with (
        patch.object(gmail_oauth, "InstalledAppFlow") as mock_flow_cls,
        patch.object(gmail_oauth, "build") as mock_build,
    ):
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = fake_creds
        mock_flow_cls.from_client_secrets_file.return_value = mock_flow
        mock_build.return_value.users.return_value.getProfile.return_value.execute.return_value = {
            "emailAddress": "wrong@gmail.com"
        }

        with pytest.raises(ValueError, match="OAuth authorized 'wrong@gmail.com'"):
            gmail_oauth.login("user@gmail.com", client_secrets_path=stub_client_secrets)


def test_login_success_persists_token_and_registers_account(
    tmp_iris: Path, mock_keyring, stub_email_store, stub_client_secrets: Path
) -> None:
    fake_creds = _make_fake_credentials()

    with (
        patch.object(gmail_oauth, "InstalledAppFlow") as mock_flow_cls,
        patch.object(gmail_oauth, "build") as mock_build,
    ):
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = fake_creds
        mock_flow_cls.from_client_secrets_file.return_value = mock_flow
        mock_build.return_value.users.return_value.getProfile.return_value.execute.return_value = {
            "emailAddress": "user@gmail.com"
        }

        account = gmail_oauth.login("USER@gmail.COM", client_secrets_path=stub_client_secrets)

    assert account.provider == "gmail"
    assert account.address == "user@gmail.com"

    stored = mock_keyring[("iris-gmail", "user@gmail.com")]
    info = json.loads(stored)
    assert info["token"] == "at-456"  # asserting mock OAuth payload shape
    assert info["refresh_token"] == "rt-123"  # asserting mock OAuth payload shape
    assert info["scopes"] == list(gmail_oauth.DEFAULT_SCOPES)


def test_login_existing_account_does_not_duplicate(
    tmp_iris: Path, mock_keyring, stub_email_store, stub_client_secrets: Path
) -> None:
    # Pre-existing account row
    stub_email_store.ensure_schema()
    pre_existing = stub_email_store.add(provider="gmail", address="user@gmail.com")

    fake_creds = _make_fake_credentials()
    with (
        patch.object(gmail_oauth, "InstalledAppFlow") as mock_flow_cls,
        patch.object(gmail_oauth, "build") as mock_build,
    ):
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = fake_creds
        mock_flow_cls.from_client_secrets_file.return_value = mock_flow
        mock_build.return_value.users.return_value.getProfile.return_value.execute.return_value = {
            "emailAddress": "user@gmail.com"
        }

        result = gmail_oauth.login("user@gmail.com", client_secrets_path=stub_client_secrets)

    assert result.id == pre_existing.id  # re-used the existing row


# ─── status() ─────────────────────────────────────────────────────────


def test_status_empty_when_no_accounts(tmp_iris: Path, mock_keyring, stub_email_store) -> None:
    assert gmail_oauth.status() == []


def test_status_reports_missing_token(tmp_iris: Path, mock_keyring, stub_email_store) -> None:
    stub_email_store.ensure_schema()
    stub_email_store.add(provider="gmail", address="orphan@gmail.com")

    snaps = gmail_oauth.status()
    assert len(snaps) == 1
    assert snaps[0].has_keychain_token is False
    assert snaps[0].token_expiry is None
    assert snaps[0].refresh_token_present is False


def test_status_reports_token_details(tmp_iris: Path, mock_keyring, stub_email_store) -> None:
    stub_email_store.ensure_schema()
    stub_email_store.add(provider="gmail", address="user@gmail.com")
    # Pre-populate keychain with a realistic token blob
    credentials.save_token(
        "gmail",
        "user@gmail.com",
        json.dumps(
            {
                "token": "at-1",
                "refresh_token": "rt-1",
                "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
                "expiry": "2026-12-31T23:59:00Z",
            }
        ),
    )

    snaps = gmail_oauth.status()
    assert len(snaps) == 1
    s = snaps[0]
    assert s.has_keychain_token is True
    assert s.refresh_token_present is True
    assert s.token_expiry is not None
    assert s.token_expiry.year == 2026
    assert s.scopes == ("https://www.googleapis.com/auth/gmail.readonly",)


def test_status_filters_non_gmail_providers(tmp_iris: Path, mock_keyring, stub_email_store) -> None:
    stub_email_store.ensure_schema()
    stub_email_store.add(provider="gmail", address="x@gmail.com")
    stub_email_store.add(provider="outlook", address="y@outlook.com")

    snaps = gmail_oauth.status()
    assert len(snaps) == 1
    assert snaps[0].email_account.provider == "gmail"


# ─── logout() ─────────────────────────────────────────────────────────


def test_logout_removes_keychain_token(tmp_iris: Path, mock_keyring, stub_email_store) -> None:
    stub_email_store.ensure_schema()
    stub_email_store.add(provider="gmail", address="user@gmail.com")
    credentials.save_token("gmail", "user@gmail.com", '{"token":"x"}')

    assert gmail_oauth.logout("USER@GMAIL.COM") is True
    assert credentials.load_token("gmail", "user@gmail.com") is None
    assert stub_email_store.get_by_address("gmail", "user@gmail.com").active is False


def test_logout_missing_account_returns_false(
    tmp_iris: Path, mock_keyring, stub_email_store
) -> None:
    stub_email_store.ensure_schema()
    assert gmail_oauth.logout("ghost@gmail.com") is False


# ─── load_credentials() ───────────────────────────────────────────────


def test_load_credentials_missing_returns_none(
    tmp_iris: Path, mock_keyring, stub_email_store
) -> None:
    assert gmail_oauth.load_credentials("nobody@gmail.com") is None


def test_load_credentials_returns_credentials_when_valid(
    tmp_iris: Path, mock_keyring, stub_email_store
) -> None:
    credentials.save_token(
        "gmail",
        "user@gmail.com",
        json.dumps(
            {
                "token": "at-1",
                "refresh_token": "rt-1",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "c",
                "client_secret": "s",
                "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
            }
        ),
    )

    with patch.object(gmail_oauth, "Credentials") as mock_creds_cls:
        fake = MagicMock()
        fake.expired = False
        fake.refresh_token = "rt-1"  # mock OAuth refresh token
        mock_creds_cls.from_authorized_user_info.return_value = fake

        result = gmail_oauth.load_credentials("USER@GMAIL.COM")

    assert result is fake


def test_load_credentials_refreshes_when_expired(
    tmp_iris: Path, mock_keyring, stub_email_store
) -> None:
    credentials.save_token(
        "gmail",
        "user@gmail.com",
        json.dumps({"token": "old", "refresh_token": "rt-1", "scopes": []}),
    )

    refreshed_json = json.dumps({"token": "new", "refresh_token": "rt-1", "scopes": []})

    with patch.object(gmail_oauth, "Credentials") as mock_creds_cls:
        fake = MagicMock()
        fake.expired = True
        fake.refresh_token = "rt-1"  # mock OAuth refresh token
        fake.to_json.return_value = refreshed_json
        mock_creds_cls.from_authorized_user_info.return_value = fake

        gmail_oauth.load_credentials("user@gmail.com")

    fake.refresh.assert_called_once()
    # Persisted-back token blob is the refreshed one
    assert mock_keyring[("iris-gmail", "user@gmail.com")] == refreshed_json


def test_load_credentials_raises_a_typed_error_when_refresh_revoked(
    tmp_iris: Path, mock_keyring, stub_email_store
) -> None:
    stale = json.dumps({"token": "old", "refresh_token": "rt-1", "scopes": []})
    credentials.save_token("gmail", "user@gmail.com", stale)

    with patch.object(gmail_oauth, "Credentials") as mock_creds_cls:
        fake = MagicMock()
        fake.expired = True
        fake.refresh_token = "rt-1"  # mock OAuth refresh token
        fake.refresh.side_effect = RefreshError("invalid_grant: Token has been expired or revoked.")
        mock_creds_cls.from_authorized_user_info.return_value = fake

        with pytest.raises(credentials.CredentialRevokedError) as caught:
            gmail_oauth.load_credentials("User@Gmail.com")

    # Typed, and still a RuntimeError: every caller that turned "no credentials"
    # into a RuntimeError keeps working, now with the reason and the fix.
    assert isinstance(caught.value, RuntimeError)
    assert caught.value.service == "Gmail"
    assert caught.value.account == "user@gmail.com"
    assert caught.value.login_command == "iris auth gmail login --user user@gmail.com"
    assert "revoked" in str(caught.value)
    assert isinstance(caught.value.__cause__, RefreshError)
    fake.refresh.assert_called_once()
    # The stale token is left untouched (not overwritten on failure).
    assert mock_keyring[("iris-gmail", "user@gmail.com")] == stale


def test_a_revoked_token_surfaces_through_a_sync_with_the_fix(
    tmp_iris: Path, mock_keyring, stub_email_store
) -> None:
    """The sweep's per-account error now says revoked + the command, not "no credentials"."""
    from iris_personal.plugins.gmail import gmail_fetch

    credentials.save_token(
        "gmail", "user@gmail.com", json.dumps({"token": "old", "refresh_token": "rt-1"})
    )
    with patch.object(gmail_oauth, "Credentials") as mock_creds_cls:
        fake = MagicMock()
        fake.expired = True
        fake.refresh_token = "rt-1"  # mock OAuth refresh token
        fake.refresh.side_effect = RefreshError("invalid_grant")
        mock_creds_cls.from_authorized_user_info.return_value = fake
        with pytest.raises(RuntimeError, match="iris auth gmail login --user user@gmail.com"):
            gmail_fetch.fetch_message_body("gmail:user@gmail.com", "msg-1")


# ─── force_refresh() — the health watch's repair (ADR-0116) ─────────────


def _stored(mock_keyring, *, refresh_token: str | None = "rt-1") -> str:  # noqa: S107
    blob = json.dumps({"token": "old", "refresh_token": refresh_token, "scopes": []})
    credentials.save_token("gmail", "user@gmail.com", blob)
    return blob


def test_force_refresh_refreshes_an_unexpired_token(
    tmp_iris: Path, mock_keyring, stub_email_store
) -> None:
    _stored(mock_keyring)
    with patch.object(gmail_oauth, "Credentials") as mock_creds_cls:
        fake = MagicMock()
        fake.expired = False  # load_credentials would NOT refresh this one
        fake.refresh_token = "rt-1"  # mock OAuth refresh token
        fake.to_json.return_value = '{"token": "new"}'
        mock_creds_cls.from_authorized_user_info.return_value = fake
        assert gmail_oauth.force_refresh("User@Gmail.com") is True
    fake.refresh.assert_called_once()
    assert mock_keyring[("iris-gmail", "user@gmail.com")] == '{"token": "new"}'


def test_force_refresh_reports_a_revoked_token(
    tmp_iris: Path, mock_keyring, stub_email_store
) -> None:
    stale = _stored(mock_keyring)
    with patch.object(gmail_oauth, "Credentials") as mock_creds_cls:
        fake = MagicMock()
        fake.refresh_token = "rt-1"  # mock OAuth refresh token
        fake.refresh.side_effect = RefreshError("invalid_grant")
        mock_creds_cls.from_authorized_user_info.return_value = fake
        assert gmail_oauth.force_refresh("user@gmail.com") is False
    assert mock_keyring[("iris-gmail", "user@gmail.com")] == stale


def test_force_refresh_without_a_token_is_none(
    tmp_iris: Path, mock_keyring, stub_email_store
) -> None:
    assert gmail_oauth.force_refresh("user@gmail.com") is None
    _stored(mock_keyring, refresh_token=None)
    with patch.object(gmail_oauth, "Credentials") as mock_creds_cls:
        fake = MagicMock()
        fake.refresh_token = None  # an access-only token cannot refresh
        mock_creds_cls.from_authorized_user_info.return_value = fake
        assert gmail_oauth.force_refresh("user@gmail.com") is None
    fake.refresh.assert_not_called()
