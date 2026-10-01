"""Tests for the Keychain-backed credentials module (ADR-0011).

Mocks the ``keyring`` library at the module level so tests never touch
the real system keychain. Redirects ``CREDENTIALS_DIR`` to a per-test
tmp_path so filesystem operations don't pollute ``~/.iris/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import keyring.errors
import pytest

from iris_harness.kernel.governance.vault import credentials


@pytest.fixture
def tmp_creds_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the module's CREDENTIALS_DIR to a tmp_path for the test."""
    redirected = tmp_path / "credentials"
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", redirected)
    return redirected


# ─── Keychain token operations ────────────────────────────────────────


def test_save_load_token_roundtrip() -> None:
    """save_token then load_token returns the same value."""
    store: dict[tuple[str, str], str] = {}

    def fake_set(service: str, account: str, value: str) -> None:
        store[(service, account)] = value

    def fake_get(service: str, account: str) -> str | None:
        return store.get((service, account))

    with (
        patch.object(credentials.keyring, "set_password", fake_set),
        patch.object(credentials.keyring, "get_password", fake_get),
    ):
        credentials.save_token("gmail", "user@example.com", "abc123")
        assert credentials.load_token("gmail", "user@example.com") == "abc123"


def test_load_token_missing_returns_none() -> None:
    """load_token returns None when no entry exists."""
    with patch.object(credentials.keyring, "get_password", return_value=None):
        assert credentials.load_token("gmail", "missing@example.com") is None


def test_delete_token_idempotent() -> None:
    """delete_token swallows PasswordDeleteError when entry is absent."""

    def raising_delete(service: str, account: str) -> None:
        raise keyring.errors.PasswordDeleteError("not found")

    with patch.object(credentials.keyring, "delete_password", raising_delete):
        # Must not raise.
        credentials.delete_token("gmail", "missing@example.com")


def test_service_name_prefix() -> None:
    """Saved tokens use the 'iris-<provider>' service prefix."""
    captured: list[tuple[str, str, str]] = []

    def fake_set(service: str, account: str, value: str) -> None:
        captured.append((service, account, value))

    with patch.object(credentials.keyring, "set_password", fake_set):
        credentials.save_token("gcalendar", "x@y.com", "tok")

    assert captured == [("iris-gcalendar", "x@y.com", "tok")]


# ─── Non-secret config-file operations ────────────────────────────────


def test_save_config_creates_dir_with_700(tmp_creds_dir: Path) -> None:
    """save_config creates the credentials/ dir with mode 0o700 on POSIX."""
    credentials.save_config("gmail", {"client_id": "x"})

    assert tmp_creds_dir.is_dir()
    if os.name == "posix":
        mode = oct(tmp_creds_dir.stat().st_mode & 0o777)
        assert mode == "0o700", f"expected 0o700, got {mode}"


def test_load_config_missing_returns_none(tmp_creds_dir: Path) -> None:
    """load_config returns None when the config file is absent."""
    assert credentials.load_config("nonexistent") is None


def test_save_config_roundtrip(tmp_creds_dir: Path) -> None:
    """save_config then load_config returns the same dict."""
    payload = {"client_id": "abc", "scopes": ["read", "write"], "last_auth_at": "2026-05-25"}
    credentials.save_config("gmail", payload)

    assert credentials.load_config("gmail") == payload


def test_list_providers_from_filesystem(tmp_creds_dir: Path) -> None:
    """list_providers enumerates config files only, sorted alphabetically."""
    credentials.save_config("gmail", {"x": 1})
    credentials.save_config("gcalendar", {"y": 2})
    # An unrelated file in the dir should be ignored.
    (tmp_creds_dir / "README.md").write_text("# notes")

    assert credentials.list_providers() == ["gcalendar", "gmail"]
