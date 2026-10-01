"""macOS Keychain-backed credential store for personal-assistant OAuth.

OAuth tokens live in the system keychain via the ``keyring`` library.
Non-secret OAuth config (client_id, scopes, last-auth timestamp,
account_email) lives in plaintext JSON under
``~/.iris/workspace/credentials/<provider>.config.json``, with the
parent directory chmod 700 as defense in depth.

Single shared entry point — connector skills never touch ``keyring``
directly, so the storage backend stays swappable.

See ADR-0003 for design rationale.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Any, cast

import keyring
import keyring.errors

logger = logging.getLogger(__name__)

# Anchored at $IRIS_HOME/workspace (``~/.iris/workspace``) per the canonical doc §4.3,
# beside SOUL.md and USER.md, and resolved on every use: this used ``Path.home()``, so a
# relocated IRIS_HOME (the test suite, an eval instance, ``iris_harness.testing``'s
# harness) still read -- and an OAuth connect wrote -- the owner's real credentials dir.
# ``CREDENTIALS_DIR`` is an override slot (``None``: resolve from IRIS_HOME).
CREDENTIALS_DIR: Path | None = None


def credentials_dir() -> Path:
    """``CREDENTIALS_DIR`` when set, else ``$IRIS_HOME/workspace/credentials``."""
    if CREDENTIALS_DIR is not None:
        return CREDENTIALS_DIR
    from iris_harness.foundation.plugin_dirs import iris_home

    return iris_home() / "workspace" / "credentials"


CONFIG_SUFFIX = ".config.json"
_KEYRING_SERVICE_PREFIX = "iris-"  # service="iris-gmail", "iris-gcalendar", etc.


class CredentialRevokedError(RuntimeError):
    """A stored OAuth token its provider refused to refresh.

    Revoked, or its refresh token expired (a Google OAuth app in "testing" mode
    rotates them about weekly). Only a re-login fixes it, and ``login_command`` is
    that exact command. It is a ``RuntimeError`` so every caller that already turned
    "no credentials" into a ``RuntimeError`` keeps working — now with the reason.
    "No token stored at all" is not this error: loaders return None for that.
    """

    def __init__(self, service: str, account: str, login_command: str) -> None:
        self.service = service
        self.account = account
        self.login_command = login_command
        super().__init__(
            f"{service} access for {account} was revoked or has expired — "
            f"run `{login_command}` to reconnect."
        )


# ─── Keychain token operations ────────────────────────────────────────


def save_token(provider: str, account: str, value: str) -> None:
    """Store an opaque token value in the OS keychain. Replaces any existing."""
    service = f"{_KEYRING_SERVICE_PREFIX}{provider}"
    keyring.set_password(service, account, value)
    logger.debug("keychain: stored token for %s/%s", service, account)


def load_token(provider: str, account: str) -> str | None:
    """Retrieve a token from the OS keychain. Returns None if absent."""
    service = f"{_KEYRING_SERVICE_PREFIX}{provider}"
    value = keyring.get_password(service, account)
    if value is None:
        logger.debug("keychain: no token for %s/%s", service, account)
    return value


def delete_token(provider: str, account: str) -> None:
    """Remove a token from the OS keychain. Idempotent — silent on missing."""
    service = f"{_KEYRING_SERVICE_PREFIX}{provider}"
    try:
        keyring.delete_password(service, account)
        logger.debug("keychain: deleted token for %s/%s", service, account)
    except keyring.errors.PasswordDeleteError:
        pass  # already gone — fine


# ─── Non-secret config-file operations ────────────────────────────────


def _ensure_credentials_dir() -> Path:
    """Create credentials/ if missing, with chmod 700 (POSIX only).

    Windows uses ACLs rather than POSIX modes; the chmod is skipped there.
    """
    directory = credentials_dir()
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(directory, stat.S_IRWXU)
    return directory


def save_config(provider: str, config: dict[str, Any]) -> Path:
    """Persist non-secret OAuth config to ``<provider>.config.json``.

    Caller MUST NOT include access_token, refresh_token, or any other
    secret material in ``config``; secrets belong in ``save_token()``.
    """
    path = _ensure_credentials_dir() / f"{provider}{CONFIG_SUFFIX}"
    path.write_text(json.dumps(config, indent=2, sort_keys=True))
    return path


def load_config(provider: str) -> dict[str, Any] | None:
    """Read non-secret OAuth config. Returns None if file absent."""
    path = credentials_dir() / f"{provider}{CONFIG_SUFFIX}"
    if not path.is_file():
        return None
    return cast("dict[str, Any] | None", json.loads(path.read_text()))


def list_providers() -> list[str]:
    """Return providers that have a config file. Does NOT touch keychain."""
    directory = credentials_dir()
    if not directory.is_dir():
        return []
    return sorted(p.name.removesuffix(CONFIG_SUFFIX) for p in directory.glob(f"*{CONFIG_SUFFIX}"))
