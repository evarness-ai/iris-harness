"""Master-key resolution shared by the credential vault and the file vault.

The Fernet master key is resolved in order:

1. ``IRIS_VAULT_MASTER_KEY`` env var (base64 Fernet key), or
2. OS keyring entry ``service=iris-vault``, ``username=master-key``
   (seeded on first run when a working keyring is present).

Extracted from ``VaultStore`` so other encrypted stores (e.g. the FileManager
document vault) share one key story instead of growing a second one.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from cryptography.fernet import Fernet

try:  # pragma: no cover - exercised via integration on host with keyring
    import keyring
# Keep vault usable in minimal envs: keyring can fail at import, not only be absent.
except Exception:  # pragma: no cover  # noqa: BLE001
    keyring = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_MASTER_KEY_ENV = "IRIS_VAULT_MASTER_KEY"
_KEYRING_SERVICE = "iris-vault"
_KEYRING_USERNAME = "master-key"


class VaultError(RuntimeError):
    """Base class for vault-related errors. Callers branch on subclasses."""


class VaultMasterKeyUnavailableError(VaultError):
    """No master key could be resolved from env or keyring."""

    def __init__(self) -> None:
        super().__init__(
            "vault master key unavailable: set IRIS_VAULT_MASTER_KEY to a "
            "Fernet key, or install/use an OS keyring. Generate a key with: "
            "python -c 'from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())'"
        )


def resolve_master_key() -> bytes:
    """Resolve the shared Fernet master key (env first, then OS keyring)."""
    env_key = os.getenv(_MASTER_KEY_ENV, "").strip()
    if env_key:
        return env_key.encode("utf-8")

    if keyring is None:
        raise VaultMasterKeyUnavailableError()

    try:
        stored = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
    except Exception as exc:  # keyring backends raise many types
        logger.debug("keyring lookup failed: %s", exc)
        raise VaultMasterKeyUnavailableError() from exc
    if stored:
        return stored.encode("utf-8")

    # First-run on a host with a working keyring: seed it.
    generated = Fernet.generate_key().decode("utf-8")
    try:
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, generated)
    except Exception as exc:  # same as above
        logger.debug("keyring write failed: %s", exc)
        raise VaultMasterKeyUnavailableError() from exc
    return generated.encode("utf-8")


# -- Read-only status + an explicit, never-overwriting store (`iris doctor`) ----------
#
# ``resolve_master_key`` above is what a governed call uses, and on a first run it SEEDS
# the keyring. A preflight must not: it reports, and writes only when the owner asks.

MasterKeySource = Literal["env", "keyring", "absent", "invalid", "no_keyring", "not_read"]


@dataclass(frozen=True)
class MasterKeyStatus:
    """Where the master key stands, found without creating one.

    ``env`` / ``keyring``: a key is there. ``absent``: a working keyring holds none.
    ``invalid``: ``IRIS_VAULT_MASTER_KEY`` is set but is not a Fernet key. ``no_keyring``:
    no usable keyring on this host (WSL2, headless Linux, containers) or it failed when
    read. ``not_read``: a keyring exists but the caller asked not to read it.
    """

    source: MasterKeySource
    detail: str = ""

    @property
    def present(self) -> bool:
        return self.source in ("env", "keyring")


def is_fernet_key(value: str) -> bool:
    """True when ``value`` is a usable Fernet key (32 url-safe base64 bytes)."""
    try:
        Fernet(value.encode("utf-8"))
    except (ValueError, TypeError):
        return False
    return True


def keyring_usable() -> bool:
    """A keyring backend is installed that claims to work here.

    Asking which backend is active opens nothing: the ``fail`` (priority 0) and ``null``
    (priority -1) backends are what keyring selects when no OS store is reachable. A
    backend that claims to work can still fail at call time (Secret Service without a
    D-Bus session); callers treat that failure as ``no_keyring`` too.
    """
    if keyring is None:
        return False
    try:
        backend = keyring.get_keyring()
    except Exception:  # noqa: BLE001 — keyring backends raise many types
        return False
    return float(getattr(backend, "priority", 0) or 0) > 0


def master_key_status(*, read_keyring: bool) -> MasterKeyStatus:
    """Where the master key is, never creating one.

    ``read_keyring=False`` stops short of reading the OS keyring: on macOS that read can
    raise a Keychain dialog, which a non-interactive caller (a script, a server) must not.
    """
    env_key = os.getenv(_MASTER_KEY_ENV, "").strip()
    if env_key:
        if is_fernet_key(env_key):
            return MasterKeyStatus("env", f"{_MASTER_KEY_ENV} is set")
        return MasterKeyStatus("invalid", f"{_MASTER_KEY_ENV} is set but is not a Fernet key")
    if not keyring_usable():
        return MasterKeyStatus("no_keyring", "no OS keyring on this host")
    if not read_keyring:
        return MasterKeyStatus("not_read", "an OS keyring is present; it was not read")
    try:
        stored = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
    except Exception as exc:  # noqa: BLE001 — keyring backends raise many types
        return MasterKeyStatus("no_keyring", f"the OS keyring could not be read ({exc})")
    if stored:
        return MasterKeyStatus("keyring", "in the OS keyring")
    return MasterKeyStatus("absent", "the OS keyring holds no IRIS master key")


def generate_master_key() -> str:
    """A new Fernet master key (not stored anywhere)."""
    return Fernet.generate_key().decode("utf-8")


class MasterKeyExistsError(VaultError):
    """Refused: a master key is already stored, and a stored key is never overwritten."""


def store_master_key_in_keyring(key: str) -> None:
    """Store ``key`` as the master key in the OS keyring, never replacing one.

    Every secret the vault holds is encrypted under the existing key, so overwriting it
    would make them unreadable. Raises :class:`MasterKeyExistsError` when an entry is
    there and :class:`VaultMasterKeyUnavailableError` when the keyring cannot be used.
    """
    if not is_fernet_key(key):
        raise ValueError("not a Fernet key")
    if not keyring_usable():
        raise VaultMasterKeyUnavailableError()
    try:
        existing = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
    except Exception as exc:  # keyring backends raise many types
        raise VaultMasterKeyUnavailableError() from exc
    if existing:
        raise MasterKeyExistsError("a master key is already in the OS keyring")
    try:
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, key)
        stored = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
    except Exception as exc:  # keyring backends raise many types
        raise VaultMasterKeyUnavailableError() from exc
    if stored != key:  # a backend that accepts writes and keeps nothing
        raise VaultMasterKeyUnavailableError()


MASTER_KEY_ENV = _MASTER_KEY_ENV
