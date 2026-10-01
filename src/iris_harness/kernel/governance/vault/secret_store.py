"""Pluggable secret store — OS keychain (default) or encrypted file vault.

ADR-0003 chose the OS keychain via ``keyring`` for OAuth tokens, on the
strength that ``keyring`` is already cross-platform (macOS Keychain,
Windows Credential Manager, Linux Secret Service). The finance build needs
to store a few more secrets — a PAN, a date of birth, per-institution PDF
passwords — and to keep working in headless/server/Docker contexts where
no desktop secret service exists. ``keyring`` degrades badly there.

So this module formalises a small ``SecretStore`` interface with two
backends behind one call site (the same swap-the-backend design
``credentials.py`` already anticipated):

* :class:`KeyringSecretStore` — default. Native OS keychain via the
  existing ``vault.credentials`` helpers. Zero config on desktops.
* :class:`FernetVaultStore` — an AES-128 (Fernet) encrypted SQLite file
  at ``data/secrets.db``. Portable (one file), works headless. The master
  key comes from a passphrase (``IRIS_VAULT_PASSPHRASE`` via scrypt) or,
  failing that, a random key kept in the OS keychain — so the bulk of
  secrets live in the portable file and only one small key touches the
  keychain.

Backend is chosen by ``IRIS_SECRET_BACKEND`` (``keyring`` | ``vault``);
default ``keyring``. Callers use :func:`get_secret_store` and never import
a backend directly.

See ADR-0032 (this module's shape + the ADR-0003 amendment).
"""

from __future__ import annotations

import base64
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from iris_harness.foundation.persistence import data_path
from iris_harness.kernel.governance.vault import credentials

# keyring "service" used to hold the vault's own master key.
_VAULT_KEYRING_NAMESPACE = "vault"
_VAULT_MASTER_ACCOUNT = "master-key"  # keychain account label, not a secret
_PASSPHRASE_ENV = "IRIS_VAULT_PASSPHRASE"  # noqa: S105 — env var name, not a secret
_BACKEND_ENV = "IRIS_SECRET_BACKEND"


@runtime_checkable
class SecretStore(Protocol):
    """Read/write/delete opaque string secrets, namespaced by domain."""

    def get(self, namespace: str, key: str) -> str | None: ...
    def set(self, namespace: str, key: str, value: str) -> None: ...
    def delete(self, namespace: str, key: str) -> None: ...


# ---------------------------------------------------------------------------
# Keyring backend
# ---------------------------------------------------------------------------


@dataclass
class KeyringSecretStore:
    """Native OS keychain backend (the ADR-0003 default), via credentials.py.

    ``namespace`` maps to the keyring *service* (``iris-<namespace>``) and
    ``key`` to the keyring *account* — the same scheme ``credentials``
    uses for OAuth, so finance secrets sit alongside tokens consistently.
    """

    def get(self, namespace: str, key: str) -> str | None:
        return credentials.load_token(namespace, key)

    def set(self, namespace: str, key: str, value: str) -> None:
        credentials.save_token(namespace, key, value)

    def delete(self, namespace: str, key: str) -> None:
        credentials.delete_token(namespace, key)


# ---------------------------------------------------------------------------
# Encrypted-file vault backend
# ---------------------------------------------------------------------------


_VAULT_SCHEMA = """
CREATE TABLE IF NOT EXISTS secrets (
    namespace   TEXT NOT NULL,
    key         TEXT NOT NULL,
    ciphertext  TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
);
"""


@dataclass
class FernetVaultStore:
    """Fernet-encrypted SQLite secret store — portable, headless-friendly.

    Values are encrypted with ``fernet`` before they touch disk; namespace
    and key are stored in clear (they are not secret — they're lookup
    coordinates). Supply ``fernet`` explicitly (tests / custom key
    management) or build one with :func:`build_fernet`.
    """

    fernet: Fernet
    db_path: Path = field(default_factory=lambda: data_path("secrets.db"))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Commit/rollback as ``with sqlite3.connect()`` did, and close the handle, which
        # it did not.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(_VAULT_SCHEMA)
            yield conn

    def get(self, namespace: str, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ciphertext FROM secrets WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
        if row is None:
            return None
        return self.fernet.decrypt(row["ciphertext"].encode()).decode()

    def set(self, namespace: str, key: str, value: str) -> None:
        token = self.fernet.encrypt(value.encode()).decode()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO secrets (namespace, key, ciphertext) VALUES (?, ?, ?) "
                "ON CONFLICT(namespace, key) DO UPDATE SET ciphertext = excluded.ciphertext",
                (namespace, key, token),
            )

    def delete(self, namespace: str, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM secrets WHERE namespace = ? AND key = ?", (namespace, key))


# ---------------------------------------------------------------------------
# Master-key derivation for the vault
# ---------------------------------------------------------------------------

# Fixed salt for passphrase derivation. A per-vault random salt would be
# marginally better, but it must itself be stored somewhere readable before
# the vault can be opened (chicken-and-egg). A constant app salt + a strong
# passphrase + scrypt's work factor is the standard pragmatic choice here;
# documented so it isn't mistaken for an oversight.
_SCRYPT_SALT = b"iris-finance-vault-v1"


def _fernet_key_from_passphrase(passphrase: str) -> bytes:
    kdf = Scrypt(salt=_SCRYPT_SALT, length=32, n=2**14, r=8, p=1)
    raw = kdf.derive(passphrase.encode())
    return base64.urlsafe_b64encode(raw)


def build_fernet(*, environ: Mapping[str, str] | None = None) -> Fernet:
    """Build the vault's Fernet from a passphrase, else a keychain-held key.

    Precedence:
      1. ``IRIS_VAULT_PASSPHRASE`` → scrypt-derived key (fully portable, no
         OS keychain needed).
      2. A random key auto-created and kept in the OS keychain (so only one
         small key depends on keyring; all secrets stay in the file).
    """
    env = environ if environ is not None else os.environ
    passphrase = env.get(_PASSPHRASE_ENV)
    if passphrase:
        return Fernet(_fernet_key_from_passphrase(passphrase))

    existing = credentials.load_token(_VAULT_KEYRING_NAMESPACE, _VAULT_MASTER_ACCOUNT)
    if existing:
        return Fernet(existing.encode())
    key = Fernet.generate_key()
    credentials.save_token(_VAULT_KEYRING_NAMESPACE, _VAULT_MASTER_ACCOUNT, key.decode())
    return Fernet(key)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_secret_store(*, environ: Mapping[str, str] | None = None) -> SecretStore:
    """Return the configured secret store. Default backend: ``keyring``.

    Set ``IRIS_SECRET_BACKEND=vault`` for the encrypted-file backend.
    """
    env = environ if environ is not None else os.environ
    backend = env.get(_BACKEND_ENV, "keyring").strip().lower()
    if backend == "vault":
        return FernetVaultStore(fernet=build_fernet(environ=env))
    if backend == "keyring":
        return KeyringSecretStore()
    raise ValueError(f"unknown {_BACKEND_ENV}={backend!r}; expected 'keyring' or 'vault'")


__all__ = [
    "SecretStore",
    "KeyringSecretStore",
    "FernetVaultStore",
    "build_fernet",
    "get_secret_store",
]
