"""Local credential vault for governance Phase 2.

Secrets are stored in a local SQLite DB with values encrypted using a
Fernet symmetric key (AES-128-CBC + HMAC-SHA256). The master key is
resolved from either:

1. ``IRIS_VAULT_MASTER_KEY`` env var (base64 Fernet key), or
2. OS keyring entry ``service=iris-vault``, ``username=master-key``.

Why Fernet+SQLite instead of sqlcipher
--------------------------------------

The design doc (``docs/architecture/unified-governance-layer.md`` §7.1)
calls for sqlcipher. We ship Fernet-over-SQLite instead for two
reasons:

- sqlcipher requires a native ``libsqlcipher`` build that isn't always
  available on developer macOS / Linux installs.
- Fernet over a SQLite blob column gives the same at-rest property
  (an attacker who steals ``vault.db`` cannot decrypt without the
  master key) with one fewer system dependency.

This decision is captured in the design doc's §7.1 note. If we ever
need full-page encryption (e.g. column-level search over encrypted
fields) the migration path is a one-shot rewrite of this module.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from iris_harness.foundation.paths import governance_config_dir
from iris_harness.kernel.governance.vault.keys import (
    VaultError,
    VaultMasterKeyUnavailableError,
    resolve_master_key,
)

__all__ = [
    "VaultCorruptionError",
    "VaultError",
    "VaultHandleAlreadyExistsError",
    "VaultMasterKeyUnavailableError",
    "VaultSecretMetadata",
    "VaultStore",
]

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = governance_config_dir() / "vault.db"


class VaultHandleAlreadyExistsError(VaultError):
    """Attempted to add a handle that is already stored (without replace=True)."""

    def __init__(self, handle: str) -> None:
        super().__init__(f"secret handle already exists: {handle}")
        self.handle = handle


class VaultCorruptionError(VaultError):
    """Stored ciphertext is not bytes or cannot be decrypted with the master key."""

    def __init__(self, handle: str, reason: str) -> None:
        super().__init__(f"vault corruption for handle {handle!r}: {reason}")
        self.handle = handle


@dataclass(frozen=True)
class VaultSecretMetadata:
    handle: str
    governor_route: str | None
    created_at: str
    updated_at: str


class VaultStore:
    """Encrypted local vault with handle-based lookup."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(resolve_master_key())
        self._init_schema()

    def add(
        self,
        *,
        handle: str,
        secret_value: str,
        governor_route: str | None = None,
        replace: bool = False,
    ) -> None:
        normalized = self._normalize_handle(handle)
        if not secret_value.strip():
            raise ValueError("secret_value must be non-empty")

        now = datetime.now(UTC).isoformat()
        encrypted = self._fernet.encrypt(secret_value.encode("utf-8"))
        with self._connect() as conn:
            if not replace:
                existing = conn.execute(
                    "SELECT 1 FROM vault_secrets WHERE handle = ?",
                    (normalized,),
                ).fetchone()
                if existing is not None:
                    raise VaultHandleAlreadyExistsError(normalized)
                conn.execute(
                    """
                    INSERT INTO vault_secrets(handle, ciphertext, governor_route, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (normalized, encrypted, governor_route, now, now),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO vault_secrets(handle, ciphertext, governor_route, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(handle) DO UPDATE SET
                        ciphertext=excluded.ciphertext,
                        governor_route=excluded.governor_route,
                        updated_at=excluded.updated_at
                    """,
                    (normalized, encrypted, governor_route, now, now),
                )
            conn.commit()

    def get(self, handle: str) -> str | None:
        normalized = self._normalize_handle(handle)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ciphertext FROM vault_secrets WHERE handle = ?",
                (normalized,),
            ).fetchone()
        if row is None:
            return None
        ciphertext = row[0]
        if not isinstance(ciphertext, bytes):
            raise VaultCorruptionError(normalized, "ciphertext is not bytes")
        try:
            return self._fernet.decrypt(ciphertext).decode("utf-8")
        except InvalidToken as exc:
            raise VaultCorruptionError(
                normalized,
                "ciphertext cannot be decrypted with the current master key "
                "(possible key rotation without re-encryption)",
            ) from exc

    def list_metadata(self) -> list[VaultSecretMetadata]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT handle, governor_route, created_at, updated_at
                FROM vault_secrets
                ORDER BY handle ASC
                """).fetchall()
        return [
            VaultSecretMetadata(
                handle=str(row[0]),
                governor_route=str(row[1]) if row[1] is not None else None,
                created_at=str(row[2]),
                updated_at=str(row[3]),
            )
            for row in rows
        ]

    def remove(self, handle: str) -> bool:
        normalized = self._normalize_handle(handle)
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM vault_secrets WHERE handle = ?",
                (normalized,),
            )
            conn.commit()
        return cursor.rowcount > 0

    def iter_secret_values(self) -> list[tuple[str, str]]:
        """Return ``[(handle, decrypted_value), ...]`` for redaction scans.

        Rows that fail to decrypt are skipped with a WARN log — a single
        corrupted entry must not poison the redaction pass and let raw
        secrets escape into a prompt.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT handle, ciphertext FROM vault_secrets ORDER BY handle ASC"
            ).fetchall()
        out: list[tuple[str, str]] = []
        for row in rows:
            handle = str(row[0])
            ciphertext = row[1]
            if not isinstance(ciphertext, bytes):
                logger.warning("vault row %s has non-bytes ciphertext; skipping", handle)
                continue
            try:
                value = self._fernet.decrypt(ciphertext).decode("utf-8")
            except InvalidToken:
                logger.warning(
                    "vault row %s cannot be decrypted with the current master key; skipping",
                    handle,
                )
                continue
            out.append((handle, value))
        return out

    @staticmethod
    def _normalize_handle(handle: str) -> str:
        raw = handle.strip()
        if not raw:
            raise ValueError("handle must be non-empty")
        return raw if raw.startswith("vault://") else f"vault://{raw}"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a sqlite connection and reassert ``0o600`` on the DB file.

        Reasserting per-open defends against an operator (or a buggy
        backup tool) widening the perms after creation.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            try:
                os.chmod(self.db_path, 0o600)
            except OSError as exc:  # pragma: no cover - non-POSIX or perms issue
                logger.warning("could not chmod %s to 0o600: %s", self.db_path, exc)
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vault_secrets (
                    handle TEXT PRIMARY KEY,
                    ciphertext BLOB NOT NULL,
                    governor_route TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)
            conn.commit()
