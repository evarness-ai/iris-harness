"""IRIS credential vault primitives."""

from iris_harness.kernel.governance.vault.handle import (
    VAULT_PREFIX,
    VaultHandleNotFoundError,
    is_vault_handle,
    reset_vault_singleton_for_tests,
    resolve_secret_value,
)
from iris_harness.kernel.governance.vault.store import (
    VaultCorruptionError,
    VaultError,
    VaultHandleAlreadyExistsError,
    VaultMasterKeyUnavailableError,
    VaultSecretMetadata,
    VaultStore,
)

__all__ = [
    "VAULT_PREFIX",
    "VaultCorruptionError",
    "VaultError",
    "VaultHandleAlreadyExistsError",
    "VaultHandleNotFoundError",
    "VaultMasterKeyUnavailableError",
    "VaultSecretMetadata",
    "VaultStore",
    "is_vault_handle",
    "reset_vault_singleton_for_tests",
    "resolve_secret_value",
]
