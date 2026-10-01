"""Where a plugin keeps a credential or a secret.

`save_token` / `load_token` / `delete_token` keep a provider's OAuth blob (keyed by
provider and account) in the OS keychain through ``keyring``, never in a plugin's own
file; `CredentialRevokedError` is what a refresh raises when the owner has revoked
access. A `SecretStore` from `get_secret_store()` holds any other secret (an app
password, an API key) in the OS keyring or the configured backend.

Every governed call needs the vault master key (#741: no key, no governed tool call).
A setup flow checks it with `master_key_status(read_keyring=...)` (a `MasterKeyStatus`;
``read_keyring=False`` never raises a Keychain dialog, for a server or a script) and,
when the owner agrees, creates one with `fix_master_key()` -- ``iris doctor --fix``'s
own function: it never replaces a key, and returns a `KeyFix` saying what it did (an
``export`` line to persist when the host has no keyring).
"""

from __future__ import annotations

from iris_harness.kernel.governance.vault.credentials import (
    CredentialRevokedError,
    delete_token,
    load_token,
    save_token,
)
from iris_harness.kernel.governance.vault.keys import MasterKeyStatus, master_key_status
from iris_harness.kernel.governance.vault.secret_store import SecretStore, get_secret_store
from iris_harness.services.system.doctor import KeyFix, fix_master_key

__all__ = [
    "CredentialRevokedError",
    "KeyFix",
    "MasterKeyStatus",
    "SecretStore",
    "delete_token",
    "fix_master_key",
    "get_secret_store",
    "load_token",
    "master_key_status",
    "save_token",
]
