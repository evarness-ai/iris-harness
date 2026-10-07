from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from iris_harness.kernel.governance.vault import (
    VaultCorruptionError,
    VaultHandleAlreadyExistsError,
    VaultMasterKeyUnavailableError,
    VaultStore,
)


@pytest.fixture()
def master_key(monkeypatch: pytest.MonkeyPatch) -> str:
    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", key)
    return key


def test_add_get_list_remove_roundtrip(tmp_path: Path, master_key: str) -> None:
    store = VaultStore(db_path=tmp_path / "vault.db")
    store.add(handle="github-token", secret_value="ghp_123", governor_route="coding/git")

    assert store.get("vault://github-token") == "ghp_123"
    rows = store.list_metadata()
    assert len(rows) == 1
    assert rows[0].handle == "vault://github-token"
    assert rows[0].governor_route == "coding/git"

    assert store.remove("github-token") is True
    assert store.get("github-token") is None


def test_add_without_replace_raises_typed_error(tmp_path: Path, master_key: str) -> None:
    store = VaultStore(db_path=tmp_path / "vault.db")
    store.add(handle="vault://openrouter", secret_value="sk-or-v1-abc")

    with pytest.raises(VaultHandleAlreadyExistsError) as excinfo:
        store.add(handle="openrouter", secret_value="sk-or-v1-new")
    assert excinfo.value.handle == "vault://openrouter"


def test_add_with_replace_updates_value(tmp_path: Path, master_key: str) -> None:
    store = VaultStore(db_path=tmp_path / "vault.db")
    store.add(handle="openai", secret_value="sk-old")
    store.add(handle="openai", secret_value="sk-new", replace=True)

    assert store.get("openai") == "sk-new"


def test_iter_secret_values_returns_decrypted_pairs(tmp_path: Path, master_key: str) -> None:
    store = VaultStore(db_path=tmp_path / "vault.db")
    store.add(handle="a", secret_value="one")
    store.add(handle="b", secret_value="two")

    assert store.iter_secret_values() == [
        ("vault://a", "one"),
        ("vault://b", "two"),
    ]


def test_db_file_is_chmod_0o600_on_create(tmp_path: Path, master_key: str) -> None:
    db_path = tmp_path / "vault.db"
    VaultStore(db_path=db_path)

    mode = stat.S_IMODE(os.stat(db_path).st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_db_file_perms_are_reasserted_on_every_open(tmp_path: Path, master_key: str) -> None:
    """An operator who accidentally widens perms gets them tightened back."""
    db_path = tmp_path / "vault.db"
    store = VaultStore(db_path=db_path)
    widened = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    os.chmod(db_path, widened)  # operator goofs
    assert stat.S_IMODE(os.stat(db_path).st_mode) == widened

    # Any read/write reasserts 0o600.
    store.add(handle="x", secret_value="y")
    assert stat.S_IMODE(os.stat(db_path).st_mode) == 0o600


def test_missing_master_key_raises_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env, no keyring -> typed error with actionable hint."""
    monkeypatch.delenv("IRIS_VAULT_MASTER_KEY", raising=False)
    import iris_harness.kernel.governance.vault.keys as keys_mod

    monkeypatch.setattr(keys_mod, "keyring", None)

    with pytest.raises(VaultMasterKeyUnavailableError) as excinfo:
        VaultStore(db_path=tmp_path / "vault.db")
    assert "IRIS_VAULT_MASTER_KEY" in str(excinfo.value)


def test_rotation_without_re_encryption_surfaces_as_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows encrypted under key A cannot be read after rotating to key B."""
    key_a = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", key_a)
    db_path = tmp_path / "vault.db"
    VaultStore(db_path=db_path).add(handle="x", secret_value="hello")

    # Rotate to a different master key without re-encrypting existing rows.
    key_b = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", key_b)
    rotated = VaultStore(db_path=db_path)

    with pytest.raises(VaultCorruptionError):
        rotated.get("x")

    # iter_secret_values skips undecryptable rows (must never break redaction).
    assert rotated.iter_secret_values() == []
