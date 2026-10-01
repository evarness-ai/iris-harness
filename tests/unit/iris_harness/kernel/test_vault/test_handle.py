from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from iris_harness.kernel.governance.vault import (
    VaultHandleNotFoundError,
    VaultStore,
    is_vault_handle,
    reset_vault_singleton_for_tests,
    resolve_secret_value,
)


@pytest.fixture()
def master_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", key)
    reset_vault_singleton_for_tests()
    yield key
    reset_vault_singleton_for_tests()


def test_is_vault_handle_recognizes_prefix() -> None:
    assert is_vault_handle("vault://github-token") is True
    assert is_vault_handle("ghp_abc") is False
    assert is_vault_handle(None) is False
    assert is_vault_handle("") is False


def test_plain_values_pass_through_unchanged(master_key: str) -> None:
    assert resolve_secret_value("ghp_plain_value") == "ghp_plain_value"
    assert resolve_secret_value(None) is None
    assert resolve_secret_value("") == ""


def test_vault_handle_resolves_to_stored_secret(tmp_path: Path, master_key: str) -> None:
    store = VaultStore(db_path=tmp_path / "vault.db")
    store.add(handle="github-token", secret_value="ghp_real_value")

    assert resolve_secret_value("vault://github-token", vault=store) == "ghp_real_value"


def test_unknown_handle_raises_typed_error(tmp_path: Path, master_key: str) -> None:
    store = VaultStore(db_path=tmp_path / "vault.db")

    with pytest.raises(VaultHandleNotFoundError) as excinfo:
        resolve_secret_value("vault://does-not-exist", vault=store)
    assert excinfo.value.handle == "vault://does-not-exist"


def test_vault_unavailable_raises_for_handle_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_VAULT_MASTER_KEY", raising=False)
    reset_vault_singleton_for_tests()

    # Force singleton construction to fail by monkeypatching VaultStore.
    import iris_harness.kernel.governance.vault.handle as handle_module

    def _broken_store() -> None:
        raise RuntimeError("no master key")

    monkeypatch.setattr(handle_module, "VaultStore", lambda: _broken_store())

    # Plain values still pass.
    assert resolve_secret_value("plain") == "plain"
    # Handles raise a clear error.
    with pytest.raises(RuntimeError, match="vault is not available"):
        resolve_secret_value("vault://nope")
