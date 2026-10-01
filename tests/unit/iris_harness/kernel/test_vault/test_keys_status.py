"""The master key's read-only status and its never-overwriting store (``iris doctor``).

Every test runs on the suite's in-memory keyring (``tests/conftest.py``), which starts
holding one generated master key; nothing here can reach the OS keyring.
"""

from __future__ import annotations

import keyring
import pytest
from cryptography.fernet import Fernet
from keyring.backends import fail

from iris_harness.kernel.governance.vault import keys

_ENTRY = ("iris-vault", "master-key")


def _backend_entries() -> dict[tuple[str, str], str]:
    return keyring.get_keyring().entries  # type: ignore[attr-defined,no-any-return]


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(keys.MASTER_KEY_ENV, raising=False)


def test_env_key_wins_and_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(keys.MASTER_KEY_ENV, Fernet.generate_key().decode())
    assert keys.master_key_status(read_keyring=True).source == "env"

    monkeypatch.setenv(keys.MASTER_KEY_ENV, "not-a-key")
    status = keys.master_key_status(read_keyring=True)
    assert status.source == "invalid"
    assert not status.present


def test_keyring_key_found_only_when_reading_is_allowed() -> None:
    assert keys.master_key_status(read_keyring=True).source == "keyring"
    # Not reading is a distinct answer, never a guess.
    assert keys.master_key_status(read_keyring=False).source == "not_read"


def test_status_never_seeds_the_keyring() -> None:
    _backend_entries().clear()
    assert keys.master_key_status(read_keyring=True).source == "absent"
    assert _ENTRY not in _backend_entries()  # resolve_master_key would have seeded one


def test_no_usable_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    keyring.set_keyring(fail.Keyring())
    assert not keys.keyring_usable()
    assert keys.master_key_status(read_keyring=True).source == "no_keyring"

    monkeypatch.setattr(keys, "keyring", None)
    assert keys.master_key_status(read_keyring=True).source == "no_keyring"


def test_a_keyring_that_fails_when_read_is_no_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(service: str, username: str) -> str:
        raise RuntimeError("no D-Bus session")

    monkeypatch.setattr(keyring.get_keyring(), "get_password", boom)
    status = keys.master_key_status(read_keyring=True)
    assert status.source == "no_keyring"
    assert "D-Bus" in status.detail


def test_store_writes_a_new_key() -> None:
    _backend_entries().clear()
    key = keys.generate_master_key()
    keys.store_master_key_in_keyring(key)
    assert _backend_entries()[_ENTRY] == key


def test_store_never_overwrites_an_existing_key() -> None:
    existing = _backend_entries()[_ENTRY]
    with pytest.raises(keys.MasterKeyExistsError):
        keys.store_master_key_in_keyring(keys.generate_master_key())
    assert _backend_entries()[_ENTRY] == existing


def test_store_refuses_without_a_keyring_or_a_real_key() -> None:
    with pytest.raises(ValueError):
        keys.store_master_key_in_keyring("not-a-key")
    keyring.set_keyring(fail.Keyring())
    with pytest.raises(keys.VaultMasterKeyUnavailableError):
        keys.store_master_key_in_keyring(keys.generate_master_key())
