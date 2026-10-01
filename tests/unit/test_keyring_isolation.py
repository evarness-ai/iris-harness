"""No test can reach the developer's OS keyring (tests/conftest.py).

Regression: with IRIS_VAULT_MASTER_KEY stripped, the vault fell back to the macOS login
Keychain, so the suite read the real `iris-vault` master key — found when a fresh
python3.13 env made macOS ask for the login password (2026-09-26).
"""

from __future__ import annotations

import os
import subprocess
import sys

import keyring

from iris_harness.kernel.governance.vault import keys

_SEEN_KEYS: list[str] = []


def test_the_keyring_a_test_sees_is_the_in_memory_one() -> None:
    backend = keyring.get_keyring()
    assert type(backend).__name__ == "_TestKeyring"
    # Nothing in it but the fixture's freshly generated vault master key (tests/conftest.py).
    assert set(backend.entries) == {("iris-vault", "master-key")}  # type: ignore[attr-defined]


def test_the_vault_seeds_its_key_into_the_test_keyring_not_the_os(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("IRIS_VAULT_MASTER_KEY", raising=False)
    keyring.delete_password("iris-vault", "master-key")

    key = keys.resolve_master_key()

    assert keyring.get_password("iris-vault", "master-key") == key.decode()
    _SEEN_KEYS.append(key.decode())


def test_each_test_starts_with_its_own_master_key() -> None:
    # The previous test seeded a master key; this one must not see it.
    key = keyring.get_password("iris-vault", "master-key")
    assert key is not None and key not in _SEEN_KEYS


def test_a_subprocess_gets_a_keyring_that_refuses() -> None:
    assert os.environ["PYTHON_KEYRING_BACKEND"] == "keyring.backends.fail.Keyring"
    cmd = [sys.executable, "-c", "import keyring; print(type(keyring.get_keyring()).__module__)"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
    assert out.stdout.strip() == "keyring.backends.fail"
