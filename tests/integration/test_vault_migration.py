"""End-to-end migration smoke test for the Phase 2 vault flow.

Covers the operator path that an operator runs in `docs/usage-guides/
vault-operations.md`:

1. Start with raw secrets in `.env`-equivalent env vars.
2. Run `iris vault import-env` -> handles end up in the vault.
3. Replace the env value with `vault://<handle>` -> LLM client still
   resolves the real secret via the vault.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from typer.testing import CliRunner

from iris_harness.kernel.governance.vault import VaultStore, reset_vault_singleton_for_tests
from iris_harness.llm.client import CodingLLMConfig, resolve_coding_llm_api_key
from iris_harness.main import app


@pytest.fixture()
def isolated_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[VaultStore]:
    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", Fernet.generate_key().decode("utf-8"))
    db_path = tmp_path / "vault.db"
    reset_vault_singleton_for_tests()

    # Force the singleton (and the CLI) to use this path.
    import iris_harness.kernel.governance.vault.handle as handle_mod
    import iris_harness.kernel.governance.vault.store as store_mod

    store = VaultStore(db_path=db_path)
    handle_mod._singleton = store
    monkeypatch.setattr(store_mod, "_DEFAULT_DB_PATH", db_path)

    try:
        yield store
    finally:
        reset_vault_singleton_for_tests()


def test_full_migration_path(isolated_vault: VaultStore, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    # Step 1: raw secret in the env, vault is empty.
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_real_value")
    assert isolated_vault.list_metadata() == []

    # Step 2: import-env moves it into the vault.
    result = runner.invoke(app, ["vault", "import-env", "--name", "GITHUB_TOKEN"])
    assert result.exit_code == 0, result.output
    rows = isolated_vault.list_metadata()
    assert [r.handle for r in rows] == ["vault://github-token"]
    assert isolated_vault.get("vault://github-token") == "ghp_real_value"

    # Step 3: operator flips env var to the vault handle; resolution still works.
    monkeypatch.setenv("GITHUB_TOKEN", "vault://github-token")
    config = CodingLLMConfig(
        provider="github",
        model="gpt-4o",
        base_url="https://models.inference.ai.azure.com",
        api_key_env="GITHUB_TOKEN",
    )
    resolved = resolve_coding_llm_api_key(config)
    assert resolved == "ghp_real_value"

    # Re-running import-env is idempotent: conflict-on-existing is counted, not raised.
    result = runner.invoke(app, ["vault", "import-env", "--name", "GITHUB_TOKEN"])
    assert result.exit_code == 0
    assert "conflicts=1" in result.output


def test_export_writes_plaintext_with_warning(isolated_vault: VaultStore, tmp_path: Path) -> None:
    runner = CliRunner()
    isolated_vault.add(handle="x", secret_value="hello")

    output = tmp_path / "snapshot.env"
    result = runner.invoke(app, ["vault", "export", "--output", str(output)])
    assert result.exit_code == 0, result.output

    contents = output.read_text(encoding="utf-8")
    assert 'X="hello"' in contents
    assert "PLAINTEXT" in contents  # safety warning in the file header
