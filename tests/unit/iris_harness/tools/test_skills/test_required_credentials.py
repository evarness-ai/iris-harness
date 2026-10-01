"""Phase 2 governance §7.1 — skill manifest required_credentials."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from cryptography.fernet import Fernet

from iris_harness.kernel.governance.vault import VaultStore, reset_vault_singleton_for_tests
from iris_harness.tools.skills.loader import load_skill_manifest, validate_skill_prerequisites
from iris_harness.tools.skills.models import RequiredCredential, SkillManifest, SkillRequirements


def _write_manifest(skill_dir: Path, *, body: dict[str, object]) -> Path:
    skill_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = skill_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return manifest_path


def _base_manifest() -> dict[str, object]:
    return {
        "name": "test-skill",
        "version": "0.1.0",
        "description": "test",
        "author": "iris",
        "license": "Apache-2.0",
        "tools": [
            {
                "name": "do_a_thing",
                "description": "demo",
                "governor_route": "system/read",
            }
        ],
    }


@pytest.fixture()
def master_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", key)
    reset_vault_singleton_for_tests()
    yield key
    reset_vault_singleton_for_tests()


def test_required_credential_validates_vault_prefix() -> None:
    with pytest.raises(ValueError, match="vault://"):
        RequiredCredential(handle="plain-handle")


def test_manifest_parses_required_credentials_block(tmp_path: Path) -> None:
    body = _base_manifest()
    body["requires"] = {
        "required_credentials": [
            {"handle": "vault://github-token", "route": "coding/git"},
            {"handle": "vault://openrouter-key"},
        ]
    }
    manifest_path = _write_manifest(tmp_path, body=body)

    manifest = load_skill_manifest(manifest_path.parent)
    assert isinstance(manifest, SkillManifest)
    handles = [c.handle for c in manifest.requires.required_credentials]
    assert handles == ["vault://github-token", "vault://openrouter-key"]
    assert manifest.requires.required_credentials[0].route == "coding/git"


def test_prerequisites_record_missing_credential_when_vault_empty(
    tmp_path: Path, master_key: str
) -> None:
    body = _base_manifest()
    body["requires"] = {"required_credentials": [{"handle": "vault://github-token"}]}
    _write_manifest(tmp_path, body=body)
    manifest = load_skill_manifest(tmp_path)

    # Vault exists but the handle is missing.
    import iris_harness.kernel.governance.vault.handle as handle_mod

    store = VaultStore(db_path=tmp_path / "vault.db")
    handle_mod._singleton = store

    try:
        missing = validate_skill_prerequisites(tmp_path, manifest)
        assert "credential:vault://github-token" in missing
    finally:
        reset_vault_singleton_for_tests()


def test_prerequisites_pass_when_vault_has_credential(tmp_path: Path, master_key: str) -> None:
    body = _base_manifest()
    body["requires"] = {"required_credentials": [{"handle": "vault://github-token"}]}
    _write_manifest(tmp_path, body=body)
    manifest = load_skill_manifest(tmp_path)

    import iris_harness.kernel.governance.vault.handle as handle_mod

    store = VaultStore(db_path=tmp_path / "vault.db")
    store.add(handle="github-token", secret_value="ghp_real")
    handle_mod._singleton = store

    try:
        missing = validate_skill_prerequisites(tmp_path, manifest)
        credential_misses = [m for m in missing if m.startswith("credential:")]
        assert credential_misses == []
    finally:
        reset_vault_singleton_for_tests()


def test_prerequisites_record_missing_credential_when_vault_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No vault available -> declared credentials are reported missing."""
    monkeypatch.delenv("IRIS_VAULT_MASTER_KEY", raising=False)
    reset_vault_singleton_for_tests()

    requires = SkillRequirements(
        required_credentials=(RequiredCredential(handle="vault://github-token"),)
    )
    manifest_body = _base_manifest()
    manifest = SkillManifest(
        name=str(manifest_body["name"]),
        version=str(manifest_body["version"]),
        description=str(manifest_body["description"]),
        author=str(manifest_body["author"]),
        license=str(manifest_body["license"]),
        tools=(),
        requires=requires,
    )

    missing = validate_skill_prerequisites(tmp_path, manifest)
    assert "credential:vault://github-token" in missing
