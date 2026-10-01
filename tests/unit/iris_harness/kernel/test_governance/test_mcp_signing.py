"""MCP server signing core — Phase 6 sub-phase 6b.1."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.kernel.governance.mcp_signing import (
    MCPSigningConfig,
    ServerSpec,
    TrustedKey,
    TrustStore,
    file_sha256,
    generate_keypair,
    public_key_for,
    resolve_signing_action,
    sign,
    verify,
    verify_server_signature,
)


def _spec(**overrides: object) -> ServerSpec:
    base: dict[str, object] = {
        "name": "filesystem",
        "transport": "stdio",
        "command": "/usr/local/bin/mcp-fs",
        "args": ("--root", "/data"),
        "env_keys": ("TOKEN", "DEBUG"),
    }
    base.update(overrides)
    return ServerSpec(**base)  # type: ignore[arg-type]


def _signed(spec: ServerSpec, key_id: str = "ops") -> tuple[str, str, TrustStore]:
    priv, pub = generate_keypair()
    signature = sign(priv, spec.canonical_bytes())
    store = TrustStore(keys={key_id: TrustedKey(key_id=key_id, public_key=pub)})
    return signature, key_id, store


# --- keys + canonical payload ------------------------------------------------


def test_sign_verify_roundtrip() -> None:
    priv, pub = generate_keypair()
    payload = b"hello mcp"
    assert verify(pub, payload, sign(priv, payload)) is True


def test_verify_rejects_wrong_payload() -> None:
    priv, pub = generate_keypair()
    assert verify(pub, b"tampered", sign(priv, b"original")) is False


def test_verify_rejects_wrong_key() -> None:
    priv, _ = generate_keypair()
    _, other_pub = generate_keypair()
    assert verify(other_pub, b"x", sign(priv, b"x")) is False


def test_verify_failsafe_on_garbage() -> None:
    assert verify("not-base64!!", b"x", "also-garbage") is False


def test_public_key_for_matches_keygen() -> None:
    priv, pub = generate_keypair()
    assert public_key_for(priv) == pub


def test_canonical_bytes_deterministic_and_env_order_insensitive() -> None:
    a = _spec(env_keys=("TOKEN", "DEBUG"))
    b = _spec(env_keys=("DEBUG", "TOKEN"))
    assert a.canonical_bytes() == b.canonical_bytes()  # env-key order ignored


def test_canonical_bytes_changes_with_command() -> None:
    assert _spec().canonical_bytes() != _spec(command="/usr/local/bin/evil").canonical_bytes()


def test_canonical_bytes_changes_with_arg_order() -> None:
    a = _spec(args=("--root", "/data"))
    b = _spec(args=("/data", "--root"))
    assert a.canonical_bytes() != b.canonical_bytes()  # arg order is significant


def test_file_sha256(tmp_path: Path) -> None:
    p = tmp_path / "bin"
    p.write_bytes(b"binary-content")
    digest = file_sha256(p)
    assert digest is not None and digest.startswith("sha256:")
    assert file_sha256(tmp_path / "missing") is None


# --- trust store -------------------------------------------------------------


def test_trust_store_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "trust.yaml"
    p.write_text(
        "version: 1\n"
        "keys:\n"
        "  - {key_id: ops, public_key: AAAA, comment: primary}\n"
        "revoked_key_ids: [old]\n",
        encoding="utf-8",
    )
    store = TrustStore.from_yaml(p)
    assert store.public_key_for("ops") == "AAAA"
    assert store.public_key_for("old") is None  # revoked
    assert store.public_key_for("nope") is None  # unknown


def test_trust_store_absent_is_empty(tmp_path: Path) -> None:
    store = TrustStore.from_yaml(tmp_path / "nope.yaml")
    assert store.public_key_for("anything") is None


def test_trust_store_rejects_duplicate_key_id(tmp_path: Path) -> None:
    p = tmp_path / "dup.yaml"
    p.write_text(
        "keys:\n" "  - {key_id: a, public_key: X}\n" "  - {key_id: a, public_key: Y}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        TrustStore.from_yaml(p)


def test_trust_store_revoked_signature() -> None:
    store = TrustStore(revoked_signatures=frozenset({"badsig"}))
    assert store.is_revoked(key_id="ops", signature="badsig") is True


# --- config ------------------------------------------------------------------


def test_packaged_config_is_shadow() -> None:
    cfg = MCPSigningConfig.from_yaml(
        Path(__file__).resolve().parents[5] / "config" / "governance" / "mcp-signing.yaml"
    )
    assert cfg.enabled is True
    assert cfg.mode == "shadow"
    assert cfg.unsigned_policy == "warn"


def test_config_absent_is_disabled(tmp_path: Path) -> None:
    assert MCPSigningConfig.from_yaml(tmp_path / "x.yaml").enabled is False


def test_config_rejects_unknown_key(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("version: 1\nbogus: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        MCPSigningConfig.from_yaml(p)


# --- verify_server_signature -------------------------------------------------


def test_verdict_verified() -> None:
    spec = _spec()
    sig, key_id, store = _signed(spec)
    verdict = verify_server_signature(spec=spec, signature=sig, signed_by=key_id, trust_store=store)
    assert verdict.status == "verified"
    assert verdict.is_trusted is True


def test_verdict_unsigned() -> None:
    verdict = verify_server_signature(
        spec=_spec(), signature=None, signed_by=None, trust_store=TrustStore.empty()
    )
    assert verdict.status == "unsigned"


def test_verdict_untrusted_unknown_signer() -> None:
    spec = _spec()
    sig, _key_id, _store = _signed(spec)
    verdict = verify_server_signature(
        spec=spec, signature=sig, signed_by="ghost", trust_store=TrustStore.empty()
    )
    assert verdict.status == "untrusted"


def test_verdict_untrusted_revoked_signer() -> None:
    spec = _spec()
    sig, key_id, store = _signed(spec)
    revoked = TrustStore(keys=store.keys, revoked_key_ids=frozenset({key_id}))
    verdict = verify_server_signature(
        spec=spec, signature=sig, signed_by=key_id, trust_store=revoked
    )
    assert verdict.status == "untrusted"


def test_verdict_invalid_on_tampered_spec() -> None:
    signed_spec = _spec()
    sig, key_id, store = _signed(signed_spec)
    tampered = _spec(command="/usr/local/bin/evil")  # attacker swapped the binary
    verdict = verify_server_signature(
        spec=tampered, signature=sig, signed_by=key_id, trust_store=store
    )
    assert verdict.status == "invalid"


# --- resolve_signing_action --------------------------------------------------


def _verdict(status: str):
    from iris_harness.kernel.governance.mcp_signing import SignatureVerdict

    return SignatureVerdict(status=status, reason="t")  # type: ignore[arg-type]


def test_action_disabled_allows() -> None:
    cfg = MCPSigningConfig.disabled()
    assert resolve_signing_action(_verdict("invalid"), cfg) == "allow"


def test_action_verified_allows() -> None:
    cfg = MCPSigningConfig(mode="enforce", unsigned_policy="deny")
    assert resolve_signing_action(_verdict("verified"), cfg) == "allow"


def test_action_shadow_downgrades_to_warn() -> None:
    cfg = MCPSigningConfig(mode="shadow", unsigned_policy="deny")
    assert resolve_signing_action(_verdict("invalid"), cfg) == "warn"


def test_action_enforce_honors_policy() -> None:
    cfg = MCPSigningConfig(mode="enforce", unsigned_policy="deny")
    assert resolve_signing_action(_verdict("unsigned"), cfg) == "deny"
    cfg_allow = MCPSigningConfig(mode="enforce", unsigned_policy="allow")
    assert resolve_signing_action(_verdict("unsigned"), cfg_allow) == "allow"
