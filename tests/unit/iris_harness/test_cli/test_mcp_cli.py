"""iris mcp keygen|sign|verify CLI — Phase 6 sub-phase 6b.3."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from iris_harness.cli.mcp import mcp_signing_app as mcp_app

runner = CliRunner()


def _write_servers(path: Path, *, signed: dict[str, str] | None = None) -> None:
    server: dict[str, object] = {
        "name": "remote",
        "enabled": True,
        "transport": "http",
        "url": "http://127.0.0.1:8787/mcp",
    }
    if signed:
        server.update(signed)
    path.write_text(
        yaml.safe_dump({"version": "1", "enabled": True, "servers": [server]}), encoding="utf-8"
    )


def _trust(path: Path) -> Path:
    return path


def test_keygen_writes_private_key_and_trust_entry(tmp_path: Path) -> None:
    trust = tmp_path / "trust.yaml"
    keys_dir = tmp_path / "keys"
    result = runner.invoke(
        mcp_app,
        ["keygen", "--id", "ops", "--trust-store", str(trust)],
        env={"IRIS_MCP_KEYS_DIR": str(keys_dir)},
    )
    assert result.exit_code == 0, result.output
    key_file = keys_dir / "ops.key"
    assert key_file.exists()
    assert oct(key_file.stat().st_mode)[-3:] == "600"
    doc = yaml.safe_load(trust.read_text())
    assert doc["keys"][0]["key_id"] == "ops"
    assert doc["keys"][0]["public_key"]


def test_keygen_rejects_duplicate_id(tmp_path: Path) -> None:
    trust = tmp_path / "trust.yaml"
    keys_dir = tmp_path / "keys"
    env = {"IRIS_MCP_KEYS_DIR": str(keys_dir)}
    runner.invoke(mcp_app, ["keygen", "--id", "ops", "--trust-store", str(trust)], env=env)
    second = runner.invoke(mcp_app, ["keygen", "--id", "ops", "--trust-store", str(trust)], env=env)
    assert second.exit_code == 1


def test_sign_then_verify_roundtrip(tmp_path: Path) -> None:
    trust = tmp_path / "trust.yaml"
    keys_dir = tmp_path / "keys"
    servers = tmp_path / "mcp-servers.yaml"
    env = {"IRIS_MCP_KEYS_DIR": str(keys_dir)}
    _write_servers(servers)

    assert (
        runner.invoke(
            mcp_app, ["keygen", "--id", "ops", "--trust-store", str(trust)], env=env
        ).exit_code
        == 0
    )
    sign_res = runner.invoke(
        mcp_app, ["sign", "remote", "--key", "ops", "--servers", str(servers)], env=env
    )
    assert sign_res.exit_code == 0, sign_res.output

    doc = yaml.safe_load(servers.read_text())
    assert doc["servers"][0]["signature"]
    assert doc["servers"][0]["signed_by"] == "ops"

    verify_res = runner.invoke(
        mcp_app,
        ["verify", "remote", "--servers", str(servers), "--trust-store", str(trust)],
    )
    assert verify_res.exit_code == 0, verify_res.output
    assert "verified" in verify_res.output


def test_verify_unsigned_exits_nonzero(tmp_path: Path) -> None:
    servers = tmp_path / "mcp-servers.yaml"
    trust = tmp_path / "trust.yaml"
    trust.write_text("version: 1\nkeys: []\n", encoding="utf-8")
    _write_servers(servers)
    result = runner.invoke(
        mcp_app, ["verify", "--servers", str(servers), "--trust-store", str(trust)]
    )
    assert result.exit_code == 1
    assert "unsigned" in result.output


def test_sign_unknown_server_errors(tmp_path: Path) -> None:
    servers = tmp_path / "mcp-servers.yaml"
    keys_dir = tmp_path / "keys"
    trust = tmp_path / "trust.yaml"
    env = {"IRIS_MCP_KEYS_DIR": str(keys_dir)}
    _write_servers(servers)
    runner.invoke(mcp_app, ["keygen", "--id", "ops", "--trust-store", str(trust)], env=env)
    result = runner.invoke(
        mcp_app, ["sign", "ghost", "--key", "ops", "--servers", str(servers)], env=env
    )
    assert result.exit_code == 1


def test_verify_detects_tampering_after_signing(tmp_path: Path) -> None:
    trust = tmp_path / "trust.yaml"
    keys_dir = tmp_path / "keys"
    servers = tmp_path / "mcp-servers.yaml"
    env = {"IRIS_MCP_KEYS_DIR": str(keys_dir)}
    _write_servers(servers)
    runner.invoke(mcp_app, ["keygen", "--id", "ops", "--trust-store", str(trust)], env=env)
    runner.invoke(mcp_app, ["sign", "remote", "--key", "ops", "--servers", str(servers)], env=env)
    # Tamper with the signed config: swap the URL.
    doc = yaml.safe_load(servers.read_text())
    doc["servers"][0]["url"] = "http://evil.test/mcp"
    servers.write_text(yaml.safe_dump(doc), encoding="utf-8")

    result = runner.invoke(
        mcp_app, ["verify", "--servers", str(servers), "--trust-store", str(trust)]
    )
    assert result.exit_code == 1
    assert "invalid" in result.output
