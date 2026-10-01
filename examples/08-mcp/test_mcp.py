"""MCP both ways, offline: IRIS serving its tools, and an external server made trustworthy.

1. **Serve** -- ``iris mcp serve`` over stdio, driven by a plain MCP client
   (``mcp_client.py``): the handshake, the tool list, tool calls -- each one governed and
   audited as ``mcp:example``, a destructive one refused, nothing run without an audit key.
2. **Consume** -- an external server (``weather_server.py``) is declared in an allowlist
   (``mcp-servers.yaml``) and signed with ``iris mcp keygen`` / ``sign``; ``verify``
   accepts it, and rejects it once its entry changes.

Every IRIS command runs as a child process in a temporary home.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from mcp_client import McpError, StdioMcpClient

from iris_harness.sdk.audit import AuditLog

HERE = Path(__file__).parent


def ledger_path(home: Path) -> Path:
    return home / ".iris" / "governance" / "audit.db"


def iris_env(home: Path, *, master_key: bool = True) -> dict[str, str]:
    """A temporary IRIS home. ``master_key``: a throwaway vault key, so audit rows can be
    keyed -- without one IRIS refuses every governed call (the last serve test)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("IRIS_")}
    env |= {
        "HOME": str(home),
        "IRIS_HOME": str(home / ".iris"),
        "IRIS_GOVERNANCE_AUDIT_DB_PATH": str(ledger_path(home)),
        # The email profile: its plugin registers `trash_email`, a destructive tool.
        "IRIS_PROFILE": "email",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
    }
    if master_key:
        env["IRIS_VAULT_MASTER_KEY"] = Fernet.generate_key().decode()
    # The owner's MCP serve config: the clients IRIS may serve, and whether each runs on
    # this machine. `example` is not declared local, so a personal result is withheld.
    overlay = home / ".iris" / "mcp-serve.yaml"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text("clients:\n  example:\n    local: false\n", encoding="utf-8")
    return env


def iris_command(*args: str) -> list[str]:
    """``iris <args>``: the ``iris`` command is ``iris_harness.main:main``."""
    return [sys.executable, "-m", "iris_harness.main", *args]


def iris(*args: str, home: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- a fixed argv, our own interpreter
        iris_command(*args), env=iris_env(home), capture_output=True, text=True, check=False
    )


# -- 1. serve ---------------------------------------------------------------------------

# What the owner writes in the client's config: which tools, and the client's name. Every
# call is governed and audited as `mcp:example`, whatever the client calls itself.
SERVE = iris_command(
    "mcp", "serve", "--client", "example",
    "--tool", "system_health", "--tool", "trash_email", "--skill", "system-status",
)  # fmt: skip


def test_iris_serves_its_tools_over_mcp(tmp_path: Path) -> None:
    with StdioMcpClient(SERVE, env=iris_env(tmp_path)) as client:
        assert client.server_info["name"] == "iris"
        tools = {tool["name"]: tool for tool in client.list_tools()}
        assert {"system_health", "trash_email", "system_status"} <= set(tools)
        assert tools["system_status"]["inputSchema"]["type"] == "object"

        assert client.call_tool("system_health").strip()
        assert client.call_tool("system_status").strip()

        with pytest.raises(McpError, match="unknown tool"):
            client.call_tool("no_such_tool")


def test_a_tool_call_served_over_mcp_is_audited(tmp_path: Path) -> None:
    with StdioMcpClient(SERVE, env=iris_env(tmp_path)) as client:
        client.call_tool("system_health")
    ledger = AuditLog(db_path=ledger_path(tmp_path)).query()
    rows = [row for row in ledger if row.hook_point.endswith("_tool_use")]

    # PRE_TOOL_USE before the call, POST_TOOL_USE over its result: one run, as the client
    # the owner named.
    assert {row.hook_point for row in rows} == {"pre_tool_use", "post_tool_use"}
    assert {row.agent_type for row in rows} == {"mcp:example"}
    assert len({row.run_id for row in rows}) == 1
    assert all(row.decision == "allow" for row in rows)


def test_a_destructive_call_is_refused_and_the_refusal_is_audited(tmp_path: Path) -> None:
    with StdioMcpClient(SERVE, env=iris_env(tmp_path)) as client:
        # An MCP client cannot answer the owner's approval, so IRIS does not run the call
        # -- nor queue it to run later, out of the client's sight.
        with pytest.raises(McpError, match="needs the owner's approval"):
            client.call_tool("trash_email", {"message_ids": ["m-1"]})
    ledger = AuditLog(db_path=ledger_path(tmp_path)).query()

    [refusal] = [row for row in ledger if row.decision == "deny"]
    assert (refusal.hook_point, refusal.plugin) == ("pre_tool_use", "destructive_approval")
    assert refusal.agent_type == "mcp:example"
    assert not [row for row in ledger if row.hook_point == "post_tool_use"]  # it never ran


def test_only_a_client_the_owner_listed_is_served(tmp_path: Path) -> None:
    unlisted = iris_command("mcp", "serve", "--client", "someone-else")
    with pytest.raises(McpError, match="closed the connection"):
        with StdioMcpClient(unlisted, env=iris_env(tmp_path)):
            pass


def test_by_default_only_contained_reads_are_served(tmp_path: Path) -> None:
    serve = iris_command("mcp", "serve", "--client", "example")
    with StdioMcpClient(serve, env=iris_env(tmp_path)) as client:
        names = {tool["name"] for tool in client.list_tools()}
    # `research` sends its query to a search engine and returns web text; the email reads
    # return text third parties wrote: each is served only when named with --tool.
    assert "system_health" in names
    assert not names & {"research", "code_exec", "search_inbox", "read_email", "trash_email"}


def test_without_a_vault_key_nothing_is_served(tmp_path: Path) -> None:
    with StdioMcpClient(SERVE, env=iris_env(tmp_path, master_key=False)) as client:
        with pytest.raises(McpError, match="no vault master key"):
            client.call_tool("system_health")


# -- 2. consume: allowlist + signing ----------------------------------------------------


def test_the_external_server_speaks_mcp() -> None:
    server = [sys.executable, str(HERE / "weather_server.py")]
    with StdioMcpClient(server) as client:
        assert [tool["name"] for tool in client.list_tools()] == ["forecast"]
        assert client.call_tool("forecast", {"city": "Lisbon"}) == "Sunny, 24 C"


def test_a_signed_server_verifies_and_a_changed_one_does_not(tmp_path: Path) -> None:
    servers = tmp_path / "mcp-servers.yaml"
    shutil.copy(HERE / "mcp-servers.yaml", servers)
    trust = tmp_path / "mcp-trust.yaml"
    key = tmp_path / "keys" / "example.key"

    made = iris(
        "mcp", "keygen", "--id", "example", "--trust-store", str(trust),
        "--private-key-out", str(key), home=tmp_path,
    )  # fmt: skip
    assert made.returncode == 0, made.stdout + made.stderr
    signed = iris(
        "mcp", "sign", "weather", "--key", "example", "--servers", str(servers),
        "--private-key", str(key), home=tmp_path,
    )  # fmt: skip
    assert signed.returncode == 0, signed.stdout + signed.stderr

    verified = iris(
        "mcp", "verify", "--servers", str(servers), "--trust-store", str(trust), home=tmp_path
    )
    assert verified.returncode == 0, verified.stdout

    # Point the signed entry at another program: the signature no longer matches.
    servers.write_text(
        servers.read_text(encoding="utf-8").replace("weather_server.py", "other.py"),
        encoding="utf-8",
    )
    tampered = iris(
        "mcp", "verify", "--servers", str(servers), "--trust-store", str(trust), home=tmp_path
    )
    assert tampered.returncode != 0
