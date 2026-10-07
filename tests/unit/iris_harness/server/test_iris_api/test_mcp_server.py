"""HTTP tests for the IRIS API MCP surface.

The security review of 2026-09-26 found that an HTTP caller could approve its own MCP
call (``approval_granted`` in the body, or ``approved`` in ``metadata``) and that the
write guard let ``/api/v1/mcp/`` through for every credential, a read-only paired
phone included. The tests below pin both closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

# A plain import -- see the note in test_governor/test_http_service.py. The API is
# `iris_harness.server.iris_api` since M6.2 layer 10, so the loader ceremony is gone.
from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance.devices import DeviceService, DeviceStore
from iris_harness.kernel.governor.audit import GovernorAuditLogger
from iris_harness.kernel.governor.policy import GovernorPolicyEngine, load_governor_policy
from iris_harness.kernel.governor.service import IRISGovernorService
from iris_harness.server.iris_api import main
from iris_harness.server.iris_api.main import create_app
from iris_harness.server.iris_api.mcp_server import APPROVAL_REQUIRED_DETAIL
from iris_harness.tools.mcp_bridge import MCPBridge


def write_governor_policy(repo_root: Path, *, requires_approval: bool = True) -> None:
    policy_dir = repo_root / "config" / "governor"
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "policy.yaml").write_text(
        "version: '1'\n"
        "routes:\n"
        "  - route: coding/mcp\n"
        "    allowed_actions:\n"
        "      - session_open\n"
        "      - call_tool\n"
        "      - invoke_server\n"
        f"    requires_approval: {'true' if requires_approval else 'false'}\n"
        "    rate_limit:\n"
        "      requests: 10\n"
        "      window_seconds: 3600\n",
        encoding="utf-8",
    )


def write_http_mcp_config(repo_root: Path, *, enabled: bool) -> None:
    config_dir = repo_root / "config" / "coding-agent"
    config_dir.mkdir(parents=True, exist_ok=True)
    enabled_value = "true" if enabled else "false"
    (config_dir / "mcp-servers.yaml").write_text(
        "version: '1'\n"
        f"enabled: {enabled_value}\n"
        "auto_export_local_tools: true\n"
        "servers:\n"
        "  - name: github\n"
        "    description: GitHub MCP test server\n"
        "    enabled: true\n"
        "    transport: http\n"
        "    url: http://127.0.0.1:8787/mcp\n",
        encoding="utf-8",
    )


def write_skill_package(repo_root: Path) -> None:
    skill_dir = repo_root / "config" / "skills" / "email_triage"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "manifest.yaml").write_text(
        "name: email_triage\n"
        "version: 1.0.0\n"
        "description: Test skill package\n"
        "author: iris-tests\n"
        "license: Apache-2.0\n"
        "tools:\n"
        "  - name: email_triage_tool\n"
        "    description: Test tool\n"
        "    governor_route: coding/skill/email_triage\n"
        "requires:\n"
        "  python: '>=3.12'\n"
        "  packages:\n"
        "    - pydantic>=2.0\n"
        "  env_vars: []\n"
        "  config_files:\n"
        "    - message_templates.yaml\n"
        "  agents:\n"
        "    - email\n",
        encoding="utf-8",
    )
    (skill_dir / "agent.md").write_text("# email_triage\n", encoding="utf-8")
    (skill_dir / "tools.py").write_text(
        "from langchain_core.tools import BaseTool\n"
        "from pydantic import BaseModel, Field\n\n"
        "class EchoArgs(BaseModel):\n"
        "    message: str = Field(description='Message to echo')\n\n"
        "class EchoTool(BaseTool):\n"
        "    name: str = 'email_triage_tool'\n"
        "    description: str = 'Echo test tool'\n"
        "    args_schema: type[BaseModel] = EchoArgs\n\n"
        "    def _run(self, message: str) -> str:\n"
        "        return message\n\n"
        "    async def _arun(self, message: str) -> str:\n"
        "        return message\n\n"
        "SKILL_TOOLS = [EchoTool]\n",
        encoding="utf-8",
    )
    (repo_root / "config" / "message_templates.yaml").write_text(
        "templates: {}\n",
        encoding="utf-8",
    )


def build_governor_service(repo_root: Path) -> IRISGovernorService:
    return IRISGovernorService(
        policy_engine=GovernorPolicyEngine(load_governor_policy(repo_root)),
        audit_logger=GovernorAuditLogger(repo_root / "data" / "audit.db"),
    )


# Every JSON-RPC method the fake server was asked to run, so a test can prove a
# refused call never reached the transport.
TRANSPORT_CALLS: list[str] = []


def fake_http_requester(_server: Any, payload: dict[str, Any]) -> Any:
    method = payload["method"]
    TRANSPORT_CALLS.append(method)
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {
                "tools": [
                    {
                        "name": "remote_echo",
                        "description": "Echo from remote MCP server",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"},
                            },
                            "required": ["message"],
                        },
                    }
                ]
            },
        }
    if method == "tools/call":
        return {
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": payload["params"]["arguments"]["message"],
                    }
                ]
            },
        }
    raise AssertionError(f"unexpected MCP method: {method}")


def build_client(repo_root: Path, *, device_service: DeviceService | None = None) -> TestClient:
    TRANSPORT_CALLS.clear()
    bridge = MCPBridge(
        repo_root,
        governor_service=build_governor_service(repo_root),
        http_requester=fake_http_requester,
    )
    return TestClient(
        create_app(mcp_bridge=bridge, device_service=device_service), headers=auth_headers()
    )


def test_healthz_config_and_local_tools_endpoints(tmp_path: Path) -> None:
    write_governor_policy(tmp_path)
    write_http_mcp_config(tmp_path, enabled=True)
    write_skill_package(tmp_path)
    client = build_client(tmp_path)

    health = client.get("/healthz")
    config = client.get("/api/v1/mcp/config")
    local_tools = client.get("/api/v1/mcp/tools/local", params={"agent_name": "email"})

    assert health.status_code == 200
    health_json = health.json()
    assert health_json["status"] == "ok"
    assert health_json["mcp_bridge_enabled"] is True
    assert health_json["enabled_server_count"] == 1
    assert health_json["runtime_ready"] is False
    assert config.status_code == 200
    assert config.json()["configured_servers"][0]["name"] == "github"
    assert local_tools.status_code == 200
    assert local_tools.json()["tools"][0]["name"] == "email_triage_tool"


def test_external_tool_discovery_and_call_flow_through_iris_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A server whose governor route needs no approval still works end to end over
    # HTTP, for a caller allowed to write.
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    write_governor_policy(tmp_path, requires_approval=False)
    write_http_mcp_config(tmp_path, enabled=True)
    client = build_client(tmp_path)

    discovery = client.post("/api/v1/mcp/servers/github/tools/list", json={})
    invocation = client.post(
        "/api/v1/mcp/servers/github/tools/remote_echo/call",
        json={"arguments": {"message": "hello from API"}},
    )

    assert discovery.status_code == 200, discovery.text
    assert discovery.json()["tools"][0]["name"] == "remote_echo"
    assert invocation.status_code == 200, invocation.text
    assert invocation.json()["invocation"]["server_name"] == "github"
    assert invocation.json()["invocation"]["result"] == {
        "content": [{"type": "text", "text": "hello from API"}],
    }


def test_external_tool_endpoints_require_governor_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    write_governor_policy(tmp_path)
    write_http_mcp_config(tmp_path, enabled=True)
    client = build_client(tmp_path)

    listing = client.post("/api/v1/mcp/servers/github/tools/list", json={})
    call = client.post(
        "/api/v1/mcp/servers/github/tools/remote_echo/call",
        json={"arguments": {"message": "x"}},
    )

    for response in (listing, call):
        assert response.status_code == 403
        assert response.json()["detail"] == APPROVAL_REQUIRED_DETAIL
    assert TRANSPORT_CALLS == []


def test_the_client_cannot_approve_its_own_mcp_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    write_governor_policy(tmp_path)
    write_http_mcp_config(tmp_path, enabled=True)
    client = build_client(tmp_path)

    # The old body flag is no longer a field: refused outright, not quietly dropped.
    for path, body in [
        ("/api/v1/mcp/servers/github/tools/list", {"approval_granted": True}),
        (
            "/api/v1/mcp/servers/github/tools/remote_echo/call",
            {"approval_granted": True, "arguments": {"message": "x"}},
        ),
        ("/api/v1/mcp/servers/github/session", {"approval_granted": True}),
    ]:
        assert client.post(path, json=body).status_code == 422, path

    # Consent smuggled in metadata is stripped before the governor sees it.
    for flag in ("approved", "approval_granted"):
        smuggled = client.post(
            "/api/v1/mcp/servers/github/tools/list", json={"metadata": {flag: True}}
        )
        assert smuggled.status_code == 403, flag
        assert smuggled.json()["detail"] == APPROVAL_REQUIRED_DETAIL
        session = client.post("/api/v1/mcp/servers/github/session", json={"metadata": {flag: True}})
        assert session.status_code == 200
        assert session.json()["decision"]["allowed"] is False
    assert TRANSPORT_CALLS == []


def test_a_read_only_phone_cannot_reach_the_mcp_write_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Even with the switch on and a route that needs no approval: a read device is
    # judged on its own scope, as for every other gated write.
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    write_governor_policy(tmp_path, requires_approval=False)
    write_http_mcp_config(tmp_path, enabled=True)
    service = DeviceService(store=DeviceStore(db_path=tmp_path / "devices.db"))
    client = build_client(tmp_path, device_service=service)
    code = client.post("/api/v1/devices/pair/start", json={"scope": "read"}).json()["code"]
    claimed = client.post(
        "/api/v1/devices/pair/claim", json={"code": code, "name": "Phone", "kind": "app"}
    ).json()
    client.cookies.clear()
    phone = {"Authorization": f"Bearer {claimed['token']}"}

    for path in (
        "/api/v1/mcp/servers/github/session",
        "/api/v1/mcp/servers/github/tools/list",
        "/api/v1/mcp/servers/github/tools/remote_echo/call",
    ):
        refused = client.post(path, json={}, headers=phone)
        assert refused.status_code == 403, path
        assert refused.json() == {"detail": "this device is paired read-only"}
    # Reads stay open to it.
    assert client.get("/api/v1/mcp/config", headers=phone).status_code == 200
    assert TRANSPORT_CALLS == []


def test_the_shared_secret_needs_the_write_switch_for_mcp_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    write_governor_policy(tmp_path, requires_approval=False)
    write_http_mcp_config(tmp_path, enabled=True)
    client = build_client(tmp_path)

    refused = client.post("/api/v1/mcp/servers/github/tools/list", json={})

    assert refused.status_code == 403
    assert "control writes are disabled" in refused.json()["detail"]
    assert TRANSPORT_CALLS == []


def test_the_default_bridge_carries_the_governance_kernel_when_mcp_is_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris_harness.kernel.governance as governance

    sentinel = object()
    monkeypatch.setattr(governance, "kernel_from_env", lambda: sentinel)
    # The bridge reads the resolved config (foundation.paths.config_root()) and keeps the
    # governor's audit log under REPO_ROOT/data; point both here.
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(main, "REPO_ROOT", tmp_path)
    write_governor_policy(tmp_path)

    write_http_mcp_config(tmp_path, enabled=True)
    assert main._default_mcp_bridge()._governance_kernel is sentinel

    # MCP off (the shipped default): no kernel is built for a bridge nobody can use.
    write_http_mcp_config(tmp_path, enabled=False)
    assert main._default_mcp_bridge()._governance_kernel is None


def test_building_the_default_bridge_writes_no_audit_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importing the API builds the app, which builds this bridge; with MCP off (the
    shipped default) that left data/audit.db in the checkout running the tests."""
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(main, "REPO_ROOT", tmp_path)
    write_governor_policy(tmp_path)
    write_http_mcp_config(tmp_path, enabled=False)

    main._default_mcp_bridge()

    assert not (tmp_path / "data" / "audit.db").exists()


def test_a_call_to_an_undeclared_tool_is_refused_over_http_and_never_reaches_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #180: an undeclared tool fails closed. The HTTP surface grants no approval
    (``_HTTP_APPROVAL``), so a gated call is refused there by design, with the reason."""
    from iris_harness.kernel.governance import build_default_kernel
    from iris_harness.kernel.governance.audit import AuditLog

    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    write_governor_policy(tmp_path, requires_approval=False)
    write_http_mcp_config(tmp_path, enabled=True)
    TRANSPORT_CALLS.clear()
    bridge = MCPBridge(
        tmp_path,
        governor_service=build_governor_service(tmp_path),
        http_requester=fake_http_requester,
        governance_kernel=build_default_kernel(audit_log=AuditLog(tmp_path / "audit.db")),
    )
    client = TestClient(create_app(mcp_bridge=bridge), headers=auth_headers())

    call = client.post(
        "/api/v1/mcp/servers/github/tools/remote_echo/call",
        json={"arguments": {"message": "x"}},
    )

    assert call.status_code == 403, call.text
    assert "github/remote_echo" in call.json()["detail"]
    assert TRANSPORT_CALLS == []
