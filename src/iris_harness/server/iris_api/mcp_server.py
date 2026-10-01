"""Unified MCP routes for the IRIS API service.

The HTTP caller can never approve an MCP action. The ``coding/mcp`` governor route
requires approval, and a flag in a request body is the caller vouching for itself:
before this, any credential that reached these routes (a read-only paired phone
included, since the write guard let ``/api/v1/mcp/`` through) could send
``approval_granted: true`` and run an external tool. The routes now pass no approval,
refuse extra body fields, and answer a call that needs approval with a 403 saying so.
Approval stays with in-process callers that asked the owner first. The approval queue
(ADR-0118) was not wired in here: it pauses and resumes an agent run, and a bare HTTP
call has no run to resume.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from iris_harness.tools import (
    MCPApprovalRequired,
    MCPBridge,
    MCPInvocationResult,
    MCPToolDefinition,
)


class MCPServerSummary(BaseModel):
    """Safe server summary for API responses."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    enabled: bool
    transport: str
    command: str | None = None
    args: tuple[str, ...] = Field(default_factory=tuple)
    url: str | None = None
    timeout_seconds: float


class MCPConfigResponse(BaseModel):
    """Top-level MCP bridge configuration summary."""

    model_config = ConfigDict(frozen=True)

    enabled: bool
    auto_export_local_tools: bool
    enabled_server_count: int
    configured_servers: tuple[MCPServerSummary, ...]


class MCPToolCatalogResponse(BaseModel):
    """Collection of tool definitions returned by one MCP endpoint."""

    model_config = ConfigDict(frozen=True)

    tools: tuple[MCPToolDefinition, ...]


class MCPServerSessionRequest(BaseModel):
    """Request payload for opening a governed MCP server session.

    ``extra="forbid"``: a body that still sends ``approval_granted`` gets a 422 rather
    than having it silently ignored, so an old client learns the flag is gone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPServerSessionResponse(BaseModel):
    """Guard result for a server session authorization."""

    model_config = ConfigDict(frozen=True)

    decision: dict[str, Any]


class MCPExternalToolsRequest(BaseModel):
    """Request payload for external tool discovery (no approval field, see above)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPToolCallRequest(BaseModel):
    """Request payload for external tool invocation (no approval field, see above)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPToolCallResponse(BaseModel):
    """Structured MCP invocation payload."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    invocation: MCPInvocationResult


# What the HTTP surface grants: nothing. A constant, not a literal at each call, so the
# rule is stated once and a test can find it.
_HTTP_APPROVAL = False

APPROVAL_REQUIRED_DETAIL = (
    "approval required: this MCP action needs the owner's approval, and the HTTP API "
    "cannot grant it. Run it from a flow that asks the owner first."
)


def create_mcp_router(bridge: MCPBridge) -> APIRouter:
    """Build the MCP routes mounted under the IRIS API service."""
    router = APIRouter(prefix="/api/v1/mcp", tags=["mcp"])

    @router.get("/config", response_model=MCPConfigResponse)
    def get_mcp_config() -> MCPConfigResponse:
        return MCPConfigResponse(
            enabled=bridge.config.enabled,
            auto_export_local_tools=bridge.config.auto_export_local_tools,
            enabled_server_count=len(bridge.list_enabled_servers()),
            configured_servers=tuple(_summarize_server(server) for server in bridge.config.servers),
        )

    @router.get("/tools/local", response_model=MCPToolCatalogResponse)
    def list_local_tools(agent_name: str | None = None) -> MCPToolCatalogResponse:
        return MCPToolCatalogResponse(tools=bridge.export_local_tools(agent_name=agent_name))

    @router.post("/servers/{server_name}/session", response_model=MCPServerSessionResponse)
    def open_server_session(
        server_name: str,
        payload: MCPServerSessionRequest,
    ) -> MCPServerSessionResponse:
        try:
            decision = bridge.authorize_server_session(
                server_name,
                approval_granted=_HTTP_APPROVAL,
                metadata=payload.metadata,
            )
        except Exception as exc:
            raise _as_http_exception(exc) from exc
        return MCPServerSessionResponse(decision=decision.model_dump(mode="json"))

    @router.post("/servers/{server_name}/tools/list", response_model=MCPToolCatalogResponse)
    def list_external_tools(
        server_name: str,
        payload: MCPExternalToolsRequest,
    ) -> MCPToolCatalogResponse:
        try:
            tools = bridge.list_external_tools(
                server_name,
                approval_granted=_HTTP_APPROVAL,
                metadata=payload.metadata,
            )
        except Exception as exc:
            raise _as_http_exception(exc) from exc
        return MCPToolCatalogResponse(tools=tools)

    @router.post(
        "/servers/{server_name}/tools/{tool_name}/call", response_model=MCPToolCallResponse
    )
    def call_external_tool(
        server_name: str,
        tool_name: str,
        payload: MCPToolCallRequest,
    ) -> MCPToolCallResponse:
        try:
            invocation = bridge.invoke_external_tool(
                server_name,
                tool_name,
                payload.arguments,
                approval_granted=_HTTP_APPROVAL,
            )
        except Exception as exc:
            raise _as_http_exception(exc) from exc
        return MCPToolCallResponse(invocation=invocation)

    return router


def _summarize_server(server: Any) -> MCPServerSummary:
    """Project a server config into a safe API response model."""
    return MCPServerSummary(
        name=server.name,
        description=server.description,
        enabled=server.enabled,
        transport=server.transport,
        command=server.command,
        args=server.args,
        url=server.url,
        timeout_seconds=server.timeout_seconds,
    )


def _as_http_exception(exc: Exception) -> HTTPException:
    """Map bridge errors into stable API responses."""
    if isinstance(exc, MCPApprovalRequired):
        return HTTPException(status_code=403, detail=APPROVAL_REQUIRED_DETAIL)
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


MCPServerSummary.model_rebuild()
MCPConfigResponse.model_rebuild()
MCPToolCatalogResponse.model_rebuild()
MCPServerSessionRequest.model_rebuild()
MCPServerSessionResponse.model_rebuild()
MCPExternalToolsRequest.model_rebuild()
MCPToolCallRequest.model_rebuild()
MCPToolCallResponse.model_rebuild()
