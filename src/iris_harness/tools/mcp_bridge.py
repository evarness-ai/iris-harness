# ruff: noqa: S603
"""Disabled-by-default MCP bridge for local skill export and governed server calls."""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urljoin, urlparse

import httpx
import yaml
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris_harness.foundation.ids import new_ulid
from iris_harness.foundation.observability.logging_setup import log_egress
from iris_harness.kernel.governance.call_context import register_call
from iris_harness.kernel.governance.mcp_signing import (
    MCPSigningConfig,
    ServerSpec,
    SignatureVerdict,
    TrustStore,
    file_sha256,
    resolve_signing_action,
    verify_server_signature,
)
from iris_harness.kernel.governance.plugins.mcp_allowlist import MCPServerGovernance
from iris_harness.kernel.governor import GovernorGuardDecision, IRISGovernorService
from iris_harness.tools.skills import SkillRegistry

if TYPE_CHECKING:
    from iris_harness.kernel.governance.kernel import GovernanceKernel

logger = logging.getLogger(__name__)


class MCPSignatureError(RuntimeError):
    """Raised when MCP signing is in ``enforce`` mode and a server's signature
    does not verify (and ``unsigned_policy`` is ``deny``) — the launch/connect
    is refused before any process starts."""


class MCPApprovalRequired(PermissionError):
    """Raised when the governor refuses an MCP action because nobody approved it.

    A subclass of :exc:`PermissionError` so existing callers keep working; the HTTP
    API tells it apart to say "approval required" instead of echoing the governor's
    wording, which names a metadata flag the client is not allowed to set."""


# Metadata keys the governor reads as consent (``kernel/governor/policy.py``,
# ``_has_approval_flag``). Only the bridge's own ``approval_granted`` argument may set one.
_APPROVAL_KEYS = frozenset({"approval_granted", "approved"})


class MCPServerConfig(BaseModel):
    """Configuration for one external MCP server integration."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1)
    description: str = Field(default="", min_length=0)
    enabled: bool = Field(default=False)
    transport: str = Field(default="stdio", min_length=1)
    command: str | None = Field(default=None)
    args: tuple[str, ...] = Field(default_factory=tuple)
    url: str | None = Field(default=None)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=10.0, gt=0)
    governance: MCPServerGovernance | None = Field(default=None)
    # Phase 6 MCP signing (6b): Ed25519 signature over the canonical launch spec
    # and the trust-store key id that produced it. Both optional — an unsigned
    # entry is allowed/warned/denied per config/governance/mcp-signing.yaml.
    signature: str | None = Field(default=None)
    signed_by: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_enabled_transport(self) -> MCPServerConfig:
        """Require the minimum connection details only when a server is enabled."""
        if not self.enabled:
            return self

        if self.transport == "stdio" and not self.command:
            raise ValueError("enabled stdio MCP servers require a command")
        if self.transport in {"http", "sse"} and not self.url:
            raise ValueError("enabled HTTP/SSE MCP servers require a url")
        return self


class MCPBridgeConfig(BaseModel):
    """Top-level MCP bridge configuration."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    version: str = Field(default="1", min_length=1)
    enabled: bool = Field(default=False)
    auto_export_local_tools: bool = Field(default=True)
    servers: tuple[MCPServerConfig, ...] = Field(default_factory=tuple)


class MCPToolDefinition(BaseModel):
    """MCP-shaped representation of one local or external tool."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    governor_route: str = Field(..., min_length=1)
    source_kind: str = Field(..., min_length=1)
    source_name: str = Field(..., min_length=1)


class MCPInvocationResult(BaseModel):
    """Structured record of a governed MCP tool invocation."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    server_name: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    governor_decision: GovernorGuardDecision = Field(...)
    result: Any = Field(...)


def load_mcp_bridge_config(
    repo_root: Path,
    *,
    config_path: Path | None = None,
) -> MCPBridgeConfig:
    """Load the MCP bridge config, defaulting to a disabled bridge when absent."""
    resolved_path = (
        config_path
        if config_path is not None
        else repo_root / "config" / "coding-agent" / "mcp-servers.yaml"
    ).resolve()
    if not resolved_path.exists():
        return MCPBridgeConfig()

    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("mcp bridge configuration must decode to a mapping")
    return MCPBridgeConfig.model_validate(payload)


def _refusal(decision: GovernorGuardDecision, *, approval_granted: bool) -> PermissionError:
    """The error for a refused MCP action: approval missing, or any other refusal."""
    if decision.requires_approval and not approval_granted:
        return MCPApprovalRequired(decision.reason)
    return PermissionError(decision.reason)


def build_server_spec(server: MCPServerConfig) -> ServerSpec:
    """Build the canonical signable spec for ``server``.

    The single source of truth shared by the bridge gate and the ``iris mcp
    sign``/``verify`` CLI, so the CLI signs exactly what the gate checks. Hashes
    the binary only for a local-path ``stdio`` command (a launcher like
    ``npx <pkg>`` is not a file — see the doc's scope boundary)."""
    package_sha256: str | None = None
    if server.transport == "stdio" and server.command:
        candidate = Path(server.command)
        if candidate.is_file():
            package_sha256 = file_sha256(candidate)
    return ServerSpec(
        name=server.name,
        transport=server.transport,
        command=server.command,
        args=server.args,
        url=server.url,
        env_keys=tuple(server.env.keys()),
        package_sha256=package_sha256,
    )


def _load_signing_config(repo_root: Path) -> MCPSigningConfig:
    """Load the MCP-signing policy; absent/unreadable → disabled (no-op gate)."""
    try:
        return MCPSigningConfig.from_yaml(repo_root / "config" / "governance" / "mcp-signing.yaml")
    except Exception:  # signing must never break bridge construction
        logger.debug("mcp-signing config load failed; signing disabled", exc_info=True)
        return MCPSigningConfig.disabled()


def _load_trust_store(repo_root: Path, config: MCPSigningConfig) -> TrustStore:
    """Load the trust store referenced by ``config``; absent/disabled → empty."""
    if not config.enabled:
        return TrustStore.empty()
    try:
        path = Path(config.trust_store)
        if not path.is_absolute():
            path = repo_root / path
        return TrustStore.from_yaml(path)
    except Exception:  # degrade to empty (every signed server untrusted)
        logger.debug("mcp-signing trust store load failed; using empty store", exc_info=True)
        return TrustStore.empty()


def declared_effect(server: MCPServerConfig, tool_name: str) -> str | None:
    """The effect the operator declared for ``tool_name`` on ``server``, else None."""
    if server.governance is None:
        return None
    declared = server.governance.tools.get(tool_name)
    return declared.effect if declared is not None else None


def effective_effect(server: MCPServerConfig, tool_name: str) -> str:
    """What the bridge treats ``tool_name`` as: the declared effect, else the server's default.

    An undeclared tool fails closed to ``destructive`` (issue #180); the operator's
    ``governance.undeclared_tools: read`` is the per-server opt-out.
    """
    declared = declared_effect(server, tool_name)
    if declared is not None:
        return declared
    if server.governance is None:
        return "destructive"
    return server.governance.undeclared_tools


def undeclared_tool_names(server: MCPServerConfig, tool_names: Iterable[str]) -> list[str]:
    """The names in ``tool_names`` the operator has not declared on ``server``."""
    return [name for name in tool_names if declared_effect(server, name) is None]


class MCPBridge:
    """Bridge local skill tools into MCP definitions and guard external tool calls."""

    def __init__(
        self,
        repo_root: Path,
        *,
        config: MCPBridgeConfig | None = None,
        skill_registry: SkillRegistry | None = None,
        governor_service: IRISGovernorService | None = None,
        http_requester: Callable[[MCPServerConfig, dict[str, Any]], Any] | None = None,
        sse_exchange_runner: (
            Callable[
                [MCPServerConfig, tuple[dict[str, Any], ...], tuple[str, ...]],
                dict[str, Any],
            ]
            | None
        ) = None,
        governance_kernel: GovernanceKernel | None = None,
        signing_config: MCPSigningConfig | None = None,
        trust_store: TrustStore | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.config = config or load_mcp_bridge_config(self.repo_root)
        self.skill_registry = skill_registry or SkillRegistry(self.repo_root)
        self.governor_service = governor_service or IRISGovernorService.from_repo_root(
            self.repo_root
        )
        self._http_requester = http_requester or _default_http_requester
        self._sse_exchange_runner = sse_exchange_runner or _default_sse_exchange_runner
        self._server_index = {server.name: server for server in self.config.servers}
        self._governance_kernel = governance_kernel
        # Phase 6 MCP signing (6b.2): verify each server's signature before
        # launch/connect. Absent config → disabled (no-op); shadow → warn-only.
        self._signing_config = signing_config or _load_signing_config(self.repo_root)
        self._trust_store = (
            trust_store
            if trust_store is not None
            else _load_trust_store(self.repo_root, self._signing_config)
        )

    @classmethod
    def from_repo_root(
        cls,
        repo_root: Path,
        *,
        config_path: Path | None = None,
        governance_kernel: GovernanceKernel | None = None,
    ) -> MCPBridge:
        """Construct an MCP bridge from repository configuration."""
        return cls(
            repo_root,
            config=load_mcp_bridge_config(repo_root, config_path=config_path),
            governance_kernel=governance_kernel,
        )

    def list_enabled_servers(self) -> tuple[MCPServerConfig, ...]:
        """Return configured external MCP servers that are explicitly enabled."""
        return tuple(server for server in self.config.servers if server.enabled)

    def export_local_tools(self, *, agent_name: str | None = None) -> tuple[MCPToolDefinition, ...]:
        """Expose loadable local skill tools as MCP-style tool definitions."""
        if not self.config.auto_export_local_tools:
            return ()

        self.skill_registry.discover()
        packages = self.skill_registry.list_packages(agent_name=agent_name, only_loadable=True)
        definitions: list[MCPToolDefinition] = []
        for package in packages:
            for manifest_tool, tool_class in zip(
                package.manifest.tools,
                package.tool_classes,
                strict=False,
            ):
                definitions.append(
                    MCPToolDefinition(
                        name=manifest_tool.name,
                        description=manifest_tool.description,
                        input_schema=_build_tool_input_schema(tool_class),
                        governor_route=manifest_tool.governor_route,
                        source_kind="skill",
                        source_name=package.manifest.name,
                    )
                )
        if agent_name is None:
            definitions.extend(self._export_persona_tools())
        return tuple(definitions)

    def _export_persona_tools(self) -> tuple[MCPToolDefinition, ...]:
        """Expose a generic persona invoker plus auto-registered per-persona shims."""
        # Personas are a coding-agent feature. The import is lazy so skill-only
        # callers pay no cold-start cost, and guarded because the coding agent is
        # not part of the harness release (OSS plan decision 2): without it there
        # is nothing to invoke, so no persona tool is exported at all.
        try:
            # absent from the public tree: mypy must pass there as well as here
            from iris_code.resource_paths import (  # type: ignore[import-not-found,unused-ignore]
                iter_persona_contract_paths,
            )
            from iris_code.sub_agents import (  # type: ignore[import-not-found,unused-ignore]
                load_persona_profile,
            )
        except ImportError:
            return ()

        definitions: list[MCPToolDefinition] = [_build_generic_persona_tool_definition()]
        for path in iter_persona_contract_paths(self.repo_root):
            try:
                profile = load_persona_profile(path, repo_root=self.repo_root)
            # a single malformed persona file must not crash the whole bridge
            except Exception:  # noqa: BLE001, S112
                continue
            for command in profile.commands:
                try:
                    definitions.append(_build_persona_command_tool_definition(profile, command))
                except Exception:  # noqa: BLE001, S112 - defensive against schema drift
                    continue
        return tuple(definitions)

    def authorize_server_session(
        self,
        server_name: str,
        *,
        approval_granted: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> GovernorGuardDecision:
        """Authorize opening a session with one configured MCP server."""
        server = self._require_enabled_server(server_name)
        payload_metadata = {"server_name": server.name, "approval_granted": approval_granted}
        if metadata:
            payload_metadata.update(metadata)
        return self._guard_mcp_action(
            "session_open",
            server_name=server.name,
            approval_granted=approval_granted,
            metadata=payload_metadata,
        )

    def list_external_tools(
        self,
        server_name: str,
        *,
        approval_granted: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MCPToolDefinition, ...]:
        """Discover tool definitions from one configured external MCP server."""
        server = self._require_enabled_server(server_name)
        payload_metadata = {"method": "tools/list"}
        if metadata:
            payload_metadata.update(metadata)
        decision = self._guard_mcp_action(
            "invoke_server",
            server_name=server.name,
            approval_granted=approval_granted,
            metadata=payload_metadata,
        )
        if not decision.allowed:
            raise _refusal(decision, approval_granted=approval_granted)

        result = self._invoke_transport(server, method="tools/list", params={})
        tool_payloads = _coerce_mcp_tools_payload(result)
        definitions = tuple(
            _coerce_external_tool_definition(server, payload) for payload in tool_payloads
        )
        undeclared = undeclared_tool_names(server, (d.name for d in definitions))
        if undeclared:
            default = server.governance.undeclared_tools if server.governance else "destructive"
            logger.warning(
                "MCP server %r has tools with no declared effect: %s; they are treated as %s "
                "(declare them under governance.tools, or set governance.undeclared_tools)",
                server.name,
                ", ".join(sorted(undeclared)),
                default,
            )
        return definitions

    def invoke_external_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        approval_granted: bool = False,
        executor: Callable[[MCPServerConfig, str, dict[str, Any]], Any] | None = None,
        persona: str | None = None,
        run_id: str | None = None,
        approved_by: str | None = None,
    ) -> MCPInvocationResult:
        """Guard an outbound MCP tool call, then delegate to the supplied executor.

        When a ``governance_kernel`` was supplied at construction, fires
        ``PreToolUse`` (route ``mcp/{server}/{tool}``) before the legacy
        governor check. ``persona`` and ``run_id`` are forwarded into the
        :class:`~iris_harness.kernel.governance.hooks.types.HookContext` so
        ``MCPAllowlistHook`` can enforce the ``(persona, server, tool)``
        tuple. A ``deny`` or ``require_approval`` outcome raises
        :exc:`PermissionError` with the governance reason (the bridge has no resume
        path, so a call that needs approval does not run); the pre step matches the
        post step. The server is called with the arguments as
        ``PreToolUse`` left them (a ``vault://`` handle resolved by the
        credential broker); everything recorded -- the governor's audit, the
        returned ``arguments`` -- keeps the arguments as the caller wrote them.

        Its result then passes ``PostToolUse``: an external server's output is
        third-party text (``content: external``), so the external-content floor
        marks and scans it (and the opt-in retrieved-content injection guard, when
        enabled). A ``deny`` or ``require_approval`` raises
        :exc:`PermissionError`; a
        ``transform`` is what the caller receives.

        A tool the operator declared in the server's ``governance.tools`` carries that
        effect through both steps (issue #102). A ``destructive`` one is refused before the
        server is reached unless ``approved_by`` names an approved queue row that pinned
        exactly this call, and it leaves a pending write-ahead ledger row before it runs,
        settled after. A tool nobody declared is treated as ``destructive`` (the server's
        ``governance.undeclared_tools``, default ``destructive``; ``read`` opts a server out),
        so it fails closed (issue #180).
        """
        server = self._require_enabled_server(server_name)
        effect = effective_effect(server, tool_name)
        call_arguments = dict(arguments)
        # One id for this outbound call, minted here and nowhere else: its PRE and POST rows
        # both carry it (#134). Not the caller's to supply.
        call_id = new_ulid()
        # Where this call sits (the governed call it runs inside, if any), for the kernel's
        # rows of the call (#134 stage 2). The HTTP route runs inside none, so it has no parent.
        register_call(call_id)

        # Governance kernel pre-check (Phase 4 — story 12.gov-4.6).
        if self._governance_kernel is not None:
            call_arguments = self._fire_governance_pre_tool(
                server_name=server.name,
                tool_name=tool_name,
                arguments=arguments,
                persona=persona,
                run_id=run_id,
                call_id=call_id,
                effect=effect,
                approved_by=approved_by,
            )

        decision = self._guard_mcp_action(
            "call_tool",
            server_name=server.name,
            approval_granted=approval_granted,
            metadata={
                "tool_name": tool_name,
                "arguments": arguments,
            },
        )
        if not decision.allowed:
            raise _refusal(decision, approval_granted=approval_granted)

        if executor is not None:
            result = executor(server, tool_name, call_arguments)
        else:
            result = self._invoke_transport(
                server,
                method="tools/call",
                params={"name": tool_name, "arguments": call_arguments},
            )
        if self._governance_kernel is not None:
            result = self._fire_governance_post_tool(
                server_name=server.name,
                tool_name=tool_name,
                result=result,
                persona=persona,
                run_id=run_id,
                call_id=call_id,
                effect=effect,
            )
        return MCPInvocationResult(
            server_name=server.name,
            tool_name=tool_name,
            arguments=dict(arguments),
            governor_decision=decision,
            result=result,
        )

    def _fire_governance_pre_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        persona: str | None,
        run_id: str | None,
        call_id: str,
        effect: str | None = None,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        """Fire ``PreToolUse`` through the governance kernel for an MCP dispatch.

        Route format: ``mcp/{server_name}/{tool_name}`` (AC-5). The
        ``mcp_server`` and ``mcp_tool`` payload fields allow
        ``MCPAllowlistHook`` to identify the dispatch without parsing the
        route string. A ``deny`` or ``require_approval`` decision raises
        :exc:`PermissionError` (the bridge has no resume path), as the post step does.
        Returns the arguments to call the server with: the final context's
        ``args`` (``tool_payload.args_of``).

        The row carries a keyed ``args_digest`` of the arguments as written
        (``kernel/governance/audit/digest.py``). With no audit key the call is
        refused here, before ``PRE_TOOL_USE`` and before the server is reached:
        :exc:`PermissionError` with the no-key message.
        """
        from iris_harness.kernel.governance.audit.digest import (
            AuditKeyUnavailable,
            audit_digester,
        )
        from iris_harness.kernel.governance.hooks.tool_payload import (
            CALL_ID,
            TOOL_CALL_ID,
            TOOL_EFFECT,
            args_of,
            pre_tool_payload,
        )
        from iris_harness.kernel.governance.hooks.types import (
            HookContext,
            HookPoint,
        )

        assert self._governance_kernel is not None  # caller guards
        try:
            digester = audit_digester()
        except AuditKeyUnavailable as exc:
            raise PermissionError(f"MCP tool '{server_name}/{tool_name}' refused: {exc}") from exc
        route = f"mcp/{server_name}/{tool_name}"
        ctx = HookContext(
            hook_point=HookPoint.PRE_TOOL_USE,
            run_id=run_id or f"mcp-{server_name}",
            agent_type="mcp",
            persona=persona,
            route=route,
            payload=pre_tool_payload(
                route,
                dict(arguments),
                mcp_server=server_name,
                mcp_tool=tool_name,
                tool_plugin=f"mcp:{server_name}",
                **digester.args_fields(arguments),
            ),
            metadata={
                CALL_ID: call_id,
                TOOL_CALL_ID: call_id,
                # What the operator declared for this tool, or None: an MCP server declares
                # no effect of its own. A destructive declaration makes the call wait for an
                # approval (``approved_by``) and gives it a write-ahead ledger row.
                TOOL_EFFECT: effect,
                **({"approved_by": approved_by} if approved_by else {}),
            },
        )
        decision, final_ctx = self._governance_kernel.fire_sync(HookPoint.PRE_TOOL_USE, ctx)
        if decision.outcome == "deny":
            raise PermissionError(
                f"MCP tool '{server_name}/{tool_name}' blocked by governance: " f"{decision.reason}"
            )
        if decision.outcome == "require_approval":
            # The bridge has no resume path of its own, so a hook asking for approval
            # stops the call here, the same as the POST step withholds the result.
            raise PermissionError(
                f"MCP tool '{server_name}/{tool_name}' needs approval before it runs: "
                f"{decision.reason}"
            )
        final_args = args_of(final_ctx.payload)
        return dict(final_args) if final_args is not None else dict(arguments)

    def _fire_governance_post_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        result: Any,
        persona: str | None,
        run_id: str | None,
        call_id: str,
        effect: str | None = None,
    ) -> Any:
        """Fire ``PostToolUse`` over an MCP server's result; what the caller may receive.

        An external server's output is third-party text, so it is declared
        ``content: external``; its effect is the one the operator declared for the tool, or
        the server's ``undeclared_tools`` default (an MCP server declares none itself).
        ``deny`` / ``require_approval`` raise :exc:`PermissionError` (the result is
        withheld); a ``transform`` is returned.
        """
        from iris_harness.kernel.governance.audit.digest import audit_digester
        from iris_harness.kernel.governance.hooks.tool_payload import (
            post_tool_payload,
            result_of,
            tool_post_metadata,
        )
        from iris_harness.kernel.governance.hooks.types import (
            HookContext,
            HookPoint,
        )

        assert self._governance_kernel is not None  # caller guards
        route = f"mcp/{server_name}/{tool_name}"
        ctx = HookContext(
            hook_point=HookPoint.POST_TOOL_USE,
            run_id=run_id or f"mcp-{server_name}",
            agent_type="mcp",
            persona=persona,
            route=route,
            # The key was resolved (and cached) by the PRE step that let this call run.
            payload=post_tool_payload(
                route,
                result,
                mcp_server=server_name,
                mcp_tool=tool_name,
                tool_plugin=f"mcp:{server_name}",
                **audit_digester().result_fields(result),
            ),
            metadata=tool_post_metadata(
                effect=effect, content="external", verify=None, tool_call_id=call_id
            ),
        )
        decision, final_ctx = self._governance_kernel.fire_sync(HookPoint.POST_TOOL_USE, ctx)
        if decision.outcome in ("deny", "require_approval"):
            raise PermissionError(
                f"MCP tool '{server_name}/{tool_name}' result withheld by governance: "
                f"{decision.reason}"
            )
        return result_of(final_ctx.payload)

    def _require_enabled_server(self, server_name: str) -> MCPServerConfig:
        """Return one enabled server, or raise a clear config error."""
        if not self.config.enabled:
            raise ValueError("MCP bridge is disabled in the effective coding MCP bridge config")

        server = self._server_index.get(server_name)
        if server is None:
            raise ValueError(f"unknown MCP server: {server_name}")
        if not server.enabled:
            raise ValueError(f"MCP server '{server_name}' is disabled")
        return server

    def _guard_mcp_action(
        self,
        action: str,
        *,
        server_name: str,
        approval_granted: bool,
        metadata: dict[str, Any] | None = None,
    ) -> GovernorGuardDecision:
        """Evaluate one outbound MCP action through the local governor service.

        Approval comes from the ``approval_granted`` argument alone. ``metadata`` is
        free-form and can arrive from an HTTP body, and the governor treats an
        ``approved`` or ``approval_granted`` key in it as consent, so both keys are
        dropped from it and the argument is written last. Before this, a caller could
        send ``{"approved": true}`` in metadata and approve its own call.
        """
        payload_metadata: dict[str, Any] = {"server_name": server_name}
        if metadata:
            payload_metadata.update(
                {key: value for key, value in metadata.items() if key not in _APPROVAL_KEYS}
            )
        payload_metadata["approval_granted"] = approval_granted
        return self.governor_service.guard(
            "coding/mcp",
            {
                "action": action,
                "metadata": payload_metadata,
            },
        )

    def _invoke_transport(
        self,
        server: MCPServerConfig,
        *,
        method: str,
        params: dict[str, Any],
    ) -> Any:
        """Invoke one MCP method over the configured transport."""
        # Phase 6 G (signing): verify the server's signature before any process
        # is launched or any endpoint is contacted. Shadow → warn + proceed;
        # enforce + deny → raise (refuse) before the request is built.
        self._enforce_signature(server)
        request_payload = _build_jsonrpc_request(
            request_id=f"iris-{server.name}-{method.replace('/', '-')}",
            method=method,
            params=params,
        )
        if server.transport == "http":
            response_payload = self._http_requester(server, request_payload)
            return _extract_mcp_result(response_payload)
        if server.transport == "stdio":
            return self._invoke_stdio_transport(server, request_payload)
        if server.transport == "sse":
            return self._invoke_sse_transport(server, request_payload)

        raise ValueError(
            f"MCP transport '{server.transport}' is not implemented yet; supported transports: http, stdio, sse"
        )

    def _verify_signature(self, server: MCPServerConfig) -> SignatureVerdict:
        return verify_server_signature(
            spec=build_server_spec(server),
            signature=server.signature,
            signed_by=server.signed_by,
            trust_store=self._trust_store,
        )

    def _enforce_signature(self, server: MCPServerConfig) -> None:
        """Verify the server signature and apply the configured policy.

        No-op when signing is disabled. In ``shadow`` mode every non-verified
        verdict logs a warning but proceeds; in ``enforce`` mode the configured
        ``unsigned_policy`` applies and ``deny`` raises ``MCPSignatureError``."""
        if not self._signing_config.enabled:
            return
        verdict = self._verify_signature(server)
        action = resolve_signing_action(verdict, self._signing_config)
        if action == "deny":
            logger.warning(
                "mcp-signing[enforce] DENY server=%s status=%s reason=%s",
                server.name,
                verdict.status,
                verdict.reason,
            )
            raise MCPSignatureError(
                f"MCP server {server.name!r} failed signature verification: "
                f"{verdict.status} ({verdict.reason})"
            )
        if not verdict.is_trusted:
            logger.warning(
                "mcp-signing[%s] server=%s status=%s reason=%s key_id=%s (proceeding)",
                self._signing_config.mode,
                server.name,
                verdict.status,
                verdict.reason,
                verdict.key_id,
            )

    def _invoke_stdio_transport(
        self,
        server: MCPServerConfig,
        request_payload: dict[str, Any],
    ) -> Any:
        """Invoke one MCP method over a one-shot stdio session with initialize handshake."""
        initialize_payload = _build_jsonrpc_request(
            request_id=f"iris-{server.name}-initialize",
            method="initialize",
            params={
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "iris", "version": "0.1.0"},
            },
        )
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        responses = _run_stdio_exchange(
            repo_root=self.repo_root,
            server=server,
            outgoing_messages=(initialize_payload, initialized_notification, request_payload),
            expected_response_ids=(
                str(initialize_payload["id"]),
                str(request_payload["id"]),
            ),
        )
        _extract_mcp_result(responses[str(initialize_payload["id"])])
        return _extract_mcp_result(responses[str(request_payload["id"])])

    def _invoke_sse_transport(
        self,
        server: MCPServerConfig,
        request_payload: dict[str, Any],
    ) -> Any:
        """Invoke one MCP method over an SSE session with initialize handshake."""
        initialize_payload = _build_jsonrpc_request(
            request_id=f"iris-{server.name}-initialize",
            method="initialize",
            params={
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "iris", "version": "0.1.0"},
            },
        )
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        responses = self._sse_exchange_runner(
            server,
            (initialize_payload, initialized_notification, request_payload),
            (str(initialize_payload["id"]), str(request_payload["id"])),
        )
        _extract_mcp_result(responses[str(initialize_payload["id"])])
        return _extract_mcp_result(responses[str(request_payload["id"])])


def _build_jsonrpc_request(
    *,
    request_id: str,
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build one JSON-RPC request payload."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }


def _default_http_requester(server: MCPServerConfig, payload: dict[str, Any]) -> Any:
    """Execute one JSON-RPC request against an HTTP MCP endpoint."""
    if server.url is None:
        raise ValueError(f"MCP server '{server.name}' is missing an HTTP URL")

    log_egress(
        destination=urlparse(server.url).netloc,
        method="POST",
        kind="mcp",
        purpose="mcp.http",
    )
    response = httpx.post(server.url, json=payload, timeout=server.timeout_seconds)
    response.raise_for_status()
    return response.json()


def _default_sse_exchange_runner(
    server: MCPServerConfig,
    outgoing_messages: tuple[dict[str, Any], ...],
    expected_response_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Run one MCP session over SSE and collect the expected JSON-RPC responses."""
    if server.url is None:
        raise ValueError(f"MCP server '{server.name}' is missing an SSE URL")

    try:
        with httpx.Client(timeout=server.timeout_seconds) as client:
            log_egress(
                destination=urlparse(server.url).netloc,
                method="GET",
                kind="mcp",
                purpose="mcp.sse",
            )
            with client.stream(
                "GET",
                server.url,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                event_stream = _iter_sse_events(response.iter_lines())
                message_endpoint = _resolve_sse_message_endpoint(server.url, event_stream)

                for message in outgoing_messages:
                    log_egress(
                        destination=urlparse(message_endpoint).netloc,
                        method="POST",
                        kind="mcp",
                        purpose="mcp.sse",
                    )
                    post_response = client.post(message_endpoint, json=message)
                    post_response.raise_for_status()

                responses: dict[str, Any] = {}
                expected_ids = set(expected_response_ids)
                for _event_name, data in event_stream:
                    if not data.strip():
                        continue
                    payload = json.loads(data)
                    if not isinstance(payload, dict):
                        continue
                    payload_id = payload.get("id")
                    if payload_id is None:
                        continue
                    normalized_id = str(payload_id)
                    if normalized_id in expected_ids:
                        responses[normalized_id] = payload
                    if len(responses) == len(expected_ids):
                        return responses
    except httpx.TimeoutException as exc:
        raise ValueError(
            f"timed out waiting for MCP SSE responses from server '{server.name}'"
        ) from exc

    raise ValueError(f"incomplete MCP SSE exchange from server '{server.name}'")


def _resolve_sse_message_endpoint(
    base_url: str,
    event_stream: Any,
) -> str:
    """Read the initial SSE endpoint event and resolve the message POST URL."""
    for event_name, data in event_stream:
        if event_name == "endpoint":
            endpoint = data.strip()
            if not endpoint:
                break
            return cast(str, urljoin(base_url, endpoint))
    raise ValueError("MCP SSE stream did not publish an endpoint event")


def _iter_sse_events(lines: Any) -> Any:
    """Yield parsed SSE events as `(event_name, data)` tuples."""
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
        stripped = line.rstrip("\r\n")
        if not stripped:
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = "message"
            data_lines = []
            continue
        if stripped.startswith(":"):
            continue
        if stripped.startswith("event:"):
            event_name = stripped.partition(":")[2].strip() or "message"
            continue
        if stripped.startswith("data:"):
            data_lines.append(stripped.partition(":")[2].lstrip())

    if data_lines:
        yield event_name, "\n".join(data_lines)


def _run_stdio_exchange(
    *,
    repo_root: Path,
    server: MCPServerConfig,
    outgoing_messages: tuple[dict[str, Any], ...],
    expected_response_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Run a one-shot stdio MCP session and collect the expected JSON-RPC responses."""
    if server.command is None:
        raise ValueError(f"MCP server '{server.name}' is missing a stdio command")

    environment = os.environ.copy()
    environment.update(server.env)
    process = subprocess.Popen(
        [server.command, *server.args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(repo_root),
        env=environment,
    )

    try:
        if process.stdin is None or process.stdout is None:
            raise ValueError(f"unable to open stdio pipes for MCP server '{server.name}'")

        for message in outgoing_messages:
            process.stdin.write(_encode_stdio_message(message))
        process.stdin.flush()

        deadline = time.monotonic() + server.timeout_seconds
        responses: dict[str, Any] = {}
        while len(responses) < len(expected_response_ids):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out waiting for MCP stdio responses from server '{server.name}'"
                )

            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError(
                    f"timed out waiting for MCP stdio responses from server '{server.name}'"
                )

            payload = _read_stdio_message(process.stdout)
            payload_id = payload.get("id")
            if payload_id is None:
                continue
            responses[str(payload_id)] = payload

        return responses
    except TimeoutError:
        _terminate_process(process)
        stderr_output = _read_process_stderr(process)
        timeout_message = f"timed out waiting for MCP stdio responses from server '{server.name}'"
        if stderr_output:
            timeout_message = f"{timeout_message}: {stderr_output}"
        raise ValueError(timeout_message) from None
    finally:
        _terminate_process(process)


def _encode_stdio_message(payload: dict[str, Any]) -> bytes:
    """Encode one MCP stdio message using the current SDK newline-delimited format."""
    return (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _read_stdio_message(stream: Any) -> dict[str, Any]:
    """Read one MCP stdio message from either newline-delimited or legacy framed output."""
    raw_line = stream.readline()
    if not raw_line:
        raise ValueError("unexpected EOF while reading MCP stdio message")

    stripped_line = raw_line.decode("utf-8", errors="ignore").strip()
    if stripped_line.startswith("{"):
        payload = json.loads(stripped_line)
        if not isinstance(payload, dict):
            raise ValueError("MCP stdio response must decode to a JSON object")
        return payload

    headers = [raw_line.decode("ascii", errors="ignore").strip()]
    while True:
        raw_line = stream.readline()
        if not raw_line:
            raise ValueError("unexpected EOF while reading MCP stdio headers")
        if raw_line in {b"\r\n", b"\n"}:
            break
        headers.append(raw_line.decode("ascii", errors="ignore").strip())

    content_length = _parse_content_length(headers)
    body = stream.read(content_length)
    if len(body) != content_length:
        raise ValueError("unexpected EOF while reading MCP stdio body")

    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MCP stdio response must decode to a JSON object")
    return payload


def _parse_content_length(headers: list[str]) -> int:
    """Extract Content-Length from MCP stdio headers."""
    for header in headers:
        name, separator, value = header.partition(":")
        if separator and name.lower() == "content-length":
            return int(value.strip())
    raise ValueError("MCP stdio message is missing Content-Length")


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate a spawned stdio MCP process safely."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


def _read_process_stderr(process: subprocess.Popen[bytes]) -> str:
    """Collect stderr output from a subprocess without raising."""
    if process.stderr is None:
        return ""
    try:
        return process.stderr.read().decode("utf-8", errors="ignore").strip()
    except Exception:  # noqa: BLE001 — stderr is diagnostic only
        return ""


def _extract_mcp_result(payload: Any) -> Any:
    """Extract the result payload from a JSON-RPC response."""
    if isinstance(payload, dict):
        error = payload.get("error")
        if error is not None:
            raise ValueError(_format_mcp_error(error))
        if "result" in payload:
            return payload["result"]
    return payload


def _format_mcp_error(error: Any) -> str:
    """Format a JSON-RPC error into a readable message."""
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message", "Unknown MCP error")
        return f"MCP error {code}: {message}" if code is not None else str(message)
    return str(error)


def _coerce_mcp_tools_payload(payload: Any) -> list[dict[str, Any]]:
    """Validate that an MCP tools/list result contains a tool array."""
    if not isinstance(payload, dict):
        raise ValueError("MCP tools/list response must be a JSON object")
    tools = payload.get("tools")
    if not isinstance(tools, list):
        raise ValueError("MCP tools/list response must include a tools array")
    if not all(isinstance(tool, dict) for tool in tools):
        raise ValueError("MCP tools/list entries must be JSON objects")
    return list(tools)


def _coerce_external_tool_definition(
    server: MCPServerConfig,
    payload: dict[str, Any],
) -> MCPToolDefinition:
    """Convert an MCP tool payload into the shared tool definition model."""
    input_schema = payload.get("inputSchema") or payload.get("input_schema")
    if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}, "additionalProperties": True}

    return MCPToolDefinition(
        name=str(payload.get("name", "")),
        description=str(payload.get("description", "Tool exposed by external MCP server")),
        input_schema=input_schema,
        governor_route="coding/mcp",
        source_kind="mcp_server",
        source_name=server.name,
    )


def _build_tool_input_schema(tool_class: type[BaseTool]) -> dict[str, Any]:
    """Convert a LangChain tool args schema into an MCP-style JSON schema."""
    schema: dict[str, Any] | None = None
    try:
        tool_instance = tool_class()
        input_schema = tool_instance.get_input_schema()
        # LangChain 1.x types this as a pydantic v2 or v1 model; only v2 has
        # model_json_schema.
        runtime_schema = (
            input_schema.model_json_schema()
            if issubclass(input_schema, BaseModel)
            else input_schema.schema()
        )
        if isinstance(runtime_schema, dict):
            return runtime_schema
    except Exception:  # noqa: BLE001 — no runtime schema falls back to the declared one
        schema = None

    model_field = getattr(tool_class, "model_fields", {}).get("args_schema")
    args_schema = getattr(model_field, "default", None)
    if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
        declared_schema = args_schema.model_json_schema()
        if isinstance(declared_schema, dict):
            schema = declared_schema
    if schema is not None:
        return schema
    return {"type": "object", "properties": {}, "additionalProperties": True}


_PERSONA_BATCH_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "for_each_story": {"type": "boolean", "default": False},
        "scope": {
            "type": "string",
            "enum": ["platform", "project"],
            "default": "platform",
        },
        "project_slug": {"type": ["string", "null"], "default": None},
        "for_each_section": {"type": ["string", "null"], "default": None},
        "for_each_glob": {"type": ["string", "null"], "default": None},
        "items": {
            "type": "array",
            "items": {"type": "object"},
            "default": [],
        },
    },
    "additionalProperties": False,
}


def _build_generic_persona_tool_definition() -> MCPToolDefinition:
    """Build the generic ``iris_invoke_persona`` tool definition."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "persona": {
                "type": ["string", "null"],
                "description": "Persona to invoke (sm, analyst, architect, developer, tester).",
            },
            "command": {
                "type": ["string", "null"],
                "description": "Command id owned by the persona.",
            },
            "batch": _PERSONA_BATCH_SPEC_SCHEMA,
            "dry_run": {"type": "boolean", "default": False},
            "allow_writes": {"type": "boolean", "default": False},
            "list_available": {
                "type": "boolean",
                "default": False,
                "description": "Return the persona → command mapping instead of invoking.",
            },
        },
        "additionalProperties": False,
    }
    return MCPToolDefinition(
        name="iris_invoke_persona",
        description=(
            "Invoke one persona + command ad-hoc over a batch of items "
            "(story/section/glob/explicit). Bypasses the bootstrap pipeline."
        ),
        input_schema=schema,
        governor_route="coding/tool/persona-invoke",
        source_kind="persona",
        source_name="generic",
    )


def _build_persona_command_tool_definition(
    profile: Any,
    command: Any,
) -> MCPToolDefinition:
    """Build a per-persona-per-command shim tool definition."""
    safe_persona = profile.name.replace("-", "_")
    safe_command = command.id.replace("-", "_")
    tool_name = f"iris_{safe_persona}_{safe_command}"
    description_lines = profile.description.splitlines()
    head = description_lines[0].strip() if description_lines else profile.name
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "batch": _PERSONA_BATCH_SPEC_SCHEMA,
            "dry_run": {"type": "boolean", "default": False},
            "allow_writes": {"type": "boolean", "default": False},
        },
        "additionalProperties": False,
    }
    return MCPToolDefinition(
        name=tool_name,
        description=f"{head} — {command.id}",
        input_schema=schema,
        governor_route=f"coding/tool/persona-{safe_persona}-{safe_command}",
        source_kind="persona",
        source_name=profile.name,
    )
