"""MCPAllowlistHook — Phase 4 ``PreToolUse`` MCP server per-persona allowlist.

Story 12.gov-4.6. Enforces the ``(persona, server, tool)`` allowlist
declared in the ``governance.allowed_for_personas`` block of
``mcp-servers.yaml``. This closes the design §8.5 gap: a ``tester``
persona cannot call a destructive MCP tool that only the ``developer``
persona should touch.

Priority **18** at ``PreToolUse`` — between ``PersonaSurface`` (15) and
``ToolPolicyHook`` (20). PersonaSurface enforces the coarse persona ×
tool surface; MCPAllowlistHook adds fine-grained ``(persona, server, tool)``
enforcement using the YAML-declared server-side rules.

The hook fires for **any** ``PreToolUse`` payload that carries
``mcp_server``. The MCP bridge is responsible for setting that field;
all other dispatches short-circuit with ``allow``.

Server governance policy:

- **No governance block** for a server → ``warn`` audit + ``allow``
  (Phase 4 ships the surface; fail-closed is a v1.1 follow-up once
  every server gets a governance block).
- **Persona not listed** in ``allowed_for_personas`` → ``deny``
  (default-deny per persona).
- **Persona listed, tool allowed** (fnmatch wildcards supported) → ``allow``.
- **Persona listed, tool not matched** → ``deny``.
- **``ctx.persona is None``** → ``deny`` (MCP bridge must carry persona).

Signature verification stub (design §8.5 footnote):
:func:`verify_mcp_signature` returns ``True`` for all servers in v1.
Phase 6 (``mcp-server-signing.md``) will replace this with real
manifest signing. Every call is audited at ``info`` severity so the
list of servers that will need signing is already visible in the audit
log before Phase 6 ships.
"""

from __future__ import annotations

import logging
from fnmatch import fnmatch
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MCPPersonaGrant(BaseModel):
    """One persona's tool grant within a single MCP server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    persona: str = Field(..., min_length=1)
    tools: tuple[str, ...] = Field(default_factory=tuple)


class MCPToolGovernance(BaseModel):
    """What the operator declares about one tool of an MCP server.

    An MCP server declares no effect of its own, and its own hints (``destructiveHint``) are
    the server's claim, not the operator's, so the bridge never trusts them. The operator
    says what a tool does here; the bridge then treats the call like a plugin tool of that
    effect (a ``destructive`` one waits for an itemised approval and leaves a write-ahead
    ledger row before it runs).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    effect: Literal["read", "write", "destructive"]


class MCPServerGovernance(BaseModel):
    """Governance block for one MCP server (``governance:`` key in YAML)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_for_personas: tuple[MCPPersonaGrant, ...] = Field(default_factory=tuple)
    # Tool name -> what the operator declares it does. A tool not listed declares nothing
    # and is handled as before: no effect, no write-ahead row.
    tools: dict[str, MCPToolGovernance] = Field(default_factory=dict)

    @field_validator("tools")
    @classmethod
    def _tool_names_are_not_empty(
        cls, value: dict[str, MCPToolGovernance]
    ) -> dict[str, MCPToolGovernance]:
        if any(not name.strip() for name in value):
            raise ValueError("an MCP tool name in governance.tools must not be empty")
        return value


# ---------------------------------------------------------------------------
# Signature verification stub (SUPERSEDED — see iris_harness.kernel.governance.mcp_signing)
# ---------------------------------------------------------------------------


def verify_mcp_signature(server_name: str, manifest_path: Path) -> bool:
    """Legacy always-True stub. **Superseded** by real Ed25519 verification in
    :mod:`iris_harness.kernel.governance.mcp_signing` (Phase 6, 6b.1/6b.2), which is wired into
    the MCP bridge launch gate (``MCPBridge._enforce_signature``).

    This shim is retained only for backward compatibility and is **not** on the
    enforcement path. It always returns ``True``; do not use it for new checks —
    call :func:`iris_harness.kernel.governance.mcp_signing.verify_server_signature` instead.
    """
    logger.info(
        "mcp_allowlist: verify_mcp_signature called for server=%r path=%s "
        "(stub always-True; Phase 6 will implement real signing)",
        server_name,
        manifest_path,
    )
    return True


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------


class MCPAllowlistHook:
    """Enforce per-persona MCP server tool allowlist at ``PreToolUse``.

    Construct with a mapping of ``server_name → MCPServerGovernance``
    built from the ``governance:`` blocks in ``mcp-servers.yaml``. Pass
    ``governance_map=None`` for degraded no-op mode (all MCP calls
    allowed, one ``WARNING`` per instance on first miss).
    """

    name: str = "mcp_allowlist"
    hook_point: HookPoint = HookPoint.PRE_TOOL_USE
    priority: int = 18  # between PersonaSurface (15) and ToolPolicyHook (20)

    ALLOW_REASON_NOT_MCP: ClassVar[str] = "mcp_allowlist: not an MCP dispatch"
    ALLOW_REASON_DEGRADED: ClassVar[str] = "mcp_allowlist: no governance map (degraded)"
    ALLOW_REASON_UNGOVERNED: ClassVar[str] = "mcp_allowlist: server has no governance block (warn)"

    def __init__(
        self,
        *,
        governance_map: dict[str, MCPServerGovernance] | None = None,
    ) -> None:
        self._map = governance_map
        self._warned_no_map = False

    async def __call__(self, ctx: HookContext) -> HookDecision:
        mcp_server = ctx.payload.get("mcp_server")
        if not isinstance(mcp_server, str) or not mcp_server:
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_NOT_MCP)

        mcp_tool = ctx.payload.get("mcp_tool")
        tool_label = str(mcp_tool) if isinstance(mcp_tool, str) else "<unknown>"

        if self._map is None:
            if not self._warned_no_map:
                logger.warning(
                    "MCPAllowlistHook: no governance map loaded; MCP tool calls "
                    "will be allowed without allowlist enforcement"
                )
                self._warned_no_map = True
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_DEGRADED)

        governance = self._map.get(mcp_server)
        if governance is None:
            logger.warning(
                "mcp_allowlist: server %r has no governance block — "
                "allowed without enforcement (ungoverned)",
                mcp_server,
            )
            return HookDecision(
                outcome="allow",
                reason=self.ALLOW_REASON_UNGOVERNED,
                severity="warn",
                audit_metadata={"mcp_server": mcp_server, "mcp_tool": tool_label},
            )

        if ctx.persona is None:
            return HookDecision(
                outcome="deny",
                reason=(
                    f"mcp_allowlist: MCP dispatch to server {mcp_server!r} "
                    "missing persona context"
                ),
                severity="error",
                audit_metadata={"mcp_server": mcp_server, "mcp_tool": tool_label},
            )

        persona_grant = _find_persona_grant(governance, ctx.persona)
        if persona_grant is None:
            return HookDecision(
                outcome="deny",
                reason=(
                    f"mcp_allowlist: persona {ctx.persona!r} not listed in "
                    f"server {mcp_server!r} allowed_for_personas"
                ),
                severity="error",
                audit_metadata={
                    "mcp_server": mcp_server,
                    "mcp_tool": tool_label,
                    "persona": ctx.persona,
                    "deny_reason": "persona_not_allowed",
                },
            )

        if not isinstance(mcp_tool, str) or not mcp_tool:
            return HookDecision(
                outcome="deny",
                reason=(
                    f"mcp_allowlist: MCP dispatch to server {mcp_server!r} "
                    "missing mcp_tool in payload"
                ),
                severity="error",
                audit_metadata={"mcp_server": mcp_server, "persona": ctx.persona},
            )

        for pattern in persona_grant.tools:
            if fnmatch(mcp_tool, pattern):
                return HookDecision(
                    outcome="allow",
                    reason=(
                        f"mcp_allowlist: persona {ctx.persona!r} allowed to call "
                        f"{mcp_server!r}/{mcp_tool!r}"
                    ),
                    audit_metadata={
                        "mcp_server": mcp_server,
                        "mcp_tool": mcp_tool,
                        "persona": ctx.persona,
                        "matched_pattern": pattern,
                    },
                )

        return HookDecision(
            outcome="deny",
            reason=(
                f"mcp_allowlist: persona {ctx.persona!r} tool {mcp_tool!r} "
                f"not in server {mcp_server!r} allowlist"
            ),
            severity="error",
            audit_metadata={
                "mcp_server": mcp_server,
                "mcp_tool": mcp_tool,
                "persona": ctx.persona,
                "deny_reason": "tool_not_allowed",
            },
        )


def _find_persona_grant(governance: MCPServerGovernance, persona: str) -> MCPPersonaGrant | None:
    for grant in governance.allowed_for_personas:
        if grant.persona == persona:
            return grant
    return None
