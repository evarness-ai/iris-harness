"""PersonaSurface — Phase 4 ``PreToolUse`` per-persona tool-surface guard.

Story 12.gov-4.2. Enforces the design §8.1 declarative tool surface:
each coding-agent persona declares ``allowed_tools`` in
``src/iris_code/resources/config/persona-policy.yaml``; PersonaSurface
denies any dispatch outside that list. The ``orchestrator`` persona is
delegation-only — hard-coded here, not policy-driven, because making it
configurable would let a future policy override accidentally re-grant
direct tool access (a known OWASP ASI03 anti-pattern).

Priority **15** at ``PreToolUse``: runs before ``ToolPolicyHook`` (20)
and the credential broker (25-ish) so a denied persona dispatch never
gets vault handles resolved.

Non-coding-agent dispatches (chat, voice, ...) short-circuit with
``allow`` — this plugin is the coding-pipeline surface only.
"""

from __future__ import annotations

import logging
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, ClassVar, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

from iris_harness.kernel.governance.hooks.types import (
    DataClassification,
    HookContext,
    HookDecision,
    HookPoint,
)

logger = logging.getLogger(__name__)

# Personas named in src/iris_code/resources/personas/*.agent.md.
# Must match the canonical roster — a policy listing an unknown persona
# (or omitting one) is a misconfiguration we surface at load time.
CANONICAL_PERSONAS: Final[frozenset[str]] = frozenset(
    {
        "analyst",
        "architect",
        "developer",
        "orchestrator",
        "sm",
        "tester",
        "ux-designer",
    }
)


class PersonaPolicy(BaseModel):
    """One persona's tool surface.

    Only ``allowed_tools`` is consumed in 12.gov-4.2. The remaining
    fields are declared so the YAML doc validates cleanly today and
    later plugins (CommandSandbox / FSJail / NetworkEgress / classifier
    integration) can read them without a schema migration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    allowed_tools: tuple[str, ...] = Field(default_factory=tuple)
    fs_read_only: bool = False
    fs_write_jail: tuple[str, ...] = Field(default_factory=tuple)
    # Consumed by CommandSandbox (12.gov-4.3):
    command_allowlist: tuple[str, ...] = Field(default_factory=tuple)
    command_denylist_args: tuple[str, ...] = Field(default_factory=tuple)
    command_shell_opt_in: bool = False
    # Forward-compat fields (inert until their owning plugin ships):
    network_egress_domains: tuple[str, ...] = Field(default_factory=tuple)
    classes_max: DataClassification | None = None


class PersonaPolicyDocument(BaseModel):
    """The parsed contents of one ``persona-policy.yaml`` file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    personas: dict[str, PersonaPolicy] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> PersonaPolicyDocument:
        """Load + validate a persona policy from disk.

        Asserts every canonical persona has a record; raises ``ValueError``
        on missing or unknown persona names so misconfigurations surface
        loudly at startup instead of producing silent permissive runs.
        """
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"persona-policy file must decode to a mapping: {path}")
        persona_block = raw.get("personas", {})
        if not isinstance(persona_block, dict):
            raise ValueError(f"persona-policy 'personas' key must be a mapping: {path}")

        parsed: dict[str, PersonaPolicy] = {}
        for name, body in persona_block.items():
            if not isinstance(body, dict):
                raise ValueError(f"persona-policy entry for {name!r} must be a mapping: {path}")
            parsed[str(name)] = PersonaPolicy(name=str(name), **body)

        names = frozenset(parsed.keys())
        missing = CANONICAL_PERSONAS - names
        extra = names - CANONICAL_PERSONAS
        if missing or extra:
            raise ValueError(
                f"persona-policy roster mismatch at {path}: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        return cls(personas=parsed)


class PersonaSurface:
    """Deny tool dispatches outside the active persona's ``allowed_tools``.

    Construct with a ``PersonaPolicyDocument`` (loaded from
    ``persona-policy.yaml`` or built in-memory for tests). Constructing
    with ``policy=None`` is a degraded-mode no-op: the plugin returns
    ``allow`` for every coding dispatch and logs a `warn` audit on the
    first miss. This is for tests and for installs without the coding
    resources bundle; ``build_default_kernel`` loads the packaged file
    by default so production deployments always enforce.
    """

    name: str = "persona_surface"
    hook_point: HookPoint = HookPoint.PRE_TOOL_USE
    priority: int = 15

    #: Hard-coded surface for the orchestrator persona. NOT policy-driven —
    #: see module docstring. Anything outside this set denies with severity
    #: ``error`` regardless of what the policy file says for ``orchestrator``.
    ORCHESTRATOR_ALLOWED: ClassVar[frozenset[str]] = frozenset({"delegate_to_persona", "read_file"})

    #: Marker reason strings — exposed as class constants so tests can match
    #: on them without coupling to the precise message wording.
    DENY_REASON_MISSING_PERSONA: ClassVar[str] = "coding-agent dispatch missing persona"
    DENY_REASON_ORCHESTRATOR_DELEGATION_ONLY: ClassVar[str] = (
        "orchestrator persona is delegation-only"
    )
    ALLOW_REASON_NON_CODING: ClassVar[str] = "not a coding-agent dispatch"

    def __init__(self, *, policy: PersonaPolicyDocument | None = None) -> None:
        self._policy = policy
        self._warned_no_policy = False

    async def __call__(self, ctx: HookContext) -> HookDecision:
        # Non-coding agent paths (chat, voice, ...) pass straight through.
        # PersonaSurface is the coding-pipeline tool-surface guard only.
        if ctx.agent_type != "coding":
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_NON_CODING)

        # Degraded mode: policy not loaded. Allow + warn once so tests and
        # bundle-less installs aren't blocked, but the operator sees the
        # warning on first dispatch.
        if self._policy is None:
            if not self._warned_no_policy:
                logger.warning(
                    "PersonaSurface: no persona-policy loaded; coding dispatches "
                    "will be allowed without surface enforcement"
                )
                self._warned_no_policy = True
            return HookDecision(
                outcome="allow", reason="persona_surface: no policy loaded (degraded)"
            )

        if ctx.persona is None:
            return HookDecision(
                outcome="deny",
                reason=self.DENY_REASON_MISSING_PERSONA,
                severity="error",
            )

        tool_name = self._extract_tool_name(ctx.payload)
        if tool_name is None:
            return HookDecision(
                outcome="deny",
                reason="persona_surface: missing tool_name in PreToolUse context",
                severity="error",
            )

        # Orchestrator: hard-coded delegation-only. Policy file ignored.
        if ctx.persona == "orchestrator":
            if tool_name in self.ORCHESTRATOR_ALLOWED:
                return HookDecision(
                    outcome="allow",
                    reason=f"orchestrator may invoke {tool_name!r}",
                )
            return HookDecision(
                outcome="deny",
                reason=(
                    f"{self.DENY_REASON_ORCHESTRATOR_DELEGATION_ONLY}; "
                    f"cannot invoke {tool_name!r}"
                ),
                severity="error",
            )

        persona_policy = self._policy.personas.get(ctx.persona)
        if persona_policy is None:
            return HookDecision(
                outcome="deny",
                reason=f"persona_surface: unknown persona {ctx.persona!r}",
                severity="error",
            )

        for pattern in persona_policy.allowed_tools:
            if fnmatch(tool_name, pattern):
                return HookDecision(
                    outcome="allow",
                    reason=f"persona {ctx.persona!r} allows {tool_name!r}",
                )

        return HookDecision(
            outcome="deny",
            reason=(f"persona {ctx.persona!r} not allowed to invoke {tool_name!r}"),
        )

    @staticmethod
    def _extract_tool_name(payload: dict[str, Any]) -> str | None:
        raw = payload.get("tool_name")
        if isinstance(raw, str) and raw:
            return raw
        return None
