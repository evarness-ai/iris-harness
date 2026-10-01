"""CommandSandbox — Phase 4 ``PreToolUse`` shell-command guard.

Story 12.gov-4.3. Enforces each persona's ``command_allowlist``,
``command_denylist_args`` and a shell-metacharacter guard for tools
that execute shell commands. Closes OWASP ASI02 (Tool Misuse) for the
coding pipeline and satisfies design §8.1 exit criterion #1: a
``developer``-persona run that attempts ``rm -rf src/`` is denied at
``PreToolUse`` before the command reaches the sandbox runner.

Priority **25** at ``PreToolUse`` — runs after ``PersonaSurface``
(15) and ``ToolPolicyHook`` (20). By the time CommandSandbox fires,
the tool is already known to be inside the persona's surface; this
hook drills into the *arguments* of allowed shell tools and rejects
disallowed binaries, dangerous argument patterns, and shell
metacharacters that would slip past argv-style execution.

Order of checks (chosen to make AC-2 fire with the production developer
policy where ``rm`` is not in the allowlist — denylist runs first so the
audit row names the dangerous *pattern*, not a generic "binary missing
from allowlist" reason):

1. Denylist — literal substring match against the joined argv. Patterns
   like ``rm -rf`` and ``chmod 777`` are denied regardless of binary.
2. Allowlist — first argv token (the binary) must match an allowlist
   entry. Wildcards via :func:`fnmatch.fnmatch`.
3. Metachar guard — any token containing an unescaped shell
   metacharacter denies unless the payload sets ``shell=True`` *and*
   the persona declares ``command_shell_opt_in: true`` (which emits a
   ``warn`` audit and allows).

Audit metadata always carries the resolved ``binary`` and ``argv`` list
— **never** the raw command string — so a malicious newline embedded in
a command token cannot inject extra log lines (the audit sink JSON-
encodes the list, turning ``\n`` into a literal escape).
"""

from __future__ import annotations

import logging
import shlex
from fnmatch import fnmatch
from typing import Any, ClassVar, Final

from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.plugins.persona_surface import PersonaPolicyDocument

logger = logging.getLogger(__name__)

#: Default set of tool names this plugin considers "shell runners". The
#: real coding-agent tool is ``run_allowed_command``; ``run_command`` /
#: ``shell`` / ``bash`` are listed for forward compatibility with future
#: tool names and to match the story spec.
DEFAULT_SHELL_TOOLS: Final[frozenset[str]] = frozenset(
    {"run_command", "shell", "bash", "run_allowed_command"}
)

#: Tokens whose presence inside any argv element trips the metachar guard.
#: Multi-char patterns (``&&``, ``||``) are checked alongside single chars
#: so ``cmd && other`` denies even when shlex keeps them as one token.
SHELL_METACHARS: Final[tuple[str, ...]] = (
    "$",
    "`",
    "|",
    ">",
    "<",
    ";",
    "&&",
    "||",
    "\n",
    "\r",
)


class CommandSandbox:
    """Enforce per-persona shell-command allow/deny + metachar guard."""

    name: str = "command_sandbox"
    hook_point: HookPoint = HookPoint.PRE_TOOL_USE
    priority: int = 25

    ALLOW_REASON_NOT_SHELL_TOOL: ClassVar[str] = "command_sandbox: not a shell tool"
    ALLOW_REASON_NON_CODING: ClassVar[str] = "command_sandbox: not a coding-agent dispatch"
    ALLOW_REASON_DEGRADED: ClassVar[str] = "command_sandbox: no policy loaded (degraded)"

    def __init__(
        self,
        *,
        policy: PersonaPolicyDocument | None = None,
        shell_tools: frozenset[str] = DEFAULT_SHELL_TOOLS,
    ) -> None:
        self._policy = policy
        self._shell_tools = shell_tools
        self._warned_no_policy = False

    async def __call__(self, ctx: HookContext) -> HookDecision:
        tool_name = ctx.payload.get("tool_name")
        if not isinstance(tool_name, str) or tool_name not in self._shell_tools:
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_NOT_SHELL_TOOL)

        if ctx.agent_type != "coding":
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_NON_CODING)

        if ctx.persona is None:
            return HookDecision(
                outcome="deny",
                reason="command_sandbox: coding-agent dispatch missing persona",
                severity="error",
            )

        # Defense in depth: orchestrator is delegation-only at PersonaSurface
        # (priority 15) and would never reach here. If wiring ever changes,
        # keep this hard floor.
        if ctx.persona == "orchestrator":
            return HookDecision(
                outcome="deny",
                reason=("command_sandbox: orchestrator persona cannot invoke " "shell tools"),
                severity="error",
            )

        if self._policy is None:
            if not self._warned_no_policy:
                logger.warning(
                    "CommandSandbox: no persona-policy loaded; shell commands "
                    "will be allowed without sandbox enforcement"
                )
                self._warned_no_policy = True
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_DEGRADED)

        persona_policy = self._policy.personas.get(ctx.persona)
        if persona_policy is None:
            return HookDecision(
                outcome="deny",
                reason=f"command_sandbox: unknown persona {ctx.persona!r}",
                severity="error",
            )

        argv = _extract_argv(ctx.payload)
        if argv is None:
            return HookDecision(
                outcome="deny",
                reason=("command_sandbox: missing or unparseable 'command' " "in payload"),
                severity="error",
            )
        if not argv:
            return HookDecision(
                outcome="deny",
                reason="command_sandbox: empty command",
                severity="error",
            )

        binary = argv[0]
        argv_joined = " ".join(argv)

        # 1) Denylist — dangerous patterns deny regardless of binary.
        for pattern in persona_policy.command_denylist_args:
            if pattern and pattern in argv_joined:
                return HookDecision(
                    outcome="deny",
                    reason=(
                        f"command_sandbox: denylist pattern {pattern!r} "
                        f"matched for persona {ctx.persona!r}"
                    ),
                    severity="critical",
                    audit_metadata={
                        "binary": binary,
                        "argv": argv,
                        "denylist_pattern": pattern,
                        "policy": "denylist",
                    },
                )

        # 2) Allowlist — binary must be permitted.
        if not _binary_allowed(binary, persona_policy.command_allowlist):
            return HookDecision(
                outcome="deny",
                reason=(
                    f"command_sandbox: binary {binary!r} not in persona "
                    f"{ctx.persona!r} command_allowlist"
                ),
                severity="error",
                audit_metadata={
                    "binary": binary,
                    "argv": argv,
                    "policy": "allowlist",
                },
            )

        # 3) Metachar guard — shell injection sentinels in any argv token.
        metachar = _find_metachar(argv)
        if metachar is not None:
            shell_opt_in_payload = bool(ctx.payload.get("shell"))
            if not shell_opt_in_payload:
                return HookDecision(
                    outcome="deny",
                    reason=(
                        f"command_sandbox: shell metacharacter "
                        f"{metachar!r} in argv; set shell=True to opt in"
                    ),
                    severity="critical",
                    audit_metadata={
                        "binary": binary,
                        "argv": argv,
                        "metachar": metachar,
                        "policy": "metachar",
                    },
                )
            if not persona_policy.command_shell_opt_in:
                return HookDecision(
                    outcome="deny",
                    reason=(
                        f"command_sandbox: shell=True requested but persona "
                        f"{ctx.persona!r} does not declare command_shell_opt_in"
                    ),
                    severity="critical",
                    audit_metadata={
                        "binary": binary,
                        "argv": argv,
                        "metachar": metachar,
                        "policy": "metachar_shell_opt_in",
                    },
                )
            return HookDecision(
                outcome="allow",
                reason=(
                    f"command_sandbox: shell=True opt-in for persona "
                    f"{ctx.persona!r}; metachar {metachar!r} permitted"
                ),
                severity="warn",
                audit_metadata={
                    "binary": binary,
                    "argv": argv,
                    "metachar": metachar,
                    "shell_opt_in": True,
                },
            )

        return HookDecision(
            outcome="allow",
            reason=(
                f"command_sandbox: binary {binary!r} permitted for persona " f"{ctx.persona!r}"
            ),
            audit_metadata={"binary": binary, "argv": argv},
        )


def _extract_argv(payload: dict[str, Any]) -> list[str] | None:
    """Parse the shell command out of a PreToolUse payload.

    Supports the story-spec flat shape (``payload["command"]``) and the
    coding-agent runtime shape (``payload["args"]["command"]``). A
    pre-tokenized list is preferred when supplied; strings are split
    with :func:`shlex.split` to avoid quoting bugs.
    """
    raw = payload.get("command")
    if raw is None:
        args = payload.get("args")
        if isinstance(args, dict):
            raw = args.get("command")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return shlex.split(raw)
        except ValueError:
            return None
    if isinstance(raw, (list, tuple)):
        if all(isinstance(t, str) for t in raw):
            return list(raw)
    return None


def _binary_allowed(binary: str, allowlist: tuple[str, ...]) -> bool:
    for pattern in allowlist:
        if fnmatch(binary, pattern):
            return True
    return False


def _find_metachar(argv: list[str]) -> str | None:
    """Return the first metacharacter found in any argv token, else None."""
    for token in argv:
        for mc in SHELL_METACHARS:
            if mc in token:
                return mc
    return None
