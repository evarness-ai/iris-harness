"""NetworkEgress — Phase 4 ``PreToolUse`` per-persona domain allowlist.

Story 12.gov-4.5. Enforces each coding-agent persona's
``network_egress_domains`` allowlist for tools that make outbound HTTP
calls. Default-deny: a persona with an empty or absent domain list is
denied on every outbound HTTP tool dispatch. This closes OWASP ASI02
(Tool Misuse) for the data-exfiltration vector: a tool that silently
POSTs code or secrets to an unexpected host is blocked at ``PreToolUse``
before the call is ever made.

Priority **35** at ``PreToolUse`` — after ``PersonaSurface`` (15),
``CredentialBroker`` (15), ``ToolPolicyHook`` (20), ``CommandSandbox``
(25), and ``FSJail`` (30). By the time this hook fires the caller
persona is already known and the tool is already within the persona's
declared allowed surface; NetworkEgress only verifies the outbound
hostname.

Check order (chosen to produce the most informative audit row):

1. **Tool-name guard** — only tools in :data:`DEFAULT_NETWORK_TOOLS`
   are inspected; others short-circuit with ``allow``. Tool patterns
   support fnmatch wildcards (``github_*``, ``mcp_*``).
2. **URL extraction + parse** — missing or scheme-less URLs deny with
   ``unparseable_url`` so a tool cannot escape inspection by omitting
   a URL.
3. **Cloud-LLM skip** — well-known LLM API hostnames in
   :data:`CLOUD_LLM_HOSTNAMES` are governed by ``EgressGate`` at
   ``PreLLMCall``; they short-circuit here with ``allow`` to prevent
   double enforcement.
4. **Default-deny gate** — empty ``network_egress_domains`` denies
   immediately; no allowlist check is needed.
5. **fnmatch allowlist** — the hostname (port and path stripped) must
   match at least one entry. Wildcards follow standard glob rules:
   ``*.github.com`` matches ``api.github.com`` but not ``github.com``
   or ``raw.githubusercontent.com``.

Audit rows always carry the resolved hostname and a query-string-free
URL so sensitive tokens embedded in ``?token=...`` params are never
persisted to the audit log.
"""

from __future__ import annotations

import logging
from fnmatch import fnmatch
from typing import Any, ClassVar, Final
from urllib.parse import urlparse, urlunparse

from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.identity_redaction import (
    owner_identity,
    reset_owner_identity,
)
from iris_harness.kernel.governance.plugins.persona_surface import PersonaPolicyDocument

logger = logging.getLogger(__name__)

#: Tool-name patterns whose payloads are inspected for outbound URLs.
#: Exact names and fnmatch wildcards are both valid entries.
DEFAULT_NETWORK_TOOLS: Final[frozenset[str]] = frozenset(
    {"web_fetch", "research", "http_request", "github_*", "mcp_*"}
)

#: Cloud-LLM API hostnames governed by ``EgressGate`` at ``PreLLMCall``.
#: Requests to these hosts short-circuit with ``allow`` so the governance
#: decision is not double-counted.
CLOUD_LLM_HOSTNAMES: Final[frozenset[str]] = frozenset(
    {
        "api.openai.com",
        "api.anthropic.com",
        "openrouter.ai",
        "models.inference.ai.azure.com",  # GitHub Models API
    }
)


class NetworkEgress:
    """Enforce per-persona outbound HTTP domain allowlist."""

    name: str = "network_egress"
    hook_point: HookPoint = HookPoint.PRE_TOOL_USE
    priority: int = 35

    ALLOW_REASON_NOT_NETWORK_TOOL: ClassVar[str] = "network_egress: not a network tool"
    ALLOW_REASON_NON_CODING: ClassVar[str] = "network_egress: not a coding-agent dispatch"
    ALLOW_REASON_DEGRADED: ClassVar[str] = "network_egress: no policy loaded (degraded)"
    ALLOW_REASON_CLOUD_LLM: ClassVar[str] = "network_egress: hostname governed by EgressGate"
    DENY_REASON_SECRET_EGRESS: ClassVar[str] = (
        "network_egress: outbound request contains a protected identity/vault secret"  # noqa: S105 — a deny message
    )

    # The owner's identity literals, both kinds (owner_identity: ``secret`` and ``link``).
    # Exfiltration guard (exp-007 GAP-14): a network tool whose args carry one of
    # these is denied for EVERY agent — defense-in-depth at the egress point, on the
    # chat path too (the kernel's domain allowlist is otherwise coding-persona-scoped).
    #
    # The kernel ASKS for the documents rather than reading them: it sits below
    # memory, and the loader is read from both sides. The one corpus every guard reads
    # is the seam's (`kernel/governance/identity_redaction.py`), which also says why an
    # unregistered provider is logged instead of passing silently.
    _WARNED_NO_PROVIDER: ClassVar[bool] = False

    @classmethod
    def _identity_secret_literals(cls) -> frozenset[str]:
        corpus = owner_identity()
        if corpus is None:
            # Nobody supplied the corpus. The guard still runs -- it just has nothing
            # of the user's to recognise -- so say so rather than look like it worked.
            if not cls._WARNED_NO_PROVIDER:
                cls._WARNED_NO_PROVIDER = True
                logger.warning(
                    "network_egress: no identity-text provider registered; the "
                    "exfiltration guard cannot recognise the user's own secrets. "
                    "Import iris_harness.runtime.identity_redaction (bootstrap does) "
                    "or call register_identity_text_provider."
                )
            return frozenset()
        return corpus.of("secret", "link")

    @classmethod
    def _reset_identity_cache(cls) -> None:
        """Drop the memoised corpus. For tests, and for a late registration."""
        reset_owner_identity()
        cls._WARNED_NO_PROVIDER = False

    def __init__(
        self,
        *,
        policy: PersonaPolicyDocument | None = None,
        network_tools: frozenset[str] = DEFAULT_NETWORK_TOOLS,
        cloud_llm_hostnames: frozenset[str] = CLOUD_LLM_HOSTNAMES,
    ) -> None:
        self._policy = policy
        self._network_tools = network_tools
        self._cloud_llm_hostnames = cloud_llm_hostnames
        self._warned_no_policy = False

    async def __call__(self, ctx: HookContext) -> HookDecision:
        tool_name = ctx.payload.get("tool_name")
        if not isinstance(tool_name, str) or not is_network_tool(tool_name, self._network_tools):
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_NOT_NETWORK_TOOL)

        # Secret-egress guard (all agents, incl. chat): never let an identity/vault
        # secret literal leave via a network tool, regardless of destination.
        secrets = self._identity_secret_literals()
        if secrets:
            haystack = str(ctx.payload.get("args", ""))
            if any(literal in haystack for literal in secrets):
                return HookDecision(
                    outcome="deny",
                    reason=self.DENY_REASON_SECRET_EGRESS,
                    severity="error",
                )

        if ctx.agent_type != "coding":
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_NON_CODING)

        if ctx.persona is None:
            return HookDecision(
                outcome="deny",
                reason="network_egress: coding-agent dispatch missing persona",
                severity="error",
            )

        # Defense in depth: orchestrator is delegation-only at PersonaSurface
        # (priority 15) and cannot hold network tools in its surface. Guard
        # here too in case wiring changes.
        if ctx.persona == "orchestrator":
            return HookDecision(
                outcome="deny",
                reason="network_egress: orchestrator persona cannot invoke network tools",
                severity="error",
            )

        if self._policy is None:
            if not self._warned_no_policy:
                logger.warning(
                    "NetworkEgress: no persona-policy loaded; network tools will be "
                    "allowed without egress enforcement"
                )
                self._warned_no_policy = True
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_DEGRADED)

        persona_policy = self._policy.personas.get(ctx.persona)
        if persona_policy is None:
            return HookDecision(
                outcome="deny",
                reason=f"network_egress: unknown persona {ctx.persona!r}",
                severity="error",
            )

        url = _extract_url(ctx.payload)
        if url is None:
            return HookDecision(
                outcome="deny",
                reason="network_egress: missing URL in network-tool payload",
                severity="error",
                audit_metadata={"tool_name": tool_name},
            )

        parsed = _parse_url(url)
        if parsed is None:
            return HookDecision(
                outcome="deny",
                reason="network_egress: unparseable_url — no scheme or netloc",
                severity="error",
                audit_metadata={"tool_name": tool_name, "url_raw": url},
            )

        hostname, clean_url = parsed

        # Cloud-LLM paths are governed by EgressGate; skip here.
        if hostname in self._cloud_llm_hostnames:
            return HookDecision(
                outcome="allow",
                reason=self.ALLOW_REASON_CLOUD_LLM,
                audit_metadata={"hostname": hostname, "url": clean_url},
            )

        # Default-deny: empty allowlist means no outbound HTTP permitted.
        if not persona_policy.network_egress_domains:
            return HookDecision(
                outcome="deny",
                reason=(
                    f"network_egress: persona {ctx.persona!r} has no "
                    "network_egress_domains configured (default-deny); "
                    "hostname_not_allowed"
                ),
                severity="error",
                audit_metadata={
                    "tool_name": tool_name,
                    "hostname": hostname,
                    "url": clean_url,
                    "deny_reason": "hostname_not_allowed",
                },
            )

        for pattern in persona_policy.network_egress_domains:
            if fnmatch(hostname, pattern):
                return HookDecision(
                    outcome="allow",
                    reason=(
                        f"network_egress: hostname {hostname!r} permitted "
                        f"for persona {ctx.persona!r}"
                    ),
                    audit_metadata={"hostname": hostname, "url": clean_url},
                )

        return HookDecision(
            outcome="deny",
            reason=(
                f"network_egress: hostname {hostname!r} not in persona "
                f"{ctx.persona!r} network_egress_domains; hostname_not_allowed"
            ),
            severity="error",
            audit_metadata={
                "tool_name": tool_name,
                "hostname": hostname,
                "url": clean_url,
                "deny_reason": "hostname_not_allowed",
            },
        )


def is_network_tool(tool_name: str, tool_set: frozenset[str]) -> bool:
    """Return True if ``tool_name`` matches any pattern in ``tool_set``.

    Public: the owner-PII shadow hook asks it too, so the calls it audits for the egress
    column are exactly the calls this guard inspects.
    """
    for pattern in tool_set:
        if fnmatch(tool_name, pattern):
            return True
    return False


def _extract_url(payload: dict[str, Any]) -> str | None:
    """Extract the target URL from a ``PreToolUse`` payload.

    Supports the flat payload shape (``payload["url"]`` /
    ``payload["endpoint"]``) and the nested coding-agent runtime shape
    (``payload["args"]["url"]`` / ``payload["args"]["endpoint"]``).
    """
    args = payload.get("args")
    containers: tuple[dict[str, Any], ...] = (
        (args, payload) if isinstance(args, dict) else (payload,)
    )
    for container in containers:
        for key in ("url", "endpoint"):
            val = container.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _parse_url(url: str) -> tuple[str, str] | None:
    """Return ``(hostname, clean_url)`` or ``None`` if the URL is unparseable.

    ``clean_url`` has the query string and fragment stripped so sensitive
    tokens embedded in ``?token=...`` params are never captured in audit
    rows. Hostname extraction removes the port if present.
    """
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return None
    if not parsed.scheme or not parsed.netloc:
        return None
    # parsed.hostname strips the port and lower-cases the result.
    hostname = parsed.hostname
    if not hostname:
        return None
    clean = urlunparse(parsed._replace(query="", fragment=""))
    return hostname, clean
