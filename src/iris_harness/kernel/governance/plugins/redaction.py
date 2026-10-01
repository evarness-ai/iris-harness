"""RedactionFilterHook — Phase 2 secret-leak scrub on ``PreLLMCall``.

Two redaction passes run in order for every outbound prompt:

1. Vault-handle substitution: any raw secret value already stored in
   the local vault is replaced with its ``vault://<handle>`` reference.
2. Regex-pack substitution: any credential-shaped substring that
   survived (e.g. an inline API key the operator never added to the
   vault) is replaced with ``[REDACTED:<pattern_name>]``.

Both passes emit ``severity='critical'`` audit events so an operator
can prove that no raw secret reached the cloud LLM provider.
"""

from __future__ import annotations

from typing import Any, Protocol

from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.plugins.regex_packs import CREDENTIAL_PATTERNS, RegexEntry


class _SecretProvider(Protocol):
    def iter_secret_values(self) -> list[tuple[str, str]]: ...


_REDACTION_PLACEHOLDER = "[REDACTED:{pattern}]"


class RedactionFilterHook:
    """Scrub raw secret values from outbound prompts.

    ``secrets`` is optional — when ``None``, only the regex-pack pass
    runs (still valuable: it catches inline keys even without a vault).
    """

    name: str = "redaction_filter"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 20  # classifier (10) -> redaction (20) -> egress gate (30)
    _TEXT_KEYS: tuple[str, ...] = ("prompt", "text", "input", "message")

    def __init__(
        self,
        *,
        secrets: _SecretProvider | None = None,
        credential_packs: list[RegexEntry] | None = None,
    ) -> None:
        self._secrets = secrets
        self._patterns: list[RegexEntry] = (
            list(CREDENTIAL_PATTERNS) if credential_packs is None else list(credential_packs)
        )

    async def __call__(self, ctx: HookContext) -> HookDecision:
        text, key = self._extract_text(ctx.payload)
        if text is None or key is None:
            return HookDecision(
                outcome="allow",
                reason="redaction_filter: no text payload to scan",
            )

        replaced, matched_handles = self._substitute_vault_values(text)
        replaced, matched_patterns = self._substitute_pattern_matches(replaced)

        if not matched_handles and not matched_patterns:
            return HookDecision(
                outcome="allow",
                reason="redaction_filter: no raw secret values detected",
            )

        transformed: dict[str, Any] = {**ctx.payload, key: replaced}
        reason_parts: list[str] = []
        if matched_handles:
            reason_parts.append(f"vault_handles={len(matched_handles)}")
        if matched_patterns:
            reason_parts.append(f"pattern_matches={len(matched_patterns)}")
        return HookDecision(
            outcome="transform",
            reason=f"redaction_filter: scrubbed outbound prompt ({', '.join(reason_parts)})",
            transformed_payload=transformed,
            severity="critical",
            audit_metadata={
                "event": "raw_secret_redacted",
                "matched_handles": matched_handles,
                "pattern_names": matched_patterns,
                "count": len(matched_handles) + len(matched_patterns),
            },
        )

    def _substitute_vault_values(self, text: str) -> tuple[str, list[str]]:
        matched: list[str] = []
        if self._secrets is None:
            return text, matched
        for handle, secret in self._secrets.iter_secret_values():
            if not secret:
                continue
            if secret in text:
                text = text.replace(secret, handle)
                matched.append(handle)
        return text, matched

    def _substitute_pattern_matches(self, text: str) -> tuple[str, list[str]]:
        matched: list[str] = []
        for name, pattern, _classification in self._patterns:
            if pattern.search(text):
                text = pattern.sub(_REDACTION_PLACEHOLDER.format(pattern=name), text)
                matched.append(name)
        return text, matched

    @classmethod
    def _extract_text(cls, payload: dict[str, Any]) -> tuple[str | None, str | None]:
        for key in cls._TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                return value, key
        return None, None
