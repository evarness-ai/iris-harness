from __future__ import annotations

from iris_harness.kernel.governance import HookContext, HookPoint
from iris_harness.kernel.governance.plugins import RedactionFilterHook


class _Secrets:
    def iter_secret_values(self) -> list[tuple[str, str]]:
        return [
            ("vault://openrouter-key", "sk-or-v1-secret-value"),
            ("vault://github-token", "ghp_secret_token"),
        ]


def _ctx(prompt: str) -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="run-redaction",
        agent_type="chat",
        payload={"prompt": prompt},
    )


async def test_redaction_replaces_raw_secret_values_with_handles() -> None:
    hook = RedactionFilterHook(secrets=_Secrets())
    decision = await hook(_ctx("deploy using token ghp_secret_token and sk-or-v1-secret-value now"))

    assert decision.outcome == "transform"
    assert decision.severity == "critical"
    transformed = decision.transformed_payload or {}
    prompt = transformed.get("prompt")
    assert isinstance(prompt, str)
    assert "ghp_secret_token" not in prompt
    assert "sk-or-v1-secret-value" not in prompt
    assert "vault://github-token" in prompt
    assert "vault://openrouter-key" in prompt
    assert decision.audit_metadata.get("matched_handles") == [
        "vault://openrouter-key",
        "vault://github-token",
    ]


async def test_redaction_allows_when_no_match() -> None:
    hook = RedactionFilterHook(secrets=_Secrets())
    decision = await hook(_ctx("hello world"))

    assert decision.outcome == "allow"
    assert decision.transformed_payload is None


async def test_redaction_allows_when_payload_has_no_text_field() -> None:
    hook = RedactionFilterHook(secrets=_Secrets())
    ctx = HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="run-redaction",
        agent_type="chat",
        payload={"not_text": "value"},
    )

    decision = await hook(ctx)
    assert decision.outcome == "allow"


async def test_regex_pack_redacts_inline_github_pat_not_in_vault() -> None:
    """A raw `ghp_` token never added to the vault is still scrubbed."""
    hook = RedactionFilterHook(secrets=_Secrets())
    raw_pat = "ghp_" + "a" * 40  # matches github_pat_classic
    decision = await hook(_ctx(f"using pat {raw_pat} for the call"))

    assert decision.outcome == "transform"
    assert decision.severity == "critical"
    transformed = decision.transformed_payload or {}
    prompt = transformed.get("prompt")
    assert isinstance(prompt, str)
    assert raw_pat not in prompt
    assert "[REDACTED:github_pat_classic]" in prompt
    assert "github_pat_classic" in decision.audit_metadata.get("pattern_names", [])


async def test_regex_pack_runs_even_without_vault() -> None:
    """The hook still scrubs known credential shapes when the vault is offline."""
    hook = RedactionFilterHook(secrets=None)
    decision = await hook(_ctx("anthropic key: " + "sk-ant-" + "x" * 40))

    assert decision.outcome == "transform"
    transformed = decision.transformed_payload or {}
    prompt = transformed.get("prompt")
    assert isinstance(prompt, str)
    assert "[REDACTED:anthropic_key]" in prompt


async def test_vault_and_regex_redaction_combine_in_one_decision() -> None:
    """Vault match + pattern match in the same prompt yield both redactions."""
    hook = RedactionFilterHook(secrets=_Secrets())
    inline_anthropic = "sk-ant-" + "z" * 40
    decision = await hook(_ctx(f"ghp_secret_token then {inline_anthropic} then trailing text"))

    assert decision.outcome == "transform"
    transformed = decision.transformed_payload or {}
    prompt = transformed.get("prompt")
    assert isinstance(prompt, str)
    assert "vault://github-token" in prompt
    assert "[REDACTED:anthropic_key]" in prompt
    assert "ghp_secret_token" not in prompt
    assert inline_anthropic not in prompt
    audit = decision.audit_metadata
    assert "vault://github-token" in audit.get("matched_handles", [])
    assert "anthropic_key" in audit.get("pattern_names", [])
    assert audit.get("count") == 2
