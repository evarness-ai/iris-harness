"""Unit tests for the shared harness LLM client (``iris_harness.llm.client``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.llm.client import PROVIDER_DEFAULTS, CodingLLMClient, LLMMessage


def test_coding_llm_client_binds_tools_and_parses_tool_calls() -> None:
    captured: dict[str, object] = {}

    class StubModel:
        def bind_tools(self, tools: object) -> StubModel:
            captured["tools"] = tools
            return self

        def invoke(self, messages: object) -> object:
            captured["messages"] = messages
            return type(
                "StubResponse",
                (),
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "mcp__github__remote_echo",
                            "args": {"message": "hello"},
                        }
                    ],
                },
            )()

    client = CodingLLMClient(
        PROVIDER_DEFAULTS["ollama"],
        model_factory=lambda **_: StubModel(),
    )

    response = client.invoke_turn(
        messages=(
            LLMMessage(role="system", content="system instructions"),
            LLMMessage(role="user", content="write the code"),
        ),
        bound_tools=(
            {
                "type": "function",
                "function": {
                    "name": "mcp__github__remote_echo",
                    "description": "Echo from remote MCP server",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                },
            },
        ),
    )

    assert response.tool_calls[0].name == "mcp__github__remote_echo"
    assert response.tool_calls[0].arguments == {"message": "hello"}
    assert captured["messages"] == [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "write the code"},
    ]


def test_build_credential_provider_rejects_unknown_auth_mode() -> None:
    import pytest

    from iris_harness.llm.client import CodingLLMConfig, build_credential_provider

    config = CodingLLMConfig(
        provider="github",
        model="gpt-4o",
        base_url="https://example/v1",
        api_key_env=None,
        auth_mode="does-not-exist",
    )

    with pytest.raises(ValueError):
        build_credential_provider(config, environ={})


def test_copilot_credential_provider_requires_opt_in() -> None:
    import pytest

    from iris_harness.llm import copilot_auth
    from iris_harness.llm.client import PROVIDER_DEFAULTS, build_credential_provider

    with pytest.raises(copilot_auth.CopilotBackendDisabledError):
        build_credential_provider(PROVIDER_DEFAULTS["copilot"], environ={})


def test_copilot_credential_provider_opts_in_when_flag_set() -> None:
    from iris_harness.llm import copilot_auth
    from iris_harness.llm.client import PROVIDER_DEFAULTS, build_credential_provider

    provider = build_credential_provider(
        PROVIDER_DEFAULTS["copilot"],
        environ={"IRIS_ENABLE_COPILOT_BACKEND": "1"},
    )

    assert isinstance(provider, copilot_auth.CopilotTokenProvider)
    assert provider.extra_headers()["Copilot-Integration-Id"] == "vscode-chat"


def test_resolve_coding_llm_api_key_resolves_vault_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vault:// handle in the env var is resolved to the stored secret."""
    from cryptography.fernet import Fernet

    from iris_harness.kernel.governance.vault import VaultStore, reset_vault_singleton_for_tests
    from iris_harness.llm.client import CodingLLMConfig, resolve_coding_llm_api_key

    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", Fernet.generate_key().decode("utf-8"))
    reset_vault_singleton_for_tests()
    try:
        store = VaultStore(db_path=tmp_path / "vault.db")
        store.add(handle="github-token", secret_value="ghp_real_secret")

        # Point the singleton at this test vault.
        import iris_harness.kernel.governance.vault.handle as handle_mod

        monkeypatch.setattr(handle_mod, "_singleton", store)

        config = CodingLLMConfig(
            provider="github",
            model="gpt-4o",
            base_url="https://models.inference.ai.azure.com",
            api_key_env="GITHUB_TOKEN",
        )

        # Vault-handle env var resolves to the real secret.
        resolved = resolve_coding_llm_api_key(
            config, environ={"GITHUB_TOKEN": "vault://github-token"}
        )
        assert resolved == "ghp_real_secret"

        # Plain key passes through unchanged.
        plain = resolve_coding_llm_api_key(config, environ={"GITHUB_TOKEN": "ghp_plain"})
        assert plain == "ghp_plain"
    finally:
        reset_vault_singleton_for_tests()


def test_coding_llm_client_raises_user_facing_error_for_rate_limits() -> None:

    class StubModel:
        def invoke(self, messages: object) -> object:
            del messages
            error = Exception(
                "Error code: 429 - {'error': {'code': 'RateLimitReached', 'message': "
                "'Rate limit of 50 per 86400s exceeded for UserByModelByDay. "
                "Please wait 52220 seconds before retrying.'}}"
            )
            error.status_code = 429
            raise error

    client = CodingLLMClient(PROVIDER_DEFAULTS["ollama"], model_factory=lambda **_: StubModel())

    with pytest.raises(ValueError) as exc_info:
        client.invoke(system_prompt="system instructions", user_prompt="write the code")

    message = str(exc_info.value)
    assert "LLM rate limit reached for provider 'ollama' model 'qwen3-coder:30b'." in message
    assert "Retry after 52220 seconds" in message
    assert "Switch to another configured backend in the coding pipeline config" in message


def test_governance_tier_is_the_routers_stamp_not_the_tier_name() -> None:
    """The tier router stamps the governed tier on every config it builds; the client
    reads that, never ``tier_name`` -- a local model named ``tier3`` governed as cloud."""
    from iris_harness.llm.client import CodingLLMConfig

    stamped = CodingLLMClient(
        CodingLLMConfig(provider="ollama", tier_name="tier2", governance_tier="tier_2")
    )
    assert stamped._governance_target_tier == "tier_2"

    named_tier3 = CodingLLMClient(CodingLLMConfig(provider="lmstudio", tier_name="tier3"))
    assert named_tier3._governance_target_tier == "tier_1"


def test_governance_tier_without_a_stamp_is_the_providers_declared_locality() -> None:
    """Unstamped (a provider profile, a /model override): local -> tier_1, cloud or a
    provider llm_tiers.yaml does not declare -> tier_3. The constructor arg wins."""
    from iris_harness.llm.client import CodingLLMConfig

    local = CodingLLMClient(CodingLLMConfig(provider="ollama"))
    assert local._governance_target_tier == "tier_1"

    cloud = CodingLLMClient(CodingLLMConfig(provider="openrouter"))
    assert cloud._governance_target_tier == "tier_3"

    undeclared = CodingLLMClient(CodingLLMConfig(provider="some-new-host"))
    assert undeclared._governance_target_tier == "tier_3"

    pinned = CodingLLMClient(
        CodingLLMConfig(provider="ollama", governance_tier="tier_2"),
        governance_target_tier="tier_3",
    )
    assert pinned._governance_target_tier == "tier_3"


def test_coerce_invocation_response_strips_think_blocks() -> None:
    """exp-004 spike 4 adoption: Tier-3 qwen3.6:35b-a3b thinks by default;
    thinking must never reach content — terminated or budget-truncated."""
    from iris_harness.llm.client import _coerce_invocation_response

    class _Msg:
        tool_calls = None
        additional_kwargs: dict = {}
        usage_metadata = None
        response_metadata: dict = {}

        def __init__(self, content: str) -> None:
            self.content = content

    done = _coerce_invocation_response(_Msg("<think>step by step...</think>\nThe answer is 4."))
    assert done.content == "The answer is 4."

    truncated = _coerce_invocation_response(_Msg("<think>still thinking when the budget ran ou"))
    assert truncated.content == ""

    untouched = _coerce_invocation_response(_Msg("plain answer, no thinking"))
    assert untouched.content == "plain answer, no thinking"


def test_stripped_public_content_downgrade_only_for_web_synthesis() -> None:
    """The cloud search-synthesis client downgrades a 'personal' classification (PII in
    fetched public web content) to 'internal' so the egress gate permits the cloud call —
    but NEVER downgrades 'secret', and the downgrade is OFF by default."""
    from iris_harness.kernel.governance.kernel import HookContext, HookDecision, HookPoint
    from iris_harness.llm.client import CodingLLMConfig

    class _FakeKernel:
        def __init__(self, classify_as: str) -> None:
            self._classify_as = classify_as
            self.pre_llm_classification: str | None = None

        def fire_sync(self, hook_point: HookPoint, ctx: HookContext):  # type: ignore[no-untyped-def]
            if hook_point == HookPoint.PRE_CLASSIFY:
                ctx = ctx.model_copy(update={"classification": self._classify_as})
                return HookDecision(outcome="allow", reason="x"), ctx
            # PRE_LLM_CALL — capture what classification the egress hook receives.
            self.pre_llm_classification = ctx.classification
            return HookDecision(outcome="allow", reason="x"), ctx

    def _make(classify_as: str, *, downgrade: bool):
        k = _FakeKernel(classify_as)
        c = CodingLLMClient(
            CodingLLMConfig(provider="copilot", tier_name="tier3"),
            governance_kernel=k,
            governance_stripped_public_content=downgrade,
        )
        c._governance_pre_llm(prompt_text="news with a name", run_id="r1")
        return k.pre_llm_classification

    assert _make("personal", downgrade=True) == "internal"  # web PII downgraded
    assert _make("personal", downgrade=False) == "personal"  # default: untouched
    assert _make("secret", downgrade=True) == "secret"  # secrets NEVER downgraded
    assert _make("public", downgrade=True) == "public"  # public unchanged
