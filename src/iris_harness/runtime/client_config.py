"""Which model config answers this turn: intent, provider profile, preferred model.

Gate-1 extraction (OSS plan M5.7). Resolving a routing intent, picking the effective
provider profile, inferring a provider from a bare model name, and building the
``CodingLLMConfig`` a profile implies.

Extracted as 4d's prerequisite, the same reorder 4b needed: the tool-calling loops could not
leave bootstrap while the client resolution they depend on stayed behind. Unlike 4b's
prerequisites these four are shared widely — ``_finalize_chat``, ``warmup``,
``_build_search_synthesis_client``, ``_build_router_classifier_for_model``,
``_make_react_handler`` and ``_routing_intent_for`` all call one or more — so bootstrap
importing them back is the point of the module, not a smell.
"""

from __future__ import annotations

from iris_harness.agent.intent_router import (
    IIntentClassifier,
    KeywordClassifier,
    KeywordFirstClassifier,
    LLMRouterClassifier,
)
from iris_harness.llm.client import CodingLLMConfig


def config_from_profile(profile_name: str, *, intent: str = "") -> CodingLLMConfig:
    """Build a CodingLLMConfig from a named profile in ~/.iris/providers.json.

    When ``intent == "coding"`` and the profile defines ``coding_model``, swap
    in that model so the coding intent stays on a coding-tuned model even when
    ``/provider`` flips the general chat default. The session-level
    ``preferred_model`` override is applied by the caller and still wins.
    """
    from iris_harness.llm.client import CodingLLMConfig
    from iris_harness.llm.providers import ProviderManager

    mgr = ProviderManager()
    profiles = {p.name: p for p in mgr.list_profiles()}
    profile = profiles.get(profile_name)
    if profile is None:
        raise ValueError(f"Unknown provider profile: {profile_name!r}")
    headers: tuple[tuple[str, str], ...] = tuple(profile.default_headers.items())
    model = profile.model
    if intent == "coding" and profile.coding_model:
        model = profile.coding_model
    return CodingLLMConfig(
        provider=profile.provider_type,
        model=model,
        base_url=profile.base_url,
        api_key_env=profile.api_key_env,
        auth_mode=profile.auth_mode,
        default_headers=headers,
        temperature=0.5,
        max_tokens=4096,
        timeout_seconds=60,
        # Not an llm_tiers.yaml key, so governance still infers its tier from the
        # provider; it names where the model came from in the session log.
        tier_name=f"profile:{profile_name}",
    )


def effective_provider_profile(
    provider_profile: str | None,
    preferred_model: str | None = None,
) -> str:
    """Return the provider profile that should override tier routing.

    The built-in ``ollama`` profile is the local default selected by the REPL,
    but normal local chat should still flow through ``config/llm_tiers.yaml`` so
    the ResourceGovernor can apply ``num_ctx``/``keep_alive`` and pre-emptive
    evictions. Users can still force a specific local model with ``/model``;
    custom/non-local provider profiles continue to override tiers.
    """
    profile = (provider_profile or "").strip()
    model = (preferred_model or "").strip()
    if profile == "ollama" and not model:
        return ""
    return profile


def infer_provider(model: str) -> tuple[str, str, str | None]:
    """Return (provider, base_url, api_key_env) inferred from a model name."""
    lower = model.lower()
    if any(x in lower for x in ("claude", "anthropic")):
        return "anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"
    if any(x in lower for x in ("gpt-", "o1", "o3", "o4")):
        return "github", "https://models.inference.ai.azure.com", "GITHUB_TOKEN"
    if any(x in lower for x in ("gemini",)):
        return "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
    if any(x in lower for x in ("mistral", "mixtral")):
        return "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
    return "ollama", "http://localhost:11434/v1", None


MULTI_STEP_ROUTING_INTENT = "task_planning"


def resolve_routing_intent(intent: str, is_multi_step: bool) -> str:
    """Pick the intent whose configured tier should drive model selection.

    Multi-step turns are compound work — borrow the tier-2 intent's model
    (``task_planning`` → qwen2.5:7b-instruct in config/llm_tiers.yaml) instead of
    the per-intent default, so heavy queries don't run on a small model.
    """
    return MULTI_STEP_ROUTING_INTENT if is_multi_step else intent


# Moved out of bootstrap.py for release gate 1 ("the composition root only"): both the
# composition root and the runtime facade build a router from a config, so the builder
# belongs with the other config-to-client builders rather than in either caller.
def _llm_router_from_config(cfg: CodingLLMConfig) -> IIntentClassifier:
    """Build a keyword-first classifier with an LLM router as the secondary."""
    from iris_harness.llm.client import CodingLLMClient

    # Routing prefers low-temperature, short responses.
    router_cfg = cfg.model_copy(update={"temperature": 0.0, "max_tokens": 256})
    client = CodingLLMClient(router_cfg)

    def invoke(system_prompt: str, user_prompt: str) -> str:
        return client.invoke(system_prompt=system_prompt, user_prompt=user_prompt)

    keyword = KeywordClassifier()
    llm_router = LLMRouterClassifier(invoke=invoke, keyword_fallback=keyword)
    return KeywordFirstClassifier(llm_router, keyword=keyword)
