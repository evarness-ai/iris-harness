"""Tests for tier_router classification, routing, and trace metadata."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from iris_harness.llm.model_metadata import ModelMetadata
from iris_harness.llm.tier_router import ModelTier, TierRouter, model_tier_for


def _patch_meta(meta: ModelMetadata | None):
    return patch("iris_harness.llm.model_metadata.get_metadata", return_value=meta)


class TestModelTierFor:
    def test_local_non_reasoning_is_small(self) -> None:
        meta = ModelMetadata(
            "llama3.2:3b", reasoning=False, tool_calling=True, context_window=8_192
        )
        with _patch_meta(meta):
            assert model_tier_for("llama3.2:3b", "ollama") is ModelTier.SMALL

    def test_local_unknown_metadata_defaults_small(self) -> None:
        with _patch_meta(None):
            assert model_tier_for("some-tiny-model", "ollama") is ModelTier.SMALL

    def test_cloud_unknown_metadata_defaults_mid(self) -> None:
        with _patch_meta(None):
            assert model_tier_for("mystery-cloud", "openrouter") is ModelTier.MID

    def test_cloud_reasoning_large_context_is_large(self) -> None:
        meta = ModelMetadata("gpt-5.4", reasoning=True, tool_calling=True, context_window=128_000)
        with _patch_meta(meta):
            assert model_tier_for("gpt-5.4", "github") is ModelTier.LARGE

    def test_cloud_reasoning_small_context_is_mid(self) -> None:
        meta = ModelMetadata(
            "reasoner-mini", reasoning=True, tool_calling=True, context_window=32_000
        )
        with _patch_meta(meta):
            assert model_tier_for("reasoner-mini", "openrouter") is ModelTier.MID

    def test_cloud_tool_calling_no_reasoning_is_mid(self) -> None:
        meta = ModelMetadata(
            "gpt-5-mini", reasoning=False, tool_calling=True, context_window=128_000
        )
        with _patch_meta(meta):
            assert model_tier_for("gpt-5-mini", "github") is ModelTier.MID

    def test_local_reasoning_model_is_mid(self) -> None:
        # A local model that genuinely supports reasoning should not be downgraded
        # to SMALL just because it runs on ollama.
        meta = ModelMetadata(
            "local-thinker", reasoning=True, tool_calling=True, context_window=32_000
        )
        with _patch_meta(meta):
            assert model_tier_for("local-thinker", "ollama") is ModelTier.MID

    def test_local_reasoning_large_context_is_large(self) -> None:
        meta = ModelMetadata(
            "local-titan", reasoning=True, tool_calling=True, context_window=200_000
        )
        with _patch_meta(meta):
            assert model_tier_for("local-titan", "ollama") is ModelTier.LARGE

    def test_metadata_lookup_failure_falls_back_safely(self) -> None:
        with patch(
            "iris_harness.llm.model_metadata.get_metadata", side_effect=RuntimeError("boom")
        ):
            assert model_tier_for("x", "ollama") is ModelTier.SMALL
            assert model_tier_for("x", "openrouter") is ModelTier.MID


class TestTierRouter:
    def test_loads_route_matrix_from_yaml(self) -> None:
        router = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))

        assert router.get_tier("general").name == "Fast"
        assert router.get_tier("task_planning").name == "Advanced"
        assert router.get_tier("skill_writing").name == "Cloud"
        assert router.get_tier("gemma_test").name == "Gemma"

    def test_unknown_intent_falls_back_to_capable_model(self) -> None:
        router = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))

        tier = router.get_tier("does_not_exist")
        assert tier.name == "Fallback"
        # Fallback is the capable tier-1 model, not the weak llama3.2:3b — an
        # unmapped/agentic intent must never silently degrade on a real task.
        assert tier.model == "granite4:latest"

    def test_system_intent_routes_to_capable_tier1(self) -> None:
        router = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))
        assert router.get_llm_config("system").model == "granite4.2:3b"

    def test_malformed_yaml_fails_safe_to_inline_default(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.yaml"
        broken.write_text("tiers: [not-a-mapping", encoding="utf-8")

        router = TierRouter.load_from_yaml(broken)
        tier = router.get_tier("general")

        assert tier.name == "Fallback"
        assert tier.provider == "ollama"
        assert tier.model == "llama3.2:3b"

    def test_trace_metadata_exposes_selected_tier(self) -> None:
        router = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))

        metadata = router.trace_metadata_for_intent("task_planning")

        assert metadata == {
            "tier_name": "Advanced",
            "tier_provider": "ollama",
            "tier_model": "qwen2.5:7b-instruct",
        }

    def test_get_llm_config_carries_resolved_tier_name(self) -> None:
        """Phase 1 fix: configs must carry the yaml tier key so clients
        report the configured governance tier instead of inferring tier_1
        from the local provider (which made tier_2 invisible in audit)."""
        router = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))

        assert router.get_llm_config("intent_classification").tier_name == "router"
        assert router.get_llm_config("general").tier_name == "tier1"
        assert router.get_llm_config("task_planning").tier_name == "tier2"
        assert router.get_llm_config("skill_writing").tier_name == "tier3"
        assert router.get_llm_config("does_not_exist").tier_name == "fallback"

    def test_get_llm_config_for_tier_builds_the_named_tier(self) -> None:
        router = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))

        cfg = router.get_llm_config_for_tier("tier2")

        assert cfg is not None
        assert cfg.tier_name == "tier2"
        assert cfg.model == router.get_tier_by_name("tier2").model

    def test_get_llm_config_for_tier_unknown_is_none(self) -> None:
        router = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))

        assert router.get_llm_config_for_tier("no_such_tier") is None

    def test_tier_name_survives_vars_round_trip_to_client_governance_tier(self) -> None:
        """Bootstrap call sites rebuild configs via CodingLLMConfig(**vars(cfg));
        the tier identity must survive and drive the governance tier label."""
        from iris_harness.llm.client import CodingLLMClient, CodingLLMConfig

        router = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))
        cfg = router.get_llm_config("task_planning")

        client = CodingLLMClient(CodingLLMConfig(**vars(cfg)))

        assert client._governance_target_tier == "tier_2"

    def test_intent_classification_routes_to_dedicated_router_tier(self) -> None:
        """Decoupling landed 2026-05-20 — intent_classification has its
        own small/fast tier (llama3.2:3b) separate from Tier 1 execution
        (granite4) so swapping the executor doesn't slow the per-turn
        routing call."""

        router = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))

        router_cfg = router.get_llm_config("intent_classification")
        general_cfg = router.get_llm_config("general")

        # Router tier — small + fast for label-picking.
        assert router_cfg.model == "llama3.2:3b"
        # Tier 1 — the executor for general chat — must NOT serve the
        # routing call (otherwise the decoupling is moot).
        assert general_cfg.model != router_cfg.model
        assert general_cfg.model == "granite4.2:3b"

    def test_tier3_serves_moe_via_lmstudio(self) -> None:
        """exp-004 adoption: Tier 3 is the 35B-A3B MoE on the MLX/LM Studio
        path (+39% decode vs same weights on Ollama, spike 4). Governance
        label stays tier_3 via the tier_name mapping."""
        router = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))
        cfg = router.get_llm_config("complex_task")

        assert cfg.provider == "lmstudio"
        assert cfg.model == "qwen/qwen3.6-35b-a3b"
        assert cfg.tier_name == "tier3"
        assert "1234" in cfg.base_url or "LM_STUDIO" in str(cfg.base_url).upper()


def _one_tier_router(tmp_path: Path, provider: str) -> TierRouter:
    tiers = tmp_path / "llm_tiers.yaml"
    tiers.write_text(
        "tiers:\n"
        "  tier1:\n"
        f"    provider: {provider}\n"
        "    model: some-model\n"
        "    use_for: [general]\n"
    )
    return TierRouter.load_from_yaml(tiers)


def test_ollama_base_url_is_read_when_the_config_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the URL was frozen at import, so a test that pointed Ollama at a
    dead port after import still reached the live server on 11434."""
    router = _one_tier_router(tmp_path, "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    assert router.get_llm_config("general").base_url == "http://127.0.0.1:9/v1"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://mac.tailnet:11434/v1/")
    assert router.get_llm_config("general").base_url == "http://mac.tailnet:11434/v1"
    monkeypatch.delenv("OLLAMA_BASE_URL")
    assert router.get_llm_config("general").base_url == "http://localhost:11434/v1"


def test_lmstudio_base_url_is_read_when_the_config_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = _one_tier_router(tmp_path, "lmstudio")
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:9")
    assert router.get_llm_config("general").base_url == "http://127.0.0.1:9/v1"


def test_unknown_provider_falls_back_to_the_current_ollama_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = _one_tier_router(tmp_path, "somethingelse")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    assert router.get_llm_config("general").base_url == "http://127.0.0.1:9/v1"


class _DownshiftEverything:
    """A governor under pressure: recommends tier1 for every tier."""

    def __init__(self) -> None:
        self.acquired: list[str] = []

    def recommend_tier_name(self, current: str) -> str:
        return "tier1"

    def acquire(self, model: str) -> None:
        self.acquired.append(model)


def test_a_pinned_tier_is_never_downshifted_or_moved_by_a_prior() -> None:
    router = TierRouter.load_from_yaml(Path("config/llm_tiers.yaml"))
    judge = router.get_tier_by_name("email_judge")
    assert judge is not None and judge.pinned
    router.governor = _DownshiftEverything()
    router.set_intent_tier_priors({"email_judge": "tier2"})

    assert router.get_llm_config("email_judge").model == judge.model
    assert router.get_llm_config_for_tier("email_judge").model == judge.model
    # An unpinned local tier still downshifts: the pin is the only exemption.
    assert router.get_llm_config("task_planning").model == router.get_tier("general").model
