"""Tests for the cost pricing module (story 12.gov-3.7 / AC-4, AC-5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.kernel.governance.cost.pricing import (
    FALLBACK_TIER_PRICING,
    TierPricing,
    estimate_cost_usd,
    estimate_tokens,
    load_default_pricing,
    pricing_for_tier,
)


def test_tier_1_and_tier_2_are_zero_cost() -> None:
    """AC-5: local tiers record zero cost."""
    assert (
        pricing_for_tier("tier_1").cost_usd(prompt_tokens=10_000, completion_tokens=10_000) == 0.0
    )
    assert (
        pricing_for_tier("tier_2").cost_usd(prompt_tokens=10_000, completion_tokens=10_000) == 0.0
    )


def test_tier_3_default_pricing_nonzero() -> None:
    price = pricing_for_tier("tier_3")
    assert price.prompt_per_1k_usd > 0
    assert price.completion_per_1k_usd > 0
    cost = price.cost_usd(prompt_tokens=1000, completion_tokens=1000)
    assert cost == pytest.approx(price.prompt_per_1k_usd + price.completion_per_1k_usd)


def test_unknown_tier_falls_back_not_to_zero() -> None:
    """AC-4: unknown tier must not silently treat as free."""
    price = pricing_for_tier("tier_99_made_up")
    assert price is FALLBACK_TIER_PRICING
    cost = price.cost_usd(prompt_tokens=1000, completion_tokens=1000)
    assert cost > 0.0


def test_none_tier_returns_fallback() -> None:
    """A caller that forgets to set tier must not bypass the cap."""
    assert pricing_for_tier(None) is FALLBACK_TIER_PRICING


def test_estimate_cost_usd_with_custom_pricing() -> None:
    custom = {"tier_3": TierPricing(prompt_per_1k_usd=1.0, completion_per_1k_usd=2.0)}
    cost = estimate_cost_usd(
        tier="tier_3", prompt_tokens=500, completion_tokens=500, pricing=custom
    )
    assert cost == pytest.approx(0.5 * 1.0 + 0.5 * 2.0)


def test_estimate_tokens_empty() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_basic() -> None:
    # len // 4 with a floor of 1 for any non-empty string.
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcd" * 100) == 100


def test_load_default_pricing_without_config_returns_defaults() -> None:
    table = load_default_pricing(config_path=None)
    assert table["tier_1"].prompt_per_1k_usd == 0.0
    assert table["tier_2"].prompt_per_1k_usd == 0.0
    assert table["tier_3"].prompt_per_1k_usd > 0


def test_load_default_pricing_with_yaml_override(tmp_path: Path) -> None:
    """Operators can override tier_3 pricing via config/llm_tiers.yaml."""
    path = tmp_path / "llm_tiers.yaml"
    path.write_text(
        "tiers:\n"
        "  tier3:\n"
        "    pricing:\n"
        "      prompt_per_1k_usd: 0.001\n"
        "      completion_per_1k_usd: 0.002\n"
    )
    table = load_default_pricing(config_path=path)
    assert table["tier_3"].prompt_per_1k_usd == pytest.approx(0.001)
    assert table["tier_3"].completion_per_1k_usd == pytest.approx(0.002)
    # Local tiers still zero — only the overridden tier was changed.
    assert table["tier_1"].prompt_per_1k_usd == 0.0


def test_load_default_pricing_with_malformed_yaml_drops_to_default(tmp_path: Path) -> None:
    """A bad pricing block must not crash startup."""
    path = tmp_path / "bad.yaml"
    path.write_text("tiers:\n  tier3:\n    pricing:\n      prompt_per_1k_usd: not-a-number\n")
    table = load_default_pricing(config_path=path)
    # tier_3 falls back to the hardcoded default rather than raising.
    assert table["tier_3"].prompt_per_1k_usd > 0
