"""Router start-tier priors override (ADR-0068 L5)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.llm.tier_router import TierRouter

_CONFIG = Path("config/llm_tiers.yaml")


def _router() -> TierRouter:
    return TierRouter.load_from_yaml(_CONFIG)


def test_prior_overrides_static_start_tier() -> None:
    router = _router()
    base = router.get_tier("general")
    # Pick a different existing tier key to redirect 'general' to.
    other_key = next(key for key, name in router.intent_tier_map().items() if name != "general")
    target_tier_key = router.intent_tier_map()[other_key]
    if target_tier_key == router.intent_tier_map().get("general"):
        # Ensure it's actually different from general's current start tier.
        target_tier_key = next(
            k for k in router._tiers if k != router.intent_tier_map().get("general")
        )

    router.set_intent_tier_priors({"general": target_tier_key})
    assert router.intent_tier_priors() == {"general": target_tier_key}
    routed = router.get_tier("general")
    assert routed.name == router.get_tier_by_name(target_tier_key).name
    assert routed.name != base.name or target_tier_key == router.intent_tier_map().get("general")


def test_unknown_prior_tier_is_ignored() -> None:
    router = _router()
    router.set_intent_tier_priors({"general": "no_such_tier"})
    assert router.intent_tier_priors() == {}  # unknown tier dropped
    # routing falls back to the static mapping
    assert router.get_tier("general").name == _router().get_tier("general").name


def test_priors_empty_by_default() -> None:
    assert _router().intent_tier_priors() == {}
