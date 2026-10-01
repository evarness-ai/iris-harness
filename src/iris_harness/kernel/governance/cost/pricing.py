"""Per-tier LLM pricing for the cost limiter (story 12.gov-3.7).

The three runtime tiers — ``tier_1`` (fast local), ``tier_2`` (advanced
local), and ``tier_3`` (cloud — opt-in) — get a ``TierPricing`` row each.
Local tiers cost zero by design (the privacy promise of the three-tier
strategy is that local stays local *and* free). Cloud pricing defaults
to a conservative GPT-4o-class figure so the ledger is never silently
under-counting; callers that want exact pricing can pass an explicit
mapping or extend ``config/llm_tiers.yaml`` with a ``pricing:`` sub-block
per tier (read by ``load_default_pricing``).

AC-4 of story 3.7: unknown tiers MUST fall back to the configurable
default ``FALLBACK_TIER_PRICING``, **not** zero. A run that bypasses
the egress gate to a misconfigured tier should still hit the cap.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# Default daily cap (USD). Operators override via env or wiring.
DEFAULT_DAILY_CAP_USD: Final[float] = 5.00


@dataclass(frozen=True)
class TierPricing:
    """USD price per 1k tokens for one LLM tier."""

    prompt_per_1k_usd: float
    completion_per_1k_usd: float

    def cost_usd(self, *, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1000.0) * self.prompt_per_1k_usd + (
            completion_tokens / 1000.0
        ) * self.completion_per_1k_usd


# Local tiers — free by design.
_LOCAL: Final[TierPricing] = TierPricing(0.0, 0.0)

# Cloud — conservative GPT-4o-class default. Operators can override
# per-tier by passing a custom ``pricing`` dict to the limiter wiring.
_CLOUD_DEFAULT: Final[TierPricing] = TierPricing(
    prompt_per_1k_usd=0.005,
    completion_per_1k_usd=0.015,
)

# Used when the requested tier is missing from the pricing table.
# Intentionally non-zero so AC-4 ("does NOT silently treat as free")
# is honoured even when ``LLMTier`` literal grows a new value before
# the pricing table is updated.
FALLBACK_TIER_PRICING: Final[TierPricing] = _CLOUD_DEFAULT


_DEFAULT_PRICING: Final[Mapping[str, TierPricing]] = {
    "tier_1": _LOCAL,
    "tier_2": _LOCAL,
    "tier_3": _CLOUD_DEFAULT,
}


def load_default_pricing(*, config_path: Path | None = None) -> dict[str, TierPricing]:
    """Return the default pricing dict, optionally overridden from YAML.

    If ``config_path`` is provided and contains a ``tiers.<key>.pricing``
    sub-block with ``prompt_per_1k_usd`` + ``completion_per_1k_usd``
    fields, those values override the defaults for the matching
    ``LLMTier`` literal (mapping rule below). The current
    ``config/llm_tiers.yaml`` uses ``tier1`` / ``tier2`` / ``tier3``
    keys — we accept either ``tierN`` or ``tier_N`` for ergonomics.
    """
    table: dict[str, TierPricing] = dict(_DEFAULT_PRICING)
    if config_path is None or not config_path.exists():
        return table

    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("cost pricing: could not load %s: %s", config_path, exc)
        return table

    tiers = (raw or {}).get("tiers") or {}
    for key, body in tiers.items():
        if not isinstance(body, dict):
            continue
        pricing = body.get("pricing")
        if not isinstance(pricing, dict):
            continue
        try:
            tp = TierPricing(
                prompt_per_1k_usd=float(pricing["prompt_per_1k_usd"]),
                completion_per_1k_usd=float(pricing["completion_per_1k_usd"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "cost pricing: malformed entry for tier=%r in %s: %s", key, config_path, exc
            )
            continue
        normalized = _normalize_tier_key(str(key))
        if normalized is None:
            continue
        table[normalized] = tp
    return table


def pricing_for_tier(
    tier: str | None, *, pricing: Mapping[str, TierPricing] | None = None
) -> TierPricing:
    """Return the pricing row for ``tier``; fall back to FALLBACK on miss."""
    table = pricing or _DEFAULT_PRICING
    if tier is None:
        return FALLBACK_TIER_PRICING
    return table.get(tier, FALLBACK_TIER_PRICING)


def estimate_cost_usd(
    *,
    tier: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: Mapping[str, TierPricing] | None = None,
) -> float:
    """Estimate USD cost from token counts using ``pricing[tier]`` (or fallback)."""
    return pricing_for_tier(tier, pricing=pricing).cost_usd(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ``len(text) // 4`` (the GPT BPE heuristic)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _normalize_tier_key(key: str) -> str | None:
    """Map ``tier1`` / ``tier_1`` → ``tier_1``; reject unknown shapes."""
    cleaned = key.strip().lower()
    if cleaned in {"tier_1", "tier_2", "tier_3"}:
        return cleaned
    if cleaned in {"tier1", "tier2", "tier3"}:
        return f"tier_{cleaned[-1]}"
    return None
