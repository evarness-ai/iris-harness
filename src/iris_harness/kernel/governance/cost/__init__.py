"""Phase 3 cost ledger + pricing.

The ledger (``store.py``) is the source of truth for per-user spend;
the pricing module (``pricing.py``) maps an LLM tier + token counts to
a dollar figure. ``CostLimiter`` (in ``governance.plugins``) reads the
ledger at ``PreLLMCall`` to deny over-budget calls and writes the
estimated cost of each allowed call back to the ledger. The
``cost_budget`` evaluator signal (in ``governance.evaluator.signals``)
reads the same ledger to surface warnings before the hard cap trips.

All three pieces share the same DB so a halted run's ledger writes are
visible to the limiter on the *next* call without an in-memory cache.
"""

from iris_harness.kernel.governance.cost.pricing import (
    DEFAULT_DAILY_CAP_USD,
    FALLBACK_TIER_PRICING,
    TierPricing,
    estimate_cost_usd,
    estimate_tokens,
    load_default_pricing,
    pricing_for_tier,
)
from iris_harness.kernel.governance.cost.store import (
    CostEntry,
    CostStore,
    default_cost_ledger_db_path,
)
from iris_harness.kernel.governance.cost.summary import (
    COST_LIMITER_ENV,
    ENABLE_HINT,
    CostSummary,
    cost_summary,
    recording_enabled,
)

__all__ = [
    "COST_LIMITER_ENV",
    "CostEntry",
    "CostStore",
    "CostSummary",
    "default_cost_ledger_db_path",
    "DEFAULT_DAILY_CAP_USD",
    "ENABLE_HINT",
    "FALLBACK_TIER_PRICING",
    "TierPricing",
    "cost_summary",
    "estimate_cost_usd",
    "estimate_tokens",
    "load_default_pricing",
    "pricing_for_tier",
    "recording_enabled",
]
