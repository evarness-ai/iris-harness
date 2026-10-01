"""Escalation priors: turn recorded escalation history into predictive routing.

ADR-0068 L5. The curator decides escalation reactively (after a Tier-1 answer);
over time its recorded `escalation_shadow` signals reveal which *intents* reliably
escalate. This analyzer mines that history into per-intent **start-tier
recommendations** so the router can start those intents higher — the predictive
version of the cascade, earned from data instead of guessed.

Pure: reads signals, returns recommendations. The runtime decides whether to
apply them (shadow vs. enabled). See `docs/architecture/escalation-judge.md` §6.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from iris_harness.services.learning.store import LearningMetricsStore, SignalRecord

logger = logging.getLogger(__name__)


def recommend_start_tiers(
    store: LearningMetricsStore,
    *,
    current_tiers: Mapping[str, str],
    window: timedelta | None = None,
    min_samples: int = 20,
    escalate_rate_threshold: float = 0.4,
    limit: int = 2000,
    now: datetime | None = None,
) -> dict[str, str]:
    """Recommend a higher start tier for intents that reliably escalate.

    Reads ``escalation_shadow`` signals (one per judged turn, carrying the
    would-be ``action`` + ``target_tier`` + ``intent``), groups by intent, and
    proposes ``intent -> dominant escalate target`` when, over a window with at
    least ``min_samples`` turns, the escalate rate clears ``escalate_rate_threshold``.
    No-ops (target == current start tier) are dropped. Best-effort; never raises.
    """
    try:
        rows = store.recent_signals(metric_name="escalation_shadow", limit=limit)
    except Exception:  # analysis must never break a tick
        logger.debug("escalation priors: signal read failed", exc_info=True)
        return {}

    if window is not None:
        cutoff = (now or datetime.now(UTC)) - window
        rows = [r for r in rows if r.ts >= cutoff]

    by_intent: dict[str, list[SignalRecord]] = defaultdict(list)
    for row in rows:
        intent = str(row.metadata.get("intent") or "").strip()
        if intent:
            by_intent[intent].append(row)

    recommendations: dict[str, str] = {}
    for intent, items in by_intent.items():
        if len(items) < min_samples:
            continue
        escalate_rows = [i for i in items if i.metadata.get("action") == "escalate"]
        rate = len(escalate_rows) / len(items)
        if rate < escalate_rate_threshold:
            continue
        targets = Counter(
            str(i.metadata.get("target_tier") or "")
            for i in escalate_rows
            if i.metadata.get("target_tier")
        )
        if not targets:
            continue
        target = targets.most_common(1)[0][0]
        if not target or current_tiers.get(intent) == target:
            continue  # already starting there — nothing to learn
        recommendations[intent] = target
    return recommendations
