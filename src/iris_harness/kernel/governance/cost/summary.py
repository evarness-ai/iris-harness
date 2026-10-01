"""What has IRIS spent — the read side of the cost ledger.

``CostStore`` answers per-user sums; this rolls them into the one shape a
caller wants (today, this month, per tier) and, crucially, says whether the
ledger is being written at all.

That last part is the point. ``CostLimiter`` is opt-in
(``IRIS_GOVERNANCE_COST_LIMITER_ENABLED``), and it is the only thing that
writes the ledger. With it off, every sum is a truthful 0.00 that *reads* as
"IRIS has spent nothing" when it means "nobody is counting". A caller that
cannot tell those apart will render the wrong one, so ``recording`` and
``enable_hint`` travel with the numbers.

Pure over an injected store and clock: no env is read except the one flag, and
that is injectable too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from iris_harness.kernel.governance.cost.store import CostStore

#: The env var that registers ``CostLimiter``, and so starts the ledger.
COST_LIMITER_ENV = "IRIS_GOVERNANCE_COST_LIMITER_ENABLED"

#: Whether passing the cap refuses the call. Independent of recording, and
#: defaults to on, so switching recording on never quietly removes a guard.
COST_ENFORCE_ENV = "IRIS_GOVERNANCE_COST_ENFORCE"

#: The cap itself, when enforcing.
COST_CAP_ENV = "IRIS_GOVERNANCE_DAILY_COST_CAP_USD"

#: The line that turns recording on. A bare assignment, not a sentence: the UI
#: offers it as a copy button, and the same rule the health module keeps for
#: ``start_action`` applies — what the owner copies has to be pasteable.
ENABLE_HINT = f"{COST_LIMITER_ENV}=1"

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CostSummary:
    """Spend so far, and whether the number can be believed."""

    recording: bool
    #: Only meaningful while recording: whether passing the cap refuses calls.
    enforcing: bool
    #: The daily cap in force, or None when not enforcing.
    daily_cap_usd: float | None
    user_id: str
    today_usd: float
    month_usd: float
    by_tier_usd: dict[str, float] = field(default_factory=dict)
    entries: int = 0
    ledger_db: str = ""
    #: Present only when nothing is being recorded.
    enable_hint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "recording": self.recording,
            "enforcing": self.enforcing,
            "daily_cap_usd": self.daily_cap_usd,
            "user_id": self.user_id,
            "today_usd": round(self.today_usd, 6),
            "month_usd": round(self.month_usd, 6),
            "by_tier_usd": {k: round(v, 6) for k, v in self.by_tier_usd.items()},
            "entries": self.entries,
            "ledger_db": self.ledger_db,
            "enable_hint": self.enable_hint,
        }


def recording_enabled(env: dict[str, str] | None = None) -> bool:
    """True when ``CostLimiter`` is configured to write the ledger."""
    source = os.environ if env is None else env
    return source.get(COST_LIMITER_ENV, "0").strip().lower() in _TRUTHY


def enforcing_enabled(env: dict[str, str] | None = None) -> bool:
    """True when passing the daily cap refuses the call.

    Defaults to on, matching the wiring: an install that already had the
    limiter must not lose its guard because recording became separable.
    """
    source = os.environ if env is None else env
    return source.get(COST_ENFORCE_ENV, "1").strip().lower() in _TRUTHY


def _cap_usd(env: dict[str, str] | None = None) -> float | None:
    """The configured cap, or the package default. None if unreadable."""
    from iris_harness.kernel.governance.cost.pricing import (
        DEFAULT_DAILY_CAP_USD,
    )

    source = os.environ if env is None else env
    raw = source.get(COST_CAP_ENV, "").strip()
    if not raw:
        return float(DEFAULT_DAILY_CAP_USD)
    try:
        return float(raw)
    except ValueError:
        # The wiring falls back to the default and logs; the card should show
        # the number that is actually in force, not the unparseable one.
        return float(DEFAULT_DAILY_CAP_USD)


def cost_summary(
    *,
    store: CostStore | None = None,
    user_id: str = "local",
    now: datetime | None = None,
    env: dict[str, str] | None = None,
) -> CostSummary:
    """Roll the ledger into today / this month / per tier.

    The ledger is read even when recording is off: it may hold history from a
    period when it was on, and hiding that would be its own kind of lying.
    A missing or unreadable ledger is zeros, never an exception — this feeds a
    status card, and a status card that 500s tells the owner less than one
    reading zero.
    """
    moment = now or datetime.now(UTC)
    on = recording_enabled(env)
    enforcing = on and enforcing_enabled(env)
    cap = _cap_usd(env) if enforcing else None

    ledger = store if store is not None else _open_store()
    if ledger is None:
        return CostSummary(
            recording=on,
            enforcing=enforcing,
            daily_cap_usd=cap,
            user_id=user_id,
            today_usd=0.0,
            month_usd=0.0,
            ledger_db="",
            enable_hint=None if on else ENABLE_HINT,
        )

    month_start = date(moment.year, moment.month, 1)
    try:
        today = ledger.sum_today(user_id=user_id, now=moment)
        month = ledger.sum_since(user_id=user_id, since=month_start)
        by_tier = ledger.sum_by_tier_since(user_id=user_id, since=month_start)
        entries = ledger.count()
    except Exception:  # noqa: BLE001 — a locked or half-written ledger is not a 500
        today, month, by_tier, entries = 0.0, 0.0, {}, 0

    return CostSummary(
        recording=on,
        enforcing=enforcing,
        daily_cap_usd=cap,
        user_id=user_id,
        today_usd=today,
        month_usd=month,
        by_tier_usd=by_tier,
        entries=entries,
        ledger_db=str(getattr(ledger, "db_path", "")),
        enable_hint=None if on else ENABLE_HINT,
    )


def _open_store() -> CostStore | None:
    """The ledger at its configured path, or None if it cannot be opened."""
    from pathlib import Path  # keep import cost off the hot path

    raw = os.getenv("IRIS_GOVERNANCE_COST_LEDGER_DB_PATH", "").strip()
    try:
        return CostStore(db_path=Path(raw) if raw else None)
    except Exception:  # noqa: BLE001 — unwritable dir, bad path, read-only mount
        return None


__all__ = [
    "COST_CAP_ENV",
    "COST_ENFORCE_ENV",
    "COST_LIMITER_ENV",
    "ENABLE_HINT",
    "CostSummary",
    "cost_summary",
    "enforcing_enabled",
    "recording_enabled",
]
