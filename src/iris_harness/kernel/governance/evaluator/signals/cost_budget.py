"""cost_budget signal — surface daily spend approaching the cap.

Design §9.1: rolling token spend × tier price vs the per-user daily
limit. Trip rules:

- ``today_spend / cap >= 1.0`` → ``halt`` (severity ``critical``).
  The ``CostLimiter`` plugin already denies the *next* LLM call, but
  the signal exists so a long-running ReAct loop that's already past
  the cap (e.g., from a checkpointed run resumed after the cap was
  lowered) gets halted cleanly between steps without waiting for the
  next ``PreLLMCall``.
- ``today_spend / cap >= 0.80`` → ``warn`` (no halt). The reason is
  surfaced in the audit trail so the operator can react.
- otherwise → ``ok``.

The signal reads the same ``CostStore`` the limiter writes to, so
ledger updates from the current step are immediately visible.
"""

from __future__ import annotations

from typing import Any

from iris_harness.kernel.governance.cost.store import CostStore
from iris_harness.kernel.governance.evaluator.types import SignalResult, StepRecord


class CostBudgetSignal:
    """Halt at 100% / warn at 80% of the per-user daily cost cap."""

    name: str = "cost_budget"
    priority: int = 30  # before semantic signals (loop_detect at 40)

    def __init__(
        self,
        *,
        store: CostStore,
        daily_cap_usd: float,
        user_id: str = "local",
        warn_ratio: float = 0.80,
    ) -> None:
        if daily_cap_usd < 0:
            raise ValueError("daily_cap_usd must be >= 0")
        if not 0.0 < warn_ratio <= 1.0:
            raise ValueError("warn_ratio must be in (0.0, 1.0]")
        self._store = store
        self._cap = float(daily_cap_usd)
        self._user_id = user_id
        self._warn_ratio = warn_ratio

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
        # A zero cap means "limiter disabled" — treat as ok always to
        # avoid divide-by-zero and surprising halts when an operator
        # opts the signal in without setting a cap.
        if self._cap <= 0.0:
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason="cost_budget: cap disabled (<= 0)",
            )

        today = self._store.sum_today(user_id=self._user_id)
        ratio = today / self._cap
        audit = {
            "today_spend_usd": today,
            "daily_cap_usd": self._cap,
            "ratio": ratio,
            "warn_ratio": self._warn_ratio,
            "user_id": self._user_id,
        }

        if ratio >= 1.0:
            return SignalResult(
                name=self.name,
                verdict="halt",
                reason=(
                    f"cost_budget: today=${today:.4f} >= cap=${self._cap:.2f} "
                    f"(ratio={ratio:.2f})"
                ),
                severity="critical",
                audit_metadata=audit,
            )

        if ratio >= self._warn_ratio:
            return SignalResult(
                name=self.name,
                verdict="warn",
                reason=(
                    f"cost_budget: today=${today:.4f} approaching cap=${self._cap:.2f} "
                    f"(ratio={ratio:.2f})"
                ),
                severity="warn",
                audit_metadata=audit,
            )

        return SignalResult(
            name=self.name,
            verdict="ok",
            reason=f"cost_budget: today=${today:.4f} / cap=${self._cap:.2f} ({ratio:.2f})",
            audit_metadata=audit,
        )
