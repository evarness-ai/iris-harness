"""EvaluatorRegistry — aggregates signal verdicts with worst-wins precedence.

Signals register at process start and are frozen via ``init_lock()``
before the first evaluation, mirroring the kernel's trust boundary
(design §5.3). Per-``run_id`` mutable state is held by the registry so
individual signals stay pure functions of ``(step, state)``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Final

from iris_harness.kernel.governance.evaluator.types import (
    Signal,
    SignalResult,
    SignalVerdict,
    StepRecord,
)

logger = logging.getLogger(__name__)


_VERDICT_RANK: Final[dict[SignalVerdict, int]] = {
    "ok": 0,
    "warn": 1,
    "require_approval": 2,
    "halt": 3,
}


class EvaluatorRegistryLockedError(RuntimeError):
    """Attempted ``register()`` after ``init_lock()``."""


class EvaluatorRegistry:
    """Registry of evaluator signals + per-run state."""

    def __init__(self) -> None:
        self._signals: list[Signal] = []
        self._init_locked: bool = False
        self._state: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))

    def register(self, signal: Signal) -> None:
        if self._init_locked:
            raise EvaluatorRegistryLockedError(
                f"signal {signal.name!r} cannot be registered after init_lock(); "
                "registration is frozen at process start (design §5.3)."
            )
        self._signals.append(signal)
        self._signals.sort(key=lambda s: s.priority)
        logger.debug(
            "registered evaluator signal name=%s priority=%d", signal.name, signal.priority
        )

    def init_lock(self) -> None:
        self._init_locked = True
        logger.info("evaluator registry locked: %d signals", len(self._signals))

    @property
    def is_locked(self) -> bool:
        return self._init_locked

    def signal_count(self) -> int:
        return len(self._signals)

    def reset_run_state(self, run_id: str) -> None:
        """Drop per-run state — call on terminal verdict for the run."""
        self._state.pop(run_id, None)

    def evaluate(self, step: StepRecord) -> tuple[SignalResult, ...]:
        """Run every registered signal against ``step``.

        Returns one ``SignalResult`` per signal. A signal that raises is
        translated into a ``halt`` verdict with ``severity='error'`` —
        an evaluator bug should fail closed, not silently allow.
        """
        results: list[SignalResult] = []
        for signal in self._signals:
            state = self._state[step.run_id][signal.name]
            try:
                result = signal(step, state=state)
            except Exception as exc:  # any signal bug fails closed
                logger.exception("evaluator signal %s raised", signal.name)
                results.append(
                    SignalResult(
                        name=signal.name,
                        verdict="halt",
                        reason=f"signal {signal.name!r} raised: {exc.__class__.__name__}",
                        severity="error",
                        audit_metadata={"exception": str(exc)},
                    )
                )
                continue
            results.append(result)
        return tuple(results)

    @staticmethod
    def worst(results: tuple[SignalResult, ...]) -> SignalResult | None:
        """Return the worst-ranked result, or ``None`` if there are none."""
        if not results:
            return None
        return max(results, key=lambda r: _VERDICT_RANK[r.verdict])
