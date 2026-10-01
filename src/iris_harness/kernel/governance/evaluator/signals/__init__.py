"""Phase 3 programmatic signals.

Each signal is a pure function of a ``StepRecord`` plus a small,
signal-private per-run state dict supplied by the registry. The
registry aggregates verdicts via worst-wins precedence and surfaces
the worst at the kernel's ``PostStep`` hook.

Cheap (no extra deps) signals:

- ``step_cap``                — halt after configurable iteration count
- ``action_repeat``           — halt after N identical (tool, args) calls
- ``tool_failure_streak``     — halt after N consecutive tool errors
- ``classification_violation`` — halt when cloud tier sees personal/secret data

Ledger-backed signals (need a ``CostStore`` — opt-in at wiring time):

- ``cost_budget`` — warn at 80% / halt at 100% of the per-user daily
  spend cap (12.gov-3.7)

Semantic signals (need an ``Embedder`` — opt-in at wiring time):

- ``loop_detect`` — halt when the last N thought embeddings are pairwise
  similar above a threshold (12.gov-3.6)
- ``goal_drift`` — request HITL approval when the current thought has
  drifted from the original task statement (12.gov-3.6)
"""

from iris_harness.kernel.governance.evaluator.signals.action_repeat import ActionRepeatSignal
from iris_harness.kernel.governance.evaluator.signals.classification_violation import (
    ClassificationViolationSignal,
)
from iris_harness.kernel.governance.evaluator.signals.cost_budget import CostBudgetSignal
from iris_harness.kernel.governance.evaluator.signals.goal_drift import GoalDriftSignal
from iris_harness.kernel.governance.evaluator.signals.loop_detect import LoopDetectSignal
from iris_harness.kernel.governance.evaluator.signals.step_cap import StepCapSignal
from iris_harness.kernel.governance.evaluator.signals.tool_failure_streak import (
    ToolFailureStreakSignal,
)

__all__ = [
    "ActionRepeatSignal",
    "ClassificationViolationSignal",
    "CostBudgetSignal",
    "GoalDriftSignal",
    "LoopDetectSignal",
    "StepCapSignal",
    "ToolFailureStreakSignal",
]
