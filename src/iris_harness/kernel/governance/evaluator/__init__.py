"""Phase 3 evaluator plane.

The evaluator runs out-of-band of the agent's main think-act-observe
loop. Programmatic signals (step_cap, action_repeat, loop_detect, ...)
fire from this plane and are surfaced to the kernel via the
``PostStep`` hook.

This package ships the in-process skeleton (registry + hook adapter).
Out-of-process isolation (Stanford 2026 critique, design §9.3) lands
in 12.gov-3.9.
"""

from iris_harness.kernel.governance.evaluator.flagged_runs import (
    ChromaFlaggedRunThoughtWriter,
    FlaggedRunThoughtWriter,
)
from iris_harness.kernel.governance.evaluator.hook import EvaluatorHook
from iris_harness.kernel.governance.evaluator.judge import (
    JUDGE_PLUGIN_NAME,
    JudgeClient,
    JudgeVerdict,
    LLMJudge,
    judge_from_env,
)
from iris_harness.kernel.governance.evaluator.registry import EvaluatorRegistry
from iris_harness.kernel.governance.evaluator.types import (
    Signal,
    SignalResult,
    SignalVerdict,
    StepRecord,
)

__all__ = [
    "EvaluatorHook",
    "EvaluatorRegistry",
    "ChromaFlaggedRunThoughtWriter",
    "FlaggedRunThoughtWriter",
    "JUDGE_PLUGIN_NAME",
    "JudgeClient",
    "JudgeVerdict",
    "LLMJudge",
    "Signal",
    "SignalResult",
    "SignalVerdict",
    "StepRecord",
    "judge_from_env",
]
