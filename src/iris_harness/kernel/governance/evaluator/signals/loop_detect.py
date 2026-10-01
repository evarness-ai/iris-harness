"""loop_detect signal — semantic loop detection over recent steps.

Design §9.1: cosine similarity of consecutive step embeddings > 0.92
for 3 steps → inject correction → halt next trip. v1 ships the halt
half (the correction-injection layer lands when the ReAct loop grows
a mid-stream correction surface, same caveat as ``action_repeat``).

What gets embedded is the *step*, not the thought alone: the thought plus
the tool name plus a bounded rendering of its args (multi-step loop plan,
decision 5, 2026-09-14). Thought-only embedding halted a legitimate
fan-out — "find the dues, then a reminder for each" produces the same
goal-level thought on every step while the action moves to the next
item — at cosine 0.93 on step 2. With the action in the text, a model
that thinks the same thing and *does* the same thing still trips; one
that thinks the same thing about different items does not. It stays
distinct from ``action_repeat`` (exact arg-hash counting) because
near-identical args — ``search_inbox("amount due")`` then
``search_inbox("amounts due on email")`` — embed alike and are caught
here, where the hash misses them.

Concretely the signal maintains a per-run rolling buffer of the last
``window`` step embeddings (default 3) and halts when every pairwise
similarity in that window is at or above ``threshold`` (default 0.92).
"Pairwise" interprets the design wording strictly — for a window of
3, that means all of ``cos(t1, t2)``, ``cos(t2, t3)``, ``cos(t1, t3)``
must clear the bar. A single mid-window outlier shouldn't trip the
signal.

State shape (per-run-id, per-signal):

    {
        "embeddings": [(step_id, text, list[float]), ...],   # bounded at window
    }
"""

from __future__ import annotations

import logging
from typing import Any

from iris_harness.kernel.governance.evaluator.embeddings import Embedder, cosine_similarity
from iris_harness.kernel.governance.evaluator.flagged_runs import FlaggedRunThoughtWriter
from iris_harness.kernel.governance.evaluator.types import SignalResult, StepRecord

logger = logging.getLogger(__name__)


def _step_text(step: StepRecord, thought: str) -> str:
    """The text a step embeds as: thought, then the action when there is one.

    A step with no tool (a Final Answer, or a malformed step) embeds the
    thought alone, exactly as before.
    """
    if not step.tool_name:
        return thought
    action = f"\nAction: {step.tool_name}"
    if step.tool_args_text:
        action += f" {step.tool_args_text}"
    return thought + action


class LoopDetectSignal:
    """Halt when the last ``window`` steps (thought + action) all look the same."""

    name: str = "loop_detect"
    priority: int = 40  # later than the cheap signals — embedding cost

    def __init__(
        self,
        *,
        embedder: Embedder,
        threshold: float = 0.92,
        window: int = 3,
        flagged_writer: FlaggedRunThoughtWriter | None = None,
    ) -> None:
        if window < 2:
            raise ValueError("loop_detect window must be >= 2")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("loop_detect threshold must be in [-1.0, 1.0]")
        self._embedder = embedder
        self._threshold = threshold
        self._window = window
        self._flagged_writer = flagged_writer

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
        thought = (step.thought or "").strip()
        if not thought:
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason="loop_detect: step had no thought",
            )
        text = _step_text(step, thought)

        try:
            vector = list(self._embedder(text))
        except Exception as exc:  # noqa: BLE001 - embedder failure must not halt the run
            logger.warning("loop_detect: embedder raised (%s); skipping step", exc)
            return SignalResult(
                name=self.name,
                verdict="warn",
                reason=f"loop_detect: embedder error ({exc.__class__.__name__})",
                severity="warn",
                audit_metadata={"error": str(exc)},
            )

        if not vector:
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason="loop_detect: empty embedding",
            )

        buffer: list[tuple[int, str, list[float]]] = state.setdefault("embeddings", [])
        buffer.append((step.step_id, text, vector))
        # Bounded ring — keep only the most recent ``window`` entries.
        if len(buffer) > self._window:
            del buffer[: len(buffer) - self._window]

        if len(buffer) < self._window:
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason=f"loop_detect: {len(buffer)}/{self._window} steps collected",
                audit_metadata={
                    "collected": len(buffer),
                    "window": self._window,
                    "threshold": self._threshold,
                },
            )

        sims = _pairwise_similarities([v for _, _, v in buffer])
        min_sim = min(sims)
        max_sim = max(sims)
        audit = {
            "step_ids": [sid for sid, _, _ in buffer],
            "min_similarity": min_sim,
            "max_similarity": max_sim,
            "threshold": self._threshold,
            "window": self._window,
        }

        if min_sim >= self._threshold:
            self._persist_flagged_window(step=step, buffer=buffer, min_similarity=min_sim)
            return SignalResult(
                name=self.name,
                verdict="halt",
                reason=(
                    f"loop_detect: last {self._window} steps pairwise cosine "
                    f">= {self._threshold} (min={min_sim:.3f})"
                ),
                severity="warn",
                audit_metadata=audit,
            )

        return SignalResult(
            name=self.name,
            verdict="ok",
            reason=f"loop_detect: min pairwise cosine {min_sim:.3f} < {self._threshold}",
            audit_metadata=audit,
        )

    def _persist_flagged_window(
        self,
        *,
        step: StepRecord,
        buffer: list[tuple[int, str, list[float]]],
        min_similarity: float,
    ) -> None:
        if self._flagged_writer is None:
            return
        for step_id, text, embedding in buffer:
            try:
                self._flagged_writer.record(
                    run_id=step.run_id,
                    signal=self.name,
                    step_id=step_id,
                    thought=text,
                    embedding=embedding,
                    classification=step.classification,
                    metadata={"min_similarity": min_similarity},
                )
            except Exception:
                logger.warning("loop_detect: failed to persist flagged thought", exc_info=True)
                return


def _pairwise_similarities(vectors: list[list[float]]) -> list[float]:
    """Return every cos(v_i, v_j) for i < j."""
    sims: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(cosine_similarity(vectors[i], vectors[j]))
    return sims
