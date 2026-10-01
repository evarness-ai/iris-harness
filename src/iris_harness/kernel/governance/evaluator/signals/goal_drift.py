"""goal_drift signal — surface drift between the current thought and the original task.

Design §9.1: cosine *distance* (``1.0 - cosine_similarity``) between
the current thought embedding and the original task statement
embedding > the threshold → ``require_approval``. Unlike ``loop_detect`` this
signal does **not** halt — instead it asks for HITL approval, because
"the agent is no longer working on what the user asked" is a judgment
call the user might endorse (e.g., the agent legitimately decomposed
the task) or reject (e.g., prompt injection has redirected it).

The original task embedding is computed once per run and cached on the
per-run state dict, so the embedder is hit at most twice per step:
once for the (cached) original task at the very first step and once
for the current thought.

Not every distant thought is drift. A step that re-issues a tool call this run has
already made is the model restating its plan, not changing it — and its restatement
is usually *terser* than the first one, which is exactly what moves an embedding far
from a long, multi-clause question. A real halted turn showed the failure mode
(session ``web-6d670ccd``, run ``b7c928e2``): asked to "analyse what's happening
around the world and give me perspective on AI and politics", the loop thought
"I need current reporting and analysis to synthesize global trends on AI and
politics" (judged ok), then re-issued the identical ``research`` call thinking
"Current reporting is required to synthesize trends and give grounded perspective".
Dropping the topic nouns put it at distance 0.769, and a turn that had not drifted at
all was halted for approval that — at the time — nothing was wired to grant.

So a repeat of an already-judged ``(tool_name, tool_args_hash)`` returns ``ok``.
This does not leave repetition unpoliced: ``action_repeat`` owns identical calls and
halts at three, which is the signal that should own them. ``goal_drift`` keeps the
question it is actually good at — has the agent turned to a *different* task.

The first thought of a run is exempt for the same reason. Turning away from a task
needs a heading to turn away *from*, and at the opening step there is none: that
thought is the model's first restatement of the request, not a change of direction.
Run ``9e444de1`` halted on it — asked "wondering how my day looks like ?", the loop
thought its way to ``daily_plan`` and was stopped at distance 0.782 before a second
step existed. Measured against the production embedder, *every* correct opening
thought for that request scores 0.61-0.86, because a six-word question embeds
diffusely and puts all of them far away. A first step cannot drift, so it is not
this signal's to judge; the pre-tool hooks police what that step actually does.

The default threshold is 0.90, not the 0.65 the design named. Measured against the
production embedder (ChromaDB MiniLM-L6), on-task thoughts reach 0.86 when the original
task is a short question and real drift starts at 0.88; 0.65 sits inside the on-task
band and halts correct turns. 0.90 clears the measured on-task ceiling and still catches
every injected control in that sample. It remains a hand-picked constant compared
against a distance whose baseline moves with the task's length, so it is a better
default rather than a solved problem — ``IRIS_GOVERNANCE_GOAL_DRIFT_MAX_DISTANCE``
retunes it, and every judged step writes its ``distance`` to the audit ledger, which is
the corpus a fitted default should come from.

State shape (per-run-id, per-signal):

    {
        "original_vector": list[float] | None,
        "original_text":   str | None,
        "judged_actions":  set[tuple[str, str]],
        "judged_a_step":   bool,
    }
"""

from __future__ import annotations

import logging
from typing import Any

from iris_harness.kernel.governance.evaluator.drift_exemptions import (
    DriftExemption,
    DriftExemptionStore,
    extract_keywords,
    keyword_match,
)
from iris_harness.kernel.governance.evaluator.embeddings import Embedder, cosine_similarity
from iris_harness.kernel.governance.evaluator.flagged_runs import FlaggedRunThoughtWriter
from iris_harness.kernel.governance.evaluator.types import SignalResult, StepRecord

logger = logging.getLogger(__name__)


class GoalDriftSignal:
    """Request HITL approval when the current thought diverges from the task."""

    name: str = "goal_drift"
    priority: int = 45  # later than loop_detect (40): the cost is one extra embed call

    def __init__(
        self,
        *,
        embedder: Embedder,
        max_distance: float = 0.90,
        flagged_writer: FlaggedRunThoughtWriter | None = None,
        exemptions: DriftExemptionStore | None = None,
    ) -> None:
        if not 0.0 <= max_distance <= 2.0:
            # cosine distance lies in [0, 2]; reject obviously bad values
            raise ValueError("goal_drift max_distance must be in [0.0, 2.0]")
        self._embedder = embedder
        self._max_distance = max_distance
        self._flagged_writer = flagged_writer
        self._exemptions = exemptions

    def __call__(self, step: StepRecord, *, state: dict[str, Any]) -> SignalResult:
        thought = (step.thought or "").strip()
        original = (step.original_task or "").strip()

        if not original:
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason="goal_drift: no original_task supplied by caller",
            )
        if not thought:
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason="goal_drift: step had no thought",
            )

        # A tool call this run has already made is not a change of task, whatever the
        # thought above it embeds to. Checked before the embed, so the cheap answer is
        # also the fast one. See the module docstring for the run this comes from.
        action_key = self._action_key(step)
        judged: set[tuple[str, str]] = state.setdefault("judged_actions", set())
        if action_key is not None and action_key in judged:
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason=(
                    f"goal_drift: repeat of an already-judged action "
                    f"({step.tool_name}) — action_repeat owns this case"
                ),
                audit_metadata={
                    "skipped": "repeat_action",
                    "tool_name": step.tool_name,
                    "tool_args_hash": step.tool_args_hash,
                },
            )

        # The opening thought of a run has no heading to have turned away from. It is
        # exempt, and consuming the exemption here — after the guards above — means a
        # step with no thought does not spend it. See the module docstring for the run.
        if not state.get("judged_a_step"):
            state["judged_a_step"] = True
            if action_key is not None:
                judged.add(action_key)
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason="goal_drift: first step of the run — no trajectory to drift from",
                audit_metadata={"skipped": "first_step"},
            )

        original_vector = self._resolve_original_vector(original, state=state)
        if original_vector is None:
            # Embedder failed on the original task — degrade to warn so
            # the operator notices, but don't request approval.
            return SignalResult(
                name=self.name,
                verdict="warn",
                reason="goal_drift: could not embed original_task",
                severity="warn",
            )

        try:
            thought_vector = list(self._embedder(thought))
        except Exception as exc:  # noqa: BLE001 - embedder failure must not halt
            logger.warning("goal_drift: embedder raised (%s); skipping step", exc)
            return SignalResult(
                name=self.name,
                verdict="warn",
                reason=f"goal_drift: embedder error ({exc.__class__.__name__})",
                severity="warn",
                audit_metadata={"error": str(exc)},
            )

        if not thought_vector:
            return SignalResult(
                name=self.name,
                verdict="ok",
                reason="goal_drift: empty thought embedding",
            )

        if action_key is not None:
            judged.add(action_key)

        similarity = cosine_similarity(original_vector, thought_vector)
        distance = 1.0 - similarity
        audit = {
            "similarity": similarity,
            "distance": distance,
            "max_distance": self._max_distance,
        }

        if distance > self._max_distance:
            # The person may already have answered this question. An exemption they
            # approved for a task like this one, whose thought used these words, is a
            # standing "yes" — used here, and named in the audit so it can be read back.
            granted = self._matching_exemption(
                thought=thought, original_vector=original_vector, state=state
            )
            if granted is not None:
                exemption, coverage = granted
                return SignalResult(
                    name=self.name,
                    verdict="ok",
                    reason=(
                        f"goal_drift: distance {distance:.3f} > {self._max_distance}, allowed by "
                        f"an approved exemption ({exemption.exemption_id})"
                    ),
                    audit_metadata={
                        **audit,
                        "exemption_id": exemption.exemption_id,
                        "exemption_run_id": exemption.run_id,
                        "keyword_coverage": coverage,
                    },
                )

            self._persist_flagged_thought(
                step=step,
                thought=thought,
                thought_vector=thought_vector,
                distance=distance,
            )
            self._record_candidate(step=step, thought=thought, original_vector=original_vector)
            return SignalResult(
                name=self.name,
                verdict="require_approval",
                reason=(
                    f"goal_drift: thought drifted from original task "
                    f"(distance={distance:.3f} > {self._max_distance})"
                ),
                severity="warn",
                audit_metadata=audit,
            )

        return SignalResult(
            name=self.name,
            verdict="ok",
            reason=f"goal_drift: distance {distance:.3f} <= {self._max_distance}",
            audit_metadata=audit,
        )

    def _approved_exemptions(self, *, state: dict[str, Any]) -> list[DriftExemption]:
        """Approved exemptions, read once per run. A store failure means none."""
        cached = state.get("approved_exemptions")
        if isinstance(cached, list):
            return cached
        rows: list[DriftExemption] = []
        if self._exemptions is not None:
            try:
                rows = self._exemptions.approved()
            except Exception:  # a store failure must not halt a turn
                logger.warning("goal_drift: could not read exemptions", exc_info=True)
        state["approved_exemptions"] = rows
        return rows

    def _matching_exemption(
        self, *, thought: str, original_vector: list[float], state: dict[str, Any]
    ) -> tuple[DriftExemption, float] | None:
        """The approved exemption that covers this thought, with its keyword coverage.

        Both tests must pass: the exemption was earned on a task like this one, and
        this thought uses the words the approved thought used. The best coverage wins,
        so the audit names the closest grant rather than whichever was written first.
        """
        rows = self._approved_exemptions(state=state)
        if not rows:
            return None
        thought_keywords = extract_keywords(thought)
        if not thought_keywords:
            return None
        best: tuple[DriftExemption, float] | None = None
        for exemption in rows:
            if not exemption.task_vector:
                continue
            task_distance = 1.0 - cosine_similarity(original_vector, list(exemption.task_vector))
            if task_distance > self._max_distance:
                continue  # earned on a different question
            coverage = keyword_match(exemption, thought_keywords)
            if coverage is None:
                continue
            if best is None or coverage > best[1]:
                best = (exemption, coverage)
        return best

    def _record_candidate(
        self, *, step: StepRecord, thought: str, original_vector: list[float]
    ) -> None:
        """Bank the halting thought so an approval has something to promote."""
        if self._exemptions is None:
            return
        try:
            self._exemptions.record_candidate(
                run_id=step.run_id,
                step_id=step.step_id,
                original_task=(step.original_task or "").strip(),
                task_vector=original_vector,
                thought=thought,
            )
        except Exception:  # never let bookkeeping halt a turn
            logger.warning("goal_drift: could not record exemption candidate", exc_info=True)

    @staticmethod
    def _action_key(step: StepRecord) -> tuple[str, str] | None:
        """The step's ``(tool, args)`` identity, or None when it called no tool."""
        if not step.tool_name or not step.tool_args_hash:
            return None
        return (step.tool_name, step.tool_args_hash)

    def _resolve_original_vector(
        self, original: str, *, state: dict[str, Any]
    ) -> list[float] | None:
        cached_text = state.get("original_text")
        if cached_text == original:
            cached = state.get("original_vector")
            if isinstance(cached, list):
                return cached

        try:
            vector = list(self._embedder(original))
        except Exception as exc:  # noqa: BLE001
            logger.warning("goal_drift: failed to embed original_task (%s)", exc)
            state["original_vector"] = None
            state["original_text"] = original
            return None

        if not vector:
            state["original_vector"] = None
            state["original_text"] = original
            return None

        state["original_vector"] = vector
        state["original_text"] = original
        return vector

    def _persist_flagged_thought(
        self,
        *,
        step: StepRecord,
        thought: str,
        thought_vector: list[float],
        distance: float,
    ) -> None:
        if self._flagged_writer is None:
            return
        try:
            self._flagged_writer.record(
                run_id=step.run_id,
                signal=self.name,
                step_id=step.step_id,
                thought=thought,
                embedding=thought_vector,
                classification=step.classification,
                metadata={"distance": distance},
            )
        except Exception:
            logger.warning("goal_drift: failed to persist flagged thought", exc_info=True)
