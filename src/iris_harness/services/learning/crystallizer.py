"""Crystallize recurring *successful* interactions into skill proposals.

Reworked from the original error-frequency heuristic (per the "model-driven over
heuristic" principle). The signal is no longer "this intent errors a lot" — it is
"this intent is handled WELL, repeatedly, and the user doesn't keep correcting
it." Those clean, recurring successes are what's worth packaging as a reusable
skill.

Two stages:

1. **Success-trace mining (measured).** Count clean successes per (intent, agent)
   from ``task_completed`` signals, gated by a low ``user_correction`` rate — both
   are §4.2 measured outcomes, not guesses.
2. **Agentic synthesis.** An injected :class:`SkillSynthesizer` (LLM, governed,
   off by default) reads a few successful example traces and writes the skill's
   real semantic content. Without a synthesizer it falls back to a scaffold.

Output stays conservative: a quarantined proposal under ``config/skills/auto/``
that a human must promote (the registry excludes that subtree).

Lived in ``skills/`` until M6.2, next to what it PRODUCES rather than what it is part
of: crystallizing is a learning stage -- it reads the learning store and runs the
pre-flight gate before a proposal is ever written (ADR-0070). Skills importing learning
was the upward edge; the module belongs on this side of it.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from iris_harness.foundation.observability.session_log import TurnRecord, iter_recent_turns
from iris_harness.services.learning.store import LearningMetricsStore
from iris_harness.tools.skills.loader import (
    AUTO_SKILLS_DIR,
    resolve_auto_skills_root,
    scaffold_skill_proposal,
)
from iris_harness.tools.skills.models import SkillProposal

if TYPE_CHECKING:
    from iris_harness.services.learning.preflight import PreflightVerdict
from iris_harness.tools.skills.skill_synthesizer import (
    SkillExample,
    SkillSynthesizer,
    SynthesizedSkill,
)

logger = logging.getLogger(__name__)

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class CrystallizeReport:
    """Summary of one crystallizer pass."""

    proposals: tuple[SkillProposal, ...]
    skipped_existing: tuple[str, ...]
    # Proposals that ran the optional replay-eval gate and FAILED on quality — they
    # are scaffolded with status ``preflight_failed`` (inert, not promotable) for
    # visibility rather than landing as ``proposed``.
    rejected: tuple[SkillProposal, ...] = field(default_factory=tuple)


class SkillCrystallizer:
    """Crystallize clean, recurring successes into quarantined skill proposals."""

    def __init__(
        self,
        *,
        store: LearningMetricsStore,
        repo_root: Path,
        min_occurrences: int = 5,
        sample_window: int = 400,
        max_correction_rate: float = 0.25,
        synthesizer: SkillSynthesizer | None = None,
        turn_reader: Callable[[], list[TurnRecord]] | None = None,
        max_examples: int = 3,
        preflight: Callable[[SkillProposal], PreflightVerdict] | None = None,
    ) -> None:
        self._store = store
        self._repo_root = repo_root.resolve()
        self._min_occurrences = max(2, min_occurrences)
        self._sample_window = max(self._min_occurrences, sample_window)
        self._max_correction_rate = min(1.0, max(0.0, max_correction_rate))
        self._synthesizer = synthesizer
        self._turn_reader = turn_reader
        self._max_examples = max(1, max_examples)
        # Optional replay-eval gate (ADR-0070). When set, each candidate is validated
        # before it lands as ``proposed``; quality failures are quarantined as
        # ``preflight_failed``. Injected (model-backed) so the crystallizer stays
        # unit-testable without a backend.
        self._preflight = preflight

    def crystallize(self) -> CrystallizeReport:
        success_counts = self._count_clean_successes()
        if not success_counts:
            return CrystallizeReport(proposals=(), skipped_existing=())
        correction_counts = self._count_corrections()

        proposals: list[SkillProposal] = []
        rejected: list[SkillProposal] = []
        skipped: list[str] = []
        auto_root = resolve_auto_skills_root(self._repo_root)
        examples_by_intent: dict[str, list[SkillExample]] | None = None

        for (intent, agent_type), count in success_counts.most_common():
            if count < self._min_occurrences:
                break
            # Quality gate: skip intents the user keeps correcting (§4.2).
            corrections = correction_counts.get(intent, 0)
            if corrections and corrections / (count + corrections) > self._max_correction_rate:
                continue
            slug = self._build_slug(intent, agent_type)
            if (auto_root / slug / "manifest.yaml").exists():
                skipped.append(slug)
                continue

            if examples_by_intent is None:
                candidate_intents = [
                    i
                    for (i, _agent), c in success_counts.most_common()
                    if c >= self._min_occurrences
                ]
                examples_by_intent = self._mine_examples(candidate_intents)
            examples = examples_by_intent.get(intent, [])
            synthesized = self._synthesize(intent, agent_type, examples)

            proposal = self._build_proposal(
                slug=slug,
                intent=intent,
                agent_type=agent_type,
                successes=count,
                corrections=corrections,
                synthesized=synthesized,
            )
            # Optional replay-eval gate: validate the intent still works before the
            # proposal lands as ``proposed`` (ADR-0070). A quality failure is
            # quarantined as ``preflight_failed`` instead.
            is_rejected = False
            if self._preflight is not None:
                proposal, is_rejected = self._apply_preflight(proposal)

            overrides = synthesized.manifest_overrides(slug=slug) if synthesized else None
            try:
                scaffold_skill_proposal(self._repo_root, proposal, manifest_overrides=overrides)
            except Exception:  # proposals are best-effort
                logger.exception("failed to scaffold skill proposal %s", slug)
                continue
            (rejected if is_rejected else proposals).append(proposal)

        return CrystallizeReport(
            proposals=tuple(proposals),
            skipped_existing=tuple(skipped),
            rejected=tuple(rejected),
        )

    def _apply_preflight(self, proposal: SkillProposal) -> tuple[SkillProposal, bool]:
        """Run the injected replay-eval gate. Returns (possibly-updated proposal,
        is_rejected). Inconclusive verdicts (no workload / infra error) DON'T block —
        only a genuine quality failure (the eval actually ran) quarantines."""
        assert self._preflight is not None
        slug = proposal.skill_slug
        try:
            verdict = self._preflight(proposal)
        except Exception:  # infra failure must not block crystallization
            logger.warning(
                "crystallizer: preflight errored for %s; landing as proposed (inconclusive)",
                slug,
                exc_info=True,
            )
            return (
                proposal.model_copy(update={"preflight_reason": "inconclusive (preflight error)"}),
                False,
            )
        if verdict.passed:
            return proposal.model_copy(update={"preflight_reason": verdict.reason}), False
        if verdict.runs == 0:
            # No workload / insufficient evidence — a data-availability gap, not a
            # quality signal. Don't penalize a sparse-trace intent.
            logger.info("crystallizer: preflight inconclusive for %s (%s)", slug, verdict.reason)
            return (
                proposal.model_copy(update={"preflight_reason": f"inconclusive: {verdict.reason}"}),
                False,
            )
        logger.info(
            "crystallizer: preflight FAILED for %s (%s) — quarantined", slug, verdict.reason
        )
        return (
            proposal.model_copy(
                update={"status": "preflight_failed", "preflight_reason": verdict.reason}
            ),
            True,
        )

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------

    def _count_clean_successes(self) -> Counter[tuple[str, str]]:
        signals = self._store.recent_signals(
            metric_name="task_completed", limit=self._sample_window
        )
        counts: Counter[tuple[str, str]] = Counter()
        for signal in signals:
            if signal.value != 1.0:
                continue
            intent = str(signal.metadata.get("intent") or "").strip()
            agent = str(signal.resolved_agent or signal.metadata.get("agent_type") or "").strip()
            if intent and agent:
                counts[(intent, agent)] += 1
        return counts

    def _count_corrections(self) -> Counter[str]:
        signals = self._store.recent_signals(
            metric_name="user_correction", limit=self._sample_window
        )
        counts: Counter[str] = Counter()
        for signal in signals:
            if signal.value != 1.0:
                continue
            intent = str(signal.metadata.get("intent") or "").strip()
            if intent:
                counts[intent] += 1
        return counts

    def _mine_examples(self, intents: list[str]) -> dict[str, list[SkillExample]]:
        # Injected reader (tests): bucket whatever it returns, by intent.
        if self._turn_reader is not None:
            try:
                turns = self._turn_reader()
            except Exception:  # example mining is best-effort
                logger.debug("crystallizer: turn reader failed", exc_info=True)
                return {}
            out: dict[str, list[SkillExample]] = {}
            for turn in turns:
                if turn.has_errors or not turn.intent or not turn.query:
                    continue
                bucket = out.setdefault(turn.intent, [])
                if len(bucket) < self._max_examples:
                    bucket.append(SkillExample(query=turn.query, response=turn.response))
            return out
        # Real path: an intent-filtered scan PER target intent so a low-volume intent
        # (e.g. calendar) isn't crowded out of the newest sample_window turns by
        # high-volume ones — which otherwise leaves synthesis with no examples and
        # silently falls back to a scaffold. Mirrors build_workload_from_traces.
        scan_cap = max(self._sample_window, 100_000)
        out = {}
        for intent in dict.fromkeys(intents):  # unique, order-preserving
            try:
                turns = iter_recent_turns(
                    limit=self._max_examples,
                    intent=intent,
                    scan_cap=scan_cap,
                    require_query=True,
                    unique_queries=True,
                )
            except Exception:  # example mining is best-effort
                logger.debug("crystallizer: turn scan failed for %s", intent, exc_info=True)
                continue
            bucket = []
            for turn in turns:
                if turn.has_errors or not turn.query:
                    continue
                bucket.append(SkillExample(query=turn.query, response=turn.response))
                if len(bucket) >= self._max_examples:
                    break
            if bucket:
                out[intent] = bucket
        return out

    def _synthesize(
        self, intent: str, agent_type: str, examples: list[SkillExample]
    ) -> SynthesizedSkill | None:
        if self._synthesizer is None:
            return None
        try:
            return self._synthesizer.synthesize(
                intent=intent, agent_type=agent_type, examples=examples
            )
        except Exception:  # synthesis is best-effort; fall back to scaffold
            logger.debug("crystallizer: synthesis failed for %s", intent, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Proposal construction
    # ------------------------------------------------------------------

    def _build_proposal(
        self,
        *,
        slug: str,
        intent: str,
        agent_type: str,
        successes: int,
        corrections: int,
        synthesized: SynthesizedSkill | None,
    ) -> SkillProposal:
        proposal_id = f"prop-{uuid4().hex[:12]}"
        proposal_dir_rel = f"config/skills/{AUTO_SKILLS_DIR}/{slug}"
        skill_name = synthesized.name if synthesized else f"{intent} ({agent_type})"
        source_description = (
            f"recurring SUCCESS: intent '{intent}' handled cleanly by '{agent_type}'"
            f" {successes} times (corrections={corrections}) in the last"
            f" {self._sample_window} turns"
        )
        if synthesized:
            source_description += f" — synthesized: {synthesized.description}"
        return SkillProposal(
            proposal_id=proposal_id,
            task_id=f"signal:{intent}",
            skill_name=skill_name,
            skill_slug=slug,
            scope="learning",
            project_slug=None,
            source_description=source_description,
            source_changed_files=(),
            tool_usage=(agent_type,),
            skill_usage=(),
            reward_summary=f"successes={successes} corrections={corrections}",
            proposal_dir=proposal_dir_rel,
            manifest_path=f"{proposal_dir_rel}/manifest.yaml",
            status="proposed",
        )

    @staticmethod
    def _build_slug(intent: str, agent_type: str) -> str:
        raw = f"{intent}-{agent_type}".lower()
        slug = _SLUG_PATTERN.sub("-", raw).strip("-")
        return slug or "unnamed-skill"
