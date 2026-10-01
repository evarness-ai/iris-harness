"""Building the skill crystallizer and its replay-eval pre-flight.

Both are off by default and both are assembled from several flags (ADR-0070), so
`build_runtime` asks for them rather than spelling the conditions out inline
(release gate 1).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iris_harness.services.learning.preflight import PreflightVerdict
    from iris_harness.tools.skills.models import SkillProposal
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from iris_harness.llm.tier_router import TierRouter
from iris_harness.tools.skills.skill_synthesizer import (
    SKILL_SYNTH_SYSTEM_PROMPT,
    SkillExample,
    SkillSynthesizer,
    SynthesizedSkill,
    build_skill_synth_prompt,
    parse_synthesized_skill,
)

logger = logging.getLogger(__name__)


class _CrystallizerLLMSynthesizer:
    """Sync adapter: synthesize a skill from success traces via a governed LLM."""

    def __init__(self, *, invoke: Callable[[str, str], str]) -> None:
        self._invoke = invoke

    def synthesize(
        self, *, intent: str, agent_type: str, examples: Sequence[SkillExample]
    ) -> SynthesizedSkill | None:
        prompt = build_skill_synth_prompt(intent=intent, agent_type=agent_type, examples=examples)
        raw = self._invoke(SKILL_SYNTH_SYSTEM_PROMPT, prompt)
        return parse_synthesized_skill(raw)


def _build_skill_synthesizer(*, tier_router: TierRouter) -> SkillSynthesizer | None:
    """Build the optional agentic skill synthesizer (crystallizer rework).

    Opt-in via ``IRIS_SKILL_SYNTHESIS``; default off so the crystallizer falls
    back to a scaffold. Uses a capable LOCAL tier (synthesis reads user content,
    so it stays on-box) through the governed CodingLLMClient path.
    """
    enabled = os.getenv("IRIS_SKILL_SYNTHESIS", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None
    try:
        from iris_harness.llm.client import CodingLLMClient

        cfg = cast(Any, tier_router.get_llm_config("task_planning"))
        cfg = cfg.model_copy(update={"temperature": 0.2, "max_tokens": 512})
        client = CodingLLMClient(cfg, governance_agent_type="chat")
    except Exception:
        logger.debug("skill synthesizer disabled: could not init LLM client", exc_info=True)
        return None
    return _CrystallizerLLMSynthesizer(
        invoke=lambda system, user: client.invoke(system_prompt=system, user_prompt=user)
    )


def _build_crystallize_preflight(
    *, repo_root: Path
) -> Callable[[SkillProposal], PreflightVerdict] | None:
    """Build the optional replay-eval gate for the crystallize tick (ADR-0070).

    Opt-in via ``IRIS_CRYSTALLIZE_PREFLIGHT``; default off. When enabled, each new
    crystallized proposal is replayed through an isolated eval runtime (sandbox mode
    from ``IRIS_EVAL_SANDBOX``, default ``effect``) before it lands as ``proposed`` —
    a quality failure is quarantined as ``preflight_failed`` instead. Returns ``None``
    when disabled. The eval needs a local model backend; infra errors are treated as
    inconclusive by the crystallizer and don't block proposals.
    """
    enabled = os.getenv("IRIS_CRYSTALLIZE_PREFLIGHT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return None
    try:
        from iris_harness.services.learning.preflight import (
            build_proposal_preflight,
        )

        return build_proposal_preflight(repo_root)
    except Exception:
        logger.debug("crystallize preflight disabled: could not build", exc_info=True)
        return None
