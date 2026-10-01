"""The governed narrative LLM call the deterministic-first agents narrate with.

Every deterministic-first answer in IRIS is built the same way: compute the
grounded text from local data, then optionally let a small model phrase it. The
model only *summarises* content that is already grounded — it can never invent or
drop a fact — and it is always optional: ``None`` here means the caller returns
its grounded text unchanged.

This lived in ``runtime/bootstrap.py``. It is core, not bootstrap-specific: the
email digest, the daily plan, the finance digests and every tool-side summary
want the same governed, tier-routed call, and after the domain extractions some
of those callers are plugins — which may not import runtime internals (OSS plan
release gate 2).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from iris_harness.llm.tier_router import TierRouterService

logger = logging.getLogger(__name__)


def make_narrative_llm_call(
    tier_router: TierRouterService,
    *,
    intent: str = "general",
    max_tokens: int = 256,
    temperature: float = 0.3,
) -> Callable[[str], str] | None:
    """Tier-routed, governed LLM call for the deterministic-first agents' narrative
    layer (email inbox digest, daily plan) and grounded tool-side summaries.

    ``intent`` selects the tier (default ``general`` → tier-1 narrator; pass
    ``communication`` for the tier-2 instruct model that summarises email bodies
    well — issue 0002 item C). Built without ``governance_handled_upstream`` so
    each ``invoke`` fires the governance hooks itself (same governed path as the
    curator judges). Returns ``None`` if the client can't be built, so callers
    fall back to their grounded text. It only *summarises* already-grounded
    content — it can never invent or drop facts. An extraction caller passes
    ``temperature=0`` so the same email reads the same way every sweep.
    """
    try:
        from iris_harness.llm.client import CodingLLMClient
    except Exception:
        logger.debug("narrative llm disabled: CodingLLMClient import failed", exc_info=True)
        return None

    try:
        cfg = cast(Any, tier_router.get_llm_config(intent))
        cfg = cfg.model_copy(update={"temperature": temperature, "max_tokens": max_tokens})
        client = CodingLLMClient(cfg, governance_agent_type="chat")
    except Exception:
        logger.debug("narrative llm disabled: could not initialize LLM client", exc_info=True)
        return None

    def _call(prompt: str) -> str:
        return client.invoke(system_prompt="", user_prompt=prompt)

    return _call
