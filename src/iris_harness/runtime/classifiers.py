"""Choosing the intent classifier the router runs with.

Which classifier a runtime gets depends on flags, on whether a model backend is
reachable, and on whether the semantic wrapper is worth its cost -- decisions with
their own logic, which is why they are not inline in `build_runtime` (release
gate 1).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from iris_harness.agent.intent_router import (
    HybridClassifier,
    IIntentClassifier,
    KeywordClassifier,
)
from iris_harness.foundation.paths import config_dir as resolve_config_dir
from iris_harness.llm.tier_router import TierRouter
from iris_harness.runtime.client_config import (
    _llm_router_from_config,
)

logger = logging.getLogger(__name__)

_INTENT_SEMANTIC_ENABLED_ENV = "IRIS_INTENT_ROUTER_SEMANTIC"

_INTENT_SEMANTIC_THRESHOLD_ENV = "IRIS_INTENT_ROUTER_SEMANTIC_THRESHOLD"

_INTENT_SEMANTIC_MARGIN_ENV = "IRIS_INTENT_ROUTER_SEMANTIC_MARGIN"


def _maybe_wrap_semantic_intent(
    base: IIntentClassifier, *, config_dir: Path | None
) -> IIntentClassifier:
    """Opt-in (``IRIS_INTENT_ROUTER_SEMANTIC=1``, default OFF): make a semantic
    (embedding) classifier the PRIMARY for all intents — it decides the confident
    cases and defers to ``base`` (keyword + LLM router) when unsure or unavailable.
    Anchors are loaded from ``config/intent_anchors.yaml`` (tunable, YAML-first).
    Any failure returns ``base`` unchanged, so this can only add behavior."""
    if os.getenv(_INTENT_SEMANTIC_ENABLED_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return base
    try:
        from iris_harness.agent.intent_router import (
            DEFAULT_INTENT_SEMANTIC_MARGIN,
            DEFAULT_INTENT_SEMANTIC_THRESHOLD,
            SemanticIntentClassifier,
            load_intent_anchors,
            load_intent_defer_anchors,
        )
        from iris_harness.kernel.governance.evaluator.embeddings import (
            DefaultEmbedder,
        )

        anchors_path = (config_dir or resolve_config_dir()) / "intent_anchors.yaml"
        anchors = load_intent_anchors(anchors_path)
        if not anchors:
            logger.warning(
                "IRIS_INTENT_ROUTER_SEMANTIC=1 but no anchors at %s; using base router",
                anchors_path,
            )
            return base

        def _env_float(name: str, default: float) -> float:
            raw = os.getenv(name, "").strip()
            try:
                return float(raw) if raw else default
            except ValueError:
                return default

        return SemanticIntentClassifier(
            DefaultEmbedder(),
            anchors,
            fallback=base,
            threshold=_env_float(_INTENT_SEMANTIC_THRESHOLD_ENV, DEFAULT_INTENT_SEMANTIC_THRESHOLD),
            margin=_env_float(_INTENT_SEMANTIC_MARGIN_ENV, DEFAULT_INTENT_SEMANTIC_MARGIN),
            defer_anchors=load_intent_defer_anchors(anchors_path),
        )
    except Exception:  # never break startup over the router
        logger.exception("semantic intent classifier init failed; using base router")
        return base


def _build_intent_classifier(
    tier_router: TierRouter,
    *,
    llm_call: Callable[[str], str] | None = None,
    config_dir: Path | None = None,
) -> IIntentClassifier:
    """Construct the default intent classifier used by the runtime.

    Prefers a Tier-1 LLM router (e.g. ``llama3.2:3b`` via Ollama) using the
    ``intent_classification`` tier tag. The router itself falls back to the
    keyword classifier on any per-call failure (no Ollama running, parse error,
    etc.) so chat keeps working offline. Tests / explicit ``llm_call`` callers
    continue to get the legacy ``HybridClassifier`` path.

    When ``IRIS_INTENT_ROUTER_SEMANTIC=1`` the result is wrapped so a semantic
    (embedding) classifier handles all intents first, deferring to this base when
    unsure (see ``_maybe_wrap_semantic_intent``).
    """
    if llm_call is not None:
        base: IIntentClassifier = HybridClassifier(llm_call=llm_call)
    else:
        try:
            from iris_harness.llm.client import CodingLLMConfig

            cfg_obj = tier_router.get_llm_config("intent_classification")
            if not isinstance(cfg_obj, CodingLLMConfig):  # defensive — protocol return type
                raise TypeError("tier_router did not return a CodingLLMConfig")
            base = _llm_router_from_config(cfg_obj)
        except Exception:  # never break startup over the router
            logger.exception("LLM router unavailable; falling back to KeywordClassifier")
            base = KeywordClassifier()
    return _maybe_wrap_semantic_intent(base, config_dir=config_dir)
