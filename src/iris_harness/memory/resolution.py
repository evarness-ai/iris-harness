"""IRIS's entity resolution: memris's Resolver with IRIS's name folding and thresholds.

memris decides *policy* (exact or alias → the same entity; a look-alike → a separate
entity plus a candidate that merges on evidence). IRIS supplies what memris leaves to
its caller: how names fold (``entity_aliases.yaml``'s aliases and trailing company
words, shared with the Map), how alike two names look, the thresholds
(``learning.yaml`` → ``resolution``), and which episode a name turned up in (the
current conversation).
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from iris_harness.memory.ontology import memory_config_dir
from memris.graph import MemoryGraph
from memris.model import name_key
from memris.resolve import Resolver


def name_similarity(a: str, b: str) -> float:
    """How alike two names look, 0–1 — case and spacing ignored. Cheap and local."""
    return SequenceMatcher(None, name_key(a), name_key(b)).ratio()


def _thresholds() -> dict[str, Any]:
    import yaml

    path = memory_config_dir() / "learning.yaml"
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    section = loaded.get("resolution") if isinstance(loaded, dict) else None
    return section if isinstance(section, dict) else {}


def resolver_for(graph: MemoryGraph) -> Resolver:
    from iris_harness.memory.graph import canonical_entity  # Map config

    settings = _thresholds()
    return Resolver(
        graph,
        similarity=name_similarity,
        normalise=canonical_entity,
        ask_similarity=float(settings.get("ask_similarity", 0.8)),
        evidence_episodes=int(settings.get("evidence_episodes", 3)),
    )


def current_episode() -> str | None:
    """The conversation a name turned up in — evidence is counted per conversation."""
    try:
        from iris_harness.foundation.observability.session_log import (
            current_session_id,
        )

        return current_session_id()
    except Exception:  # noqa: BLE001 — silent-ok: no session (CLI, tests), no evidence count
        return None


__all__ = ["current_episode", "name_similarity", "resolver_for"]
