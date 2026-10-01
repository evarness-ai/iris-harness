"""The allowlist of keys that may become durable facts, plus the first-person gate.

Two filters, both cheap, both in front of the review queue:

- **Key allowlist**: extraction accepted whatever key the model returned, so the
  store collected `topic=murder plot`, `error=it shows error`, `source=CBS News`,
  `number=4` — the subject of a conversation mistaken for a property of the person.
  Since memris plan PR 3 the allowlist IS the ontology: a key is allowed when
  ``config/memory/mappings.yaml`` maps it onto a property; the mapping's first key is
  the canonical one and its others are aliases. (``fact_keys.yaml`` is retired.)
- **First-person gate**: a fact about the user has to come from the user talking
  about themselves. `employer=Department of Justice` (confidence 0.9) was mined from
  a news article the user pasted; `name=ollama` (confidence 1.0) from a message about
  configuring a model.

The confirmation tiers (ADR-0114) are read from ``config/memory/learning.yaml``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from iris_harness.foundation.process_state import track_globals
from iris_harness.memory.ontology import memory_config_dir, memory_ontology, normalise_fact_key

logger = logging.getLogger(__name__)

_CACHE: tuple[frozenset[str], dict[str, str]] | None = None
_RAW_CACHE: dict[str, Any] | None = None
_AUTO_PATTERNS: list[re.Pattern[str]] | None = None

# "I …", "my …", "we …" and the contractions. The gate is deliberately simple: a
# sentence about the user almost always carries one of these, and everything it lets
# through still has to clear validation and then a human.
_FIRST_PERSON_RE = re.compile(
    r"(?:^|[\s\"'(\[])(?:i|i'm|im|i've|ive|i'll|id|i'd|my|mine|me|myself|we|our|ours|us)"
    r"(?:[\s,.'!?;:)\]]|$)",
    re.IGNORECASE,
)

# Quoted or pasted content. First person inside someone else's text is not the user
# speaking: a fenced block is dropped whole, and quoted lines are dropped line by line.
_FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_QUOTED_LINE_RE = re.compile(r"^\s*(?:>|\|)")


def _load() -> tuple[frozenset[str], dict[str, str]]:
    """(canonical keys, alias → canonical) from the fact mappings (cached).

    A mapping lists the keys that feed one property; the first is the canonical key.
    A broken ontology allows nothing — the safe direction: it should stop proposals,
    not wave everything through.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    allowed: set[str] = set()
    aliases: dict[str, str] = {}
    try:
        for rule in memory_ontology().mappings:
            if rule.source_type != "fact" or not rule.keys:
                continue
            canonical = normalise_fact_key(rule.keys[0])
            allowed.add(canonical)
            for other in rule.keys[1:]:
                aliases[normalise_fact_key(other)] = canonical
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory ontology unreadable (%s) — no fact keys are allowed", exc)
    _CACHE = (frozenset(allowed), aliases)
    return _CACHE


def _load_raw() -> dict[str, Any]:
    """``config/memory/learning.yaml`` (cached) — the confirmation tiers live there."""
    global _RAW_CACHE
    if _RAW_CACHE is not None:
        return _RAW_CACHE
    data: dict[str, Any] = {}
    path = memory_config_dir() / "learning.yaml"
    if path.exists():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s", path, exc)
    _RAW_CACHE = data
    return data


def learning_config() -> dict[str, Any]:
    """``config/memory/learning.yaml`` as loaded (cached) — for its other sections."""
    return _load_raw()


def reset_cache() -> None:
    """Drop the cached config (tests, and a reload after editing the YAML)."""
    global _CACHE, _RAW_CACHE, _AUTO_PATTERNS
    _CACHE = None
    _RAW_CACHE = None
    _AUTO_PATTERNS = None


def allowed_keys() -> frozenset[str]:
    return _load()[0]


def canonical_key(key: str) -> str | None:
    """Canonical form of ``key``, or None when it is not an allowed profile key.

    An empty allowlist means nothing is allowed. That is the safe direction: a
    missing config should stop proposals, not wave everything through.
    """
    raw = normalise_fact_key(key)
    if not raw:
        return None
    allowed, aliases = _load()
    if raw in allowed:
        return raw
    mapped = aliases.get(raw)
    if mapped and mapped in allowed:
        return mapped
    return None


def is_self_statement(message: str) -> bool:
    """True when the message reads as the user talking about themselves."""
    text = (message or "").strip()
    if not text:
        return False
    # Strip quoted/pasted content, then look for first person in what is left.
    text = _FENCED_BLOCK_RE.sub(" ", text)
    own_lines = [ln for ln in text.splitlines() if not _QUOTED_LINE_RE.match(ln)]
    own = "\n".join(own_lines).strip()
    if not own:
        return False
    return bool(_FIRST_PERSON_RE.search(own))


AUTO = "auto"
ASK = "ask"
QUEUE = "queue"


def _confirmation_config() -> dict[str, Any]:
    raw = _load_raw().get("confirmation") or {}
    return raw if isinstance(raw, dict) else {}


def auto_confirm_patterns() -> list[re.Pattern[str]]:
    """Compiled Tier-A patterns (cached with the config)."""
    global _AUTO_PATTERNS
    if _AUTO_PATTERNS is None:
        compiled: list[re.Pattern[str]] = []
        for raw in _confirmation_config().get("auto_confirm_patterns") or []:
            try:
                compiled.append(re.compile(str(raw)))
            except re.error as exc:
                logger.warning("bad auto-confirm pattern %r: %s", raw, exc)
        _AUTO_PATTERNS = compiled
    return _AUTO_PATTERNS


def ask_min_confidence() -> float:
    try:
        return float(_confirmation_config().get("ask_min_confidence", 0.6))
    except (TypeError, ValueError):
        return 0.6


def ask_max_per_session() -> int:
    try:
        return int(_confirmation_config().get("ask_max_per_session", 1))
    except (TypeError, ValueError):
        return 1


def repeat_confirms_after() -> int:
    try:
        return int(_confirmation_config().get("repeat_confirms_after", 3))
    except (TypeError, ValueError):
        return 3


def subject_max_hops() -> int:
    """How far from the owner a fact's subject may be (``subject_scope.max_hops``).

    0 keeps capture to the owner; 1 admits someone the owner names through their own
    relation to them ("my wife Petra works at Infosys"; ADR-0115 decision 6).
    """
    raw = _load_raw().get("subject_scope") or {}
    try:
        return max(0, int(raw.get("max_hops", 1))) if isinstance(raw, dict) else 1
    except (TypeError, ValueError):
        return 1


def classify_capture(
    message: str, key: str, value: str, confidence: float, *, known: bool | None = None
) -> str:
    """How this fact should be confirmed: ``AUTO``, ``ASK`` or ``QUEUE``.

    Tier A (AUTO) needs all of: a plain self-statement shape from the config, an
    allowed key, and the value present verbatim in the message. That last condition is
    what keeps `name=ollama` out — "i am running ollama for local models" carries the
    value but not the shape, and "my name is ollama" would carry both, which is a
    sentence nobody types by accident.
    """
    text = message or ""
    # ``known``: the caller already knows the key is kept — a learned key (memris PR 7)
    # is not in the YAML allowlist, and follows the same tiers once it is learned.
    allowed = known if known is not None else canonical_key(key) is not None
    if not is_self_statement(text) or not allowed:
        return QUEUE
    grounded = value.strip().lower() in text.lower()
    if grounded and any(p.search(text) for p in auto_confirm_patterns()):
        return AUTO
    if confidence >= ask_min_confidence():
        return ASK
    return QUEUE


__all__ = [
    "ASK",
    "AUTO",
    "QUEUE",
    "allowed_keys",
    "ask_max_per_session",
    "ask_min_confidence",
    "auto_confirm_patterns",
    "classify_capture",
    "repeat_confirms_after",
    "canonical_key",
    "subject_max_hops",
    "is_self_statement",
    "learning_config",
    "reset_cache",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_CACHE", "_RAW_CACHE", "_AUTO_PATTERNS")
