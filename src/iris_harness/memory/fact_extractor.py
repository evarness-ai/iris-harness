"""LLM-driven user fact extraction (Slice 5 of the four-layer memory plan).

Replaces the brittle regex extractor in ``runtime/bootstrap.py`` with a
tier-1 LLM call that returns a structured list of facts.  The extractor is
opt-in via the ``IRIS_FACT_EXTRACTION_MODE=llm`` env var so the regex path
remains the default until this is battle-tested in production.

Design notes
------------
* The prompt asks for **strict JSON only** so cheap parsing is enough.
* Tier-1 models (3B class) sometimes wrap output in ``` fences or prepend a
  short preamble.  ``_parse_facts_json`` strips both before decoding.
* We never trust the LLM's confidence blindly: values outside ``[0.0, 1.0]``
  are clamped, and any fact missing ``key`` or ``value`` is dropped.
* The extractor is *additive*.  Callers persist facts via the existing
  ``MemoryStore`` / ``SemanticIndex`` plumbing — this module is purely a
  parser around an LLM call.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _prompts_raw() -> dict[str, Any]:
    """``config/memory/extraction.yaml`` — prompts and the declarative patterns."""
    import yaml

    from iris_harness.memory.ontology import memory_config_dir

    path = memory_config_dir() / "extraction.yaml"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _prompts() -> dict[str, str]:
    raw = _prompts_raw()
    return {k: str(raw[k]) for k in ("system_prompt", "user_prompt")}


def keys_block(extra: tuple[str, ...] = ()) -> str:
    """One line per fact key the ontology keeps, with what it means (closed extraction).

    Generated from the mappings and the ontology's labels, so the prompt can only offer
    keys that exist — and offers a new one the moment it is added to the YAML. ``extra``:
    keys memory learned on its own (memris PR 7), offered like the declared ones.
    """
    from iris_harness.memory.ontology import memory_ontology, normalise_fact_key

    ontology = memory_ontology()
    lines: list[str] = []
    declared: set[str] = set()
    for rule in ontology.mappings:
        if rule.source_type != "fact" or not rule.keys:
            continue
        key = normalise_fact_key(rule.keys[0])
        term = ontology.relations.get(rule.predicate) or ontology.attributes.get(rule.predicate)
        label = term.label if term is not None else rule.predicate
        line = f"- {key}" if normalise_fact_key(label) == key else f"- {key}: {label}"
        if rule.object_class is not None:
            line += f" → {ontology.classes[rule.object_class].label}"
        lines.append(line)
        declared.add(key)
    lines.extend(f"- {key}" for key in extra if key not in declared)
    return "\n".join(lines)


def system_prompt(extra_keys: tuple[str, ...] = ()) -> str:
    return _prompts()["system_prompt"].format(keys=keys_block(extra_keys))


def user_prompt(message: str) -> str:
    return _prompts()["user_prompt"].format(message=message)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedFact:
    """One fact returned by the LLM extractor.

    ``about`` names who the fact is about when it is not the owner — someone one hop
    away ("my wife Petra works at Infosys": employer=Infosys about Petra). Whether that
    someone is in scope is capture's decision (ADR-0115 decision 6), not the model's.
    """

    key: str
    value: str
    confidence: float
    about: str | None = None


class _SupportsInvoke(Protocol):
    """Protocol matching ``CodingLLMClient.invoke`` so tests can supply a fake."""

    def invoke(self, *, system_prompt: str, user_prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_facts_with_llm(
    message: str, *, client: _SupportsInvoke, extra_keys: tuple[str, ...] = ()
) -> list[ExtractedFact]:
    """Call the LLM and return the parsed list of facts (or empty on failure).

    Errors are swallowed and logged — fact extraction must never break a
    chat turn.  Callers can compare ``len(result)`` to decide whether to
    fall back to the regex path.
    """
    cleaned = (message or "").strip()
    if not cleaned:
        return []

    try:
        raw = client.invoke(
            system_prompt=system_prompt(extra_keys),
            user_prompt=user_prompt(cleaned),
        )
    except Exception:
        logger.exception("LLM fact extraction failed — falling back to no-op")
        return []

    return _parse_facts_json(raw)


# ---------------------------------------------------------------------------
# Deterministic declarative capture (issue 0020)
# ---------------------------------------------------------------------------
# A reliable floor UNDER the LLM extractor for unambiguous self-statements the
# tier-1 model keeps missing — the value is lifted verbatim from the message (so it
# passes grounding) and is a single URL/handle token (so it passes plausibility).

# A bare domain or URL: requires a dotted TLD, so "Local AI" / "agentic harness"
# never match. Keeps the user's own form (with or without scheme / www).
_URL_RE = re.compile(
    r"((?:https?://)?(?:www\.)?[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)+(?:/[^\s,;]*)?)",
    re.IGNORECASE,
)


def _declarative() -> tuple[list[tuple[str, re.Pattern[str]]], re.Pattern[str] | None]:
    """The patterns in extraction.yaml's ``declarative`` block — no key is named here."""
    config = _prompts_raw().get("declarative") or {}
    in_context = [
        (str(rule["key"]), re.compile(str(rule["context"]), re.IGNORECASE))
        for rule in config.get("url_in_context") or []
        if isinstance(rule, dict) and rule.get("key") and rule.get("context")
    ]
    nouns = [re.escape(str(n)) for n in config.get("my_noun_is") or []]
    # "my <noun> is/at/= <token>" — the noun becomes the key, the token the value.
    my_noun_is = (
        re.compile(
            r"\bmy\s+(" + "|".join(nouns) + r")\b\s*(?:is|are|:|=|at)?\s*(\S+)", re.IGNORECASE
        )
        if nouns
        else None
    )
    return in_context, my_noun_is


def _clean_token(tok: str) -> str:
    return tok.strip().strip(".,!?;:()[]\"'")


def extract_declarative_facts(message: str) -> list[ExtractedFact]:
    """High-signal self-statements the LLM extractor is unreliable at — captured
    deterministically so they're never lost: a URL stated in a configured context, and
    "my <noun> is <url/handle>" (both in extraction.yaml). Values are verbatim."""
    text = (message or "").strip()
    if not text:
        return []
    facts: list[ExtractedFact] = []
    seen: set[str] = set()

    def _add(key: str, value: str, conf: float) -> None:
        value = _clean_token(value)
        if key not in seen and value and "." in value:
            facts.append(ExtractedFact(key=key, value=value, confidence=conf))
            seen.add(key)

    in_context, my_noun_is = _declarative()
    urls = [m.group(1) for m in _URL_RE.finditer(text)]
    # e.g. "I write ... on my blog ... <url>" → that key's value is the URL.
    for key, context in in_context:
        if urls and context.search(text):
            _add(key, urls[0], 0.9)

    if my_noun_is is not None:
        for m in my_noun_is.finditer(text):
            _add(m.group(1).lower(), m.group(2), 0.9)

    return facts


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Strips a leading ```json / ``` fence and any trailing fence.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
# Finds the first ``[ ... ]`` JSON array in a longer body.
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_facts_json(raw: str) -> list[ExtractedFact]:
    """Defensively parse the LLM's raw output into ``ExtractedFact`` objects.

    Handles three common deviations from the contract:
      1. ``` ``` code fences around the JSON.
      2. A short preamble or trailing comment outside the array.
      3. Confidence values out of range or missing entirely.
    """
    if not raw:
        return []

    text = raw.strip()
    text = _FENCE_RE.sub("", text).strip()

    if not text.startswith("["):
        match = _ARRAY_RE.search(text)
        if match is None:
            logger.warning("fact extractor: no JSON array found in output: %r", raw[:200])
            return []
        text = match.group(0)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("fact extractor: JSON decode failed for output: %r", raw[:200])
        return []

    if not isinstance(payload, list):
        return []

    facts: list[ExtractedFact] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip().lower()
        value = str(item.get("value") or "").strip().rstrip(".,!?")
        if not key or not value:
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        about = " ".join(str(item.get("about") or "").split()).strip(".,!?;:\"'") or None
        facts.append(ExtractedFact(key=key, value=value, confidence=confidence, about=about))
    return facts
