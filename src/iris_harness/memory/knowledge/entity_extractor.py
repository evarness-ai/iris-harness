"""Entity extraction from agent activity via regex patterns and optional LLM."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable

from iris_harness.kernel.governance import HookContext, HookPoint, kernel_from_env
from iris_harness.kernel.governance.turn_label import apply_turn_floor
from iris_harness.llm.client import GovernedPromptCall

from .models import Entity, EntityType

logger = logging.getLogger(__name__)

# Regex patterns for common entity types
_PERSON_RE = re.compile(r"\b(?:Mr\.|Ms\.|Dr\.|Prof\.)?\s*([A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,25})\b")
_EMAIL_DOMAIN_RE = re.compile(r"@([\w\-]+\.[a-z]{2,6})")
_INSTITUTION_RE = re.compile(
    r"\b([A-Z][A-Za-z&\s]{2,40}(?:Bank|Corp|Inc|LLC|Ltd|Co\.|Foundation|Institute|University|Hospital|Agency))\b"
)
_CONCEPT_RE = re.compile(
    r"\b(monthly budget|annual report|quarterly statement|spending trend|portfolio|investment strategy)\b",
    re.IGNORECASE,
)

# Bigram stopwords — if EITHER word of a _PERSON_RE match is in this
# combined set, the match is rejected unless the exact pair is in
# _BIGRAM_ALLOWLIST. Derived from analysis of ~390 entity pages in
# data/wiki/entities/ — these are the recurring tokens across noise
# pages that the regex captured by accident. See ADR-0010.
_BIGRAM_STOPWORDS_ARTICLES: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "our",
        "their",
        "his",
        "her",
        "its",
        "who",
        "what",
        "when",
        "where",
        "which",
        "why",
        "how",
        "in",
        "on",
        "from",
        "for",
        "with",
        "of",
        "to",
        "by",
    }
)
_BIGRAM_STOPWORDS_INTERJECTIONS: frozenset[str] = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "yes",
        "no",
        "ok",
        "okay",
        "none",
        "please",
        "final",
    }
)
_BIGRAM_STOPWORDS_RECENCY: frozenset[str] = frozenset(
    {
        "new",
        "latest",
        "today",
        "tomorrow",
        "yesterday",
        "recent",
        "weekly",
        "monthly",
        "daily",
        "morning",
        "evening",
        "breaking",
        "upcoming",
        "trending",
        "top",
        "last",
        "next",
    }
)
_BIGRAM_STOPWORDS: frozenset[str] = (
    _BIGRAM_STOPWORDS_ARTICLES | _BIGRAM_STOPWORDS_INTERJECTIONS | _BIGRAM_STOPWORDS_RECENCY
)

# Legitimate proper-noun bigrams whose components include a stopword.
# Overrides the denylist. Lowercase for case-insensitive matching.
_BIGRAM_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("new", "york"),
        ("new", "delhi"),
        ("new", "jersey"),
        ("hong", "kong"),
        ("san", "francisco"),
    }
)


def _is_likely_person_bigram(first: str, second: str) -> bool:
    """Reject (first, second) if either word is a bigram-stopword,
    unless the exact pair is allowlisted. See ADR-0010."""
    pair = (first.lower(), second.lower())
    if pair in _BIGRAM_ALLOWLIST:
        return True
    return first.lower() not in _BIGRAM_STOPWORDS and second.lower() not in _BIGRAM_STOPWORDS


# Snake_case detector — rejects tool names like 'web_search', 'data_storage',
# 'vector_search' from being treated as entities. fullmatch() ensures
# 'New_York' (mixed case) or 'gmail' (no underscore) are NOT matched.
_SNAKE_CASE_RE = re.compile(r"[a-z]+(?:_[a-z]+)+")


def _is_valid_hint(hint: str) -> bool:
    """Return True if hint is plausibly an entity name. See ADR-0010.

    Rejection rules:
      - length not in 3..60 (rejects empty, single-char, runaway strings)
      - pure snake_case (tool names)
      - two-word bigram failing the person-bigram filter (stopword pair)
    """
    if not 3 <= len(hint) <= 60:
        return False
    if _SNAKE_CASE_RE.fullmatch(hint):
        return False
    parts = hint.split()
    if len(parts) == 2 and not _is_likely_person_bigram(parts[0], parts[1]):
        return False
    return True


class EntityExtractor:
    """Extract named entities from text using regex patterns and optional LLM."""

    def __init__(self, llm_call: Callable[[str], str] | None = None) -> None:
        self._llm = llm_call
        self._kernel = kernel_from_env()

    def extract(self, text: str, *, hints: list[str] | None = None) -> list[Entity]:
        """Return deduplicated entities found in text."""
        entities: dict[str, Entity] = {}

        # Hint-based entities take priority (agent pre-identified them).
        # Validation rules: see _is_valid_hint / ADR-0010.
        for hint in hints or []:
            slug = hint.strip()
            if not _is_valid_hint(slug):
                logger.debug("entity hint rejected: %r", slug)
                continue
            entities[slug.lower()] = Entity(
                name=slug,
                entity_type="topic",
                source_text=text[:100],
                confidence=0.95,
            )

        # Regex-based extraction
        for match in _PERSON_RE.finditer(text):
            name = match.group(1).strip()
            parts = name.split(None, 1)
            if len(parts) != 2 or not _is_likely_person_bigram(parts[0], parts[1]):
                continue
            entities.setdefault(
                name.lower(),
                Entity(name=name, entity_type="person", source_text=text[:100], confidence=0.8),
            )

        for match in _INSTITUTION_RE.finditer(text):
            name = match.group(1).strip()
            entities.setdefault(
                name.lower(),
                Entity(
                    name=name, entity_type="institution", source_text=text[:100], confidence=0.75
                ),
            )

        for match in _EMAIL_DOMAIN_RE.finditer(text):
            domain = match.group(1)
            org = domain.split(".")[0].title()
            entities.setdefault(
                org.lower(),
                Entity(name=org, entity_type="institution", source_text=text[:100], confidence=0.6),
            )

        for match in _CONCEPT_RE.finditer(text):
            name = match.group(1).strip().lower()
            entities.setdefault(
                name,
                Entity(
                    name=name.title(), entity_type="concept", source_text=text[:100], confidence=0.7
                ),
            )

        # LLM-based extraction as supplementary pass
        if self._llm and len(text) >= 50:
            llm_entities = self._extract_with_llm(text)
            for e in llm_entities:
                entities.setdefault(e.name.lower(), e)

        return list(entities.values())

    def _extract_with_llm(self, text: str) -> list[Entity]:
        prompt = (
            "Extract named entities from this text. Reply with one entity per line:\n"
            "NAME: <name> | TYPE: <person|institution|concept|event|topic>\n\n"
            f"Text: {text[:500]}\n\nEntities:"
        )
        try:
            raw = self._invoke_with_governance(prompt)
            return _parse_llm_entities(raw, text)
        except Exception as exc:  # the rule-based entities still stand
            logger.warning(
                "entity extraction: LLM pass failed (%s); keeping the rule-based entities",
                type(exc).__name__,
                exc_info=True,
            )
            return []

    def _invoke_with_governance(self, prompt: str) -> str:
        assert self._llm is not None
        # A GovernedPromptCall governs itself, at the tier it goes to (llm/client.py).
        if self._kernel is None or isinstance(self._llm, GovernedPromptCall):
            return self._llm(prompt)

        run_id = str(uuid.uuid4())
        classify_ctx = HookContext(
            hook_point=HookPoint.PRE_CLASSIFY,
            run_id=run_id,
            agent_type="entity_extractor",
            payload={"prompt": prompt},
        )
        classify_decision, classified_ctx = self._kernel.fire_sync(
            HookPoint.PRE_CLASSIFY, classify_ctx
        )
        if classify_decision.outcome in ("deny", "require_approval"):
            raise RuntimeError(
                f"governance blocked entity extraction call: {classify_decision.reason}"
            )

        llm_ctx = HookContext(
            hook_point=HookPoint.PRE_LLM_CALL,
            run_id=run_id,
            agent_type="entity_extractor",
            # Floored at the turn's label: this prompt may look tamer than the data the
            # turn holds (kernel/governance/turn_label.apply_turn_floor).
            classification=apply_turn_floor(classified_ctx.classification),
            # An opaque callable: where it sends the prompt is unknown, so the call is
            # governed as leaving the machine (fail closed), not guessed to be local.
            tier="tier_3",
            payload={"prompt": prompt},
        )
        decision, _ = self._kernel.fire_sync(HookPoint.PRE_LLM_CALL, llm_ctx)
        if decision.outcome in ("deny", "require_approval"):
            raise RuntimeError(f"governance blocked entity extraction call: {decision.reason}")

        return self._llm(prompt)


def _parse_llm_entities(raw: str, source_text: str) -> list[Entity]:
    entities: list[Entity] = []
    for line in raw.splitlines():
        m = re.match(r"NAME:\s*(.+?)\s*\|\s*TYPE:\s*(\w+)", line, re.IGNORECASE)
        if not m:
            continue
        name = m.group(1).strip()
        raw_type = m.group(2).strip().lower()
        entity_type: EntityType = raw_type if raw_type in ("person", "institution", "concept", "event", "topic") else "topic"  # type: ignore[assignment]
        entities.append(
            Entity(
                name=name, entity_type=entity_type, source_text=source_text[:100], confidence=0.85
            )
        )
    return entities
