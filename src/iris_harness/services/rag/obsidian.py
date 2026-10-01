"""Obsidian / markdown front-matter + wikilink parsing (RAG R1).

Pure functions so vault ingestion is deterministic and unit-testable with no
filesystem. Handles the markdown conventions an Obsidian vault adds over a
flat folder: YAML front-matter (title/tags/aliases), ``[[wikilinks]]`` (with
``|alias`` and ``#heading`` forms), and inline ``#tags``. Front-matter is
stripped from the indexed body; title/tags/links become source metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
# Inline #tag — not a heading (#... at line start with space) and not inside a word.
_INLINE_TAG_RE = re.compile(r"(?:^|\s)#([A-Za-z0-9][\w/-]*)")


@dataclass(frozen=True)
class ParsedNote:
    title: str
    body: str  # front-matter stripped
    tags: tuple[str, ...] = field(default_factory=tuple)
    links: tuple[str, ...] = field(default_factory=tuple)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split leading YAML front-matter. Returns (metadata, body-without-fm)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return meta, text[m.end() :]


def extract_wikilinks(text: str) -> tuple[str, ...]:
    """Return unique ``[[link]]`` targets (alias/heading stripped), in order."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in _WIKILINK_RE.findall(text):
        target = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if target and target not in seen:
            seen.add(target)
            out.append(target)
    return tuple(out)


def _normalise_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # "a, b" or "a b" or "#a #b"
        return [t.lstrip("#") for t in re.split(r"[,\s]+", value.strip()) if t]
    if isinstance(value, (list, tuple)):
        return [str(t).lstrip("#").strip() for t in value if str(t).strip()]
    return []


def extract_tags(metadata: dict[str, Any], body: str) -> tuple[str, ...]:
    """Front-matter ``tags`` plus inline ``#tags``, deduped in first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for tag in _normalise_tags(metadata.get("tags")) + _INLINE_TAG_RE.findall(body):
        t = tag.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return tuple(out)


def parse_note(text: str, *, file_title: str) -> ParsedNote:
    """Parse a markdown/Obsidian note into title, stripped body, tags, links."""
    meta, body = parse_frontmatter(text)
    title = str(meta.get("title") or _first_alias(meta) or file_title).strip() or file_title
    return ParsedNote(
        title=title,
        body=body,
        tags=extract_tags(meta, body),
        links=extract_wikilinks(body),
    )


def _first_alias(meta: dict[str, Any]) -> str | None:
    aliases = meta.get("aliases") or meta.get("alias")
    if isinstance(aliases, str):
        return aliases
    if isinstance(aliases, (list, tuple)) and aliases:
        return str(aliases[0])
    return None


def context_line(note: ParsedNote) -> str:
    """A compact searchable line embedding tags + linked notes, so a note is
    retrievable by its tags and the things it links to (the vault graph)."""
    parts: list[str] = []
    if note.tags:
        parts.append("tags: " + ", ".join(note.tags))
    if note.links:
        parts.append("links: " + ", ".join(note.links))
    return " | ".join(parts)


__all__ = [
    "ParsedNote",
    "parse_frontmatter",
    "extract_wikilinks",
    "extract_tags",
    "parse_note",
    "context_line",
]
