"""Domain models for the IRIS Knowledge Wiki."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

PageType = Literal["entity", "concept", "source", "synthesis"]
EntityType = Literal["person", "institution", "concept", "event", "topic"]
LintSeverity = Literal["error", "warning", "info", "suggestion"]


@dataclass
class WikiPage:
    """A single knowledge wiki page backed by a markdown file."""

    slug: str
    page_type: PageType
    title: str
    body: str
    frontmatter: dict[str, object] = field(default_factory=dict)
    created: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def wikilinks(self) -> list[str]:
        """Return all [[wikilink]] targets found in the body."""
        import re

        return re.findall(r"\[\[([^\]]+)\]\]", self.body)

    @property
    def source_count(self) -> int:
        val = self.frontmatter.get("source_count", 0)
        if isinstance(val, int):
            return val
        try:
            return int(str(val))
        except (TypeError, ValueError):
            return 0


@dataclass(frozen=True)
class Entity:
    """A named entity extracted from agent activity."""

    name: str
    entity_type: EntityType
    source_text: str = ""
    confidence: float = 0.8


@dataclass
class WikiIngestEvent:
    """Event emitted by agents to trigger wiki updates."""

    source_agent: str
    source_id: str
    content: str
    entities_hint: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class LintResult:
    """A single wiki lint finding."""

    slug: str
    severity: LintSeverity
    check: str
    message: str


@dataclass
class LintReport:
    """Aggregated results from a wiki lint pass."""

    findings: list[LintResult] = field(default_factory=list)
    pages_checked: int = 0
    ran_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_warnings(self) -> bool:
        return any(f.severity in ("error", "warning") for f in self.findings)

    def by_severity(self, severity: LintSeverity) -> list[LintResult]:
        return [f for f in self.findings if f.severity == severity]
