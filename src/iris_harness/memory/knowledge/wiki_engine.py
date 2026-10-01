"""IRIS Knowledge Wiki Engine — ingest, query, and lint file-backed wiki pages."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from iris_harness.foundation.env import env_flag_on

from .entity_extractor import EntityExtractor
from .models import (
    Entity,
    LintReport,
    LintResult,
    PageType,
    WikiIngestEvent,
    WikiPage,
)
from .page_manager import PageManager, slugify
from .wiki_index import WikiIndex

if TYPE_CHECKING:
    from iris_harness.memory.semantic_index import SemanticIndex


logger = logging.getLogger(__name__)


class WikiEngine:
    """Orchestrate ingest, query, and lint across file-backed wiki pages.

    When a ``SemanticIndex`` is provided every saved page is indexed into
    the ``iris_wiki_pages`` ChromaDB collection.  ``query()`` then uses
    embedding similarity as the primary ranking strategy, falling back to
    keyword search when the index is empty or unavailable.  Cross-references
    are also derived from semantic similarity rather than requiring explicit
    ``[[wikilinks]]`` in the body.
    """

    def __init__(
        self,
        wiki_root: Path,
        llm_call: Callable[[str], str] | None = None,
        semantic_index: SemanticIndex | None = None,
        *,
        ingest_enabled: bool | None = None,
    ) -> None:
        self.wiki_root = wiki_root
        self._pages = PageManager(wiki_root)
        self._index = WikiIndex(wiki_root)
        self._extractor = EntityExtractor(llm_call=llm_call)
        self._llm = llm_call
        self._semantic = semantic_index
        # Automatic ingest is OFF by default. Every chat turn and every classified
        # email wrote pages that nothing read back — 2,243 ingests against 0 logged
        # queries, because the only reader is the optional ``wiki_search`` tool and it
        # rarely survives the ReAct tool shortlist. ``IRIS_WIKI_INGEST=1`` restores the
        # old behavior; callers that ARE an explicit user action (the reingest CLI)
        # pass ``ingest_enabled=True`` instead of relying on the env flag.
        self.ingest_enabled = (
            env_flag_on("IRIS_WIKI_INGEST") if ingest_enabled is None else ingest_enabled
        )
        # Re-embedding every page at startup is a write on that same dead path (~2.5k
        # upserts per boot), so it follows the same switch.
        if self._semantic and self._semantic.is_ready and self.ingest_enabled:
            # Startup default is non-blocking so API readiness is not held by
            # large wiki re-indexing runs. Use IRIS_SYNC_WIKI_BLOCKING=1 to
            # force the previous synchronous behavior.
            if os.getenv("IRIS_SYNC_WIKI_BLOCKING", "").strip() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                self._sync_pages_to_index()
            else:
                threading.Thread(
                    target=self._sync_pages_to_index,
                    name="iris-wiki-index-sync",
                    daemon=True,
                ).start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_pages_to_index(self) -> None:
        """Bulk-index all existing pages at startup (idempotent via upsert)."""
        assert self._semantic is not None
        try:
            pages = self._pages.load_all()
            tuples = [
                (p.slug, p.page_type, p.title, p.body, p.last_updated.isoformat()) for p in pages
            ]
            self._semantic.sync_wiki_pages(tuples)
        except Exception:
            logger.exception("wiki semantic startup sync failed")

    def _index_page(self, page: WikiPage) -> None:
        """Index a single page — called after every save."""
        if self._semantic and self._semantic.is_ready:
            self._semantic.index_wiki_page(
                page.slug,
                page.page_type,
                page.title,
                page.body,
                page.last_updated.isoformat(),
            )

    def _render_with_refs(
        self,
        page: WikiPage,
        *,
        max_cross_refs: int,
        semantic_hits: list[tuple[str, str]],
    ) -> str:
        """Render a page body with cross-references.

        ``semantic_hits`` is a pre-ranked list of (slug, page_type) from
        ChromaDB; they supplement explicit ``[[wikilinks]]`` in the body.
        """
        parts = [f"**{page.title}**\n\n{page.body.strip()}"]

        # Use semantic hits as cross-refs when available, fall back to wikilinks
        ref_candidates: list[tuple[str, str]] = []
        for slug, ptype in semantic_hits:
            if slug != page.slug:
                ref_candidates.append((slug, ptype))
        if not ref_candidates:
            ref_candidates = [(s, "") for s in page.wikilinks]

        for ref_slug, ref_type in ref_candidates[:max_cross_refs]:
            ref = (
                self._pages.load(ref_slug, ref_type)  # type: ignore[arg-type]
                if ref_type
                else self._find_page_by_slug(ref_slug)
            )
            if ref is not None:
                parts.append(f"\n---\n**Related: {ref.title}**\n{ref.body[:300].strip()}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, event: WikiIngestEvent, *, force: bool = False) -> list[WikiPage]:
        """Extract entities from an agent event and upsert wiki pages.

        A no-op unless ingest is enabled (``IRIS_WIKI_INGEST`` / ``ingest_enabled``)
        or the caller passes ``force=True`` for an explicit user-invoked backfill.
        """
        if not (self.ingest_enabled or force):
            logger.debug(
                "wiki ingest skipped: automatic ingest is off (source=%s agent=%s)",
                event.source_id,
                event.source_agent,
            )
            return []
        if len(event.content) < 50:
            return []

        entities = self._extractor.extract(event.content, hints=event.entities_hint)
        upserted: list[WikiPage] = []

        for entity in entities[:15]:
            page = self._upsert_entity_page(entity, event)
            upserted.append(page)

        self._index.rebuild(self._pages.load_all())
        self._index.append_log(
            "ingest",
            f"agent={event.source_agent} source={event.source_id} pages={len(upserted)}",
        )
        return upserted

    def _upsert_entity_page(self, entity: Entity, event: WikiIngestEvent) -> WikiPage:
        slug = slugify(entity.name)
        page_type: PageType = "entity"
        existing = self._pages.load(slug, page_type)
        ts = event.timestamp.isoformat()

        if existing is not None:
            new_fact = f"- {ts}: [{event.source_agent}] {event.content[:120].strip()}"
            self._pages.append_section(slug, page_type, f"## Recent Activity\n{new_fact}")
            updated = self._pages.load(slug, page_type) or existing
            self._index_page(updated)
            return updated

        page = WikiPage(
            slug=slug,
            page_type=page_type,
            title=entity.name,
            body=(
                f"## Overview\n{entity.name} — {entity.entity_type}.\n\n"
                f"## Recent Activity\n- {ts}: [{event.source_agent}] {event.content[:120].strip()}\n"
            ),
            frontmatter={
                "entity_type": entity.entity_type,
                "tags": [entity.entity_type],
                "source_count": 1,
            },
            created=event.timestamp,
        )
        self._pages.save(page)
        self._index_page(page)
        return page

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, *, max_cross_refs: int = 3) -> str:
        """Return a compiled answer using semantic search with keyword fallback.

        Semantic path (ChromaDB available):
          1. Query ``iris_wiki_pages`` for top ``1 + max_cross_refs`` hits.
          2. Use the top hit as the main answer page.
          3. Use remaining hits as cross-references (no wikilinks required).

        Keyword fallback:
          1. Scan ``index.md`` for the best keyword match.
          2. Fall back to full-text search across all page bodies.
          3. Cross-reference via explicit ``[[wikilinks]]`` in the body.
        """
        if self._semantic and self._semantic.is_ready:
            hits = self._semantic.query_wiki(question, n=1 + max_cross_refs)
            if hits:
                main_slug, main_type = hits[0]
                page = self._pages.load(main_slug, main_type)  # type: ignore[arg-type]
                if page is not None:
                    result = self._render_with_refs(
                        page,
                        max_cross_refs=max_cross_refs,
                        semantic_hits=hits[1:],
                    )
                    self._index.append_log(
                        "query", f"q={question[:80]} slug={main_slug} mode=semantic"
                    )
                    return result

        # Keyword fallback
        slug = self._index.find_slug(question)
        if slug is None:
            return self._full_text_search(question)

        page = self._find_page_by_slug(slug)
        if page is None:
            return self._full_text_search(question)

        result = self._render_with_refs(page, max_cross_refs=max_cross_refs, semantic_hits=[])
        self._index.append_log("query", f"q={question[:80]} slug={slug} mode=keyword")
        return result

    def _full_text_search(self, query: str) -> str:
        """Keyword search across all page bodies."""
        tokens = set(query.lower().split())
        matches: list[tuple[int, WikiPage]] = []
        for page in self._pages.load_all():
            text = (page.title + " " + page.body).lower()
            score = sum(1 for t in tokens if t in text)
            if score:
                matches.append((score, page))
        if not matches:
            return ""
        matches.sort(key=lambda x: x[0], reverse=True)
        best = matches[0][1]
        return f"**{best.title}**\n\n{best.body.strip()}"

    def _find_page_by_slug(self, slug: str) -> WikiPage | None:
        for page_type in ("entity", "concept", "source", "synthesis"):
            page = self._pages.load(slug, page_type)
            if page is not None:
                return page
        return None

    # ------------------------------------------------------------------
    # Lint
    # ------------------------------------------------------------------

    def lint(self, *, staleness_days: int = 30) -> LintReport:
        """Run health checks across all wiki pages."""
        report = LintReport()
        pages = self._pages.load_all()
        report.pages_checked = len(pages)
        all_slugs = {p.slug for p in pages}
        threshold = datetime.now(UTC) - timedelta(days=staleness_days)

        # Build inbound link map
        inbound: dict[str, int] = {p.slug: 0 for p in pages}
        for page in pages:
            for ref in page.wikilinks:
                if ref in inbound:
                    inbound[ref] += 1

        for page in pages:
            if inbound.get(page.slug, 0) == 0 and not page.wikilinks:
                report.findings.append(
                    LintResult(
                        slug=page.slug,
                        severity="warning",
                        check="orphan",
                        message=f"'{page.slug}' has no inbound or outbound links.",
                    )
                )

            if page.last_updated < threshold:
                days_old = (datetime.now(UTC) - page.last_updated).days
                report.findings.append(
                    LintResult(
                        slug=page.slug,
                        severity="info",
                        check="stale",
                        message=f"'{page.slug}' not updated in {days_old} days.",
                    )
                )

            for ref in page.wikilinks:
                if ref not in all_slugs:
                    report.findings.append(
                        LintResult(
                            slug=page.slug,
                            severity="warning",
                            check="broken_link",
                            message=f"'{page.slug}' links to non-existent page '[[{ref}]]'.",
                        )
                    )

            if not page.body.strip():
                report.findings.append(
                    LintResult(
                        slug=page.slug,
                        severity="error",
                        check="empty_page",
                        message=f"'{page.slug}' has an empty body.",
                    )
                )

        self._index.append_log(
            "lint", f"pages={report.pages_checked} findings={len(report.findings)}"
        )
        return report

    # ------------------------------------------------------------------
    # Legacy dict-based API (backward compat with old KnowledgeWikiEngine)
    # ------------------------------------------------------------------

    def add_page(self, title: str, content: str) -> None:
        slug = slugify(title)
        page = WikiPage(slug=slug, page_type="concept", title=title, body=content)
        self._pages.save(page)
        self._index_page(page)

    def get_page(self, title: str) -> str:
        slug = slugify(title)
        page = self._find_page_by_slug(slug)
        return page.body if page else "Page not found."

    def list_pages(self) -> list[str]:
        return [p.title for p in self._pages.load_all()]

    def overview(self) -> str:
        count = len(self._pages.load_all())
        return f"Knowledge Wiki contains {count} pages."

    def query_pages(self, keyword: str) -> list[str]:
        return [
            p.title for p in self._pages.load_all() if keyword.lower() in (p.title + p.body).lower()
        ]

    def validate_pages(self) -> bool:
        return all(bool(p.body.strip()) for p in self._pages.load_all())


# Alias for backward compat with imports that use the old class name
KnowledgeWikiEngine = WikiEngine
