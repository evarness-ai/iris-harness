"""Behavioral tests for the WikiEngine — ingest, query, lint, and semantic search."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.memory.knowledge.models import WikiIngestEvent, WikiPage
from iris_harness.memory.knowledge.wiki_engine import WikiEngine
from iris_harness.memory.semantic_index import SemanticIndex


def _engine(tmp_path: Path | None = None) -> tuple[WikiEngine, Path]:
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    # These tests exercise the ingest path itself, so they opt in explicitly
    # (automatic ingest is off by default — see WikiEngine.ingest_enabled).
    return WikiEngine(tmp_path, ingest_enabled=True), tmp_path


def _engine_with_index(tmp_path: Path) -> tuple[WikiEngine, SemanticIndex]:
    idx = SemanticIndex(persist_dir=tmp_path / "chroma")
    engine = WikiEngine(tmp_path / "wiki", semantic_index=idx, ingest_enabled=True)
    return engine, idx


def _event(
    content: str = "", agent: str = "email", source_id: str = "s1", hints: list[str] | None = None
) -> WikiIngestEvent:
    return WikiIngestEvent(
        source_agent=agent,
        source_id=source_id,
        content=content
        or "John Smith sent an invoice from ACME Inc for the monthly budget review today. This is a sufficiently long content item.",
        entities_hint=hints or [],
        timestamp=datetime.now(UTC),
    )


def test_ingest_creates_pages_for_extracted_entities() -> None:
    engine, _ = _engine()
    pages = engine.ingest(_event(hints=["Chase Bank"]))

    assert any(p.title == "Chase Bank" for p in pages)


def test_ingest_short_content_skipped() -> None:
    engine, _ = _engine()
    pages = engine.ingest(_event(content="Hi there"))

    assert pages == []


def test_ingest_upserts_existing_page() -> None:
    engine, _ = _engine()
    engine.ingest(_event(hints=["Alice"]))
    engine.ingest(
        _event(
            hints=["Alice"],
            content="Alice sent another email about the project update in the office today.",
        )
    )

    page = engine._pages.load("alice", "entity")
    assert page is not None
    assert page.source_count >= 2


def test_query_returns_content_for_known_entity() -> None:
    engine, _ = _engine()
    engine.ingest(_event(hints=["Chase Bank"]))

    result = engine.query("Chase Bank")

    assert "Chase Bank" in result


def test_query_falls_back_to_full_text_search() -> None:
    engine, _ = _engine()
    engine.add_page("Monthly Budget", "Track expenses and income every month.")

    result = engine.query("budget expenses")

    assert "budget" in result.lower() or "expenses" in result.lower()


def test_query_empty_wiki_returns_empty_string() -> None:
    engine, _ = _engine()
    result = engine.query("anything")

    assert result == ""


def test_lint_detects_orphan_pages() -> None:
    engine, _ = _engine()
    engine.add_page("Orphan Page", "This page has no links and no one links to it.")

    report = engine.lint()

    orphan_checks = [f for f in report.findings if f.check == "orphan"]
    assert len(orphan_checks) >= 1


def test_lint_detects_broken_wikilinks() -> None:
    engine, _ = _engine()
    engine._pages.save(
        WikiPage(
            slug="linked",
            page_type="concept",
            title="Linked",
            body="See [[nonexistent-page]] for more info.",
        )
    )

    report = engine.lint()

    broken = [f for f in report.findings if f.check == "broken_link"]
    assert any("nonexistent-page" in f.message for f in broken)


def test_lint_detects_stale_pages() -> None:
    engine, _ = _engine()
    old_page = WikiPage(
        slug="old-doc", page_type="concept", title="Old Doc", body="Ancient content."
    )
    old_page.last_updated = datetime.now(UTC) - timedelta(days=60)
    engine._pages.save(old_page, touch=False)
    report = engine.lint(staleness_days=30)

    stale_checks = [f for f in report.findings if f.check == "stale"]
    assert len(stale_checks) >= 1


def test_lint_empty_wiki_has_no_findings() -> None:
    engine, _ = _engine()
    report = engine.lint()

    assert report.pages_checked == 0
    assert report.findings == []


def test_legacy_api_compatibility() -> None:
    engine, _ = _engine()
    engine.add_page("Python", "Python is a programming language.")

    assert engine.get_page("Python") != "Page not found."
    assert "Python" in engine.list_pages()
    assert engine.overview().startswith("Knowledge Wiki contains")
    assert engine.validate_pages()


def test_log_is_appended_after_operations() -> None:
    engine, tmp = _engine()
    engine.ingest(_event(hints=["Test Entity"]))
    engine.query("Test Entity")
    engine.lint()

    log = engine._index.recent_log_lines()
    events = [line for line in log if any(k in line for k in ("ingest", "query", "lint"))]
    assert len(events) >= 2


# ---------------------------------------------------------------------------
# Semantic search tests (require ChromaDB)
# ---------------------------------------------------------------------------


@pytest.mark.real_embeddings
class TestSemanticQuery:
    def test_semantic_query_finds_relevant_page(self, tmp_path: Path) -> None:
        engine, idx = _engine_with_index(tmp_path)
        engine.add_page(
            "OAuth Authentication",
            "OAuth is a standard for delegated access. Use it to connect third-party services.",
        )
        engine.add_page(
            "Git Branching",
            "Git branches allow parallel development. Use feature branches for each task.",
        )

        result = engine.query("how do I authorise third-party apps")

        assert "OAuth" in result

    def test_semantic_query_returns_cross_refs_from_hits(self, tmp_path: Path) -> None:
        engine, idx = _engine_with_index(tmp_path)
        engine.add_page(
            "Python",
            "Python is a high-level programming language used for scripting and data science.",
        )
        engine.add_page(
            "FastAPI", "FastAPI is a Python web framework for building REST APIs quickly."
        )
        engine.add_page(
            "SQLite", "SQLite is an embedded relational database used in Python applications."
        )

        result = engine.query("Python web development database")

        # Main hit and at least one related page should appear
        assert "Related:" in result

    def test_semantic_query_logs_mode_as_semantic(self, tmp_path: Path) -> None:
        engine, idx = _engine_with_index(tmp_path)
        engine.add_page(
            "Memory Management", "Memory management controls how data is stored and retrieved."
        )

        engine.query("how memory is handled")

        log = engine._index.recent_log_lines(5)
        assert any("mode=semantic" in line for line in log)

    def test_semantic_ingest_indexes_page(self, tmp_path: Path) -> None:
        engine, idx = _engine_with_index(tmp_path)
        assert idx._wiki.count() == 0

        engine.ingest(_event(hints=["Anthropic"]))

        assert idx._wiki.count() >= 1

    def test_semantic_add_page_indexes_page(self, tmp_path: Path) -> None:
        engine, idx = _engine_with_index(tmp_path)
        engine.add_page(
            "ChromaDB", "ChromaDB is an open-source vector database for AI applications."
        )

        assert idx._wiki.count() == 1
        hits = idx.query_wiki("vector database for embeddings")
        assert hits[0][0] == "chromadb"

    def test_startup_sync_indexes_existing_pages(self, tmp_path: Path) -> None:
        # Pre-populate wiki without semantic index
        engine_plain = WikiEngine(tmp_path / "wiki", ingest_enabled=True)
        engine_plain.add_page(
            "Existing Topic", "This page existed before semantic indexing was enabled."
        )

        # Now create engine with semantic index — should auto-sync
        idx = SemanticIndex(persist_dir=tmp_path / "chroma")
        engine_semantic = WikiEngine(tmp_path / "wiki", semantic_index=idx, ingest_enabled=True)

        assert idx._wiki.count() == 1
        result = engine_semantic.query("existing page topic")
        assert "Existing Topic" in result

    def test_no_index_falls_back_to_keyword(self, tmp_path: Path) -> None:
        engine, _ = _engine(tmp_path / "wiki")
        engine.add_page("Keyword Match", "This content contains the exact search term banana.")

        result = engine.query("banana")

        assert "banana" in result.lower()

    def test_empty_wiki_with_index_returns_empty_string(self, tmp_path: Path) -> None:
        engine, idx = _engine_with_index(tmp_path)
        result = engine.query("anything at all")
        assert result == ""


class TestIngestSwitch:
    """Automatic ingest is off unless opted in (IRIS_WIKI_INGEST / ingest_enabled).

    Every chat turn and every classified email used to write pages that only the
    optional ``wiki_search`` tool could read.
    """

    def test_default_is_off_and_ingest_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("IRIS_WIKI_INGEST", raising=False)
        engine = WikiEngine(tmp_path / "wiki")

        assert engine.ingest_enabled is False
        assert engine.ingest(_event("Remy Kumar sent an invoice for the Springfield flat.")) == []
        # PageManager creates the folder tree up front, so assert on written files.
        assert list((tmp_path / "wiki" / "entities").glob("*.md")) == []
        assert not (tmp_path / "wiki" / "log.md").exists()

    def test_env_flag_enables_ingest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IRIS_WIKI_INGEST", "1")
        engine = WikiEngine(tmp_path / "wiki")

        assert engine.ingest_enabled is True
        assert engine.ingest(_event("Remy Kumar sent an invoice for the Springfield flat.")) != []

    def test_force_ingests_while_switch_is_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The reingest-wiki CLI is an explicit user action, so it forces.
        monkeypatch.delenv("IRIS_WIKI_INGEST", raising=False)
        engine = WikiEngine(tmp_path / "wiki")

        pages = engine.ingest(
            _event("Remy Kumar sent an invoice for the Springfield flat."), force=True
        )

        assert pages != []

    def test_query_still_reads_existing_pages_while_ingest_is_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("IRIS_WIKI_INGEST", raising=False)
        engine = WikiEngine(tmp_path / "wiki")
        engine.add_page("Kept Topic", "This page contains the exact search term banana.")

        assert "banana" in engine.query("banana").lower()

    def test_startup_semantic_sync_follows_the_switch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Re-embedding every page at boot (~2.5k upserts) is a write on the same
        # dead path. A stub index keeps this test off the ML extras.
        synced: list[list[tuple[str, str, str, str, str]]] = []

        class _StubIndex:
            is_ready = True

            def sync_wiki_pages(self, tuples: list[tuple[str, str, str, str, str]]) -> None:
                synced.append(tuples)

        monkeypatch.setenv("IRIS_SYNC_WIKI_BLOCKING", "1")
        monkeypatch.delenv("IRIS_WIKI_INGEST", raising=False)
        WikiEngine(tmp_path / "wiki", semantic_index=_StubIndex())  # type: ignore[arg-type]
        assert synced == []

        monkeypatch.setenv("IRIS_WIKI_INGEST", "1")
        WikiEngine(tmp_path / "wiki", semantic_index=_StubIndex())  # type: ignore[arg-type]
        assert len(synced) == 1
