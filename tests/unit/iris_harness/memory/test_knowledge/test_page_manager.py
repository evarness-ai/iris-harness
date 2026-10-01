"""Behavioral tests for the wiki page manager."""

from __future__ import annotations

import tempfile
from pathlib import Path

from iris_harness.memory.knowledge.models import WikiPage
from iris_harness.memory.knowledge.page_manager import PageManager, slugify


def _manager() -> tuple[PageManager, Path]:
    tmp = tempfile.mkdtemp()
    return PageManager(Path(tmp)), Path(tmp)


def test_slugify_converts_to_kebab_case() -> None:
    assert slugify("Chase Bank") == "chase-bank"
    assert slugify("John Smith Jr.").startswith("john-smith-jr")


def test_save_and_load_roundtrip() -> None:
    mgr, _ = _manager()
    page = WikiPage(
        slug="chase-bank", page_type="entity", title="Chase Bank", body="Primary checking account."
    )
    mgr.save(page)
    loaded = mgr.load("chase-bank", "entity")

    assert loaded is not None
    assert loaded.slug == "chase-bank"
    assert loaded.title == "Chase Bank"
    assert "Primary checking" in loaded.body


def test_load_returns_none_for_missing_page() -> None:
    mgr, _ = _manager()
    assert mgr.load("nonexistent", "entity") is None


def test_delete_removes_file() -> None:
    mgr, _ = _manager()
    page = WikiPage(slug="test", page_type="concept", title="Test", body="content")
    mgr.save(page)
    assert mgr.exists("test", "concept")
    mgr.delete("test", "concept")
    assert not mgr.exists("test", "concept")


def test_frontmatter_is_preserved() -> None:
    mgr, _ = _manager()
    page = WikiPage(
        slug="alice",
        page_type="entity",
        title="Alice",
        body="content",
        frontmatter={"entity_type": "person", "tags": ["finance"], "source_count": 3},
    )
    mgr.save(page)
    loaded = mgr.load("alice", "entity")

    assert loaded is not None
    assert loaded.frontmatter.get("entity_type") == "person"
    assert loaded.source_count == 3


def test_append_section_adds_content() -> None:
    mgr, _ = _manager()
    page = WikiPage(slug="bob", page_type="entity", title="Bob", body="Initial content.")
    mgr.save(page)
    mgr.append_section("bob", "entity", "## Update\nNew fact added.")
    loaded = mgr.load("bob", "entity")

    assert loaded is not None
    assert "New fact added." in loaded.body


def test_wikilinks_parsed_from_body() -> None:
    mgr, _ = _manager()
    page = WikiPage(
        slug="budget",
        page_type="concept",
        title="Budget",
        body="See [[chase-bank]] and [[monthly-budget]] for details.",
    )
    mgr.save(page)
    loaded = mgr.load("budget", "concept")

    assert loaded is not None
    assert "chase-bank" in loaded.wikilinks
    assert "monthly-budget" in loaded.wikilinks


def test_load_all_returns_all_saved_pages() -> None:
    mgr, _ = _manager()
    mgr.save(WikiPage(slug="p1", page_type="entity", title="P1", body="x"))
    mgr.save(WikiPage(slug="p2", page_type="concept", title="P2", body="y"))
    pages = mgr.load_all()

    assert {p.slug for p in pages} == {"p1", "p2"}
