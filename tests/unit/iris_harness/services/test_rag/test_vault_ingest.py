"""Tests for Obsidian-aware vault ingestion (RAG R1)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from iris_harness.services.rag.ingest import _source_id, ingest_path, sync_all
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(db_path=tmp_path / "rag.db")
    s.ensure_schema()
    return s


def _note(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "review.md").write_text(
        "---\ntitle: Quarterly Review\ntags: [finance, q3]\n---\n"
        "# Q3\n\nRevenue grew. See [[Revenue Model]] for the assumptions."
    )
    return vault


def test_frontmatter_title_and_metadata_stored(store: DocumentStore, tmp_path: Path) -> None:
    ingest_path(_note(tmp_path), store=store, index=None, kind="obsidian")
    source = store.list_sources()[0]
    assert source.title == "Quarterly Review"  # not "review"
    assert "finance" in source.tags and "q3" in source.tags
    assert source.links == ("Revenue Model",)
    assert source.kind == "obsidian"


def test_frontmatter_not_indexed_but_tags_links_searchable(
    store: DocumentStore, tmp_path: Path
) -> None:
    ingest_path(_note(tmp_path), store=store, index=None, kind="obsidian")
    # Body retrievable, YAML fence not present in chunk text.
    chunk = store.get_chunk(f"{_source_id((tmp_path / 'vault' / 'review.md').resolve())}:0")
    assert chunk is not None
    assert "title:" not in chunk.text  # frontmatter stripped
    # The context line embeds tags + links → retrievable by a linked note's name.
    hits = search_documents("Revenue Model", store=store, index=None)
    assert hits and "Revenue Model" in hits[0].text


def test_mtime_unchanged_skips_without_reindex(store: DocumentStore, tmp_path: Path) -> None:
    vault = _note(tmp_path)
    ingest_path(vault, store=store, index=None, kind="obsidian")
    # Second sync with no edits: mtime matches → skipped.
    r = sync_all(store=store, index=None)
    assert r.sources_skipped == 1 and r.sources_updated == 0


def test_mtime_bump_without_content_change_refreshes_not_reindexes(
    store: DocumentStore, tmp_path: Path
) -> None:
    vault = _note(tmp_path)
    ingest_path(vault, store=store, index=None, kind="obsidian")
    note = vault / "review.md"
    future = note.stat().st_mtime + 100
    os.utime(note, (future, future))  # touch: mtime changes, content identical
    r = ingest_path(vault, store=store, index=None, kind="obsidian")
    assert r.sources_skipped == 1 and r.sources_updated == 0
    assert store.list_sources()[0].mtime == future  # mtime refreshed
