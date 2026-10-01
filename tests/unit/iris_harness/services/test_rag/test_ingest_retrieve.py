"""Tests for ingestion + cited retrieval (RAG R0), keyword-fallback path."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.rag.ingest import SecretIngestError, ingest_path, sync_all
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(db_path=tmp_path / "rag.db")
    s.ensure_schema()
    return s


def _vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "alpha.md").write_text("# Alpha\n\nThe mitochondria is the powerhouse of the cell.")
    (vault / "beta.md").write_text("# Beta\n\nQuarterly revenue grew on strong cloud margins.")
    (vault / "ignore.bin").write_bytes(b"\x00 binary not a document")
    return vault


def test_ingest_folder_indexes_text_files_only(store: DocumentStore, tmp_path: Path) -> None:
    r = ingest_path(_vault(tmp_path), store=store, index=None, kind="folder")
    assert r.sources_added == 2  # the .bin (unsupported) is skipped
    assert r.chunks_indexed >= 2
    assert {Path(p).name for p in r.paths} == {"alpha.md", "beta.md"}


def test_reingest_is_idempotent_on_unchanged_content(store: DocumentStore, tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    ingest_path(vault, store=store, index=None)
    r2 = ingest_path(vault, store=store, index=None)
    assert r2.sources_added == 0 and r2.sources_updated == 0 and r2.sources_skipped == 2


def test_changed_file_is_reindexed(store: DocumentStore, tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    ingest_path(vault, store=store, index=None)
    (vault / "alpha.md").write_text("# Alpha\n\nUpdated: photosynthesis happens in chloroplasts.")
    r = sync_all(store=store, index=None)
    assert r.sources_updated == 1 and r.sources_skipped == 1
    hits = search_documents("chloroplasts", store=store, index=None)
    assert hits and "photosynthesis" in hits[0].text


def test_retrieval_returns_cited_chunks(store: DocumentStore, tmp_path: Path) -> None:
    ingest_path(_vault(tmp_path), store=store, index=None)
    hits = search_documents("mitochondria powerhouse", store=store, index=None, limit=3)
    assert hits
    top = hits[0]
    assert "mitochondria" in top.text.lower()
    assert top.citation.source_path.endswith("alpha.md")
    assert "alpha.md#" in top.citation.label()


def test_empty_query_returns_nothing(store: DocumentStore, tmp_path: Path) -> None:
    ingest_path(_vault(tmp_path), store=store, index=None)
    assert search_documents("   ", store=store, index=None) == []


def test_search_empty_store_returns_nothing(store: DocumentStore) -> None:
    assert search_documents("anything", store=store, index=None) == []


def test_secret_classification_is_refused(store: DocumentStore, tmp_path: Path) -> None:
    doc = tmp_path / "secret.md"
    doc.write_text("do not index this")
    with pytest.raises(SecretIngestError):
        ingest_path(doc, store=store, index=None, classification="secret")


def test_retrieval_carries_chunk_classification(store: DocumentStore, tmp_path: Path) -> None:
    doc = tmp_path / "note.md"
    doc.write_text("# Note\n\nQuarterly revenue improved")
    ingest_path(doc, store=store, index=None, classification="personal")

    hits = search_documents("quarterly revenue", store=store, index=None, limit=3)

    assert hits
    assert hits[0].classification == "personal"
