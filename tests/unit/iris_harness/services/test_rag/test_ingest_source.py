"""The RAG ingest seam (OSS plan M6.1b): what crosses it, and what happens without it.

The file domain leaves the core tree (OSS plan M6, decision 2), so the core must
ingest correctly with no source registered, and must never let a source's failure
reach the ingest. The catalog implementation is tested in
tests/unit/test_filemanager/test_rag_ingest_source.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.rag.ingest import ingest_path
from iris_harness.services.rag.ingest_source import (
    IndexedDocument,
    IngestSource,
    KnownFile,
    current_ingest_source,
    register_ingest_source,
)
from iris_harness.services.rag.store import DocumentStore

_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"  # fake key the classifier flags


class Recorder:
    def __init__(self) -> None:
        self.indexed: list[IndexedDocument] = []

    def known_file(self, path: Path) -> KnownFile | None:
        return None

    def record_indexed(self, doc: IndexedDocument) -> None:
        self.indexed.append(doc)


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(db_path=tmp_path / "rag.db")
    s.ensure_schema()
    return s


def test_indexed_document_carries_what_the_domain_needs(
    store: DocumentStore, tmp_path: Path
) -> None:
    doc = tmp_path / "note.md"
    doc.write_text("# Title\n\nSome notes here.")
    rec = Recorder()

    ingest_path(doc, store=store, index=None, kind="file", source=rec)

    assert len(rec.indexed) == 1
    reported = rec.indexed[0]
    assert reported.path == doc
    assert reported.kind == "file"
    assert reported.classification == "public"
    assert reported.byte_size == doc.stat().st_size
    assert reported.content_sha and reported.source_id
    assert reported.head == doc.read_bytes()[:64]


def test_classification_is_scanned_in_the_core(store: DocumentStore, tmp_path: Path) -> None:
    """A core-only install still classifies content: the scanner is the kernel's."""
    doc = tmp_path / "contact.md"
    doc.write_text("Reach me at jane@example.com")
    rec = Recorder()

    ingest_path(doc, store=store, index=None, kind="file", source=rec)

    assert rec.indexed[0].classification == "personal"


def test_secret_content_is_refused_and_never_reported(store: DocumentStore, tmp_path: Path) -> None:
    """Secret content never enters RAG, so the file domain hears nothing of it."""
    doc = tmp_path / "creds.md"
    doc.write_text(f"aws_secret_access_key = {_AWS_KEY}")
    rec = Recorder()

    result = ingest_path(doc, store=store, index=None, kind="file", source=rec)

    assert result.sources_denied == 1
    assert rec.indexed == [] and store.list_sources() == []


def test_a_failing_source_never_breaks_the_ingest(store: DocumentStore, tmp_path: Path) -> None:
    class Broken:
        def known_file(self, path: Path) -> KnownFile | None:
            return None

        def record_indexed(self, doc: IndexedDocument) -> None:
            raise RuntimeError("catalog is down")

    doc = tmp_path / "note.md"
    doc.write_text("content")

    result = ingest_path(doc, store=store, index=None, kind="file", source=Broken())

    assert result.sources_added == 1
    assert store.list_sources()


def test_the_registry_round_trips_and_clears() -> None:
    # Save and restore: a plugin's setup() may have registered one already, and this
    # test must not depend on -- or damage -- that.
    previous = current_ingest_source()
    rec = Recorder()
    try:
        register_ingest_source(rec)
        assert current_ingest_source() is rec
        assert isinstance(rec, IngestSource)  # runtime_checkable protocol
        register_ingest_source(None)
        assert current_ingest_source() is None  # the core-only shape
    finally:
        register_ingest_source(previous)
