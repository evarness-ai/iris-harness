"""Tests for the RAG ingest gate (FMX8): propose/execute seam, secret denial,
classification ratchet, and the TOCTOU re-scan at execute time.

The gate talks to an ``IngestSource`` (OSS plan M6.1b), not to a file catalog: the
file domain leaves the core tree, so these tests stand a fake in its place. What the
real one does with a catalog is tests/unit/test_filemanager/test_rag_ingest_source.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.rag.ingest import SecretIngestError, ingest_path
from iris_harness.services.rag.ingest_gate import (
    IngestDeniedError,
    execute_rag_ingest,
    propose_rag_ingest,
)
from iris_harness.services.rag.ingest_source import IndexedDocument, KnownFile
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore

# A realistic AWS access key id (matches the credential regex pack) -> secret.
_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
_PERSONAL_TEXT = "# Contact\n\nReach me at jane@example.com about the mitochondria notes."
_PUBLIC_TEXT = "# Notes\n\nThe mitochondria is the powerhouse of the cell."


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(db_path=tmp_path / "rag.db")
    s.ensure_schema()
    return s


class FakeSource:
    """A file domain that knows exactly what a test says it knows."""

    def __init__(self, known: dict[Path, KnownFile] | None = None) -> None:
        self.known = known or {}
        self.indexed: list[IndexedDocument] = []

    def known_file(self, path: Path) -> KnownFile | None:
        return self.known.get(Path(path).resolve())

    def record_indexed(self, doc: IndexedDocument) -> None:
        self.indexed.append(doc)


def _knows(path: Path, *, classification: str, document_type: str | None = None) -> FakeSource:
    return FakeSource(
        {path.resolve(): KnownFile(classification=classification, document_type=document_type)}
    )


# ─── propose ─────────────────────────────────────────────────────────────


def test_propose_secret_file_is_denied_with_vault_pointer(tmp_path: Path) -> None:
    f = tmp_path / "creds.md"
    f.write_text(f"# Keys\n\naws_key={_AWS_KEY}\n")
    with pytest.raises(IngestDeniedError, match="vault"):
        propose_rag_ingest(f)


def test_propose_personal_file_carries_classification(tmp_path: Path) -> None:
    f = tmp_path / "contact.md"
    f.write_text(_PERSONAL_TEXT)
    p = propose_rag_ingest(f)
    assert p.classification == "personal"
    assert p.resolved_path == str(f.resolve())
    assert p.size == f.stat().st_size
    assert f.name in p.reason


def test_propose_ratchets_known_classification_above_scan(tmp_path: Path) -> None:
    f = tmp_path / "plain.md"
    f.write_text(_PUBLIC_TEXT)  # content scans public...
    p = propose_rag_ingest(
        f, source=_knows(f, classification="personal", document_type="statement")
    )
    assert p.classification == "personal"  # ...but what the domain knows ratchets it up
    assert p.document_type == "statement"


def test_propose_secret_known_file_denies_even_when_scan_is_clean(tmp_path: Path) -> None:
    f = tmp_path / "innocuous.md"
    f.write_text(_PUBLIC_TEXT)
    with pytest.raises(IngestDeniedError, match="vault"):
        propose_rag_ingest(f, source=_knows(f, classification="secret"))


def test_propose_survives_a_source_that_raises(tmp_path: Path) -> None:
    """An advisory lookup never fails an ingest: the scan alone decides."""

    class Broken:
        def known_file(self, path: Path) -> KnownFile | None:
            raise RuntimeError("catalog is down")

        def record_indexed(self, doc: IndexedDocument) -> None: ...

    f = tmp_path / "plain.md"
    f.write_text(_PUBLIC_TEXT)
    assert propose_rag_ingest(f, source=Broken()).classification == "public"


def test_propose_missing_file_is_denied(tmp_path: Path) -> None:
    with pytest.raises(IngestDeniedError, match="not an existing file"):
        propose_rag_ingest(tmp_path / "ghost.md")


# ─── execute (TOCTOU guard) ──────────────────────────────────────────────


def test_execute_denies_when_file_swapped_to_secret(tmp_path: Path, store: DocumentStore) -> None:
    f = tmp_path / "notes.md"
    f.write_text(_PUBLIC_TEXT)
    p = propose_rag_ingest(f)
    f.write_text(f"# Keys\n\naws_key={_AWS_KEY}\n")  # swap after approval
    with pytest.raises(IngestDeniedError):
        execute_rag_ingest(p, store=store, index=None)
    assert store.list_sources() == []  # nothing was ingested


def test_execute_denies_when_content_changed_even_if_still_clean(
    tmp_path: Path, store: DocumentStore
) -> None:
    f = tmp_path / "notes.md"
    f.write_text(_PUBLIC_TEXT)
    p = propose_rag_ingest(f)
    f.write_text("# Notes\n\nHarmless but DIFFERENT content than what was approved.")
    with pytest.raises(IngestDeniedError, match="changed"):
        execute_rag_ingest(p, store=store, index=None)


def test_execute_denies_when_file_vanished(tmp_path: Path, store: DocumentStore) -> None:
    f = tmp_path / "notes.md"
    f.write_text(_PUBLIC_TEXT)
    p = propose_rag_ingest(f)
    f.unlink()
    with pytest.raises(IngestDeniedError, match="no longer exists"):
        execute_rag_ingest(p, store=store, index=None)


def test_execute_ingests_with_classification_metadata(tmp_path: Path, store: DocumentStore) -> None:
    f = tmp_path / "contact.md"
    f.write_text(_PERSONAL_TEXT)
    p = propose_rag_ingest(f)
    r = execute_rag_ingest(p, store=store, index=None)
    assert r.sources_added == 1 and r.chunks_indexed >= 1
    source = store.list_sources()[0]
    chunk = store.get_chunk(f"{source.id}:0")
    assert chunk is not None and chunk.classification == "personal"
    # Retrieval surfaces the classification on every hit (keyword fallback path).
    hits = search_documents("mitochondria", store=store, index=None)
    assert hits and hits[0].classification == "personal"


# ─── ingest_path defense in depth + classification without the gate ─────


def test_ingest_path_refuses_secret_classification(tmp_path: Path, store: DocumentStore) -> None:
    f = tmp_path / "notes.md"
    f.write_text(_PUBLIC_TEXT)
    with pytest.raises(SecretIngestError):
        ingest_path(f, store=store, index=None, classification="secret")
    assert store.list_sources() == []


@pytest.mark.parametrize(
    ("text", "label"), [(_PUBLIC_TEXT, "public"), (_PERSONAL_TEXT, "personal")]
)
def test_ingest_path_without_the_gate_still_classifies(
    tmp_path: Path, store: DocumentStore, text: str, label: str
) -> None:
    """Every ingest classifies, not only the gated one (the old contract left it None)."""
    f = tmp_path / "notes.md"
    f.write_text(text)
    ingest_path(f, store=store, index=None)
    source = store.list_sources()[0]
    assert source.classification == label
    chunk = store.get_chunk(f"{source.id}:0")
    assert chunk is not None and chunk.classification == label
    hits = search_documents("mitochondria", store=store, index=None)
    assert hits and hits[0].classification == label


def test_ingest_path_without_the_gate_refuses_secret(tmp_path: Path, store: DocumentStore) -> None:
    f = tmp_path / "creds.md"
    f.write_text(f"# Keys\n\naws_key={_AWS_KEY}\n")
    result = ingest_path(f, store=store, index=None)
    assert result.sources_denied == 1 and result.sources_added == 0
    assert store.list_sources() == []
