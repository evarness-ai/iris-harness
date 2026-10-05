"""A document the ingest gate classified stays classified when it is re-ingested (FMX8).

Re-ingest (``sync_all``, or ``ingest_path`` on a path already indexed) used to pass no
classification, so an edited, gated document came back with unclassified chunks and
retrieval stopped carrying its label to egress gating. The edited content is now
classified by the gate's rule again and ratcheted with the stamp the source already
carries: the label stays or rises, never falls, and secret content never enters RAG.

The CLI and API surfaces are covered in their own suites (test_docs_cli.py,
test_rag_endpoints.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from iris_harness.services.rag.ingest import _source_id, ingest_path, sync_all
from iris_harness.services.rag.ingest_gate import execute_rag_ingest, propose_rag_ingest
from iris_harness.services.rag.ingest_source import IndexedDocument, KnownFile
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore

# A realistic AWS access key id (matches the credential regex pack) -> secret.
_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
_PERSONAL_TEXT = "# Contact\n\nReach me at jane@example.com about the mitochondria notes."
_PERSONAL_EDIT = "# Contact\n\nNow reach me at jane.doe@example.com about the mitochondria."
_PUBLIC_TEXT = "# Notes\n\nThe mitochondria is the powerhouse of the cell."
_PUBLIC_EDIT = "# Notes\n\nThe mitochondria is the powerhouse of every cell."


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(db_path=tmp_path / "rag.db")
    s.ensure_schema()
    return s


class FakeSource:
    def __init__(self, known: dict[Path, KnownFile] | None = None) -> None:
        self.known = known or {}

    def known_file(self, path: Path) -> KnownFile | None:
        return self.known.get(Path(path).resolve())

    def record_indexed(self, doc: IndexedDocument) -> None:
        pass


def _edit(f: Path, text: str) -> None:
    """Rewrite ``f`` and move its mtime on, so the mtime fast path cannot skip it."""
    before = f.stat().st_mtime
    f.write_text(text)
    os.utime(f, (before + 10, before + 10))


def _gated(f: Path, text: str, store: DocumentStore) -> None:
    f.write_text(text)
    execute_rag_ingest(propose_rag_ingest(f), store=store, index=None)


def _labels(store: DocumentStore, f: Path) -> set[str | None]:
    sid = _source_id(f.resolve())
    chunks = [store.get_chunk(f"{sid}:{i}") for i in range(store.count_chunks(sid))]
    return {c.classification for c in chunks if c is not None}


def test_sync_reclassifies_an_edited_gated_document(tmp_path: Path, store: DocumentStore) -> None:
    """The regression: on main the edited document came back unclassified (None)."""
    f = tmp_path / "contact.md"
    _gated(f, _PERSONAL_TEXT, store)
    assert _labels(store, f) == {"personal"}

    _edit(f, _PERSONAL_EDIT)
    result = sync_all(store=store, index=None)

    assert result.sources_updated == 1
    assert _labels(store, f) == {"personal"}
    hits = search_documents("mitochondria", store=store, index=None)
    assert hits and {h.classification for h in hits} == {"personal"}


def test_ingest_path_on_an_indexed_path_reclassifies_too(
    tmp_path: Path, store: DocumentStore
) -> None:
    """``iris docs add`` / the upload route re-ingest through ``ingest_path`` directly."""
    f = tmp_path / "contact.md"
    _gated(f, _PERSONAL_TEXT, store)
    _edit(f, _PERSONAL_EDIT)

    result = ingest_path(f, store=store, index=None)

    assert result.sources_updated == 1
    assert _labels(store, f) == {"personal"}


def test_new_content_is_classified_not_the_old_stamp_copied(
    tmp_path: Path, store: DocumentStore
) -> None:
    """A public document edited to hold personal data rises to personal."""
    f = tmp_path / "notes.md"
    _gated(f, _PUBLIC_TEXT, store)
    assert _labels(store, f) == {"public"}

    _edit(f, _PERSONAL_TEXT)
    sync_all(store=store, index=None)

    assert _labels(store, f) == {"personal"}


def test_the_label_never_falls_on_reingest(tmp_path: Path, store: DocumentStore) -> None:
    """Personal data edited out does not lower the label: the stamp is ratcheted with."""
    f = tmp_path / "contact.md"
    _gated(f, _PERSONAL_TEXT, store)

    _edit(f, _PUBLIC_EDIT)
    sync_all(store=store, index=None)

    assert _labels(store, f) == {"personal"}


def test_what_the_file_domain_knows_raises_the_label(tmp_path: Path, store: DocumentStore) -> None:
    f = tmp_path / "notes.md"
    _gated(f, _PUBLIC_TEXT, store)
    _edit(f, _PUBLIC_EDIT)

    source = FakeSource({f.resolve(): KnownFile(classification="personal")})
    sync_all(store=store, index=None, source=source)

    assert _labels(store, f) == {"personal"}


def test_an_edit_to_secret_content_is_removed_not_indexed(
    tmp_path: Path, store: DocumentStore
) -> None:
    f = tmp_path / "notes.md"
    _gated(f, _PUBLIC_TEXT, store)

    _edit(f, f"# Notes\n\nmitochondria aws_key={_AWS_KEY}\n")
    result = sync_all(store=store, index=None)

    assert result.sources_denied == 1 and result.chunks_indexed == 0
    assert "secret" in result.summary()
    assert store.list_sources() == []
    assert search_documents("mitochondria", store=store, index=None) == []


def test_an_unclassified_source_stays_unclassified(tmp_path: Path, store: DocumentStore) -> None:
    """Ingest without the gate (no stamp) keeps its prior behaviour on re-ingest."""
    f = tmp_path / "notes.md"
    f.write_text(_PERSONAL_TEXT)
    ingest_path(f, store=store, index=None)

    _edit(f, _PERSONAL_EDIT)
    result = sync_all(store=store, index=None)

    assert result.sources_updated == 1 and result.sources_denied == 0
    assert _labels(store, f) == {None}


def test_a_touched_but_unchanged_file_keeps_its_stamp(tmp_path: Path, store: DocumentStore) -> None:
    f = tmp_path / "contact.md"
    _gated(f, _PERSONAL_TEXT, store)
    _edit(f, _PERSONAL_TEXT)  # same bytes, new mtime

    result = sync_all(store=store, index=None)

    assert result.sources_skipped == 1
    assert _labels(store, f) == {"personal"}
