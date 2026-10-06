"""Every ingest classifies, and a document keeps its label across re-ingests (FMX8).

Re-ingest (``sync_all``, or ``ingest_path`` on a path already indexed) used to pass no
classification, so an edited, gated document came back with unclassified chunks and
retrieval stopped carrying its label to egress gating. Every ingest now classifies with
the gate's rule (``sensitivity.classify``), ratcheted with the label the source carries
(kept per source in rag.db): the label stays or rises, never falls, and secret content
is refused.

The CLI and API surfaces are covered in their own suites (test_docs_cli.py,
test_rag_endpoints.py).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from iris_harness.services.rag.ingest import _source_id, ingest_path, sync_all
from iris_harness.services.rag.ingest_gate import execute_rag_ingest, propose_rag_ingest
from iris_harness.services.rag.ingest_source import IndexedDocument, KnownFile, RemovedDocument
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


def _forget_labels(store: DocumentStore) -> None:
    """Make every source look like one indexed before classification existed."""
    with store._connect() as conn:
        conn.execute("UPDATE document_sources SET classification = NULL")
        conn.execute("UPDATE document_chunks SET classification = NULL")


def test_sync_labels_an_unlabelled_source_even_when_unchanged(
    tmp_path: Path, store: DocumentStore
) -> None:
    """A source with no label is never skipped as unchanged: sync classifies it."""
    f = tmp_path / "contact.md"
    f.write_text(_PERSONAL_TEXT)
    ingest_path(f, store=store, index=None)
    _forget_labels(store)

    result = sync_all(store=store, index=None)  # same bytes, same mtime

    assert result.sources_updated == 1
    assert store.list_sources()[0].classification == "personal"
    assert _labels(store, f) == {"personal"}


def test_sync_refuses_an_unlabelled_source_that_is_secret(
    tmp_path: Path, store: DocumentStore
) -> None:
    """An unlabelled source whose content is now secret is refused and removed."""
    f = tmp_path / "old.md"
    f.write_text(_PUBLIC_TEXT)
    ingest_path(f, store=store, index=None)
    _forget_labels(store)
    _edit(f, f"# Keys\n\nmitochondria aws_key={_AWS_KEY}\n")

    result = sync_all(store=store, index=None)

    assert result.sources_denied == 1
    assert store.list_sources() == []


def test_a_first_ingest_of_a_folder_classifies_each_file(
    tmp_path: Path, store: DocumentStore
) -> None:
    folder = tmp_path / "vault"
    folder.mkdir()
    (folder / "a.md").write_text(_PUBLIC_TEXT)
    (folder / "b.md").write_text(_PERSONAL_TEXT)
    (folder / "c.md").write_text(f"# Keys\n\naws_key={_AWS_KEY}\n")

    result = ingest_path(folder, store=store, index=None)

    assert (result.sources_added, result.sources_denied) == (2, 1)
    labels = {Path(s.path).name: s.classification for s in store.list_sources()}
    assert labels == {"a.md": "public", "b.md": "personal"}


def test_the_label_survives_a_zero_chunk_ingest(tmp_path: Path, store: DocumentStore) -> None:
    """The label is kept per source, so emptying a document does not drop it."""
    f = tmp_path / "contact.md"
    _gated(f, _PERSONAL_TEXT, store)

    _edit(f, "")
    sync_all(store=store, index=None)
    sid = _source_id(f.resolve())
    assert store.count_chunks(sid) == 0
    assert store.get_source(sid).classification == "personal"  # type: ignore[union-attr]

    _edit(f, _PUBLIC_TEXT)
    sync_all(store=store, index=None)
    assert _labels(store, f) == {"personal"}


def test_removing_a_source_removes_its_label(tmp_path: Path, store: DocumentStore) -> None:
    f = tmp_path / "contact.md"
    _gated(f, _PERSONAL_TEXT, store)
    sid = _source_id(f.resolve())

    store.delete_source(sid)
    assert store.get_source(sid) is None

    _edit(f, _PUBLIC_TEXT)  # a fresh ingest starts from its own content again
    ingest_path(f, store=store, index=None)
    assert store.get_source(sid).classification == "public"  # type: ignore[union-attr]


def test_an_older_rag_db_backfills_each_source_label_from_its_chunks(tmp_path: Path) -> None:
    """The migration derives a source's label from its chunks, the most sensitive one."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE document_sources (
            id TEXT PRIMARY KEY, path TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL,
            content_sha TEXT NOT NULL, added_at TEXT NOT NULL, last_synced_at TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]', links TEXT NOT NULL DEFAULT '[]',
            mtime REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE document_chunks (
            id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_path TEXT NOT NULL,
            title TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL,
            page INTEGER, classification TEXT
        );
        """)
    now = "2026-01-01T00:00:00+00:00"
    for sid in ("mixed", "plain", "legacy"):
        conn.execute(
            "INSERT INTO document_sources VALUES (?, ?, 'file', 't', 'sha', ?, ?, '[]', '[]', 1)",
            (sid, f"/{sid}.md", now, now),
        )
    for cid, sid, label in (
        ("mixed:0", "mixed", "public"),
        ("mixed:1", "mixed", "personal"),
        ("mixed:2", "mixed", "internal"),
        ("plain:0", "plain", "internal"),
        ("legacy:0", "legacy", None),
    ):
        conn.execute(
            "INSERT INTO document_chunks VALUES (?, ?, '/x.md', 't', 0, 'x', NULL, ?)",
            (cid, sid, label),
        )
    conn.commit()
    conn.close()

    store = DocumentStore(db_path=db)
    store.ensure_schema()

    labels = {s.id: s.classification for s in store.list_sources()}
    assert labels == {"mixed": "personal", "plain": "internal", "legacy": None}


def test_a_touched_but_unchanged_file_keeps_its_stamp(tmp_path: Path, store: DocumentStore) -> None:
    f = tmp_path / "contact.md"
    _gated(f, _PERSONAL_TEXT, store)
    _edit(f, _PERSONAL_TEXT)  # same bytes, new mtime

    result = sync_all(store=store, index=None)

    assert result.sources_skipped == 1
    assert _labels(store, f) == {"personal"}


# ─── the file domain is told when RAG denies or drops a file (issue 101) ──────────────


class RemovalSource(FakeSource):
    """A source that records removals; the catalog side of the seam."""

    def __init__(self) -> None:
        super().__init__()
        self.removed: list[RemovedDocument] = []

    def record_removed(self, doc: RemovedDocument) -> None:
        self.removed.append(doc)


def test_a_denied_edit_is_reported_to_the_file_domain(tmp_path: Path, store: DocumentStore) -> None:
    f = tmp_path / "notes.md"
    _gated(f, _PUBLIC_TEXT, store)
    _edit(f, f"# Notes\n\nmitochondria aws_key={_AWS_KEY}\n")
    source = RemovalSource()

    result = sync_all(store=store, index=None, source=source)

    assert result.sources_denied == 1
    assert [(d.path, d.reason, d.classification) for d in source.removed] == [
        (f.resolve(), "denied", "secret")
    ]
    assert source.removed[0].source_id == _source_id(f.resolve())


def test_a_first_ingest_denied_as_secret_is_reported_too(
    tmp_path: Path, store: DocumentStore
) -> None:
    """The catalog may hold a record RAG's own store never had (an earlier, removed index)."""
    f = tmp_path / "keys.md"
    f.write_text(f"# Keys\n\naws_key={_AWS_KEY}\n")
    source = RemovalSource()

    result = ingest_path(f, store=store, index=None, source=source)

    assert result.sources_denied == 1
    assert [d.reason for d in source.removed] == ["denied"]


def test_an_indexed_file_is_not_reported_removed(tmp_path: Path, store: DocumentStore) -> None:
    f = tmp_path / "notes.md"
    f.write_text(_PUBLIC_TEXT)
    source = RemovalSource()

    ingest_path(f, store=store, index=None, source=source)

    assert source.removed == []


def test_a_source_without_record_removed_still_works(tmp_path: Path, store: DocumentStore) -> None:
    """The hook is optional: a source written before it existed is not broken."""
    f = tmp_path / "notes.md"
    _gated(f, _PUBLIC_TEXT, store)
    _edit(f, f"# Notes\n\nmitochondria aws_key={_AWS_KEY}\n")

    result = sync_all(store=store, index=None, source=FakeSource())

    assert result.sources_denied == 1


def test_a_source_that_raises_on_removal_never_breaks_ingest(
    tmp_path: Path, store: DocumentStore
) -> None:
    class Raising(RemovalSource):
        def record_removed(self, doc: RemovedDocument) -> None:
            raise RuntimeError("catalog down")

    f = tmp_path / "notes.md"
    _gated(f, _PUBLIC_TEXT, store)
    _edit(f, f"# Notes\n\nmitochondria aws_key={_AWS_KEY}\n")

    result = sync_all(store=store, index=None, source=Raising())

    assert result.sources_denied == 1 and store.list_sources() == []


# ─── the gate's own denials of an already-indexed file (review of PR 128) ─────────────


def _approved_then_relabelled_secret(
    f: Path, store: DocumentStore, index: object | None = None
) -> tuple[object, RemovalSource]:
    """Index ``f``, approve a re-ingest, then have the file domain relabel it secret."""
    _gated(f, _PUBLIC_TEXT, store)
    proposal = propose_rag_ingest(f)
    source = RemovalSource()
    source.known[f.resolve()] = KnownFile(classification="secret")
    return proposal, source


def test_gate_scan_denial_of_an_indexed_file_drops_its_chunks_and_tells_the_domain(
    tmp_path: Path, store: DocumentStore
) -> None:
    """The byte-scan denial in ``execute_rag_ingest`` is not only the never-indexed case.

    A file RAG already holds can reach it (the domain now says secret); it raised before
    ``ingest_path``, so the old chunks stayed searchable and the catalog kept "indexed".
    """
    from iris_harness.services.rag.ingest_gate import IngestDeniedError

    f = tmp_path / "notes.md"
    proposal, source = _approved_then_relabelled_secret(f, store)
    assert store.count_chunks(_source_id(f.resolve())) > 0

    with pytest.raises(IngestDeniedError, match="now scans as secret"):
        execute_rag_ingest(proposal, store=store, index=None, source=source)  # type: ignore[arg-type]

    assert store.get_source(_source_id(f.resolve())) is None
    assert search_documents("mitochondria", store=store, index=None) == []
    assert [(d.path, d.reason, d.classification) for d in source.removed] == [
        (f.resolve(), "denied", "secret")
    ]
    assert source.removed[0].source_id == _source_id(f.resolve())


def test_gate_scan_denial_of_a_never_indexed_file_still_tells_the_domain(
    tmp_path: Path, store: DocumentStore
) -> None:
    """Same as ``ingest_path``'s denial: the catalog may hold what RAG never did."""
    from iris_harness.services.rag.ingest_gate import IngestDeniedError

    f = tmp_path / "notes.md"
    f.write_text(_PUBLIC_TEXT)
    proposal = propose_rag_ingest(f)
    source = RemovalSource()
    source.known[f.resolve()] = KnownFile(classification="secret")

    with pytest.raises(IngestDeniedError):
        execute_rag_ingest(proposal, store=store, index=None, source=source)

    assert [d.reason for d in source.removed] == ["denied"]


def test_gate_changed_since_approved_leaves_the_held_document_alone(
    tmp_path: Path, store: DocumentStore
) -> None:
    """A TOCTOU refusal says nothing about the held (approved) content: it stays, unreported."""
    from iris_harness.services.rag.ingest_gate import IngestDeniedError

    f = tmp_path / "notes.md"
    _gated(f, _PUBLIC_TEXT, store)
    proposal = propose_rag_ingest(f)
    _edit(f, _PUBLIC_EDIT)
    source = RemovalSource()

    with pytest.raises(IngestDeniedError, match="changed since"):
        execute_rag_ingest(proposal, store=store, index=None, source=source)

    assert store.count_chunks(_source_id(f.resolve())) > 0
    assert source.removed == []
