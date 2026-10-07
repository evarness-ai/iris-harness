"""``sync_all`` prunes a registered file that is gone from disk (issue #130).

It used to re-ingest only what was still there, so a deleted file stayed searchable and the
file domain was never told. Pruning removes RAG's own entries (chunks, source row, index
entries) and reports ``removed`` to the source; it never touches a file on disk (ADR-0067:
a file IRIS did not create is never deleted by it).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from iris_harness.services.rag.ingest import _source_id, ingest_path, sync_all
from iris_harness.services.rag.ingest_source import IndexedDocument, KnownFile, RemovedDocument
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(db_path=tmp_path / "rag.db")
    s.ensure_schema()
    return s


class Source:
    def __init__(self) -> None:
        self.removed: list[RemovedDocument] = []

    def known_file(self, path: Path) -> KnownFile | None:
        return None

    def record_indexed(self, doc: IndexedDocument) -> None:
        pass

    def record_removed(self, doc: RemovedDocument) -> None:
        self.removed.append(doc)


class Index:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_source(self, source_id: str) -> None:
        self.deleted.append(source_id)

    def upsert_chunks(self, *args: object, **kwargs: object) -> None:
        pass


def _two_notes(tmp_path: Path) -> tuple[Path, Path]:
    gone = tmp_path / "gone.md"
    kept = tmp_path / "kept.md"
    gone.write_text("# Gone\n\nThe zeppelin schedule for Tuesday.")
    kept.write_text("# Kept\n\nThe bakery opens at seven.")
    return gone, kept


def test_a_file_deleted_between_two_syncs_is_pruned_and_reported(
    store: DocumentStore, tmp_path: Path
) -> None:
    gone, kept = _two_notes(tmp_path)
    ingest_path(gone, store=store)
    ingest_path(kept, store=store)
    source, index = Source(), Index()
    assert sync_all(store=store, source=source).sources_removed == 0  # nothing vanished yet

    gone.unlink()
    result = sync_all(store=store, index=index, source=source)  # type: ignore[arg-type]

    gone_id = _source_id(gone.resolve())
    assert result.sources_removed == 1 and "1 removed" in result.summary()
    assert store.get_source(gone_id) is None and store.count_chunks(gone_id) == 0
    assert index.deleted == [gone_id]
    assert [(d.path, d.source_id, d.reason) for d in source.removed] == [
        (gone.resolve(), gone_id, "removed")
    ]
    assert store.get_source(_source_id(kept.resolve())) is not None
    assert [h.source_path for h in search_documents("zeppelin", store=store, index=None)] == []


def test_pruning_never_touches_a_file_on_disk(store: DocumentStore, tmp_path: Path) -> None:
    gone, kept = _two_notes(tmp_path)
    bystander = tmp_path / "unindexed.md"
    bystander.write_text("not registered with RAG")
    ingest_path(gone, store=store)
    ingest_path(kept, store=store)
    gone.unlink()
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.md")}

    sync_all(store=store, source=Source())

    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.md")} == before
    assert not gone.exists()  # pruning is not a restore either


def _note_in(folder: Path) -> Path:
    folder.mkdir()
    note = folder / "note.md"
    note.write_text("# Remote\n\nA note on a drive that may be unmounted.")
    return note


def test_a_missing_parent_directory_is_not_pruned(store: DocumentStore, tmp_path: Path) -> None:
    """An unmounted volume or dropped share: the file and its folder are both unreachable,
    so the file may well be there. Nothing is pruned or reported; the entry is counted unavailable.
    """
    note = _note_in(tmp_path / "volume")
    ingest_path(note, store=store)
    sid = _source_id(note.resolve())
    shutil.rmtree(tmp_path / "volume")  # stand-in for the unmount
    source, index = Source(), Index()

    result = sync_all(store=store, index=index, source=source)  # type: ignore[arg-type]

    assert result.sources_removed == 0 and result.sources_unavailable == 1
    assert result.sources_skipped == 0 and "1 skipped: location unavailable" in result.summary()
    assert source.removed == [] and index.deleted == []
    assert store.get_source(sid) is not None and store.count_chunks(sid) > 0


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 folder")
def test_an_unreadable_parent_directory_is_not_pruned(store: DocumentStore, tmp_path: Path) -> None:
    """A parent the owner cannot read cannot show the file is gone (a missing-looking file
    in a denied folder). A guard that only asked "does the file exist" would prune it."""
    folder = tmp_path / "locked"
    note = _note_in(folder)
    ingest_path(note, store=store)
    sid = _source_id(note.resolve())
    source = Source()
    folder.chmod(0o000)
    try:
        result = sync_all(store=store, source=source)
    finally:
        folder.chmod(0o700)

    assert result.sources_removed == 0 and result.sources_unavailable == 1
    assert result.sources_skipped == 0 and "1 skipped: location unavailable" in result.summary()
    assert source.removed == []
    assert store.get_source(sid) is not None


def test_a_file_missing_from_a_reachable_parent_is_still_pruned(
    store: DocumentStore, tmp_path: Path
) -> None:
    note = _note_in(tmp_path / "drive")
    ingest_path(note, store=store)
    note.unlink()  # the folder stays, so the volume is there and the file really is gone
    source = Source()

    result = sync_all(store=store, source=source)

    assert result.sources_removed == 1 and result.sources_unavailable == 0
    assert [d.reason for d in source.removed] == ["removed"]
