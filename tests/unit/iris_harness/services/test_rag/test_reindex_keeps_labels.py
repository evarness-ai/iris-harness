"""A rebuild of the RAG index keeps every source's label (never lowers, never drops one).

The label is canonical in ``rag.db`` (``document_sources.classification`` per source,
``document_chunks.classification`` per chunk); the Chroma collection only mirrors the chunk
label in its metadata, and retrieval reads the label back from the store. The one rebuild
path, ``reindex_all`` (``iris docs reindex``), re-reads chunks from the store, so a label
stays. These tests pin that for each way the index gets rebuilt:

* the index is lost (a fresh, empty collection) and ``reindex_all`` refills it;
* the embedding model changes (a different embedder over the same persisted collection);
* the same rebuild through the ``iris docs reindex`` command;
* a forced rebuild of an empty store (the documented way to empty the index);
* the explicit `--reset-collection` repair (issue 144) for a changed embedding model.

``iris docs sync`` has no ``--force``, and ``sync_all`` is not a rebuild (it skips unchanged
files), so it is pinned only as "does not touch a label". The embedder stub is the hashing
one of ``test_index_rebuild``; no model is loaded.
"""

from __future__ import annotations

import inspect
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from chromadb import Documents, EmbeddingFunction
from typer.testing import CliRunner

import iris_harness.foundation.persistence.embedding as embedding
from iris_harness.cli.docs import docs_app
from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.ingest import (
    EmbedderConflict,
    EmptyStoreRefused,
    IndexRebuildIncomplete,
    _source_id,
    ingest_path,
    reindex_all,
    reset_and_reindex,
    sync_all,
)
from iris_harness.services.rag.ingest_source import (
    IndexedDocument,
    KnownFile,
    RemovedDocument,
    register_ingest_source,
)
from iris_harness.services.rag.models import DocumentChunk
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore

from .test_index_rebuild import _HashEmbedder, _snapshot

_PERSONAL = "# Contact\n\nReach me at jane@example.com about the mitochondria notes."
_PUBLIC = "# Notes\n\nThe mitochondria is the powerhouse of the cell."
_INTERNAL = "# Plan\n\nThe mitochondria roadmap is internal."


class _OtherEmbedder(EmbeddingFunction[Documents]):
    """A different model: another name and another vector size than the hashing one."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Any:
        vectors = []
        for text in input:
            vec = [0.0] * 32
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[sum(map(ord, word)) % 32] += 1.0
            vectors.append(vec)
        return vectors

    @staticmethod
    def name() -> str:
        return "iris-test-other"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> _OtherEmbedder:
        return _OtherEmbedder()


class _Recorder:
    """A file-domain source that records every call the seam makes."""

    def __init__(self, known: dict[Path, KnownFile] | None = None) -> None:
        self.known = known or {}
        self.indexed: list[IndexedDocument] = []
        self.removed: list[RemovedDocument] = []

    def known_file(self, path: Path) -> KnownFile | None:
        return self.known.get(Path(path).resolve())

    def record_indexed(self, doc: IndexedDocument) -> None:
        self.indexed.append(doc)

    def record_removed(self, doc: RemovedDocument) -> None:
        self.removed.append(doc)


@pytest.fixture(autouse=True)
def _real_chroma(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("IRIS_TEST_NULL_EMBEDDINGS", raising=False)
    monkeypatch.setattr(embedding, "_shared", _HashEmbedder())
    yield
    register_ingest_source(None)


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(db_path=tmp_path / "rag.db")
    s.ensure_schema()
    return s


@pytest.fixture
def docs(tmp_path: Path) -> dict[str, Path]:
    folder = tmp_path / "vault"
    folder.mkdir()
    files = {"personal": _PERSONAL, "public": _PUBLIC, "internal": _INTERNAL}
    paths = {}
    for label, text in files.items():
        paths[label] = folder / f"{label}.md"
        paths[label].write_text(text)
    return paths


def _state(store: DocumentStore) -> dict[str, Any]:
    """Every label rag.db holds: per source, and per chunk."""
    return {
        "sources": {s.id: s.classification for s in store.list_sources()},
        "chunks": {c.id: c.classification for c in store.iter_chunks()},
    }


def _mirror(index: DocumentIndex) -> dict[str, Any]:
    """The label the index mirrors for each chunk (None when the metadata has none)."""
    return {cid: meta.get("classification") for cid, (_, meta) in _snapshot(index).items()}


def _indexed(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], source: _Recorder | None = None
) -> DocumentIndex:
    index = DocumentIndex(persist_dir=tmp_path / "chroma")
    assert index.is_ready
    ingest_path(docs["personal"].parent, store=store, index=index, source=source)
    return index


def test_the_fixture_labels_are_what_the_tests_assume(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path]
) -> None:
    """Guard the premise: three different labels, at least one above the default."""
    _indexed(tmp_path, store, docs)
    by_name = {Path(s.path).stem: s.classification for s in store.list_sources()}
    assert by_name["personal"] == "personal"
    assert by_name["public"] == "public"
    assert len(set(by_name.values())) >= 2


def test_a_rebuild_into_a_lost_index_keeps_every_label(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path]
) -> None:
    original = _indexed(tmp_path, store, docs)
    before_store, before_mirror = _state(store), _mirror(original)
    assert "personal" in before_store["sources"].values()

    lost = DocumentIndex(persist_dir=tmp_path / "chroma_lost")
    assert _mirror(lost) == {}
    assert reindex_all(store=store, index=lost) == len(before_store["chunks"])

    assert _state(store) == before_store  # canonical labels untouched
    assert _mirror(lost) == before_mirror == before_store["chunks"]  # mirror re-derived from them


def test_a_rebuild_never_lowers_a_chunk_label_the_store_holds(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path]
) -> None:
    """The mirror is re-read from the store, so a label that differs from a fresh scan stays.

    The file was labelled ``personal`` and then edited to read as public; the ratchet keeps
    ``personal``. A rebuild that re-scanned or reset labels would show ``public``.
    """
    index = _indexed(tmp_path, store, docs)
    docs["personal"].write_text(_PUBLIC)
    st = docs["personal"].stat()
    os.utime(docs["personal"], (st.st_mtime + 10, st.st_mtime + 10))
    sync_all(store=store, index=index)
    sid = _source_id(docs["personal"].resolve())
    assert store.get_source(sid).classification == "personal"  # type: ignore[union-attr]

    reindex_all(store=store, index=index)

    assert store.get_source(sid).classification == "personal"  # type: ignore[union-attr]
    assert {v for k, v in _mirror(index).items() if k.startswith(sid)} == {"personal"}


def _model_changed(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> tuple[DocumentIndex, dict[str, Any], _Recorder]:
    """Index under one embedder, then open the same collection under another."""
    recorder = _Recorder()
    _indexed(tmp_path, store, docs, source=recorder)
    before = _state(store)
    monkeypatch.setattr(embedding, "_shared", _OtherEmbedder())
    return DocumentIndex(persist_dir=tmp_path / "chroma"), before, recorder


def test_a_model_change_leaves_every_label_and_source_in_the_store(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chroma refuses the persisted collection under a new embedder (the reset test below repairs it).

    That must not cost a label: the store is untouched, the sources stay, retrieval falls
    back to keyword search and still carries each chunk's label, and the file domain is not
    told anything was removed.
    """
    changed, before, recorder = _model_changed(tmp_path, store, docs, monkeypatch)
    indexed_before = len(recorder.indexed)

    assert not changed.is_ready
    with pytest.raises(RuntimeError, match="unavailable"):
        reindex_all(store=store, index=changed)

    assert _state(store) == before
    hits = search_documents("mitochondria", store=store, index=changed)
    assert {h.classification for h in hits} == set(before["sources"].values())
    assert len(recorder.indexed) == indexed_before and recorder.removed == []


def test_a_rebuild_into_a_fresh_collection_after_a_model_change_keeps_every_label(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repair that works today: rebuild into a collection created under the new model."""
    _, before, recorder = _model_changed(tmp_path, store, docs, monkeypatch)
    fresh = DocumentIndex(persist_dir=tmp_path / "chroma_new_model")
    assert fresh.is_ready

    reindex_all(store=store, index=fresh)
    once = _snapshot(fresh)
    reindex_all(store=store, index=fresh)

    assert _state(store) == before
    assert _mirror(fresh) == before["chunks"]
    assert _snapshot(fresh) == once
    hits = search_documents("mitochondria", store=store, index=fresh)
    assert {h.classification for h in hits} == set(before["sources"].values())
    assert recorder.removed == []


def test_reindex_can_rebuild_after_the_embedding_model_changes(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 144: the explicit reset deletes the unopenable collection and refills it."""
    changed, before, recorder = _model_changed(tmp_path, store, docs, monkeypatch)
    indexed_before = len(recorder.indexed)

    assert reset_and_reindex(store=store, index=changed) == len(before["chunks"])

    assert changed.is_ready
    assert _mirror(changed) == before["chunks"]
    assert _state(store) == before
    assert len(recorder.indexed) == indexed_before and recorder.removed == []
    hits = search_documents("mitochondria", store=store, index=changed)
    assert {h.classification for h in hits} == set(before["sources"].values())


def test_plain_reindex_after_a_model_change_names_the_reset_flag(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    changed, _, _ = _model_changed(tmp_path, store, docs, monkeypatch)

    assert changed.embedder_conflict
    with pytest.raises(EmbedderConflict, match=r"iris docs reindex --reset-collection"):
        reindex_all(store=store, index=changed)
    assert isinstance(EmbedderConflict("x"), RuntimeError)  # callers catching RuntimeError still do


def test_other_open_failures_are_not_called_an_embedder_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")
    index = DocumentIndex(persist_dir=tmp_path / "chroma")
    assert not index.is_ready and not index.embedder_conflict
    with pytest.raises(RuntimeError, match="cannot be reset"):
        index.reset_collection()


def test_the_reset_is_idempotent_and_a_healthy_index_can_be_reset(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, before, _ = _model_changed(tmp_path, store, docs, monkeypatch)
    index = DocumentIndex(persist_dir=tmp_path / "chroma")

    reset_and_reindex(store=store, index=index)
    once = _snapshot(index)
    reset_and_reindex(store=store, index=index)  # now a healthy collection: reset again

    assert _snapshot(index) == once
    assert _mirror(index) == before["chunks"]
    assert _state(store) == before
    assert {s.id for s in store.list_sources()} == set(before["sources"])


def test_the_reset_keeps_a_label_above_a_fresh_scan(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file labelled personal then edited to read as public keeps personal through a reset."""
    index = _indexed(tmp_path, store, docs)
    docs["personal"].write_text(_PUBLIC)
    st = docs["personal"].stat()
    os.utime(docs["personal"], (st.st_mtime + 10, st.st_mtime + 10))
    sync_all(store=store, index=index)
    sid = _source_id(docs["personal"].resolve())
    before = _state(store)
    monkeypatch.setattr(embedding, "_shared", _OtherEmbedder())
    changed = DocumentIndex(persist_dir=tmp_path / "chroma")

    reset_and_reindex(store=store, index=changed)

    assert _state(store) == before
    assert store.get_source(sid).classification == "personal"  # type: ignore[union-attr]
    assert {v for k, v in _mirror(changed).items() if k.startswith(sid)} == {"personal"}


def test_the_reset_touches_only_the_docs_collection(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import chromadb

    _indexed(tmp_path, store, docs)
    raw = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    other = raw.get_or_create_collection("some_other_collection")
    other.add(ids=["k"], documents=["keep me"], embeddings=[[0.1, 0.2]])
    monkeypatch.setattr(embedding, "_shared", _OtherEmbedder())
    changed = DocumentIndex(persist_dir=tmp_path / "chroma")
    files_before = {p.name: p.read_text() for p in docs["personal"].parent.iterdir()}
    rows_before = [(c.id, c.text, c.classification) for c in store.iter_chunks()]

    reset_and_reindex(store=store, index=changed)

    names = {getattr(c, "name", c) for c in changed._client.list_collections()}
    assert names == {"iris_documents", "some_other_collection"}
    survivor = changed._client.get_collection("some_other_collection")
    assert survivor.get(include=["documents"])["documents"] == ["keep me"]
    assert {p.name: p.read_text() for p in docs["personal"].parent.iterdir()} == files_before
    assert [(c.id, c.text, c.classification) for c in store.iter_chunks()] == rows_before


def test_the_reset_refuses_an_empty_store_before_deleting_anything(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _indexed(tmp_path, store, docs)
    populated = _snapshot(index)
    empty = DocumentStore(db_path=tmp_path / "empty.db")
    empty.ensure_schema()

    with pytest.raises(EmptyStoreRefused, match="--force"):
        reset_and_reindex(store=empty, index=index)
    assert _snapshot(index) == populated  # nothing was deleted

    assert reset_and_reindex(store=empty, index=index, force=True) == 0
    assert _snapshot(index) == {}


def test_the_reset_refuses_an_empty_store_when_the_collection_is_unopenable(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unopenable collection reports count 0, so only ``is_ready`` shows it holds vectors."""
    changed, _, _ = _model_changed(tmp_path, store, docs, monkeypatch)
    empty = DocumentStore(db_path=tmp_path / "empty.db")
    empty.ensure_schema()

    with pytest.raises(EmptyStoreRefused):
        reset_and_reindex(store=empty, index=changed)

    assert not changed.is_ready  # nothing was deleted or recreated


def test_the_reset_logs_one_info_line_with_counts_and_no_text(
    tmp_path: Path,
    store: DocumentStore,
    docs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    changed, before, _ = _model_changed(tmp_path, store, docs, monkeypatch)
    caplog.set_level("INFO", logger="iris_harness.services.rag.ingest")

    reset_and_reindex(store=store, index=changed)

    lines = [r for r in caplog.records if r.name == "iris_harness.services.rag.ingest"]
    assert len(lines) == 1 and lines[0].levelname == "INFO"
    assert f"chunks={len(before['chunks'])}" in lines[0].getMessage()
    assert "mitochondria" not in lines[0].getMessage()


def _cli_on(tmp_path: Path, store: DocumentStore, monkeypatch: pytest.MonkeyPatch) -> DocumentIndex:
    """Make the CLI open the test's persisted collection under whichever embedder is set."""
    index = DocumentIndex(persist_dir=tmp_path / "chroma")
    monkeypatch.setattr("iris_harness.cli.docs._store_and_index", lambda: (store, index))
    return index


def test_the_cli_explains_a_model_change_and_names_the_flag(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, before, _ = _model_changed(tmp_path, store, docs, monkeypatch)
    _cli_on(tmp_path, store, monkeypatch)

    result = CliRunner().invoke(docs_app, ["reindex"])

    assert result.exit_code == 1
    assert "iris docs reindex --reset-collection" in result.output
    assert "different embedding model" in " ".join(result.output.split())
    assert "Traceback" not in result.output
    assert _state(store) == before


def test_the_cli_reset_collection_flag_rebuilds_and_exits_zero(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, before, recorder = _model_changed(tmp_path, store, docs, monkeypatch)
    index = _cli_on(tmp_path, store, monkeypatch)
    register_ingest_source(recorder)
    seen = (len(recorder.indexed), len(recorder.removed))

    result = CliRunner().invoke(docs_app, ["reindex", "--reset-collection"])

    assert result.exit_code == 0, result.output
    assert f"{len(before['chunks'])} chunk(s)" in result.output
    assert "restart" in result.output
    assert _mirror(index) == before["chunks"] and _state(store) == before
    assert (len(recorder.indexed), len(recorder.removed)) == seen


def test_the_reset_flag_is_off_by_default(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plain `reindex` never resets: it neither calls the reset nor touches the collection."""
    index = _indexed(tmp_path, store, docs)
    _cli_on(tmp_path, store, monkeypatch)
    before = _snapshot(index)

    def boom(*_: Any, **__: Any) -> None:
        raise AssertionError("reset must be explicit")

    monkeypatch.setattr(DocumentIndex, "reset_collection", boom)

    result = CliRunner().invoke(docs_app, ["reindex"])

    assert result.exit_code == 0, result.output
    assert "restart" not in result.output
    assert _snapshot(index) == before


def test_the_cli_reset_collection_refuses_an_empty_store_without_force(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _indexed(tmp_path, store, docs)
    before = _snapshot(index)
    empty = DocumentStore(db_path=tmp_path / "empty.db")
    empty.ensure_schema()
    monkeypatch.setattr("iris_harness.cli.docs._store_and_index", lambda: (empty, index))

    refused = CliRunner().invoke(docs_app, ["reindex", "--reset-collection"])
    assert refused.exit_code == 1 and "--force" in refused.output
    assert _snapshot(index) == before

    forced = CliRunner().invoke(docs_app, ["reindex", "--reset-collection", "--force"])
    assert forced.exit_code == 0, forced.output
    assert _snapshot(index) == {}


def test_the_cli_reindex_keeps_every_label_and_does_not_touch_the_file_domain(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _indexed(tmp_path, store, docs)
    before_store, before_mirror = _state(store), _mirror(index)
    recorder = _Recorder()
    register_ingest_source(recorder)
    monkeypatch.setattr("iris_harness.cli.docs._store_and_index", lambda: (store, index))

    result = CliRunner().invoke(docs_app, ["reindex"])

    assert result.exit_code == 0, result.output
    assert _state(store) == before_store
    assert _mirror(index) == before_mirror
    assert recorder.indexed == [] and recorder.removed == []  # reads no file, drops nothing


def test_reindex_all_takes_no_ingest_source(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path]
) -> None:
    """The rebuild has no seam to the file domain at all, so it cannot report a removal."""
    assert "source" not in inspect.signature(reindex_all).parameters
    recorder = _Recorder()
    index = _indexed(tmp_path, store, docs, source=recorder)
    indexed_before = len(recorder.indexed)

    reindex_all(store=store, index=index)

    assert len(recorder.indexed) == indexed_before and recorder.removed == []


def test_a_rebuild_keeps_every_source_and_is_idempotent(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path]
) -> None:
    index = _indexed(tmp_path, store, docs)
    sources_before = {s.id for s in store.list_sources()}
    state = _state(store)

    reindex_all(store=store, index=index)
    once_store, once_index = _state(store), _snapshot(index)
    reindex_all(store=store, index=index)

    assert {s.id for s in store.list_sources()} == sources_before
    assert once_store == state == _state(store)
    assert _snapshot(index) == once_index


def test_sync_after_a_rebuild_keeps_the_labels(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path]
) -> None:
    """``sync_all`` (the only other re-index command; it has no force) never lowers one."""
    index = _indexed(tmp_path, store, docs)
    before = _state(store)

    reindex_all(store=store, index=index)
    sync_all(store=store, index=index)

    assert _state(store) == before
    assert _mirror(index) == before["chunks"]


def test_a_forced_rebuild_of_an_empty_store_drops_the_mirror_not_the_labels_elsewhere(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path]
) -> None:
    """``--force`` on an empty store empties the index; a populated store is never so rebuilt."""
    index = _indexed(tmp_path, store, docs)
    other = DocumentStore(db_path=tmp_path / "empty.db")
    other.ensure_schema()
    before = _state(store)

    reindex_all(store=other, index=index, force=True)  # the wrong store, deliberately forced

    assert _snapshot(index) == {}
    assert _state(store) == before  # the real store's labels are not the rebuild's to touch


# -- a running process's handle after another process reset the collection -------------------

_STALE = "iris_harness.services.rag.index"


def _warned(caplog: pytest.LogCaptureFixture) -> list[Any]:
    """Records at WARNING or above (other INFO lines may share the logger in a full run)."""
    return [r for r in caplog.records if r.levelno >= 30]


def _stale_handle(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path]
) -> tuple[DocumentIndex, DocumentIndex]:
    """(the 'server' handle opened before a reset, the 'CLI' index that then resets)."""
    server = _indexed(tmp_path, store, docs)
    cli = DocumentIndex(persist_dir=tmp_path / "chroma")
    reset_and_reindex(store=store, index=cli)  # deletes the collection the server holds
    return server, cli


def test_a_handle_opened_before_a_reset_recovers_with_one_warning(
    tmp_path: Path,
    store: DocumentStore,
    docs: dict[str, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, cli = _stale_handle(tmp_path, store, docs)
    caplog.set_level("WARNING", logger=_STALE)

    hits = search_documents("mitochondria", store=store, index=server)

    assert hits and {h.classification for h in hits} <= {"personal", "public", "internal"}
    assert server.is_ready
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "reset or removed" in warnings[0].getMessage()
    caplog.clear()
    search_documents("mitochondria", store=store, index=server)  # now healthy: silent
    assert _warned(caplog) == []
    assert set(_snapshot(server)) == set(_snapshot(cli))


def test_index_chunks_after_a_reset_recovers(
    tmp_path: Path,
    store: DocumentStore,
    docs: dict[str, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, _ = _stale_handle(tmp_path, store, docs)
    caplog.set_level("WARNING", logger=_STALE)
    extra = DocumentChunk("zz:0", "zz", "/zz.md", "ZZ", 0, "a new chunk", classification="public")

    server.index_chunks([extra])

    assert "zz:0" in _snapshot(server)  # the retry landed in the rebuilt collection
    assert [r.levelname for r in _warned(caplog)] == ["WARNING"]


def test_a_failed_reopen_degrades_to_keyword_search_with_a_warning(
    tmp_path: Path,
    store: DocumentStore,
    docs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, _ = _stale_handle(tmp_path, store, docs)

    def boom(*_: Any, **__: Any) -> None:
        raise ValueError("cannot reopen")

    monkeypatch.setattr(server._client, "get_or_create_collection", boom)
    caplog.set_level("WARNING", logger=_STALE)

    hits = search_documents("mitochondria", store=store, index=server)

    assert not server.is_ready
    assert hits  # keyword fallback still answers, labels intact
    assert any("unavailable" in r.getMessage() for r in caplog.records)
    caplog.clear()
    server.index_chunks([DocumentChunk("q:0", "q", "/q.md", "Q", 0, "secret words here")])
    assert _warned(caplog) == []  # an unavailable index is skipped, not re-warned per call


def test_index_chunks_failure_is_a_warning_with_counts_and_no_text(
    tmp_path: Path,
    store: DocumentStore,
    docs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    index = _indexed(tmp_path, store, docs)

    def boom(*_: Any, **__: Any) -> None:
        raise ValueError("upsert exploded")

    monkeypatch.setattr(index._col, "upsert", boom)
    caplog.set_level("WARNING", logger=_STALE)

    index.index_chunks([DocumentChunk("t:0", "t", "/t.md", "T", 0, "tripwire text")])

    (record,) = [r for r in caplog.records if r.levelname == "WARNING"]
    assert "1 chunk(s)" in record.getMessage() and "tripwire" not in record.getMessage()


def test_a_healthy_collection_logs_nothing_extra(
    tmp_path: Path,
    store: DocumentStore,
    docs: dict[str, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    index = _indexed(tmp_path, store, docs)
    caplog.set_level("WARNING", logger=_STALE)

    search_documents("mitochondria", store=store, index=index)
    index.index_chunks([DocumentChunk("h:0", "h", "/h.md", "H", 0, "healthy chunk")])
    index.delete_source("h")
    reindex_all(store=store, index=index)

    assert _warned(caplog) == []


def test_only_the_collection_not_found_error_triggers_a_reopen(
    tmp_path: Path,
    store: DocumentStore,
    docs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _indexed(tmp_path, store, docs)
    reopened: list[int] = []
    monkeypatch.setattr(
        index._client, "get_or_create_collection", lambda *a, **k: reopened.append(1)
    )

    def boom(*_: Any, **__: Any) -> None:
        raise ValueError("some other chroma failure")

    monkeypatch.setattr(index._col, "query", boom)

    assert index.query("mitochondria") == []
    assert reopened == []


def test_a_rebuild_that_fails_after_the_delete_says_the_index_is_incomplete(
    tmp_path: Path, store: DocumentStore, docs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _indexed(tmp_path, store, docs)

    def _boom(_chunks: Any) -> int:
        raise OSError("disk full")

    # A context, not monkeypatch.undo(): undo would also drop the autouse fixture's hashing
    # embedder, and the rerun would try to download Chroma's default model (network).
    with monkeypatch.context() as patched:
        patched.setattr(index, "rebuild", _boom)
        with pytest.raises(IndexRebuildIncomplete, match="incomplete.*Rerun"):
            reset_and_reindex(store=store, index=index)
    assert reset_and_reindex(store=store, index=index) > 0  # the rerun heals it
