"""The RAG document index is a projection, rebuildable from the canonical store.

Two stores back RAG: ``rag.db`` (``DocumentStore``) holds the canonical chunk text, and
the ChromaDB collection in ``data/chroma_docs`` (``DocumentIndex``) mirrors it. These tests
pin both halves of "rebuildable":

* lose the vector index  -> ``reindex_all`` rebuilds it from ``rag.db``, same content;
* lose ``rag.db``        -> re-ingesting the source files rebuilds it, same chunks.

They use a real ChromaDB with a deterministic hashing embedder, so no model is loaded.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest
from chromadb import Documents, EmbeddingFunction
from typer.testing import CliRunner

import iris_harness.foundation.persistence.embedding as embedding
from iris_harness.cli.docs import docs_app
from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.ingest import EmptyStoreRefused, ingest_path, reindex_all, sync_all
from iris_harness.services.rag.models import DocumentChunk
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore


class _HashEmbedder(EmbeddingFunction[Documents]):
    """Bag-of-words hashing: deterministic, offline, and similar texts stay close."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Any:
        vectors = []
        for text in input:
            vec = [0.0] * 64
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.sha1(word.encode()).hexdigest(), 16) % 64] += 1.0
            vectors.append(vec)
        return vectors

    @staticmethod
    def name() -> str:
        return "iris-test-hash"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> _HashEmbedder:
        return _HashEmbedder()


@pytest.fixture(autouse=True)
def _real_chroma(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real ChromaDB, hashing embedder (the suite's default stubs Chroma out entirely)."""
    monkeypatch.delenv("IRIS_TEST_NULL_EMBEDDINGS", raising=False)
    monkeypatch.setattr(embedding, "_shared", _HashEmbedder())


def _vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "alpha.md").write_text(
        "---\ntags: [biology]\n---\n# Alpha\n\nThe mitochondria is the powerhouse of the cell.\n\n"
        "## Second\n\nRibosomes assemble proteins from amino acids."
    )
    (vault / "beta.md").write_text("# Beta\n\nQuarterly revenue grew on strong cloud margins.")
    return vault


def _store(tmp_path: Path, name: str = "rag.db") -> DocumentStore:
    store = DocumentStore(db_path=tmp_path / name)
    store.ensure_schema()
    return store


def _snapshot(index: DocumentIndex) -> dict[str, tuple[str, dict[str, Any]]]:
    """Everything the index holds: id -> (text, metadata)."""
    got = index._col.get(include=["documents", "metadatas"])
    return {
        i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
    }


def _chunks(store: DocumentStore) -> list[DocumentChunk]:
    return list(store.iter_chunks())


def test_index_rebuilds_from_the_store_with_identical_content(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = DocumentIndex(persist_dir=tmp_path / "chroma_a")
    assert original.is_ready
    ingest_path(_vault(tmp_path), store=store, index=original, classification="personal")
    before = _snapshot(original)
    assert len(before) >= 3

    # The index is lost: a fresh, empty one at a new location.
    lost = DocumentIndex(persist_dir=tmp_path / "chroma_b")
    assert _snapshot(lost) == {}

    assert reindex_all(store=store, index=lost) == len(before)

    assert _snapshot(lost) == before
    assert all(meta["classification"] == "personal" for _, meta in _snapshot(lost).values())


def test_search_gives_the_same_hits_before_and_after_a_rebuild(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = DocumentIndex(persist_dir=tmp_path / "chroma_a")
    ingest_path(_vault(tmp_path), store=store, index=original)
    query = "mitochondria powerhouse of the cell"
    before = search_documents(query, store=store, index=original)

    rebuilt = DocumentIndex(persist_dir=tmp_path / "chroma_b")
    reindex_all(store=store, index=rebuilt)
    after = search_documents(query, store=store, index=rebuilt)

    assert before and [(h.chunk_id, h.score) for h in after] == [
        (h.chunk_id, h.score) for h in before
    ]


def test_sync_alone_does_not_refill_a_lost_index(tmp_path: Path) -> None:
    """Why ``reindex_all`` exists: ``sync_all`` skips unchanged files, so it refills nothing."""
    store = _store(tmp_path)
    ingest_path(_vault(tmp_path), store=store, index=DocumentIndex(persist_dir=tmp_path / "a"))
    lost = DocumentIndex(persist_dir=tmp_path / "b")

    result = sync_all(store=store, index=lost)

    assert result.chunks_indexed == 0
    assert _snapshot(lost) == {}


def test_rebuild_drops_entries_the_store_no_longer_holds(tmp_path: Path) -> None:
    store = _store(tmp_path)
    index = DocumentIndex(persist_dir=tmp_path / "chroma")
    ingest_path(_vault(tmp_path), store=store, index=index)
    expected = _snapshot(index)
    # Drift: a chunk the store does not know (e.g. a source removed while the index was down).
    index.index_chunks(
        [DocumentChunk("ghost:0", "ghost", "/gone.md", "Ghost", 0, "no longer in the store")]
    )
    assert "ghost:0" in _snapshot(index)

    reindex_all(store=store, index=index)

    assert _snapshot(index) == expected


def test_rebuild_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    index = DocumentIndex(persist_dir=tmp_path / "chroma")
    ingest_path(_vault(tmp_path), store=store, index=index)
    reindex_all(store=store, index=index)
    once = _snapshot(index)

    reindex_all(store=store, index=index)

    assert _snapshot(index) == once


def test_rebuild_of_an_empty_store_is_refused_and_leaves_the_index(tmp_path: Path) -> None:
    index = DocumentIndex(persist_dir=tmp_path / "chroma")
    index.index_chunks([DocumentChunk("x:0", "x", "/x.md", "X", 0, "stale")])
    before = _snapshot(index)

    with pytest.raises(EmptyStoreRefused, match="--force"):
        reindex_all(store=_store(tmp_path), index=index)

    assert _snapshot(index) == before != {}


def test_forced_rebuild_of_an_empty_store_empties_the_index(tmp_path: Path) -> None:
    index = DocumentIndex(persist_dir=tmp_path / "chroma")
    index.index_chunks([DocumentChunk("x:0", "x", "/x.md", "X", 0, "stale")])

    assert reindex_all(store=_store(tmp_path), index=index, force=True) == 0
    assert _snapshot(index) == {}


def test_rebuild_of_an_empty_store_and_empty_index_is_a_noop(tmp_path: Path) -> None:
    index = DocumentIndex(persist_dir=tmp_path / "chroma")

    assert reindex_all(store=_store(tmp_path), index=index) == 0


def test_rebuild_batches_large_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import iris_harness.services.rag.index as index_module

    monkeypatch.setattr(index_module, "_REBUILD_BATCH", 2)
    store = _store(tmp_path)
    index = DocumentIndex(persist_dir=tmp_path / "chroma")
    ingest_path(_vault(tmp_path), store=store, index=index)
    expected = _snapshot(index)
    assert len(expected) > 2
    for stray in ("s:0", "s:1", "s:2"):
        index.index_chunks([DocumentChunk(stray, "s", "/s.md", "S", 0, "stray")])

    reindex_all(store=store, index=index)

    assert _snapshot(index) == expected


def test_reindex_all_refuses_when_the_index_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")
    index = DocumentIndex(persist_dir=tmp_path / "chroma")
    assert not index.is_ready

    with pytest.raises(RuntimeError, match="unavailable"):
        reindex_all(store=_store(tmp_path), index=index)


def test_store_rebuilds_from_source_files_with_identical_chunks(tmp_path: Path) -> None:
    """Lose ``rag.db``: re-ingesting the untouched source files restores the same chunks."""
    vault = _vault(tmp_path)
    original = _store(tmp_path, "rag_a.db")
    ingest_path(vault, store=original, index=None)

    rebuilt = _store(tmp_path, "rag_b.db")
    ingest_path(vault, store=rebuilt, index=None)

    assert len(_chunks(original)) >= 3
    assert _chunks(rebuilt) == _chunks(original)
    assert [
        (s.id, s.path, s.title, s.content_sha, s.tags, s.links) for s in rebuilt.list_sources()
    ] == [(s.id, s.path, s.title, s.content_sha, s.tags, s.links) for s in original.list_sources()]


def test_docs_reindex_cli_rebuilds_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    runner = CliRunner()
    vault = _vault(tmp_path)
    added = runner.invoke(docs_app, ["add", str(vault)])
    assert added.exit_code == 0, added.stdout

    from iris_harness.services.rag.index import DocumentIndex as CliIndex

    index = CliIndex()  # the CLI's own: data/chroma_docs under IRIS_DATA_DIR
    before = _snapshot(index)
    assert before
    index._col.delete(ids=list(before))  # the index is emptied behind the store's back
    assert _snapshot(index) == {}

    result = runner.invoke(docs_app, ["reindex"])

    assert result.exit_code == 0, result.stdout
    assert f"{len(before)} chunk(s)" in result.stdout
    assert _snapshot(CliIndex()) == before


def test_docs_reindex_cli_reports_an_unavailable_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")

    result = CliRunner().invoke(docs_app, ["reindex"])

    assert result.exit_code == 1
    assert "unavailable" in result.output


def _seed_index_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A populated index beside an empty rag.db: the wrong-IRIS_DATA_DIR shape."""
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    from iris_harness.services.rag.index import DocumentIndex as CliIndex

    CliIndex().index_chunks([DocumentChunk("x:0", "x", "/x.md", "X", 0, "stale")])


def test_docs_reindex_cli_refuses_an_empty_store_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_index_only(monkeypatch, tmp_path)
    from iris_harness.services.rag.index import DocumentIndex as CliIndex

    result = CliRunner().invoke(docs_app, ["reindex"])

    assert result.exit_code == 1
    assert "--force" in result.output
    assert _snapshot(CliIndex()) != {}


def test_docs_reindex_cli_force_empties_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_index_only(monkeypatch, tmp_path)
    from iris_harness.services.rag.index import DocumentIndex as CliIndex

    result = CliRunner().invoke(docs_app, ["reindex", "--force"])

    assert result.exit_code == 0, result.output
    assert _snapshot(CliIndex()) == {}


def test_docs_reindex_cli_reports_any_failure_not_just_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))

    def boom(**_: Any) -> int:
        raise ValueError("chroma exploded")

    monkeypatch.setattr("iris_harness.services.rag.ingest.reindex_all", boom)

    result = CliRunner().invoke(docs_app, ["reindex"])

    assert result.exit_code == 1
    assert "chroma exploded" in result.output
