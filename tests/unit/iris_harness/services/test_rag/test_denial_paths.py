"""Denial paths of RAG ingest (FMX8): the extracted-text scan, the index cleanup, and
the store's handling of an omitted classification."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import pytest
from chromadb import Documents, EmbeddingFunction

import iris_harness.foundation.persistence.embedding as embedding
from iris_harness.services.rag import ingest as ingest_mod
from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.ingest import _source_id, ingest_path
from iris_harness.services.rag.ingest_gate import (
    IngestDeniedError,
    execute_rag_ingest,
    propose_rag_ingest,
)
from iris_harness.services.rag.sensitivity import classify
from iris_harness.services.rag.store import DocumentStore

_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
_PUBLIC = "# Notes\n\nThe mitochondria is the powerhouse of the cell."


class _HashEmbedder(EmbeddingFunction[Documents]):
    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Any:
        out = []
        for text in input:
            vec = [0.0] * 64
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.sha1(word.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(vec)
        return out

    @staticmethod
    def name() -> str:
        return "iris-test-hash"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> _HashEmbedder:
        return _HashEmbedder()


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(db_path=tmp_path / "rag.db")
    s.ensure_schema()
    return s


@pytest.fixture
def real_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DocumentIndex:
    monkeypatch.delenv("IRIS_TEST_NULL_EMBEDDINGS", raising=False)
    monkeypatch.setattr(embedding, "_shared", _HashEmbedder())
    return DocumentIndex(persist_dir=tmp_path / "chroma")


def _secret_text_loader(real: Any) -> Any:
    """ingest's extractor, but the extracted text carries a secret the bytes do not."""

    def load(file: Path, raw: bytes) -> Any:
        loaded = real(file, raw)
        loaded.units = [(None, f"extracted aws_key={_AWS_KEY}")]
        return loaded

    return load


def test_classify_scans_extracted_texts(tmp_path: Path) -> None:
    f = tmp_path / "scan.md"
    f.write_text(_PUBLIC)
    assert classify(f, None)[0] != "secret"
    assert classify(f, None, texts=(f"aws_key={_AWS_KEY}",))[0] == "secret"


def test_deny_removes_the_source_from_a_real_index(
    tmp_path: Path, store: DocumentStore, real_index: DocumentIndex
) -> None:
    f = tmp_path / "notes.md"
    f.write_text(_PUBLIC)
    ingest_path(f, store=store, index=real_index)
    assert real_index.query("mitochondria powerhouse", n=3)

    f.write_text(f"# Notes\n\nmitochondria aws_key={_AWS_KEY}\n")
    os.utime(f, (f.stat().st_mtime + 10,) * 2)
    result = ingest_path(f, store=store, index=real_index)

    assert result.sources_denied == 1
    assert store.get_source(_source_id(f.resolve())) is None
    assert real_index.query("mitochondria powerhouse", n=3) == []


def test_gate_raises_when_only_the_extracted_text_is_secret(
    tmp_path: Path, store: DocumentStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "scan.md"
    f.write_text(_PUBLIC)  # the bytes scan clean, so the proposal is approved
    proposal = propose_rag_ingest(f)
    monkeypatch.setattr(ingest_mod, "_load", _secret_text_loader(ingest_mod._load))

    with pytest.raises(IngestDeniedError, match="extracted"):
        execute_rag_ingest(proposal, store=store, index=None)

    assert store.list_sources() == []


def test_upsert_without_a_classification_keeps_the_stored_label(store: DocumentStore) -> None:
    kw: dict[str, Any] = {
        "id": "s1",
        "path": "/x.md",
        "kind": "file",
        "title": "x",
        "content_sha": "a",
    }
    store.upsert_source(**kw, classification="personal")

    returned = store.upsert_source(**{**kw, "content_sha": "b"})

    assert returned.classification == "personal"
    assert store.get_source("s1").classification == "personal"  # type: ignore[union-attr]
    store.upsert_source(**kw, classification="confidential")  # a new label replaces it
    assert store.get_source("s1").classification == "confidential"  # type: ignore[union-attr]


def test_a_new_source_has_no_classification_by_default(store: DocumentStore) -> None:
    src = store.upsert_source(id="s2", path="/y.md", kind="file", title="y", content_sha="a")
    assert src.classification is None
    assert store.get_source("s2").classification is None  # type: ignore[union-attr]
