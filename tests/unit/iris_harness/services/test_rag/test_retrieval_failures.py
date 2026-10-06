"""RAG degrades on a broken index or ingest source, and now says so (review 2026-09-26).

- a failed document-index query is "no hits" and warns (it was DEBUG);
- a failed ingest-source lookup falls back to the content scan alone and warns;
- a failed classification scan still records the file as ``public`` and warns — the
  fail-open label itself is left for the owner (see the PR's recommendations).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NoReturn

import pytest

from iris_harness.services.rag import sensitivity
from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.ingest import ingest_path
from iris_harness.services.rag.ingest_source import IndexedDocument, KnownFile
from iris_harness.services.rag.store import DocumentStore


def _boom(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise RuntimeError("backend down")


def _warned(caplog: pytest.LogCaptureFixture, logger: str, text: str) -> bool:
    return any(
        r.name == logger and r.levelno == logging.WARNING and text in r.getMessage()
        for r in caplog.records
    )


class _BrokenCollection:
    def count(self) -> int:
        return 2

    query = _boom


def test_a_failed_index_query_is_empty_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")
    index = DocumentIndex(persist_dir=tmp_path / "chroma_docs")
    index._col = _BrokenCollection()
    index._ok = True

    with caplog.at_level(logging.WARNING, logger="iris_harness.services.rag.index"):
        assert index.query("anything") == []

    assert _warned(caplog, "iris_harness.services.rag.index", "document index: query failed")


class _BrokenSource:
    def known_file(self, path: Path) -> KnownFile | None:
        raise RuntimeError("catalog locked")

    def record_indexed(self, doc: IndexedDocument) -> None:
        pass


def test_a_failed_source_lookup_is_none_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="iris_harness.services.rag.sensitivity"):
        assert sensitivity.known_file(_BrokenSource(), tmp_path / "note.md") is None

    assert _warned(caplog, "iris_harness.services.rag.sensitivity", "ingest source lookup failed")


class _Recorder:
    def __init__(self) -> None:
        self.indexed: list[IndexedDocument] = []

    def known_file(self, path: Path) -> KnownFile | None:
        return None

    def record_indexed(self, doc: IndexedDocument) -> None:
        self.indexed.append(doc)


def test_a_failed_classification_scan_records_personal_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = DocumentStore(db_path=tmp_path / "rag.db")
    store.ensure_schema()
    doc = tmp_path / "note.md"
    doc.write_text("# Title\n\nSome notes here.")
    recorder = _Recorder()
    monkeypatch.setattr(sensitivity, "scan_text", _boom)

    with caplog.at_level(logging.WARNING, logger="iris_harness.services.rag.sensitivity"):
        ingest_path(doc, store=store, source=recorder)

    [recorded] = recorder.indexed
    # Fails closed: an unscanned file is never recorded as public.
    assert recorded.classification == "personal"
    assert store.list_sources()[0].classification == "personal"  # the stored label too
    assert _warned(caplog, "iris_harness.services.rag.sensitivity", "classification scan failed")
