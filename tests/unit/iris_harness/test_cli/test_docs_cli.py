"""`iris docs` stays core, and reaches the file domain only through the ingest seam.

`iris docs` is the RAG surface, so it survives on a core-only install (OSS plan M6,
decision 2). What it must not do is import a domain: it hands ingestion whatever
`rag.ingest_source` has registered — the file_organizer plugin registers one at CLI
start-up, and nothing is registered without it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.cli.docs import docs_app
from iris_harness.services.rag.ingest import _source_id
from iris_harness.services.rag.ingest_gate import execute_rag_ingest, propose_rag_ingest
from iris_harness.services.rag.ingest_source import (
    IndexedDocument,
    KnownFile,
    current_ingest_source,
    register_ingest_source,
)
from iris_harness.services.rag.store import DocumentStore

runner = CliRunner()


class Recorder:
    def __init__(self) -> None:
        self.indexed: list[IndexedDocument] = []

    def known_file(self, path: Path) -> KnownFile | None:
        return None

    def record_indexed(self, doc: IndexedDocument) -> None:
        self.indexed.append(doc)


@pytest.fixture
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")


def test_docs_add_reports_to_the_registered_source(_isolated: None, tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("# Weekend\n\nGroceries and a walk.")
    rec = Recorder()
    previous = current_ingest_source()  # a plugin's CLI registration, if one ran
    try:
        register_ingest_source(rec)
        result = runner.invoke(docs_app, ["add", str(note)])
    finally:
        register_ingest_source(previous)

    assert result.exit_code == 0, result.stdout
    assert [d.path for d in rec.indexed] == [note]


def test_docs_add_works_with_no_source_registered(_isolated: None, tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("# Weekend\n\nGroceries and a walk.")
    previous = current_ingest_source()
    try:
        register_ingest_source(None)
        result = runner.invoke(docs_app, ["add", str(note)])
    finally:
        register_ingest_source(previous)

    assert result.exit_code == 0, result.stdout
    assert "1 added" in result.stdout


# ─── re-ingest keeps a gated document classified (FMX8) ─────────────────

_PERSONAL = "# Contact\n\nReach me at jane@example.com about the herons."
_PERSONAL_EDIT = "# Contact\n\nNow reach me at jane.doe@example.com about the herons."


def _bump_mtime(note: Path) -> None:
    """Move the mtime on, so the mtime fast path cannot skip an edit made in the same tick."""
    later = note.stat().st_mtime + 10
    os.utime(note, (later, later))


def _gated_then_edited(tmp_path: Path) -> Path:
    """A note the ingest gate stamped personal, then edited on disk."""
    note = tmp_path / "contact.md"
    note.write_text(_PERSONAL)
    store = DocumentStore()  # rag.db under IRIS_DATA_DIR: the store the CLI opens
    store.ensure_schema()
    execute_rag_ingest(propose_rag_ingest(note), store=store, index=None)
    note.write_text(_PERSONAL_EDIT)
    _bump_mtime(note)
    return note


def _labels(note: Path) -> set[str | None]:
    store = DocumentStore()
    sid = _source_id(note.resolve())
    chunks = [store.get_chunk(f"{sid}:{i}") for i in range(store.count_chunks(sid))]
    return {c.classification for c in chunks if c is not None}


@pytest.mark.parametrize("command", [["sync"], ["add", "{note}"]])
def test_docs_reingest_keeps_the_classification(
    _isolated: None, tmp_path: Path, command: list[str]
) -> None:
    note = _gated_then_edited(tmp_path)
    previous = current_ingest_source()
    try:
        register_ingest_source(None)
        result = runner.invoke(docs_app, [a.format(note=note) for a in command])
    finally:
        register_ingest_source(previous)

    assert result.exit_code == 0, result.stdout
    assert "1 updated" in result.stdout
    assert _labels(note) == {"personal"}


def test_docs_sync_reports_an_edit_to_secret(_isolated: None, tmp_path: Path) -> None:
    note = _gated_then_edited(tmp_path)
    note.write_text("# Contact\n\naws_key=AKIAIOSFODNN7EXAMPLE\n")
    _bump_mtime(note)
    previous = current_ingest_source()
    try:
        register_ingest_source(None)
        result = runner.invoke(docs_app, ["sync"])
    finally:
        register_ingest_source(previous)

    assert result.exit_code == 0, result.stdout
    assert "1 removed: now classified secret" in " ".join(result.stdout.split())
    assert DocumentStore().list_sources() == []
