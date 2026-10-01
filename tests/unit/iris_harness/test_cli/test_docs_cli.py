"""`iris docs` stays core, and reaches the file domain only through the ingest seam.

`iris docs` is the RAG surface, so it survives on a core-only install (OSS plan M6,
decision 2). What it must not do is import a domain: it hands ingestion whatever
`rag.ingest_source` has registered — the file_organizer plugin registers one at CLI
start-up, and nothing is registered without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.cli.docs import docs_app
from iris_harness.services.rag.ingest_source import (
    IndexedDocument,
    KnownFile,
    current_ingest_source,
    register_ingest_source,
)

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
