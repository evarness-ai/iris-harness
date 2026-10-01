"""A failing semantic query degrades to "no hits", and now says so at WARNING.

Review 2026-09-26: query failures were logged at DEBUG, so a broken ChromaDB looked
exactly like an empty one to the agent and to anyone reading the log. No embedding
model is loaded here: the collections are stand-ins that raise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NoReturn

import pytest

from iris_harness.memory.semantic_index import SemanticIndex

_LOGGER = "iris_harness.memory.semantic_index"


class _BrokenCollection:
    def count(self) -> int:
        return 3

    def _fail(self, *_args: Any, **_kwargs: Any) -> NoReturn:
        raise RuntimeError("hnsw index is corrupt")

    query = _fail
    get = _fail
    delete = _fail
    upsert = _fail


@pytest.fixture
def index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SemanticIndex:
    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")
    idx = SemanticIndex(persist_dir=tmp_path / "chroma")
    broken = _BrokenCollection()
    idx._facts = idx._signals = idx._turns = idx._wiki = idx._episodic = broken
    idx._ok = True
    return idx


def _warned(caplog: pytest.LogCaptureFixture, text: str) -> bool:
    return any(
        r.name == _LOGGER and r.levelno == logging.WARNING and text in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda i: i.query_facts("q"), "fact query failed"),
        (lambda i: i.query_facts_text("q"), "fact-text query failed"),
        (lambda i: i.query_signals("q"), "signal query failed"),
        (lambda i: i.query_turns_detailed("q"), "turn query failed"),
        (lambda i: i.query_wiki("q"), "wiki query failed"),
        (lambda i: i.query_episodic("q"), "episodic query failed"),
    ],
)
def test_a_failed_query_is_empty_and_warns(
    index: SemanticIndex, caplog: pytest.LogCaptureFixture, call: Any, message: str
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert call(index) == []

    assert _warned(caplog, message)


def test_an_unlistable_episodic_collection_still_indexes_and_warns(
    index: SemanticIndex, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        count = index.sync_episodic_patterns([("p1", "asks early")])

    assert count == 1
    assert _warned(caplog, "could not list episodic ids")
