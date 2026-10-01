from __future__ import annotations

import sys
import types
from pathlib import Path

from iris_harness.kernel.governance.evaluator.flagged_runs import ChromaFlaggedRunThoughtWriter


class _FakeCollection:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def upsert(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _FakeClient:
    def __init__(self) -> None:
        self.collection = _FakeCollection()

    def get_or_create_collection(self, _name: str) -> _FakeCollection:
        return self.collection


def test_writer_records_flagged_row(monkeypatch, tmp_path: Path) -> None:
    client = _FakeClient()
    fake_module = types.SimpleNamespace(PersistentClient=lambda path: client)
    monkeypatch.setitem(sys.modules, "chromadb", fake_module)

    writer = ChromaFlaggedRunThoughtWriter(persist_dir=tmp_path / "chroma")
    assert writer.is_ready is True

    writer.record(
        run_id="r1",
        signal="goal_drift",
        step_id=7,
        thought="off topic",
        embedding=[0.1, 0.2],
        classification="personal",
        metadata={"distance": 0.9, "ignored": {"nested": True}},
    )

    assert len(client.collection.calls) == 1
    call = client.collection.calls[0]
    assert call["ids"] == ["r1:goal_drift:7"]
    assert call["documents"] == ["off topic"]
    assert call["embeddings"] == [[0.1, 0.2]]
    metadata = call["metadatas"][0]
    assert metadata["run_id"] == "r1"
    assert metadata["signal"] == "goal_drift"
    assert metadata["classification"] == "personal"
    assert metadata["distance"] == 0.9
    assert "ignored" not in metadata


def test_writer_is_not_ready_when_chromadb_is_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delitem(sys.modules, "chromadb", raising=False)
    monkeypatch.setitem(sys.modules, "chromadb", types.SimpleNamespace())

    writer = ChromaFlaggedRunThoughtWriter(persist_dir=tmp_path / "chroma")
    assert writer.is_ready is False
