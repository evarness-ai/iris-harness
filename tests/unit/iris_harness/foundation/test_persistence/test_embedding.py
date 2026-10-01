"""Shared embedding-function provider (Phase 3 footprint)."""

from __future__ import annotations

import pytest

from iris_harness.foundation.persistence import embedding


@pytest.fixture(autouse=True)
def _reset() -> None:
    embedding._reset_for_tests()
    yield
    embedding._reset_for_tests()


def test_null_embeddings_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")
    assert embedding.default_embedding_function() is None
    assert embedding.collection_kwargs() == {}


def test_load_failure_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_TEST_NULL_EMBEDDINGS", raising=False)

    # Force the ONNX import to fail; the provider must degrade, not raise.
    import builtins

    real_import = builtins.__import__

    def _boom(name: str, *args: object, **kw: object):
        if name == "chromadb.utils" or name.startswith("chromadb.utils"):
            raise ImportError("simulated")
        return real_import(name, *args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert embedding.default_embedding_function() is None
    assert embedding.collection_kwargs() == {}
    # Cached failure: a second call does not retry the import.
    assert embedding.default_embedding_function() is None


def test_shared_instance_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_TEST_NULL_EMBEDDINGS", raising=False)
    ef = embedding.default_embedding_function()
    if ef is None:  # ONNX model not installed in this environment — skip
        pytest.skip("default embedding function unavailable")
    assert embedding.default_embedding_function() is ef
    assert embedding.collection_kwargs()["embedding_function"] is ef
