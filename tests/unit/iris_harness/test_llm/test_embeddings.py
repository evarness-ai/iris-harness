"""Unit tests for the shared sentence embedder (OSS plan M3.1).

Moved here with ``embed_corpus`` itself: it is a generic embedder the email
workflows, the email semantic index and finance categorisation all call, so it
lives in the core ``llm`` package rather than behind the email plugin.
"""

from __future__ import annotations

import numpy as np
import pytest

from iris_harness.llm import embeddings as emb
from iris_harness.llm.embeddings import embed_corpus, reset_embedder_cache


def test_embed_corpus_caches_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The expensive SentenceTransformer constructor should only run once
    per (process, model_name). Regression test for the Track 1G e2e
    finding where 14 sequential triage calls each re-loaded the model.
    """

    monkeypatch.setenv(emb.EMBED_BACKEND_ENV, emb.EMBED_BACKEND_ST)
    reset_embedder_cache()
    construction_count = [0]

    class _StubModel:
        def encode(self, texts, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings):  # type: ignore[no-untyped-def]
            return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)

    def _fake_constructor(model_name):  # type: ignore[no-untyped-def]
        construction_count[0] += 1
        return _StubModel()

    # Patch the lazy import target inside _get_embedder. The function
    # imports SentenceTransformer at call time, so we replace the module
    # attribute that the import resolves to.
    import sys
    import types

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _fake_constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    embed_corpus(["one"], model_name="stub-model")
    embed_corpus(["two"], model_name="stub-model")
    embed_corpus(["three"], model_name="stub-model")
    # Three calls, ONE construction
    assert construction_count[0] == 1

    # Different model_name forces a new construction
    embed_corpus(["x"], model_name="other-model")
    assert construction_count[0] == 2

    reset_embedder_cache()


def test_default_backend_is_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset, the switch picks the ONNX model the process already holds for ChromaDB."""
    monkeypatch.delenv(emb.EMBED_BACKEND_ENV, raising=False)
    assert emb.embed_backend() == emb.EMBED_BACKEND_ONNX


def test_unknown_backend_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(emb.EMBED_BACKEND_ENV, "torch")
    with pytest.raises(RuntimeError, match="not a backend"):
        embed_corpus(["x"])


def test_onnx_backend_uses_shared_function_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The onnx path calls the one shared embedding function and L2-normalises its output."""
    monkeypatch.delenv(emb.EMBED_BACKEND_ENV, raising=False)
    calls: list[list[str]] = []

    def _fake_shared():  # type: ignore[no-untyped-def]
        def _ef(texts):  # type: ignore[no-untyped-def]
            calls.append(list(texts))
            return [[3.0, 4.0] for _ in texts]

        return _ef

    monkeypatch.setattr(
        "iris_harness.foundation.persistence.embedding.default_embedding_function", _fake_shared
    )
    vecs = embed_corpus(["a", "b", "c", "d", "e"])
    # Small calls: a padded batch's buffers can stay resident (549 MiB at 32 on the Mac).
    assert calls == [["a", "b", "c", "d"], ["e"]]
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(vecs, [[0.6, 0.8]] * 5, atol=1e-6)


def test_onnx_backend_serves_only_the_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(emb.EMBED_BACKEND_ENV, raising=False)
    with pytest.raises(RuntimeError, match="serves only"):
        embed_corpus(["x"], model_name="BAAI/bge-small-en")


def test_onnx_backend_reports_missing_shared_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(emb.EMBED_BACKEND_ENV, raising=False)
    monkeypatch.setattr(
        "iris_harness.foundation.persistence.embedding.default_embedding_function", lambda: None
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        embed_corpus(["x"])


def test_onnx_backend_on_no_texts_returns_an_empty_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(emb.EMBED_BACKEND_ENV, raising=False)
    monkeypatch.setattr(
        "iris_harness.foundation.persistence.embedding.default_embedding_function",
        lambda: (lambda texts: [[1.0, 0.0] for _ in texts]),
    )
    assert embed_corpus([]).shape[0] == 0
