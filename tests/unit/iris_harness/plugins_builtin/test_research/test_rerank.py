"""Tests for the opt-in cross-encoder reranker.

Hermetic: no real model is ever downloaded. The model is monkeypatched via
``_ensure_model`` to either ``None`` (unavailable) or a fake with ``.predict``.
"""

from __future__ import annotations

import iris_harness.plugins_builtin.research.rerank as rerank_mod
from iris_harness.plugins_builtin.research.models import SearchResult
from iris_harness.plugins_builtin.research.rerank import CrossEncoderReranker, build_reranker


def _make_results() -> list[SearchResult]:
    return [
        SearchResult(title="Alpha", url="https://a.com", snippet="first", score=0.5),
        SearchResult(title="Beta", url="https://b.com", snippet="second", score=0.4),
        SearchResult(title="Gamma", url="https://c.com", snippet="third", score=0.3),
    ]


def test_build_reranker_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("IRIS_RESEARCH_RERANKER", raising=False)
    assert build_reranker() is None


def test_build_reranker_none_for_explicit_none(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_RESEARCH_RERANKER", "none")
    assert build_reranker() is None


def test_build_reranker_returns_instance_for_bge(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_RESEARCH_RERANKER", "bge")
    reranker = build_reranker()
    assert isinstance(reranker, CrossEncoderReranker)


def test_build_reranker_returns_instance_for_cross_encoder(monkeypatch) -> None:
    monkeypatch.setenv("IRIS_RESEARCH_RERANKER", "cross-encoder")
    assert isinstance(build_reranker(), CrossEncoderReranker)


def test_rerank_noop_when_model_unavailable(monkeypatch) -> None:
    reranker = CrossEncoderReranker()
    monkeypatch.setattr(reranker, "_ensure_model", lambda: None)

    results = _make_results()
    before = [r.url for r in results]

    # Must not raise and must leave order + scores untouched.
    reranker.rerank("query", results)

    assert [r.url for r in results] == before
    assert all("cross_score" not in r.metadata for r in results)


def test_rerank_reorders_with_fake_model(monkeypatch) -> None:
    class _FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            # Score so the LAST result (Gamma) gets the highest cross score.
            return [float(i) for i in range(len(pairs))]

    reranker = CrossEncoderReranker()
    monkeypatch.setattr(reranker, "_ensure_model", lambda: _FakeModel())

    results = _make_results()
    reranker.rerank("query", results)

    # Gamma had the highest cross score -> must now be first.
    assert results[0].title == "Gamma"
    assert all("cross_score" in r.metadata for r in results)


def test_rerank_reorders_descending_fake_model(monkeypatch) -> None:
    class _FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            # Descending: the FIRST result scores highest.
            return [float(len(pairs) - i) for i in range(len(pairs))]

    reranker = CrossEncoderReranker()
    monkeypatch.setattr(reranker, "_ensure_model", lambda: _FakeModel())

    results = _make_results()
    reranker.rerank("query", results)

    assert results[0].title == "Alpha"


def test_rerank_empty_is_noop() -> None:
    CrossEncoderReranker().rerank("query", [])


def test_ensure_model_returns_none_on_import_failure(monkeypatch) -> None:
    # sentence_transformers may not be importable as CrossEncoder; force a failure
    # path by pointing at a bogus model name and confirm graceful degradation.
    reranker = CrossEncoderReranker(model_name="definitely/not-a-real-model-xyz")

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("no model")

    # Patch the lazily-imported symbol if present; otherwise the import itself fails.
    monkeypatch.setattr(rerank_mod, "CrossEncoderReranker", CrossEncoderReranker)
    # Drive the real _ensure_model with a guaranteed-failing constructor via a fake.
    import sys

    fake = type(sys)("sentence_transformers")
    fake.CrossEncoder = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    assert reranker._ensure_model() is None
