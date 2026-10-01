"""Unit tests for the semantic skill router."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from iris_harness.tools.skills.semantic_router import (
    DEFAULT_SIMILARITY_THRESHOLD,
    SemanticSkillRouter,
)


@dataclass
class _StubManifest:
    name: str
    description: str
    tools: tuple[Any, ...] = ()


@dataclass
class _StubPackage:
    manifest: _StubManifest
    agent_context: str | None = None
    is_loadable: bool = True


_STUB_DIM = 64


class _DeterministicEmbedder:
    """Stub embedder: hashes tokens into a fixed-dim bag-of-words vector.

    Tests don't need real embeddings — they need a deterministic distance
    function so the router's threshold logic can be exercised in isolation.
    Fixed vector dim avoids the length-mismatch path in cosine_similarity.
    """

    def __call__(self, text: str) -> list[float]:
        tokens = {t.lower() for t in text.split() if t}
        vec = [0.0] * _STUB_DIM
        for tok in tokens:
            # Use stable hashlib digest — Python's builtin hash() is
            # salted per process so collisions move between runs.
            digest = hashlib.sha1(tok.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "big") % _STUB_DIM] = 1.0
        return vec


def _make_packages() -> list[_StubPackage]:
    return [
        _StubPackage(_StubManifest("morning-briefing", "daily morning briefing summary")),
        _StubPackage(_StubManifest("daily-repo-brief", "trending github repositories report")),
        _StubPackage(_StubManifest("calendar-reminders", "schedule calendar reminders")),
    ]


def test_best_match_returns_top_package_above_threshold():
    router = SemanticSkillRouter(_DeterministicEmbedder(), threshold=0.2)
    packages = _make_packages()
    match = router.best_match("morning briefing", packages)
    assert match is not None
    assert match.manifest.name == "morning-briefing"


def test_best_match_returns_none_when_below_threshold():
    router = SemanticSkillRouter(_DeterministicEmbedder(), threshold=0.9)
    packages = _make_packages()
    # Threshold is unrealistically high — nothing should match.
    assert router.best_match("morning briefing", packages) is None


def test_best_match_returns_none_for_unrelated_query():
    router = SemanticSkillRouter(_DeterministicEmbedder(), threshold=0.2)
    packages = _make_packages()
    assert router.best_match("weather forecast tomorrow", packages) is None


def test_rank_orders_by_descending_similarity():
    router = SemanticSkillRouter(_DeterministicEmbedder(), threshold=0.0)
    packages = _make_packages()
    ranked = router.rank("trending github repositories", packages)
    assert ranked[0][0].manifest.name == "daily-repo-brief"
    assert ranked[0][1] > ranked[1][1]


def test_unloadable_packages_are_skipped():
    router = SemanticSkillRouter(_DeterministicEmbedder(), threshold=0.0)
    packages = _make_packages()
    packages[0].is_loadable = False  # morning-briefing
    ranked = router.rank("morning briefing", packages)
    names = [p.manifest.name for p, _ in ranked]
    assert "morning-briefing" not in names


def test_empty_query_returns_no_matches():
    router = SemanticSkillRouter(_DeterministicEmbedder())
    assert router.rank("", _make_packages()) == []
    assert router.best_match("", _make_packages()) is None


def test_score_returns_zero_for_empty_query():
    router = SemanticSkillRouter(_DeterministicEmbedder())
    pkg = _make_packages()[0]
    assert router.score("", pkg) == 0.0


def test_embedding_cache_avoids_recomputation():
    embedder = _DeterministicEmbedder()
    call_count = [0]
    original = embedder.__call__

    def counting(text: str) -> list[float]:
        call_count[0] += 1
        return original(text)

    embedder.__call__ = counting  # type: ignore[method-assign]
    router = SemanticSkillRouter(embedder, threshold=0.0)
    packages = _make_packages()
    router.rank("morning briefing", packages)
    first_calls = call_count[0]
    router.rank("morning briefing", packages)
    # Second call should be served entirely from cache.
    assert call_count[0] == first_calls


def test_default_threshold_is_sensible_for_chroma_minilm():
    # The shipped default is tuned for ChromaDB ONNX MiniLM-L6; this test
    # just pins the documented value so regressions are visible.
    assert 0.3 <= DEFAULT_SIMILARITY_THRESHOLD <= 0.6


def test_rank_texts_orders_by_similarity() -> None:
    # ADR-0077 P2: generic (key, text) ranking used to shortlist ReAct tools.
    router = SemanticSkillRouter(_DeterministicEmbedder())
    items = [
        ("finance_lookup", "what you owe dues bills balances holdings net worth"),
        ("code_exec", "run python code in a sandbox"),
        ("search_inbox", "search the email inbox for messages"),
    ]
    ranked = router.rank_texts("what are my dues and bills", items)
    assert ranked[0][0] == "finance_lookup"  # most relevant to the money query
    assert {k for k, _ in ranked} == {"finance_lookup", "code_exec", "search_inbox"}


def test_rank_texts_empty_inputs() -> None:
    router = SemanticSkillRouter(_DeterministicEmbedder())
    assert router.rank_texts("", [("a", "x")]) == []
    assert router.rank_texts("q", []) == []
