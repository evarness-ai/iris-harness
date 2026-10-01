"""Tests for the shared semantic near-duplicate filter (behaviors + intentions)."""

from __future__ import annotations

from dataclasses import dataclass

from iris_harness.services.learning.dedup import cosine, dedupe_by_text


@dataclass
class _Item:
    name: str


def _embed_from(mapping: dict[str, list[float]]):  # type: ignore[no-untyped-def]
    return lambda texts: [mapping[t] for t in texts]


def test_cosine_basic() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero vector -> 0, no div error


def test_drops_paraphrase_of_existing() -> None:
    items = [_Item("checks london weather each morning")]
    embed = _embed_from(
        {
            "checks london weather each morning": [1.0, 0.0],
            "daily weather in london": [0.99, 0.02],  # cosine ~1 -> duplicate
        }
    )
    kept = dedupe_by_text(
        items, key=lambda i: i.name, existing_texts=["daily weather in london"], embed=embed
    )
    assert kept == []


def test_keeps_distinct_items() -> None:
    items = [_Item("a"), _Item("b")]
    embed = _embed_from({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    kept = dedupe_by_text(items, key=lambda i: i.name, embed=embed)
    assert len(kept) == 2


def test_drops_internal_paraphrase() -> None:
    items = [_Item("x1"), _Item("x2")]  # near-duplicates of each other
    embed = _embed_from({"x1": [1.0, 0.0], "x2": [1.0, 0.0]})
    kept = dedupe_by_text(items, key=lambda i: i.name, embed=embed)
    assert [i.name for i in kept] == ["x1"]  # first wins


def test_embeddings_unavailable_passes_through() -> None:
    items = [_Item("a"), _Item("b")]
    kept = dedupe_by_text(items, key=lambda i: i.name, embed=lambda texts: [])  # wrong length
    assert len(kept) == 2  # best-effort: never silently drops everything


def test_empty_items() -> None:
    assert dedupe_by_text([], key=lambda i: i.name, embed=lambda texts: []) == []
