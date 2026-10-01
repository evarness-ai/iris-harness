"""Tests for the markdown-aware chunker (RAG R0)."""

from __future__ import annotations

from iris_harness.services.rag.chunker import chunk_markdown


def test_headings_title_their_sections() -> None:
    text = "# Top\n\nIntro para.\n\n## Section A\n\nBody of A.\n\n## Section B\n\nBody of B."
    chunks = chunk_markdown(text, file_title="doc", max_chars=40)
    titles = [c.title for c in chunks]
    # Intro is titled by the H1; sections by their headings.
    assert "Top" in titles
    assert "Section A" in titles and "Section B" in titles
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_large_block_splits_by_size() -> None:
    paras = "\n\n".join(f"paragraph number {i} with some filler text" for i in range(20))
    chunks = chunk_markdown(paras, file_title="big", max_chars=120)
    assert len(chunks) > 1
    assert all(c.title == "big" for c in chunks)  # no headings → file title


def test_plain_text_single_chunk() -> None:
    chunks = chunk_markdown("just one short line", file_title="note")
    assert len(chunks) == 1
    assert chunks[0].title == "note"
    assert "just one short line" in chunks[0].text


def test_empty_text_yields_nothing() -> None:
    assert chunk_markdown("", file_title="empty") == []
