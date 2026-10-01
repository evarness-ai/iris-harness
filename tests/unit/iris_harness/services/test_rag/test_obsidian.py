"""Tests for Obsidian front-matter + wikilink parsing (RAG R1)."""

from __future__ import annotations

from iris_harness.services.rag.obsidian import (
    context_line,
    extract_tags,
    extract_wikilinks,
    parse_frontmatter,
    parse_note,
)

NOTE = """---
title: Quarterly Review
tags: [finance, q3]
aliases: [QR]
---
# Q3

Revenue up. See [[Revenue Model]] and [[People/Alice|Alice]].
Also relates to [[Roadmap#H2]]. Tagged #planning inline.
"""


def test_parse_frontmatter_splits_yaml_and_body() -> None:
    meta, body = parse_frontmatter(NOTE)
    assert meta["title"] == "Quarterly Review"
    assert meta["tags"] == ["finance", "q3"]
    assert body.lstrip().startswith("# Q3")
    assert "---" not in body.splitlines()[0]


def test_parse_frontmatter_absent_is_passthrough() -> None:
    meta, body = parse_frontmatter("# Plain\n\nNo frontmatter here.")
    assert meta == {}
    assert body.startswith("# Plain")


def test_extract_wikilinks_strips_alias_and_heading() -> None:
    links = extract_wikilinks(NOTE)
    assert links == ("Revenue Model", "People/Alice", "Roadmap")  # |alias and #heading stripped


def test_extract_tags_merges_frontmatter_and_inline() -> None:
    meta, body = parse_frontmatter(NOTE)
    tags = extract_tags(meta, body)
    assert "finance" in tags and "q3" in tags and "planning" in tags


def test_parse_note_title_from_frontmatter() -> None:
    note = parse_note(NOTE, file_title="2026-q3")
    assert note.title == "Quarterly Review"  # frontmatter wins over filename
    assert "Revenue Model" in note.links
    assert "# Q3" in note.body and "title:" not in note.body


def test_parse_note_falls_back_to_filename() -> None:
    note = parse_note("# Body only\n\ntext", file_title="my-file")
    assert note.title == "my-file"


def test_context_line_embeds_tags_and_links() -> None:
    note = parse_note(NOTE, file_title="x")
    line = context_line(note)
    assert "tags:" in line and "finance" in line
    assert "links:" in line and "Revenue Model" in line
