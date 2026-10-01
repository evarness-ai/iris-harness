"""Markdown-aware deterministic chunker (RAG R0).

Splits a document into retrievable chunks on paragraph boundaries, packing
blocks up to a target size, and tags each chunk with the nearest markdown
heading as its title. Deterministic (no model) so ingestion is reproducible.
Plain text falls through the same path (no headings → file title).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_MAX_CHARS = 1000  # target chunk size; a block that alone exceeds it stands alone

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$")


@dataclass(frozen=True)
class Chunk:
    title: str
    text: str
    index: int


def _blocks(text: str) -> list[str]:
    # Split on blank lines into paragraph/code-fence-ish blocks; keep order.
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def chunk_markdown(
    text: str, *, file_title: str, max_chars: int = DEFAULT_MAX_CHARS
) -> list[Chunk]:
    """Chunk ``text`` into <= ``max_chars`` windows, titled by nearest heading."""
    chunks: list[Chunk] = []
    heading = file_title
    buf: list[str] = []
    buf_len = 0
    buf_heading = file_title

    def _flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append(Chunk(title=buf_heading, text="\n\n".join(buf), index=len(chunks)))
            buf = []
            buf_len = 0

    for block in _blocks(text):
        m = _HEADING_RE.match(block.splitlines()[0]) if block else None
        if m:
            # A heading starts a new section → flush the current buffer first.
            _flush()
            heading = m.group(1)
            buf_heading = heading
            # Headings themselves aren't emitted as standalone chunks; they title
            # the following content. Carry the heading text into the next chunk.
            buf.append(block)
            buf_len = len(block)
            continue
        if buf and buf_len + len(block) + 2 > max_chars:
            _flush()
            buf_heading = heading
        if not buf:
            buf_heading = heading
        buf.append(block)
        buf_len += len(block) + 2
    _flush()
    return chunks


__all__ = ["Chunk", "chunk_markdown", "DEFAULT_MAX_CHARS"]
