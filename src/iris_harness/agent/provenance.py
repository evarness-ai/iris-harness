"""Retrieval-provenance ledger (Phase 5 grounding prerequisite, P1).

Accumulates the text returned by retrieval-class tools during a single request
so the ResponseCurator's grounding judge (P2) can check whether the final
response is supported by what was actually retrieved. Pure + side-effect free;
the ReAct loop records into it, the curator reads the rendered context off
``AgentResult.metadata``. See ``docs/architecture/grounding-judge.md``.
"""

from __future__ import annotations

#: Tools whose successful output IS retrieved context the response should ground
#: in (D2). Non-retrieval tools (code_exec, skill side effects, …) don't record.
RETRIEVAL_TOOLS: frozenset[str] = frozenset(
    {
        "research",
        "memory_search",
        "memory_graph",
        "wiki_search",
        "docs_search",
        "rag_search",
        "read_file",
        "file_read",
    }
)


class ProvenanceLedger:
    """Per-request accumulator of retrieved context, capped + deduped."""

    def __init__(self, *, max_entries: int = 12, max_chars: int = 4000) -> None:
        self.max_entries = max_entries
        self.max_chars = max_chars
        self._entries: list[tuple[str, str]] = []  # (tool, content)
        self._seen: set[str] = set()

    def record(self, tool: str, content: str) -> None:
        """Record a retrieval-class tool's output. No-op for non-retrieval tools,
        empty content, duplicates, or once ``max_entries`` is reached."""
        if tool not in RETRIEVAL_TOOLS:
            return
        text = (content or "").strip()
        if not text:
            return
        key = f"{tool}\x00{text}"
        if key in self._seen or len(self._entries) >= self.max_entries:
            return
        self._seen.add(key)
        self._entries.append((tool, text))

    @property
    def is_empty(self) -> bool:
        return not self._entries

    def sources(self) -> tuple[str, ...]:
        """Distinct contributing tool names, in first-seen order."""
        out: list[str] = []
        for tool, _ in self._entries:
            if tool not in out:
                out.append(tool)
        return tuple(out)

    def render(self) -> str:
        """Capped concatenation of recorded context, each block tool-labeled."""
        parts: list[str] = []
        total = 0
        for tool, text in self._entries:
            block = f"[{tool}]\n{text}"
            if total + len(block) > self.max_chars:
                remaining = self.max_chars - total
                if remaining > 0:
                    parts.append(block[:remaining] + "\n...[truncated]")
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)
