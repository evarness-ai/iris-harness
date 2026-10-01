"""Grounded, cited Q&A over the user's documents (RAG R3) — NotebookLM-style.

Retrieval (deterministic, R0/R1) selects the evidence; an LLM then *only*
summarises that evidence into an answer that cites its sources by number. The
facts come from the retrieved chunks, never the model's parametric memory —
the same deterministic-first contract as Finance/Calendar/Planner. With no
LLM (or on error) it degrades to presenting the cited passages directly, so
an answer is always grounded and attributable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.models import Citation, RetrievedChunk
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore

logger = logging.getLogger(__name__)

_ANSWER_PROMPT = (
    "Answer the question using ONLY the numbered sources below. Cite the "
    "sources you use inline as [1], [2], etc. If the sources do not contain "
    "the answer, say so plainly — do not use outside knowledge.\n\n"
    "Question: {question}\n\nSources:\n{sources}\n\nAnswer:"
)


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: tuple[Citation, ...]
    chunks: tuple[RetrievedChunk, ...]
    grounded: bool  # True = LLM-synthesised; False = deterministic passage list

    def render(self) -> str:
        """Answer followed by a numbered Sources list."""
        if not self.citations:
            return self.answer
        lines = [self.answer, "", "Sources:"]
        lines += [f"[{i}] {c.label()}" for i, c in enumerate(self.citations, start=1)]
        return "\n".join(lines)


def _sources_block(hits: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.citation.label()})\n{h.text}" for i, h in enumerate(hits, start=1)
    )


def _passage_answer(hits: list[RetrievedChunk]) -> str:
    """Deterministic fallback: the most relevant passages, lightly framed."""
    head = "Based on your documents (no synthesis model available):"
    bodies = [f"[{i}] {h.text.strip()}" for i, h in enumerate(hits, start=1)]
    return head + "\n\n" + "\n\n".join(bodies)


def answer_question(
    query: str,
    *,
    store: DocumentStore,
    index: DocumentIndex | None = None,
    llm_call: Callable[[str], str] | None = None,
    limit: int = 5,
    tags: tuple[str, ...] = (),
) -> GroundedAnswer:
    """Retrieve evidence for ``query`` and (if an LLM is given) synthesise a
    cited answer grounded strictly in that evidence."""
    hits = search_documents(query, store=store, index=index, limit=limit, tags=tags)
    if not hits:
        return GroundedAnswer("No relevant documents found.", (), (), grounded=False)

    citations = tuple(h.citation for h in hits)
    if llm_call is None:
        return GroundedAnswer(_passage_answer(hits), citations, tuple(hits), grounded=False)

    prompt = _ANSWER_PROMPT.format(question=query, sources=_sources_block(hits))
    try:
        answer = llm_call(prompt).strip()
    except Exception:  # synthesis is best-effort over real evidence
        logger.exception("rag: answer synthesis failed; returning passages")
        return GroundedAnswer(_passage_answer(hits), citations, tuple(hits), grounded=False)
    if not answer:
        return GroundedAnswer(_passage_answer(hits), citations, tuple(hits), grounded=False)
    return GroundedAnswer(answer, citations, tuple(hits), grounded=True)


def default_llm_call() -> Callable[[str], str] | None:
    """Best-effort local Tier-1 LLM call for the CLI; None if unavailable.

    Mirrors the standalone client pattern used by finance extraction — no full
    runtime needed. Synthesis stays local (the documents never leave the box).
    """
    try:
        from iris_harness.llm.client import CodingLLMClient
        from iris_harness.llm.tier_router import TierRouter

        cfg = TierRouter().get_llm_config("general")
        client = CodingLLMClient(cfg)  # type: ignore[arg-type]

        def _call(prompt: str) -> str:
            return str(client.invoke(system_prompt="", user_prompt=prompt))

        return _call
    except Exception:  # no local model configured → deterministic CLI
        logger.debug("rag: no local LLM available for synthesis", exc_info=True)
        return None


__all__ = ["GroundedAnswer", "answer_question", "default_llm_call"]
