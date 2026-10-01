"""LangChain tool for the docs-search skill (RAG R0).

Thin wrapper over ``iris_harness.services.rag.search_documents`` — returns the top matching
chunks from the user's indexed documents with citations, so the agent can
ground and attribute answers. Domain logic + tests live in src/iris/rag.
"""

from __future__ import annotations

from iris_harness.sdk.rag import DocumentIndex, DocumentStore, search_documents
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SearchDocsInput(BaseModel):
    query: str = Field(..., description="What to look for in the user's documents.")
    limit: int = Field(default=5, description="Max chunks to return.")


class SearchDocsTool(BaseTool):
    name: str = "search_documents"
    description: str = (
        "Search the user's indexed documents (notes, Obsidian vault, etc.) and "
        "return matching passages WITH citations (source file + section). Use this "
        "to ground answers in the user's own knowledge base."
    )
    args_schema: type[BaseModel] = SearchDocsInput

    def _run(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        store = DocumentStore()
        store.ensure_schema()
        hits = search_documents(query, store=store, index=DocumentIndex(), limit=limit)
        return [
            {
                "citation": h.citation.label(),
                "source": h.citation.source_path,
                "snippet": h.text.replace("\n", " ")[:400],
                "score": f"{h.score:.2f}",
            }
            for h in hits
        ]


SKILL_TOOLS = [SearchDocsTool]
