"""Searching IRIS's own shipped documentation (the ``search_docs`` tool).

`search_core_docs(args)` is the tool body: given ``{"query": ..., "section": ..., "limit":
...}`` it returns the best-matching sections of the in-repo docs (document, heading, line,
a short snippet), by deterministic keyword match -- no model, no network, no vector store.

The corpus is an allow-list in ``config/docs_search.yaml`` (architecture, concepts,
guides, reference and usage-guides under the source checkout). It cannot reach the
identity files, the vault, ``.env`` files, the owner's data or any store, and a document
that classifies ``secret`` is dropped. On an install without the docs (a wheel) it says so.
It is an INTERNAL read tool: register it with the manifest's default ``content: internal``.
"""

from __future__ import annotations

from iris_harness.services.docs_search import search_core_docs

__all__ = ["search_core_docs"]
