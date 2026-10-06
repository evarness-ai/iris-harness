"""Search IRIS's own shipped documentation: deterministic, local, bounded.

The core could read a fixed handful of its docs (``iris_doc``) but not find which one
answers a question. This package searches the in-repo documentation -- the architecture,
concepts, guides, reference and usage-guides trees named in ``config/docs_search.yaml`` --
by keyword at section level. No vector store, no model, no network.

It is an INTERNAL tool and the corpus is an allow-list: the roots in the YAML are the only
places it reads, each path is resolved and contained, and a document that classifies
``secret`` is dropped. It cannot reach the identity files, the vault, ``.env`` files, the
owner's data directories or any store. See :mod:`.corpus` for the containment rules.
"""

from __future__ import annotations

from iris_harness.services.docs_search.search import search_core_docs

__all__ = ["search_core_docs"]
