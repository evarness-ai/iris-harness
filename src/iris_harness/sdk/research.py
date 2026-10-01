"""Search providers: a web-search backend joins the ``research`` tool's chain.

A plugin implements :class:`SearchProvider` (``is_available()`` and ``search(query, *,
max_results, ...) -> list[SearchHit]``) and registers it in ``setup``::

    api.register_search_provider("my_search", MySearch(), priority=450)

and declares the name in its manifest (``search_providers: [my_search]``) -- an
undeclared name is refused and charged to the plugin, as an undeclared tool is.
:class:`SearchHit` is all a provider returns: url, title, snippet, an optional
``published`` datetime, its ``source`` name and short ``extra`` labels. Scoring, trust
and page content are the engine's; a list of anything else is refused on the call.

The provider then serves the built-in ``research`` tool, after the providers ahead of it,
and inherits everything the chain does: the egress guards (a question about the owner's
own finances never reaches it, their name and email addresses are stripped from the
query), the result cache, the rerank, the injected-instruction scan of every result and
the audit of each call. No tool of its own is needed. ``config/search_providers.yaml``
orders the chain (lower priority first) and can turn a provider off; ``priority`` is the
place a provider takes when the file does not name it.

:func:`search_provider_chain` is the chain as the research engine reads it: live (the
registering plugin is still mounted), on, and available, in order.
"""

from __future__ import annotations

from iris_harness.services.research.providers import (
    ChainLink,
    Freshness,
    SearchHit,
    SearchProvider,
    SearchType,
    search_provider_chain,
)

__all__ = [
    "ChainLink",
    "Freshness",
    "SearchHit",
    "SearchProvider",
    "SearchType",
    "search_provider_chain",
]
