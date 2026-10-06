"""__tmpl_title: a web-search backend in IRIS's ``research`` chain.

``setup(api)`` registers :class:`Backend` as a search provider. The built-in ``research``
tool then tries it in the chain's order (``config/search_providers.yaml``: after the
configured built-ins, before the keyless DuckDuckGo floor) and it inherits what the chain
does for every provider: a question about the owner's own finances never reaches it,
their name and email addresses are stripped from the query, results are cached, reranked
and marked untrusted and tripwire-scanned before the model reads them, and every call is
audited. There is no tool of its own to declare.

:class:`Backend` serves canned results, so the plugin works offline. Replace
:meth:`Backend.search` with your service's API; log each call with
``iris_harness.sdk.logging.log_egress(destination=HOST, kind="search", ...)`` (the host,
never the query) and return ``[]`` rather than raising when the service fails. Each hit
is a :class:`SearchHit` -- the url, title and snippet your service gave, ``published`` as
a datetime when it dates the result; the research engine does the scoring.
"""

from __future__ import annotations

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.research import Freshness, SearchHit, SearchProvider, SearchType

#: The provider's name in the chain: the research result's ``provider``, and its key in
#: ``config/search_providers.yaml``. Declared under ``search_providers:`` in the manifest.
PROVIDER = "__tmpl_tool"
HOST = "search.example.org"


class Backend(SearchProvider):
    """The search service."""

    def is_available(self) -> bool:
        """Configured and usable: check your key or URL here (never the network)."""
        return True

    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_type: SearchType = "web",
        freshness: Freshness = "any",
        safe_search: bool = True,
        language: str | None = None,
    ) -> list[SearchHit]:
        """Up to ``max_results`` hits for ``query``. Never raises: ``[]`` on failure."""
        slug = query.replace(" ", "_")
        demo = [
            SearchHit(
                title=f"About {query}",
                url=f"https://{HOST}/wiki/{slug}",
                snippet=f"An overview of {query}.",
                source=PROVIDER,
            ),
            SearchHit(
                title=f"{query}: recent news",
                url=f"https://{HOST}/news/{slug}",
                snippet=f"What changed lately around {query}.",
                source=PROVIDER,
            ),
        ]
        return demo[:max_results]


def setup(api: PluginAPI) -> None:
    api.register_search_provider(PROVIDER, Backend())
