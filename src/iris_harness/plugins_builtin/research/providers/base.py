"""Search-provider base for the built-in providers.

The contract is the SDK's: ``iris_harness.sdk.research.SearchProvider``. The built-ins
subclass it explicitly -- as a third party's provider may -- so a type checker holds them
to the signature the chain calls. This base adds the short ``name`` each is registered
under (used in ``config/search_providers.yaml``, the logs and
``ResearchResult.provider``). Caching, extraction and ranking are the engine's; a
provider only turns a query into raw hits.
"""

from __future__ import annotations

from iris_harness.sdk.research import SearchProvider as _SearchProviderProtocol


class SearchProvider(_SearchProviderProtocol):
    """A built-in provider: the SDK protocol plus the name it is registered under."""

    #: Stable short id used in the chain config, logs, and ResearchResult.provider.
    name: str = "base"


__all__ = ["SearchProvider"]
