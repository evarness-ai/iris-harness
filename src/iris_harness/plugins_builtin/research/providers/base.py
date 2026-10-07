"""Search-provider base for the built-in providers.

The contract is the SDK's: ``iris_harness.sdk.research.SearchProvider``. The built-ins
subclass it explicitly -- as a third party's provider may -- so a type checker holds them
to the signature the chain calls. This base adds the short ``name`` each is registered
under (used in ``config/search_providers.yaml``, the logs and
``ResearchResult.provider``). Caching, extraction and ranking are the engine's; a
provider only turns a query into raw hits.

The keyed providers reach their service through the governed HTTP client
(``iris_harness.sdk.http``, issue #172): the request is checked against the plugin's
``egress`` declaration and recorded in the ledger. :meth:`SearchProvider.client` is that
client for the tool call that is running; a call outside one is refused and the provider
returns no hits, so nothing goes out ungoverned.
"""

from __future__ import annotations

import json
from typing import Any

from iris_harness.sdk.http import GovernedHttp, current_http
from iris_harness.sdk.research import SearchProvider as _SearchProviderProtocol


class SearchProvider(_SearchProviderProtocol):
    """A built-in provider: the SDK protocol plus the name it is registered under."""

    #: Stable short id used in the chain config, logs, and ResearchResult.provider.
    name: str = "base"

    #: A test installs a stand-in client here; production leaves it ``None``.
    _http: GovernedHttp | None = None

    def client(self) -> GovernedHttp:
        """The governed client for the plugin whose tool is running (raises ``EgressDenied``
        outside a governed tool call)."""
        return self._http if self._http is not None else current_http()


class ProviderHTTPError(RuntimeError):
    """The service answered with an error status (what ``urllib`` raised as ``HTTPError``)."""


def json_body(response: Any) -> Any:
    """The JSON body of a governed response; an error status raises :class:`ProviderHTTPError`."""
    if response.status_code >= 400:
        raise ProviderHTTPError(f"HTTP {response.status_code}")
    return json.loads(response.content)


__all__ = ["ProviderHTTPError", "SearchProvider", "json_body"]
