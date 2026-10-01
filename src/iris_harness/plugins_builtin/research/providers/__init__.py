"""Search-provider package: the built-in backends, and the chain they join.

The five built-ins are registered by the plugin's ``setup`` through
``PluginAPI.register_search_provider`` -- the seam any plugin's provider uses, so there is
no second path. Their order is ``config/search_providers.yaml`` (SearXNG, Tavily, Exa,
Brave, then DuckDuckGo last as the keyless floor); ``select_providers()`` is the chain a
research call tries now.
"""

from __future__ import annotations

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.research import ChainLink, search_provider_chain

from .base import SearchProvider
from .brave import BraveProvider
from .duckduckgo import DuckDuckGoProvider
from .exa import ExaProvider
from .searxng import SearxngProvider
from .tavily import TavilyProvider

#: The built-in providers, in registration order (the config file decides the chain's).
BUILTIN_PROVIDERS: tuple[type[SearchProvider], ...] = (
    SearxngProvider,
    TavilyProvider,
    ExaProvider,
    BraveProvider,
    DuckDuckGoProvider,
)


def register_builtin_providers(api: PluginAPI) -> None:
    """Register the built-ins on the chain, each under its ``name``."""
    for provider_class in BUILTIN_PROVIDERS:
        api.register_search_provider(provider_class.name, provider_class())


def select_providers() -> list[ChainLink]:
    """The providers a research call tries now, in chain order (live, on, available).

    With the built-ins registered and no key or URL set this is ``[ddg]``: DuckDuckGo is
    always available and the config puts it last."""
    return search_provider_chain()


__all__ = [
    "BUILTIN_PROVIDERS",
    "BraveProvider",
    "DuckDuckGoProvider",
    "ExaProvider",
    "SearchProvider",
    "SearxngProvider",
    "TavilyProvider",
    "register_builtin_providers",
    "select_providers",
]
