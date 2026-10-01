"""The search-provider chain the ``research`` tool tries, as one registry.

A search provider turns a query into raw :class:`SearchHit` hits. It knows nothing
about caching, extraction, ranking or the owner's identifiers -- the research engine owns
those, and applies them to whichever provider answered. Every provider, the built-in five
included, joins the chain through :func:`register_search_provider` (a plugin calls
``PluginAPI.register_search_provider``, which binds the owner and the fault boundary);
there is no second path, so a plugin's provider is guarded, cached, reranked and logged
exactly as DuckDuckGo is.

**Order is config.** ``config/search_providers.yaml`` gives each provider a priority
(lower is tried first) and may turn one off; a provider the file does not name runs at
the priority it registered with, else the file's ``default_priority``. Ties keep
registration order. :func:`search_provider_chain` reads the file on each call (it is
small, and a research call goes to the network anyway), so an edit applies to the next
query.

**A provider leaves with its plugin.** Each entry carries a ``live`` check -- for a
plugin, "is my plugin still mounted" -- and the chain skips an entry that is not. The
registry is process-wide state (put back when a harness run ends), keyed by name: a
second registration of a name replaces the first only from the same owner.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from iris_harness.foundation.paths import config_path, default_config_dir
from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

CONFIG_FILE = "search_providers.yaml"

# The search "lens". Maps to provider-specific query shaping (site filters, ranking),
# never to a hardcoded source list -- the provider decides how to honor it.
SearchType = Literal["web", "news", "github", "reddit", "docs"]

# Freshness window hint, passed to providers that support time filtering.
Freshness = Literal["day", "week", "month", "year", "any"]

_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True, kw_only=True, slots=True)
class SearchHit:
    """One raw hit, as a search provider returns it: what the service said, nothing more.

    ``url`` and ``title`` are the result; ``snippet`` its summary text; ``published`` when
    the service dates it (parse the service's string yourself -- a provider knows its
    format, the engine does not); ``source`` the provider's own name (the chain fills in
    the registered name when it is empty); ``extra`` short provider-specific labels
    (SearXNG's upstream engine), passed to the model beside the result.

    Frozen and keyword-only: a hit is the provider's word, and the research engine copies
    it into its own result before it scores, reranks and extracts. Relevance, trust and
    page content are the engine's, so a provider has nowhere to put them -- the engine
    scores every hit the same way, whichever provider answered.
    """

    url: str
    title: str
    snippet: str = ""
    published: datetime | None = None
    source: str = ""
    extra: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        # A read-only copy: the caller's dict cannot change a hit after the fact.
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))


def check_hits(provider: str, found: object) -> list[SearchHit]:
    """``found`` as the chain's hits, else ``TypeError`` naming ``provider`` and what it sent.

    The one check of a provider's return value, made where it crosses into the chain: a
    list of :class:`SearchHit`. Anything else -- the engine's own result type,
    which a provider written before 0.1.0 returned, a dict, a generator -- is a contract
    break, refused whole rather than adapted, so its author sees it on the first call.
    """
    if not isinstance(found, list):
        raise TypeError(
            f"search provider {provider!r} returned {type(found).__name__}, not a list of "
            "SearchHit (iris_harness.sdk.research.SearchHit)"
        )
    for hit in found:
        if not isinstance(hit, SearchHit):
            raise TypeError(
                f"search provider {provider!r} returned a {type(hit).__name__} hit, not a "
                "SearchHit (iris_harness.sdk.research.SearchHit)"
            )
    return list(found)


@runtime_checkable
class SearchProvider(Protocol):
    """One search backend in the research chain."""

    def is_available(self) -> bool:
        """True when the provider is configured and usable (its key or URL is set).

        Cheap: configuration checks only, never a network call. Read the configuration
        here, not in ``__init__``: a provider is built once, and a key set later must
        still count."""
        ...

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
        """Up to ``max_results`` raw hits for ``query``. Return ``[]`` on any failure, so
        the chain falls through to the next provider; log the call with
        ``iris_harness.sdk.logging.log_egress`` (the host, never the query).

        ``language`` is a two-letter ISO 639-1 code the results should be in, or None: a
        hint to pass on where the service has one (the engine also filters by script).
        The query the provider receives has already been through the research tool's
        guards: a question about the owner's own finances never arrives, and the owner's
        name and email addresses are stripped from it."""
        ...


@dataclass(frozen=True)
class SearchProviderEntry:
    """One registered provider: who registered it, and its place in the chain."""

    name: str
    provider: SearchProvider
    owner: str
    priority: int | None
    seq: int
    live: Callable[[], bool]


@dataclass(frozen=True)
class ChainLink:
    """A provider as the chain hands it to the engine: its registered name, and the calls."""

    name: str
    provider: SearchProvider

    def is_available(self) -> bool:
        return self.provider.is_available()

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
        """The provider's hits, held to the contract (:func:`check_hits`)."""
        if language:
            found = self.provider.search(
                query,
                max_results=max_results,
                search_type=search_type,
                freshness=freshness,
                safe_search=safe_search,
                language=language,
            )
        else:
            # ``language`` only when set, so a provider written without it still works.
            found = self.provider.search(
                query,
                max_results=max_results,
                search_type=search_type,
                freshness=freshness,
                safe_search=safe_search,
            )
        return check_hits(self.name, found)


class _ProviderSetting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priority: int | None = None
    enabled: bool = True


class ChainConfig(BaseModel):
    """``config/search_providers.yaml``."""

    model_config = ConfigDict(extra="forbid")

    default_priority: int = 500
    providers: dict[str, _ProviderSetting] = Field(default_factory=dict)

    def priority_of(self, entry: SearchProviderEntry) -> int:
        setting = self.providers.get(entry.name)
        if setting is not None and setting.priority is not None:
            return setting.priority
        return entry.priority if entry.priority is not None else self.default_priority

    def enabled(self, name: str) -> bool:
        setting = self.providers.get(name)
        return setting is None or setting.enabled


def _config_file() -> Path | None:
    """The owner's file, else the shipped one (an override directory may hold only the
    files it changes), else none."""
    for path in (config_path(CONFIG_FILE), default_config_dir() / CONFIG_FILE):
        if path.is_file():
            return path
    return None


def load_chain_config() -> ChainConfig:
    """The chain's order and switches; defaults when no file exists.

    A file that is there but malformed raises ``ValueError`` naming it: the order decides
    where the owner's queries go, so a typo is never silently the default order."""
    path = _config_file()
    if path is None:
        return ChainConfig()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return ChainConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ValueError(f"search provider config invalid: {path}: {exc}") from exc


_lock = Lock()
_entries: dict[str, SearchProviderEntry] = {}
_seq = itertools.count()


def _always() -> bool:
    return True


def register_search_provider(
    name: str,
    provider: SearchProvider,
    *,
    owner: str,
    priority: int | None = None,
    live: Callable[[], bool] | None = None,
) -> None:
    """Add ``provider`` to the chain as ``name``.

    ``owner`` names who registered it (``plugin:<name>``); ``live`` says whether that owner
    is still there (the chain skips the entry when it returns False). ``priority`` is the
    provider's place when ``config/search_providers.yaml`` does not give one. Refused
    (``ValueError``) for a bad name, or a name a different live owner holds; ``TypeError``
    when ``provider`` is not a :class:`SearchProvider`.
    """
    if not _NAME.match(name):
        raise ValueError(f"search provider name {name!r} is not valid ([a-z][a-z0-9_-]*)")
    if not isinstance(provider, SearchProvider):
        raise TypeError(
            f"search provider {name!r} is not a SearchProvider "
            "(it needs is_available() and search(query, *, max_results, ...))"
        )
    with _lock:
        held = _entries.get(name)
        if held is not None and held.owner != owner and held.live():
            raise ValueError(f"search provider {name!r} is already registered by {held.owner}")
        _entries[name] = SearchProviderEntry(
            name=name,
            provider=provider,
            owner=owner,
            priority=priority,
            seq=next(_seq),
            live=live or _always,
        )


def unregister_search_provider(name: str, *, owner: str) -> bool:
    """Remove ``name`` when ``owner`` holds it; True when something was removed."""
    with _lock:
        held = _entries.get(name)
        if held is None or held.owner != owner:
            return False
        del _entries[name]
        return True


def registered_search_providers() -> list[SearchProviderEntry]:
    """Every live entry the config leaves on, in chain order (available or not)."""
    config = load_chain_config()
    with _lock:
        entries = list(_entries.values())
    kept = [e for e in entries if e.live() and config.enabled(e.name)]
    return sorted(kept, key=lambda e: (config.priority_of(e), e.seq))


def search_provider_chain() -> list[ChainLink]:
    """The providers a research call tries now, in order: live, on, and available.

    A provider whose ``is_available`` raises is left out (and logged): one broken
    provider never takes the chain down."""
    chain: list[ChainLink] = []
    for entry in registered_search_providers():
        try:
            available = entry.provider.is_available()
        except Exception:  # noqa: BLE001 - a plugin's check; its host recorded the failure
            logger.warning("search provider %s: is_available() raised; skipped", entry.name)
            continue
        if available:
            chain.append(ChainLink(entry.name, entry.provider))
    return chain


__all__ = [
    "CONFIG_FILE",
    "ChainConfig",
    "ChainLink",
    "Freshness",
    "SearchProvider",
    "SearchProviderEntry",
    "check_hits",
    "SearchHit",
    "SearchType",
    "load_chain_config",
    "register_search_provider",
    "registered_search_providers",
    "search_provider_chain",
    "unregister_search_provider",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_entries")
