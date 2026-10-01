"""Source policy for the ``news`` lens, read from ``news_sources.yaml``.

One job: turn the plugin's tier/source YAML into the flat ``domain -> 0..10`` map the
ranker already knows how to use. Adding a source or re-scoring a tier is a YAML edit —
no code change, which is the whole point of the file existing.

Read once per process and cached. Never raises: a missing or malformed file logs and
yields an empty map, which scores every news domain at ``DEFAULT_TRUST``. That is flat
rather than inverted, so a broken file degrades the ranking instead of breaking the turn
— research is the tool the loop reaches for constantly, and a config typo must not take
web access down with it.

The file ships inside the plugin (``pyproject`` packages ``plugins_builtin/**/*.yaml``),
so a core-only install carries it. ``IRIS_RESEARCH_NEWS_SOURCES`` overrides the path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from iris_harness.sdk.process_state import track_globals

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_NEWS_SOURCES_PATH", "load_news_trust", "news_trust_scores"]

DEFAULT_NEWS_SOURCES_PATH = Path(__file__).parent / "news_sources.yaml"
_PATH_ENV = "IRIS_RESEARCH_NEWS_SOURCES"

_CACHE: dict[str, float] | None = None
_CACHE_KEY: str | None = None


def _resolved_path() -> Path:
    override = os.environ.get(_PATH_ENV, "").strip()
    return Path(override) if override else DEFAULT_NEWS_SOURCES_PATH


def load_news_trust(path: Path) -> dict[str, float]:
    """Flatten ``path``'s tiers + sources into ``domain -> score``. Never raises.

    A domain in a tier the ``tiers`` block does not name is skipped with a warning
    rather than guessed at — a silent default would be a scoring decision this module
    has no business making.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("news source policy not found at %s; news trust is flat", path)
        return {}
    except Exception:  # a config typo must not break web access
        logger.warning("news source policy at %s could not be parsed", path, exc_info=True)
        return {}
    if not isinstance(raw, dict):
        logger.warning("news source policy at %s is not a mapping; news trust is flat", path)
        return {}

    tiers_raw: Any = raw.get("tiers") or {}
    sources_raw: Any = raw.get("sources") or {}
    if not isinstance(tiers_raw, dict) or not isinstance(sources_raw, dict):
        logger.warning("news source policy at %s: 'tiers'/'sources' must be mappings", path)
        return {}

    tiers: dict[str, float] = {}
    for name, score in tiers_raw.items():
        try:
            tiers[str(name)] = float(score)
        except (TypeError, ValueError):
            logger.warning("news source policy: tier %r has a non-numeric score %r", name, score)

    scores: dict[str, float] = {}
    for tier, domains in sources_raw.items():
        score = tiers.get(str(tier))
        if score is None:
            logger.warning(
                "news source policy: tier %r has no score; its domains are skipped", tier
            )
            continue
        if not isinstance(domains, list):
            logger.warning("news source policy: tier %r does not list domains", tier)
            continue
        for domain in domains:
            host = str(domain).strip().lower()
            if host:
                scores[host] = score
    return scores


def news_trust_scores() -> dict[str, float]:
    """The cached ``domain -> 0..10`` map for the news lens.

    Cached against the resolved path, so a test pointing ``IRIS_RESEARCH_NEWS_SOURCES``
    at a fixture gets that fixture rather than a neighbour's leftovers.
    """
    global _CACHE, _CACHE_KEY
    path = _resolved_path()
    key = str(path)
    if _CACHE is None or _CACHE_KEY != key:
        _CACHE = load_news_trust(path)
        _CACHE_KEY = key
    return _CACHE


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_CACHE", "_CACHE_KEY")
