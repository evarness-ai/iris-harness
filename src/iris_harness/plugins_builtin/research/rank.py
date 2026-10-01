"""Ranking for the research engine.

Two responsibilities:

1. **Domain trust** — map a result's host to a 0..10 authority score
   (:func:`trust_for`), so canonical docs outrank SEO aggregators.
2. **Final blend** — combine semantic similarity, trust, and freshness into a
   single ``score`` and sort in place (:func:`score_results`).

The semantic component prefers a caller-supplied embedder; when none is given (or
it fails) it falls back to a cheap lexical Jaccard overlap. Nothing here ever
raises — a degraded signal beats an aborted research turn.
"""

from __future__ import annotations

import logging
import math
import re
import urllib.parse
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from iris_harness.plugins_builtin.research.models import SearchResult
from iris_harness.plugins_builtin.research.news_policy import news_trust_scores

logger = logging.getLogger(__name__)

# Score knocked off a source the user has repeatedly marked "not useful" (issue
# 0028). Blend scores are ~0..1, so a penalty this large reliably sinks a
# suppressed source below every un-suppressed one without dropping it outright —
# research is pull, never proactive, so the user still gets it if nothing better.
_SUPPRESS_PENALTY = 10.0

# Domain -> 0..10 authority score. Two match modes (see ``trust_for``): exact host,
# then suffix (``endswith``) for wildcard families like ``*.readthedocs.io``.
TRUST_SCORES: dict[str, float] = {
    # Canonical docs / first-party references.
    "docs.python.org": 10.0,
    "readthedocs.io": 10.0,
    "developer.mozilla.org": 10.0,
    "developer.apple.com": 9.0,
    "learn.microsoft.com": 9.0,
    "microsoft.com": 9.0,
    "kubernetes.io": 9.0,
    "golang.org": 9.0,
    "go.dev": 9.0,
    "rust-lang.org": 9.0,
    # Code + research.
    "github.com": 9.0,
    "gitlab.com": 8.0,
    "arxiv.org": 9.0,
    # Reference / Q&A.
    "stackoverflow.com": 8.0,
    "stackexchange.com": 8.0,
    "wikipedia.org": 8.0,
    # Generic non-profit / standards bodies (suffix match).
    ".org": 9.0,
    # Community / opinion — useful but lower authority.
    "reddit.com": 6.0,
    "news.ycombinator.com": 6.0,
    "medium.com": 5.0,
    "substack.com": 5.0,
    "dev.to": 5.0,
}

# The ``news`` lens reads its own source policy from ``news_sources.yaml`` (see
# news_policy.py). A SEPARATE table, not an overlay on TRUST_SCORES, because the web
# table is actively wrong for news: it carries no newsroom at all, so reuters.com and
# apnews.com fall to DEFAULT_TRUST (4.0) while its ``.org`` suffix rule hands any .org
# 9.0. Trust is 0.40 of the blend, so on a news query the web table ranks a random .org
# ABOVE the wire service that .org is quoting.
#
# In YAML rather than here so adding a source or re-scoring a tier is a config edit.

# Unknown but plausible domain.
DEFAULT_TRUST: float = 4.0

# Obvious SEO / AI-content-farm aggregators.
SPAM_TRUST: float = 2.0
SPAM_MARKERS: tuple[str, ...] = ("aiflashreport", "pricepertoken")

# Blend weights. ``sim`` (semantic/lexical relevance) dominates; trust contributes
# twice — once directly (0.30) and once as "domain authority" (0.10) which reuses
# the same normalized trust signal, for an effective 0.40 trust weight. Freshness
# is a light tie-breaker. Weights sum to 1.0.
_W_SIM = 0.45
_W_TRUST = 0.30
_W_FRESH = 0.15
_W_AUTHORITY = 0.10

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _host(url: str) -> str:
    """Lowercase host with a leading ``www.`` stripped. Empty string if unparseable."""
    try:
        netloc = urllib.parse.urlsplit(url).netloc.lower()
    except Exception:  # never raise on a malformed URL
        logger.debug("trust_for: failed to parse url %r", url, exc_info=True)
        return ""
    # Drop any userinfo/port.
    netloc = netloc.split("@")[-1].split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def research_source_dims(url: str, search_type: str = "web") -> dict[str, str]:
    """Surface-feedback suppression key for a research source (issue 0028).

    Keyed on host + search lens, so "this source isn't useful for this kind of
    query" downranks ``(host, search_type)`` pairs. Wired: :func:`score_results`
    consults it and applies ``_SUPPRESS_PENALTY``. See
    docs/architecture/surface-feedback-suppression.md.
    """
    return {"host": _host(url), "search_type": search_type}


def _lookup_trust(host: str, table: dict[str, float]) -> float | None:
    """Exact host first, then suffix. None when ``table`` knows nothing about ``host``."""
    if host in table:
        return table[host]
    for suffix, score in table.items():
        if host == suffix or host.endswith(suffix if suffix.startswith(".") else "." + suffix):
            return score
    return None


def trust_for(url: str, *, search_type: str = "web") -> float:
    """Return a 0..10 domain-authority score for ``url``. Never raises.

    The ``news`` lens reads the YAML source policy and does NOT fall through to the web
    table. Falling through would re-admit the very rule that inverts a news ranking
    (``".org": 9.0``), and the web table's high scorers — canonical docs, arxiv, GitHub —
    are not news sources, so an unknown-domain score is the honest answer for them here.
    """
    host = _host(url)
    if not host:
        return DEFAULT_TRUST
    if any(marker in host for marker in SPAM_MARKERS):
        return SPAM_TRUST
    table = news_trust_scores() if search_type == "news" else TRUST_SCORES
    matched = _lookup_trust(host, table)
    return DEFAULT_TRUST if matched is None else matched


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _lexical_overlap(query: str, text: str) -> float:
    """Jaccard overlap of query tokens vs ``text`` tokens, in [0, 1]."""
    q = _tokens(query)
    t = _tokens(text)
    if not q or not t:
        return 0.0
    inter = len(q & t)
    union = len(q | t)
    return inter / union if union else 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, in [-1, 1]. 0.0 if degenerate."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _result_text(r: SearchResult) -> str:
    return f"{r.title} {r.snippet}".strip()


def _semantic_scores(
    query: str,
    results: list[SearchResult],
    embed: Callable[[list[str]], list[list[float]]] | None,
) -> list[float]:
    """Per-result relevance in [0, 1]. Embedder cosine when available, else lexical.

    Falls back to lexical on any embedder failure or shape mismatch. Never raises.
    """
    texts = [_result_text(r) for r in results]
    if embed is not None:
        try:
            vectors = embed([query, *texts])
            if len(vectors) == len(results) + 1:
                qv = vectors[0]
                sims = [_cosine(qv, vectors[i + 1]) for i in range(len(results))]
                # Map cosine [-1, 1] -> [0, 1] so it blends with the other components.
                return [max(0.0, min(1.0, (s + 1.0) / 2.0)) for s in sims]
            logger.debug("rerank: embed returned %d vectors for %d texts", len(vectors), len(texts))
        except Exception:  # degrade to lexical, never abort
            logger.debug("rerank: embed failed, falling back to lexical", exc_info=True)
    return [_lexical_overlap(query, t) for t in texts]


def rerank(
    query: str,
    results: list[SearchResult],
    *,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> None:
    """Semantic rerank in place: store the relevance signal in ``metadata['sim']``.

    Uses ``embed`` (cosine) when supplied, else a lexical Jaccard overlap. Never raises.
    """
    sims = _semantic_scores(query, results, embed)
    for r, sim in zip(results, sims, strict=False):
        r.metadata["sim"] = sim


def _freshness_score(published: datetime | None, *, now: datetime) -> float:
    """Recency component in [~0.2, 1.0]. ``None`` -> 0.5 (unknown, neutral).

    1.0 within ~7 days, linearly decaying to ~0.2 at >1 year old.
    """
    if published is None:
        return 0.5
    # Treat naive timestamps as UTC so the subtraction never raises.
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    age_days = (now - published).total_seconds() / 86400.0
    if age_days <= 7:
        return 1.0
    if age_days >= 365:
        return 0.2
    # Linear decay from 1.0 (day 7) to 0.2 (day 365).
    span = 365 - 7
    return 1.0 - 0.8 * ((age_days - 7) / span)


def score_results(
    query: str,
    results: list[SearchResult],
    *,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
    now: datetime | None = None,
    search_type: str = "web",
    feedback_store: Any = None,
) -> None:
    """Blend trust + semantic relevance + freshness into ``score``; sort in place.

    Sets ``r.trust_score`` and ``r.score`` on every result, then sorts descending by
    ``score``. Deterministic and never raises.

    ``feedback_store`` (the surface-feedback spine) downranks sources the user has
    marked "not useful" for this ``search_type`` (issue 0028) — keyed on ``(host,
    search_type)`` so the signal is lens-specific.
    """
    if now is None:
        now = datetime.now(UTC)

    rerank(query, results, embed=embed)

    for r in results:
        r.trust_score = trust_for(r.url, search_type=search_type)
        sim = float(r.metadata.get("sim", 0.0))  # type: ignore[arg-type]
        trust_norm = r.trust_score / 10.0
        freshness = _freshness_score(r.published, now=now)
        # domain_authority reuses the normalized trust signal (see weight comment).
        r.score = (
            _W_SIM * sim + _W_TRUST * trust_norm + _W_FRESH * freshness + _W_AUTHORITY * trust_norm
        )
        if feedback_store is not None and _is_suppressed_source(feedback_store, r.url, search_type):
            r.score -= _SUPPRESS_PENALTY

    results.sort(key=lambda r: r.score, reverse=True)


def _is_suppressed_source(feedback_store: Any, url: str, search_type: str) -> bool:
    """Best-effort suppression check; never raises into ranking."""
    try:
        return bool(
            feedback_store.should_suppress(
                "research", "source", research_source_dims(url, search_type)
            )
        )
    except Exception:  # ranking must stay deterministic + non-raising
        logger.debug("research suppression check failed for %s", url, exc_info=True)
        return False


__all__ = [
    "DEFAULT_TRUST",
    "SPAM_TRUST",
    "TRUST_SCORES",
    "rerank",
    "score_results",
    "trust_for",
]
