"""Keyword, section-level search over the allow-listed corpus (:mod:`.corpus`).

Deterministic: a query is tokenised, each section scored by a capped term frequency
weighted by how rare the term is across the corpus (a heading hit counts triple), and
ties break on document name then line. No regex is built from the query, so its cost is
linear in the (capped) corpus. The answer is the top few matches -- document, heading,
line, a short snippet -- never a document.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iris_harness.kernel.governance.file_scan import scan_text
from iris_harness.services.docs_search.corpus import (
    Config,
    Corpus,
    DocsSearchConfigError,
    Limits,
    Section,
    build_corpus,
    load_config,
)

_TOKEN = re.compile(r"\w+")
_HEADING_WEIGHT = 3.0

NOT_PRESENT = (
    "search_docs: the docs are not present on this install (they ship with a source "
    "checkout, not the installed package), so there is nothing to search."
)


@dataclass(frozen=True)
class _Hit:
    score: float
    doc: str
    section: Section
    line: int  # the best-matching line, 1-based


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _query_terms(query: str, limits: Limits) -> list[str]:
    seen: dict[str, None] = {}
    for tok in _tokens(query[: limits.max_query_chars]):
        if len(tok) >= 2 or tok.isdigit():
            seen.setdefault(tok)
    return list(seen)[: limits.max_query_terms]


def _score_corpus(corpus: Corpus, terms: Sequence[str], section_filter: str) -> list[_Hit]:
    needle = section_filter.lower()
    sections: list[tuple[str, Section]] = [
        (doc.name, sec)
        for doc in corpus.docs
        for sec in doc.sections
        if not needle or needle in sec.heading.lower()
    ]
    if not sections:
        return []
    counts: list[Counter[str]] = []
    heads: list[set[str]] = []
    df: Counter[str] = Counter()
    for _name, sec in sections:
        c = Counter(_tokens("\n".join(sec.lines)))
        h = set(_tokens(sec.heading))
        counts.append(c)
        heads.append(h)
        for t in terms:
            if c.get(t) or t in h:
                df[t] += 1
    n = len(sections)
    hits: list[_Hit] = []
    for (name, sec), c, h in zip(sections, counts, heads, strict=True):
        matched = [t for t in terms if c.get(t) or t in h]
        if not matched:
            continue
        score = 0.0
        for t in matched:
            idf = math.log(1.0 + n / (1.0 + df[t]))
            tf = 1.0 + math.log(c[t]) if c.get(t) else 0.0
            score += idf * (tf + (_HEADING_WEIGHT if t in h else 0.0))
        score *= (len(matched) / len(terms)) ** 2
        hits.append(_Hit(score, name, sec, _best_line(sec, matched)))
    hits.sort(key=lambda x: (-x.score, x.doc, x.section.start_line))
    return hits


def _best_line(sec: Section, matched: Sequence[str]) -> int:
    best, best_n = sec.start_line, -1
    for offset, line in enumerate(sec.lines):
        toks = set(_tokens(line))
        k = sum(1 for t in matched if t in toks)
        if k > best_n:
            best, best_n = sec.start_line + 1 + offset, k
    return best


def _snippet(sec: Section, line_no: int, limit: int) -> str:
    index = max(0, min(len(sec.lines) - 1, line_no - sec.start_line - 1))
    text = " ".join(sec.lines[index].split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def render(corpus: Corpus, query: str, section: str, limit: int | None, config: Config) -> str:
    lim = config.limits
    terms = _query_terms(query, lim)
    if not terms:
        return 'Error: search_docs requires a "query" with at least one word to look for.'
    want = lim.default_results if limit is None else max(1, min(limit, lim.max_results))
    hits = _score_corpus(corpus, terms, section.strip()[: lim.max_query_chars])
    shown = f'"{" ".join(terms)}"'
    searched = f"{len(corpus.docs)} docs searched"
    status = (
        f"{corpus.dropped} documents withheld (secret/personal/over scan budget), "
        f"{corpus.pending} not yet scanned"
    )
    if not hits:
        return f"search_docs: no section matches {shown} ({searched}).\n{status}"
    out: list[str] = []
    used = 0
    for hit in hits:
        if len(out) >= want:
            break
        snippet = _snippet(hit.section, hit.line, lim.snippet_chars)
        # The text leaving this tool is classified again; the document already was.
        if scan_text(f"{hit.section.heading}\n{snippet}").is_secret:
            continue
        entry = f"{len(out) + 1}. {hit.doc} > {hit.section.heading} (line {hit.line})\n   {snippet}"
        if used + len(entry) > lim.max_output_chars:
            break
        used += len(entry)
        out.append(entry)
    if not out:
        return f"search_docs: no section matches {shown} ({searched}).\n{status}"
    notes = f"{searched}; showing {len(out)} of {len(hits)} matching sections"
    if corpus.truncated:
        notes += "; the corpus hit a size cap"
    return f"search_docs: {shown} ({notes})\n" + "\n".join(out) + f"\n{status}"


def search_core_docs(args: dict[str, Any]) -> str:
    """The ``search_docs`` tool body. ``args``: ``query``, optional ``section``, ``limit``.

    The public entry takes no corpus location: it always searches the checkout's docs.
    """
    return search_docs_in(args)


def search_docs_in(args: dict[str, Any], *, base: Path | None = None) -> str:
    """:func:`search_core_docs` over ``base`` (a temp tree in tests); internal, not the SDK's.

    Never raises into the loop: a bad config or an unreadable tree is an answer.
    """
    query = str(args.get("query") or args.get("q") or args.get("input") or "").strip()
    if not query:
        return 'Error: search_docs requires a "query" argument, e.g. {"query": "tier routing"}.'
    section = str(args.get("section") or "")
    raw_limit = args.get("limit")
    limit = raw_limit if isinstance(raw_limit, int) and not isinstance(raw_limit, bool) else None
    try:
        config = load_config()
    except DocsSearchConfigError as exc:
        return f"search_docs is unavailable: {exc}"
    corpus = build_corpus(config, base)
    if corpus.roots_missing:
        return NOT_PRESENT
    return render(corpus, query, section, limit, config)
