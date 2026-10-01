"""Unit tests for the research ranking module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from iris_harness.plugins_builtin.research.models import SearchResult
from iris_harness.plugins_builtin.research.rank import (
    _freshness_score,
    research_source_dims,
    score_results,
    trust_for,
)


def test_trust_for_known_and_unknown_domains() -> None:
    assert trust_for("https://docs.python.org/3/library/json.html") == 10.0
    assert trust_for("https://github.com/python/cpython") == 9.0
    assert trust_for("https://unknown.example/page") == 4.0
    assert trust_for("https://aiflashreport.com/gpt5-leaked") == 2.0


def test_trust_for_handles_www_prefix() -> None:
    assert trust_for("https://www.github.com/foo") == 9.0
    assert trust_for("https://www.stackoverflow.com/q/1") == 8.0


def test_trust_for_never_raises_on_bad_url() -> None:
    assert trust_for("") == 4.0
    assert trust_for("not a url at all") == 4.0
    assert trust_for("://broken") == 4.0


def test_trust_for_suffix_match() -> None:
    # *.readthedocs.io and *.org suffix families.
    assert trust_for("https://requests.readthedocs.io/en/latest/") == 10.0
    assert trust_for("https://www.djangoproject.org/") == 9.0


def _result(
    title: str, url: str, snippet: str = "", published: datetime | None = None
) -> SearchResult:
    return SearchResult(title=title, url=url, snippet=snippet, published=published)


def test_score_results_lexical_ranks_relevant_high_trust_first() -> None:
    query = "python json parsing"
    results = [
        _result("Random spam page", "https://aiflashreport.com/x", "buy now cheap"),
        _result("python json parsing", "https://docs.python.org/3/library/json.html", "parse json"),
        _result("unrelated topic", "https://medium.com/@a/cats", "cats are nice"),
    ]
    score_results(query, results, embed=None)

    # The title-matching, high-trust docs result must sort first.
    assert results[0].url == "https://docs.python.org/3/library/json.html"
    # Trust + score populated on all.
    for r in results:
        assert r.trust_score > 0.0
        assert r.score > 0.0

    # Deterministic: a second run yields identical ordering.
    again = [
        _result("Random spam page", "https://aiflashreport.com/x", "buy now cheap"),
        _result("python json parsing", "https://docs.python.org/3/library/json.html", "parse json"),
        _result("unrelated topic", "https://medium.com/@a/cats", "cats are nice"),
    ]
    score_results(query, again, embed=None)
    assert [r.url for r in again] == [r.url for r in results]


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic vector: [length, vowel_count] per text."""
    vowels = set("aeiou")
    return [[float(len(t)), float(sum(c in vowels for c in t.lower()))] for t in texts]


def test_score_results_with_embed_does_not_raise_and_orders() -> None:
    query = "machine learning"
    results = [
        _result("short", "https://medium.com/a", "x"),
        _result("machine learning tutorial guide", "https://github.com/ml/repo", "deep dive"),
        _result("aaaa", "https://reddit.com/r/ml", "bbbb"),
    ]
    score_results(query, results, embed=_fake_embed)
    assert all(r.score > 0.0 for r in results)
    # Ordering is by blended score; just assert it's a stable, full permutation.
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_score_results_falls_back_when_embed_raises() -> None:
    def boom(_texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedder down")

    query = "python json"
    results = [
        _result("python json", "https://docs.python.org/3/", "json"),
        _result("spam", "https://aiflashreport.com/x", "spam"),
    ]
    # Must not raise; falls back to lexical.
    score_results(query, results, embed=boom)
    assert results[0].url == "https://docs.python.org/3/"


def test_freshness_score() -> None:
    now = datetime(2026, 6, 25, tzinfo=UTC)
    assert _freshness_score(now, now=now) > 0.8
    assert abs(_freshness_score(None, now=now) - 0.5) < 1e-9
    two_years_ago = now - timedelta(days=730)
    assert _freshness_score(two_years_ago, now=now) < 0.3


def test_score_results_downranks_suppressed_source(tmp_path) -> None:
    """A source the user marked "not useful" for this lens sinks below others (issue 0028)."""
    from iris_harness.services.learning.suppression import NOT_USEFUL, SurfaceFeedbackStore

    fb = SurfaceFeedbackStore(db_path=tmp_path / "learning.db")
    fb.ensure_schema()

    def fresh() -> list[SearchResult]:
        return [
            _result("doc", "https://docs.python.org/3/library/json.html", "parse json"),
            _result("doc", "https://unknown.example/page", "parse json"),
        ]

    # Baseline: the high-trust docs source ranks first.
    base = fresh()
    score_results("json", base, search_type="web", feedback_store=fb)
    assert base[0].url.startswith("https://docs.python.org")

    # Mark docs.python.org "not useful" for web → it sinks to the bottom.
    fb.record(
        "research", "source", research_source_dims("https://docs.python.org/x", "web"), NOT_USEFUL
    )
    after = fresh()
    score_results("json", after, search_type="web", feedback_store=fb)
    assert after[0].url.startswith("https://unknown.example")
    assert after[-1].url.startswith("https://docs.python.org")


def test_score_results_suppression_is_lens_specific(tmp_path) -> None:
    """Suppressing a host for "news" must not downrank it for "web"."""
    from iris_harness.services.learning.suppression import NOT_USEFUL, SurfaceFeedbackStore

    fb = SurfaceFeedbackStore(db_path=tmp_path / "learning.db")
    fb.ensure_schema()
    fb.record(
        "research", "source", research_source_dims("https://docs.python.org/x", "news"), NOT_USEFUL
    )

    results = [
        _result("doc", "https://docs.python.org/3/library/json.html", "parse json"),
        _result("doc", "https://unknown.example/page", "parse json"),
    ]
    score_results("json", results, search_type="web", feedback_store=fb)
    assert results[0].url.startswith("https://docs.python.org")  # web lens unaffected


# ── the news lens has its own trust table ─────────────────────────────────────
#
# The web table carries no newsroom, so on a news query it ranked a random .org (9.0 by
# the suffix rule) above Reuters (DEFAULT_TRUST). Trust is 0.40 of the blend.


def test_news_lens_scores_wire_services_above_everything() -> None:
    assert trust_for("https://www.reuters.com/world/story", search_type="news") == 10.0
    assert trust_for("https://apnews.com/article/abc", search_type="news") == 10.0


def test_news_lens_does_not_inherit_the_web_org_suffix_rule() -> None:
    """The inversion this table exists to fix: any .org outranked every wire service."""
    org = trust_for("https://some-foundation.org/post", search_type="news")
    wire = trust_for("https://reuters.com/x", search_type="news")
    assert org == 4.0  # DEFAULT_TRUST, not the web table's 9.0
    assert wire > org
    # The web lens is unchanged — the suffix rule still applies there.
    assert trust_for("https://some-foundation.org/post") == 9.0


def test_news_lens_sinks_the_syndicators() -> None:
    """msn/yahoo/aol carried five of the six results in the 2026-09-16 failure."""
    for url in (
        "https://www.msn.com/en-us/sports/hockey/ar-AA2clbSv",
        "https://www.yahoo.com/entertainment/celebrity/articles/x.html",
        "https://www.aol.com/articles/prevent-weed-eater-line-breaking.html",
    ):
        assert trust_for(url, search_type="news") < 4.0  # below an unknown domain
        assert trust_for(url, search_type="news") < trust_for(
            "https://bbc.com/news/x", search_type="news"
        )


def test_news_lens_keeps_gov_primary_sources_high() -> None:
    assert trust_for("https://www.cisa.gov/advisory/2026-01", search_type="news") == 9.5
    # Scoped to news: the web lens has no .gov rule and must not gain one.
    assert trust_for("https://www.cisa.gov/advisory/2026-01") == 4.0


def test_web_lens_trust_is_untouched() -> None:
    assert trust_for("https://docs.python.org/3/") == 10.0
    assert trust_for("https://github.com/python/cpython") == 9.0
    assert trust_for("https://reddit.com/r/x") == 6.0
    assert trust_for("https://aiflashreport.com/x", search_type="news") == 2.0  # spam wins


def test_news_ranking_puts_the_wire_above_the_syndicator() -> None:
    """End to end through score_results: same words, the source decides the order."""
    now = datetime.now(UTC)
    results = [
        SearchResult(
            title="Rangers breaking down the season",
            url="https://www.msn.com/en-us/sports/ar-AA2clbSv",
            snippet="breaking down the biggest question",
            published=now - timedelta(hours=2),
        ),
        SearchResult(
            title="Rangers breaking down the season",
            url="https://www.reuters.com/world/us/story",
            snippet="breaking down the biggest question",
            published=now - timedelta(hours=2),
        ),
    ]
    score_results("breaking news today", results, search_type="news", now=now)
    assert results[0].url.startswith("https://www.reuters.com")
