"""The news source policy is data: adding a source or re-scoring a tier is a YAML edit.

These tests exist to keep it that way. If a domain ever has to be added in Python for a
news query to rank correctly, one of them should fail first.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from iris_harness.plugins_builtin.research import news_policy
from iris_harness.plugins_builtin.research.rank import DEFAULT_TRUST, trust_for


@pytest.fixture(autouse=True)
def _clear_policy_cache() -> Iterator[None]:
    """The map is cached per process. Cleared on both sides so neither this file's tests
    inherit a neighbour's fixture nor leave one behind for test_rank.py."""
    news_policy._CACHE = None
    news_policy._CACHE_KEY = None
    yield
    news_policy._CACHE = None
    news_policy._CACHE_KEY = None


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "news_sources.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# ── the shipped file ──────────────────────────────────────────────────────────


def test_the_shipped_policy_ships_with_the_plugin() -> None:
    """pyproject packages plugins_builtin/**/*.yaml, so a core-only install has it."""
    assert news_policy.DEFAULT_NEWS_SOURCES_PATH.is_file()


def test_the_shipped_policy_has_no_org_suffix_rule() -> None:
    """The inversion this table exists to fix. Authority is the newsroom, not the TLD."""
    scores = news_policy.load_news_trust(news_policy.DEFAULT_NEWS_SOURCES_PATH)
    assert ".org" not in scores
    assert scores["npr.org"] > DEFAULT_TRUST  # listed by name, not by TLD


def test_the_shipped_policy_ranks_wire_above_syndicator() -> None:
    scores = news_policy.load_news_trust(news_policy.DEFAULT_NEWS_SOURCES_PATH)
    assert scores["reuters.com"] > scores["msn.com"]
    assert scores["msn.com"] < DEFAULT_TRUST  # below an unknown domain, on purpose


# ── the YAML actually drives the ranking ──────────────────────────────────────


def test_a_new_source_needs_only_a_yaml_line(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The contract the user asked for: no code change to add a source."""
    path = _write(
        tmp_path,
        "tiers:\n  wire: 10.0\nsources:\n  wire:\n    - example-wire.test\n",
    )
    monkeypatch.setenv(news_policy._PATH_ENV, str(path))

    assert trust_for("https://example-wire.test/a", search_type="news") == 10.0
    # A domain the file does not name is unknown, not guessed at.
    assert trust_for("https://reuters.com/a", search_type="news") == DEFAULT_TRUST
    # And the web lens is untouched by any of it.
    assert trust_for("https://github.com/x") == 9.0


def test_rescoring_a_tier_moves_every_domain_in_it(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    body = "tiers:\n  beat: {score}\nsources:\n  beat:\n    - a.test\n    - b.test\n"
    path = _write(tmp_path, body.format(score="7.5"))
    monkeypatch.setenv(news_policy._PATH_ENV, str(path))
    assert trust_for("https://a.test/x", search_type="news") == 7.5

    _write(tmp_path, body.format(score="2.0"))
    news_policy._CACHE = None
    news_policy._CACHE_KEY = None
    assert trust_for("https://a.test/x", search_type="news") == 2.0
    assert trust_for("https://b.test/x", search_type="news") == 2.0


def test_a_leading_dot_is_a_suffix_match(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = _write(tmp_path, "tiers:\n  official: 9.5\nsources:\n  official:\n    - .gov\n")
    monkeypatch.setenv(news_policy._PATH_ENV, str(path))
    assert trust_for("https://www.cisa.gov/advisory", search_type="news") == 9.5
    assert trust_for("https://notagov.test/x", search_type="news") == DEFAULT_TRUST


# ── a broken file degrades, never raises ──────────────────────────────────────


def test_a_missing_file_is_flat_not_fatal(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Research is the tool the loop reaches for constantly. A config typo must not
    take web access down with it — flat beats inverted, and both beat an exception."""
    monkeypatch.setenv(news_policy._PATH_ENV, str(tmp_path / "absent.yaml"))
    assert news_policy.news_trust_scores() == {}
    assert trust_for("https://reuters.com/a", search_type="news") == DEFAULT_TRUST


@pytest.mark.parametrize(
    "body",
    [
        "tiers: [not, a, mapping]\nsources: {}\n",
        "just a string\n",
        "tiers:\n  wire: 10.0\nsources:\n  wire: not-a-list\n",
        ": : :\n  bad yaml\n",
        "",
    ],
)
def test_a_malformed_file_is_flat_not_fatal(body: str, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(news_policy._PATH_ENV, str(_write(tmp_path, body)))
    assert news_policy.news_trust_scores() == {}


def test_a_tier_with_no_score_is_skipped_not_guessed(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = _write(
        tmp_path,
        "tiers:\n  wire: 10.0\nsources:\n  wire:\n    - a.test\n  mystery:\n    - b.test\n",
    )
    monkeypatch.setenv(news_policy._PATH_ENV, str(path))
    scores = news_policy.news_trust_scores()
    assert scores == {"a.test": 10.0}


def test_a_non_numeric_tier_score_is_skipped(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = _write(
        tmp_path,
        "tiers:\n  wire: high\n  beat: 7.0\nsources:\n  wire:\n    - a.test\n  beat:\n    - b.test\n",
    )
    monkeypatch.setenv(news_policy._PATH_ENV, str(path))
    assert news_policy.news_trust_scores() == {"b.test": 7.0}
