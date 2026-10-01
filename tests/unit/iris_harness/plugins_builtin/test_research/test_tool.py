"""Tests for the research tool surfaces (ReAct callable + formatting)."""

from __future__ import annotations

import pytest

from iris_harness.plugins_builtin.research import tool as tool_mod
from iris_harness.plugins_builtin.research.models import ResearchResult, SearchResult


def _stub_engine(monkeypatch, captured: dict[str, object]) -> None:  # type: ignore[no-untyped-def]
    class _Eng:
        def research(self, inp: object) -> ResearchResult:
            captured["input"] = inp
            return ResearchResult(
                query=getattr(inp, "query", ""),
                results=[SearchResult(title="T", url="http://x.com", snippet="snip")],
                provider="searxng",
            )

    monkeypatch.setattr(tool_mod, "get_engine", lambda **_: _Eng())


def test_run_research_formats_results(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    _stub_engine(monkeypatch, captured)
    out = tool_mod.run_research({"query": "meta llm framework"})
    assert "provider: searxng" in out
    assert "**T**" in out and "http://x.com" in out
    assert captured["input"].query == "meta llm framework"


def test_run_research_requires_query() -> None:
    assert "requires a 'query'" in tool_mod.run_research({})


def test_run_research_coerces_aliases(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    _stub_engine(monkeypatch, captured)
    tool_mod.run_research({"input": "x", "type": "news", "limit": 3})
    inp = captured["input"]
    assert inp.search_type == "news" and inp.max_results == 3


def test_format_result_empty_uses_error() -> None:
    out = tool_mod.format_result(
        ResearchResult(query="x", results=[], provider="none", error="boom")
    )
    assert "boom" in out


# ── the news lens is derived, not remembered ──────────────────────────────────
#
# search_type defaults to "web", which is the provider's plain keyword endpoint. Asked
# for "breaking news today" on 2026-09-16 that returned a hockey analysis piece, an
# Emmys recap and a how-to on weed-eater line — every one a literal match on "breaking".
# The model was never told a news question needs search_type="news", so it never passed it.


@pytest.mark.parametrize(
    ("query", "freshness"),
    [
        ("what's some of the break news today", "day"),
        ("breaking news", "day"),
        ("any breaking news today", "day"),
        ("top headlines right now", "day"),
        ("what happened today", "day"),
        ("latest news on the port strike", "week"),
        ("ai news", "week"),
        ("what's happening with nvidia", "week"),
        ("latest on the trade talks", "week"),
    ],
)
def test_a_news_question_gets_the_news_lens(query: str, freshness: str) -> None:
    inp = tool_mod._coerce_input({"query": query})
    assert inp is not None
    assert inp.search_type == "news"
    assert inp.freshness == freshness


@pytest.mark.parametrize(
    "query",
    [
        "python asyncio newspaper parsing",  # "newspaper" is not "news"
        "how do i renew my passport",  # "renew" is not "news"
        "kubernetes ingress controller comparison",
        "what is the capital of peru",
        "openai pricing per token",
    ],
)
def test_an_ordinary_question_keeps_the_web_lens(query: str) -> None:
    inp = tool_mod._coerce_input({"query": query})
    assert inp is not None
    assert inp.search_type == "web"


def test_an_explicit_search_type_is_never_overridden() -> None:
    """A caller that named the lens keeps it, even when the query reads as news."""
    inp = tool_mod._coerce_input({"query": "breaking news today", "search_type": "web"})
    assert inp is not None and inp.search_type == "web"


def test_an_explicit_freshness_is_never_overridden() -> None:
    inp = tool_mod._coerce_input({"query": "breaking news today", "freshness": "month"})
    assert inp is not None
    assert inp.search_type == "news" and inp.freshness == "month"


def test_the_lens_survives_a_bad_sibling_argument() -> None:
    """A model that invents `max_results: "ten"` must not fall back to a keyword search."""
    inp = tool_mod._coerce_input({"query": "breaking news today", "max_results": "ten"})
    assert inp is not None
    assert inp.search_type == "news" and inp.freshness == "day"


def test_the_derived_lens_reaches_the_engine(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The seam that matters: what the engine is actually asked for."""
    captured: dict[str, object] = {}
    _stub_engine(monkeypatch, captured)
    tool_mod.run_research({"query": "what's some of the break news today"})
    assert captured["input"].search_type == "news"  # type: ignore[union-attr]
    assert captured["input"].freshness == "day"  # type: ignore[union-attr]


def test_the_langchain_path_derives_the_same_lens(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The skills surface, not the ReAct one — the sibling path the coverage rule asks for.

    This passes on the pinned LangChain whichever way the schema declares its defaults,
    because `invoke` filters unsupplied keys back out before `_run`. It is here to pin the
    behaviour, not the mechanism: if that filtering ever changes, a news question routed
    through a skill must still reach the news lens.
    """
    captured: dict[str, object] = {}
    _stub_engine(monkeypatch, captured)
    lc = tool_mod.build_langchain_tool()

    # Through `invoke`, not `_run`: only `invoke` validates against args_schema, which is
    # where the defaults are materialised. Calling `_run` directly would pass this test
    # with the bug still in place.
    lc.invoke({"query": "any breaking news today"})
    assert captured["input"].search_type == "news"  # type: ignore[union-attr]
    assert captured["input"].freshness == "day"  # type: ignore[union-attr]

    # An ordinary question is still a web search on this path.
    lc.invoke({"query": "kubernetes ingress controller comparison"})
    assert captured["input"].search_type == "web"  # type: ignore[union-attr]

    # And a caller that does name the lens still wins.
    lc.invoke({"query": "any breaking news today", "search_type": "web"})
    assert captured["input"].search_type == "web"  # type: ignore[union-attr]
