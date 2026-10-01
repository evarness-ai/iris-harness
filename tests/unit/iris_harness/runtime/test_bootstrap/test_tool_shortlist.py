"""ADR-0077 P2: relevance shortlist + hard N-cap on the ReAct toolset.

As the universal tool pool grows, a small local model faces too many choices and
fumbles tool calls. The shortlist keeps an always-on recall/search core and fills the
remaining cap with the tools most relevant to the turn — never silently (dropped names
are returned for logging), and fail-open (no embedder → no fewer tools than today).
"""

from __future__ import annotations

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.runtime.handlers.react import _REACT_CORE_TOOL_NAMES, _shortlist_react_tools


class _StubRouter:
    """Ranks (key, text) pairs by how many query words appear in the text."""

    def rank_texts(self, query: str, items):
        q = set(query.lower().split())
        scored = [(k, len(q & set(t.lower().split()))) for k, t in items]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored


def _tool(name: str, desc: str = "", *, pinned: bool = False) -> ToolSpec:
    return ToolSpec(name, desc or name, lambda _a: "", pinned=pinned)


def test_noop_when_under_cap() -> None:
    tools = [_tool("a"), _tool("b")]
    kept, dropped = _shortlist_react_tools(tools, "anything", cap=12, router=_StubRouter())
    assert kept == tools and dropped == []


def test_noop_when_query_empty() -> None:
    tools = [_tool(f"t{i}") for i in range(20)]
    kept, dropped = _shortlist_react_tools(tools, "  ", cap=5, router=_StubRouter())
    assert kept == tools and dropped == []


def test_core_always_kept_and_cap_enforced() -> None:
    core = [_tool(n) for n in _REACT_CORE_TOOL_NAMES]
    extras = [_tool(f"skill_{w}", f"about {w}") for w in ("alpha", "beta", "gamma", "delta")]
    tools = core + extras
    kept, dropped = _shortlist_react_tools(
        tools, "tell me about gamma", cap=len(core) + 1, router=_StubRouter()
    )
    kept_names = {t.name for t in kept}
    # Every core tool survives; exactly one extra (the most relevant) fills the slot.
    assert _REACT_CORE_TOOL_NAMES <= kept_names
    assert "skill_gamma" in kept_names
    assert len(kept) == len(core) + 1
    assert set(dropped) == {"skill_alpha", "skill_beta", "skill_delta"}


def test_fail_open_without_router_reports_dropped() -> None:
    tools = [_tool(f"t{i}") for i in range(10)]
    kept, dropped = _shortlist_react_tools(tools, "q", cap=4, router=None)
    assert len(kept) == 4
    assert len(dropped) == 6
    assert {t.name for t in kept} | set(dropped) == {t.name for t in tools}


def test_search_inbox_is_pinned_by_its_declaration_under_cap() -> None:
    # finance's guidance can say "call search_inbox now" on a miss; the email plugin's
    # manifest pins it (ADR-0110), so the shortlist keeps it — the core names nothing.
    tools = [
        _tool("memory_search"),
        _tool("research", pinned=True),
        _tool("search_inbox", pinned=True),
        _tool("finance_lookup", "dues balances statements"),
        _tool("inbox_digest", "summarize inbox"),
        _tool("read_email", "read one email"),
    ]
    kept, _ = _shortlist_react_tools(
        tools,
        "find Woodgrove bank statement and balances",
        cap=4,
        router=_StubRouter(),
    )
    kept_names = {t.name for t in kept}
    assert {"memory_search", "research", "search_inbox"} <= kept_names


def test_ask_user_is_pinned_core_under_cap() -> None:
    """The confirmation step must be on the menu whatever the query sounds like (PR 4)."""
    assert "ask_user" in _REACT_CORE_TOOL_NAMES
    tools = [_tool(f"t{i}") for i in range(10)] + [_tool("ask_user", "ask the user a question")]
    kept, dropped = _shortlist_react_tools(tools, "find my bills", cap=4, router=None)
    assert "ask_user" in {t.name for t in kept}
