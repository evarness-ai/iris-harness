"""ADR-0077 P2: relevance shortlist + hard N-cap on the ReAct toolset.

As the universal tool pool grows, a small local model faces too many choices and
fumbles tool calls. The shortlist keeps an always-on recall/search core and fills the
remaining cap with the tools most relevant to the turn — never silently (dropped names
are returned for logging). With no embedder the cap still holds, and the slots are
ranked without a model: the turn's own domain first, then lexical overlap, then pool
order (ADR-0077 addendum, 2026-10-01).
"""

from __future__ import annotations

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.runtime.handlers.react import _REACT_CORE_TOOL_NAMES, _shortlist_react_tools
from iris_harness.runtime.plugin_host.manifest import PluginManifest, RegistrationKind
from iris_harness.runtime.plugin_host.registry import (
    PluginRecord,
    PluginRegistry,
    PluginStatus,
    Registration,
)


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


# --- no embedder: a ranking that needs no model (ADR-0077 addendum, 2026-10-01) ------
#
# An offline install with no MiniLM on disk has no embedder. The menu used to fall to
# pool order, which put the builtins first and dropped every email tool from an email
# turn. The fixture pool below has the same shape: the domain's tools come last.

_EMAIL = ("read_email", "trash_email", "list_by_category", "restore_email")


def _email_turn_pool() -> list[ToolSpec]:
    builtins = [_tool(f"builtin_{i}", f"core helper number {i}") for i in range(8)]
    finance = [_tool("upcoming_dues", "bills and dues owed"), _tool("portfolio", "holdings")]
    email = [
        _tool("read_email", "read one email by id or sender"),
        _tool("trash_email", "move emails to the trash, after the owner approves"),
        _tool("list_by_category", "list emails in one category such as promotions"),
        _tool("restore_email", "bring trashed emails back"),
    ]
    return [_tool("memory_search"), *builtins, *finance, *email]


def test_without_an_embedder_the_turns_own_domain_tools_are_kept() -> None:
    kept, dropped = _shortlist_react_tools(
        _email_turn_pool(),
        "trash the promo emails from this week",
        cap=5,
        router=None,
        domain_tools=frozenset(_EMAIL),
    )
    assert [t.name for t in kept] == ["memory_search", *_EMAIL]
    assert "builtin_0" in dropped


def test_without_an_embedder_lexical_overlap_ranks_the_named_tool_first() -> None:
    # No domain (a general turn): the words of the query pick the tool.
    kept, _ = _shortlist_react_tools(
        _email_turn_pool(), "move the promo emails to the trash", cap=2, router=None
    )
    assert [t.name for t in kept] == ["memory_search", "trash_email"]


def test_lexical_overlap_also_orders_tools_inside_the_domain() -> None:
    kept, _ = _shortlist_react_tools(
        _email_turn_pool(),
        "bring back what you trashed",
        cap=2,
        router=None,
        domain_tools=frozenset(_EMAIL),
    )
    assert [t.name for t in kept] == ["memory_search", "restore_email"]


def test_a_word_every_tool_carries_decides_nothing() -> None:
    tools = [_tool(f"t{i}", "email helper") for i in range(5)] + [
        _tool("archive", "email helper that files mail away")
    ]
    kept, _ = _shortlist_react_tools(tools, "email: file this away", cap=1, router=None)
    assert [t.name for t in kept] == ["archive"]


def test_without_an_embedder_the_order_is_deterministic() -> None:
    def run() -> list[str]:
        kept, _ = _shortlist_react_tools(_email_turn_pool(), "anything at all", cap=4, router=None)
        return [t.name for t in kept]

    first = run()
    assert first == run() == ["memory_search", "builtin_0", "builtin_1", "builtin_2"]


class _CannotEmbed:
    """A router whose embedder fails (no model, offline): it ranks nothing."""

    def rank_texts(self, query: str, items):
        return []


def test_a_router_that_cannot_embed_ranks_like_no_router() -> None:
    # Production offline: the router exists but the query never embeds.
    args = ("trash the promo emails",)
    with_router, _ = _shortlist_react_tools(
        _email_turn_pool(), *args, cap=5, router=_CannotEmbed(), domain_tools=frozenset(_EMAIL)
    )
    without, _ = _shortlist_react_tools(
        _email_turn_pool(), *args, cap=5, router=None, domain_tools=frozenset(_EMAIL)
    )
    assert [t.name for t in with_router] == [t.name for t in without]


def test_with_an_embedder_the_domain_does_not_reorder_the_ranking() -> None:
    # The embedder's ranking is unchanged: the domain only orders what it did not rank.
    tools = _email_turn_pool()
    kept, _ = _shortlist_react_tools(
        tools,
        "core helper number 3",
        cap=2,
        router=_StubRouter(),
        domain_tools=frozenset(_EMAIL),
    )
    assert [t.name for t in kept] == ["memory_search", "builtin_3"]


def _plugin(name: str, tools: tuple[str, ...], **manifest: object) -> PluginRecord:
    record = PluginRecord(
        name=name,
        source="test",
        status=PluginStatus.LOADED,
        manifest=PluginManifest(name=name, **manifest),  # type: ignore[arg-type]
    )
    record.registrations.extend(Registration(name, RegistrationKind.TOOL, t) for t in tools)
    return record


def test_the_registry_names_the_tools_of_the_plugins_serving_the_turn() -> None:
    reg = PluginRegistry()
    reg.add_plugin(_plugin("mail", ("read_email",), read_first_intents=("communication",)))
    reg.add_plugin(_plugin("money", ("upcoming_dues",)))
    reg.add_plugin(_plugin("cal", ("calendar_lookup",)))
    reg._record("money", RegistrationKind.INTENT_HANDLER, "finance")
    reg.declare_seam("cal", "loop_intent", "calendar")

    assert reg.tools_serving(("communication", "email")) == frozenset({"read_email"})
    assert reg.tools_serving(("finance",)) == frozenset({"upcoming_dues"})
    assert reg.tools_serving(("calendar", "")) == frozenset({"calendar_lookup"})
    assert reg.tools_serving(("general", "system")) == frozenset()


def test_an_unmounted_plugin_serves_nothing() -> None:
    reg = PluginRegistry()
    off = _plugin("mail", ("read_email",), read_first_intents=("email",))
    off.status = PluginStatus.DISABLED
    reg.add_plugin(off)
    assert reg.tools_serving(("email",)) == frozenset()
