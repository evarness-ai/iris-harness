"""Tests for manifest-driven natural-language routine authoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.kernel.governance.evaluator.embeddings import DefaultEmbedder
from iris_harness.services.routines import (
    create_routine_spec,
    detect_capability_swap_intent,
    extract_routine_refinements,
    extract_user_named_title,
    format_tool_arg_prompt_line,
    match_brief_slots,
    parse_routine_authoring,
    parse_tool_arg_reply,
    resolve_routine_pick,
)
from iris_harness.tools.skills.models import ToolArg
from iris_harness.tools.skills.registry import SkillRegistry
from iris_harness.tools.skills.semantic_router import SemanticSkillRouter

REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture(scope="module")
def candidate_packages() -> tuple:
    """Every loadable skill — brief and non-brief — so the router sees
    the same candidate set the runtime does."""

    registry = SkillRegistry(repo_root=REPO_ROOT)
    registry.discover()
    return registry.list_packages(only_loadable=True)


@pytest.fixture(scope="module")
def brief_packages(candidate_packages: tuple) -> tuple:
    """Subset of the candidate set restricted to brief skills (legacy
    fixture name used by the brief-specific tests)."""

    return tuple(
        p for p in candidate_packages if p.manifest.kind == "brief" and p.manifest.brief is not None
    )


@pytest.fixture(scope="module")
def router() -> SemanticSkillRouter:
    return SemanticSkillRouter(embedder=DefaultEmbedder(), threshold=0.45)


@pytest.mark.minilm
def test_parse_morning_briefing_with_composite_request_drafts_routine(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """The bug we set out to fix: a composite morning-briefing message must
    bind to ``morning-briefing``, not collapse to a single slice."""

    parsed = parse_routine_authoring(
        (
            "send me a morning briefing every day at 9 am, with my due reminders, "
            "activities, top 10 git repositories of the day, top 10 ai news and "
            "top 10 global news, finally top 10 stocks for the day to look for"
        ),
        brief_packages=brief_packages,
        router=router,
    )

    assert parsed.action == "draft"
    assert parsed.schedule == "daily:09:00"
    assert parsed.template == "morning-briefing"
    assert "top_repos" in parsed.source_preferences
    # AI news and global news are their own news groups (digest v5).
    assert {"news_ai", "news_global"} <= set(parsed.source_preferences)
    assert "stocks" in parsed.source_preferences
    assert "fetch_web_content" in parsed.metadata["tool_callbacks"]


@pytest.mark.minilm
def test_slot_selection_excludes_weak_neighbours(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """Winner-band selection: "due reminders and top 10 stocks" selects
    ``reminders`` + ``stocks`` and must NOT drag in the weakly-related
    ``due_today`` / ``bills_due`` slots that merely share the "due" token.
    Regression for the over-generous flat-threshold behavior surfaced in live
    testing 2026-06-26. (``portfolio`` is a legitimate strong match for "stocks"
    and may co-select — the test guards the *weak* neighbours, not exact equality.)"""

    parsed = parse_routine_authoring(
        "every day at 9am send me a brief with just my due reminders and top 10 stocks",
        brief_packages=brief_packages,
        router=router,
    )
    assert parsed.action == "draft"
    assert parsed.template == "morning-briefing"
    prefs = set(parsed.source_preferences)
    assert {"reminders", "stocks"}.issubset(prefs)
    assert "due_today" not in prefs
    assert "bills_due" not in prefs
    assert "portfolio" not in prefs  # market "stocks" must not pull in my holdings


@pytest.mark.minilm
def test_schedule_phrasing_does_not_leak_into_section_selection(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """The schedule clause ("every day at 8am") must not bias section selection:
    "...just the top 10 stocks" selects ``stocks`` and must NOT leak ``reminders``
    / ``due_today`` (whose time-bearing templates otherwise weakly matched the
    '8am' tokens). Regression for the schedule-phrasing leak found in live testing
    2026-06-26. (``portfolio`` is a legitimate strong match for "stocks" and may
    co-select; the guard is that schedule phrasing adds no time-bearing slot.)"""

    parsed = parse_routine_authoring(
        "every day at 8am send me a brief with just the top 10 stocks",
        brief_packages=brief_packages,
        router=router,
    )
    assert parsed.action == "draft"
    assert parsed.template == "morning-briefing"
    prefs = set(parsed.source_preferences)
    assert "stocks" in prefs
    assert "reminders" not in prefs
    assert "due_today" not in prefs


@pytest.mark.minilm
def test_stocks_and_portfolio_sections_are_distinct(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """Trending ``stocks`` (market most-active) and ``portfolio`` (the user's own
    holdings) are semantically close but distinct sections — disambiguated by the
    slot ``summary`` descriptors. A "stocks" request must not pull in the
    portfolio, and a "portfolio" request must not pull in trending stocks."""

    stocks = parse_routine_authoring(
        "every day at 8am send me a brief with the top 10 stocks",
        brief_packages=brief_packages,
        router=router,
    )
    assert "stocks" in stocks.source_preferences
    assert "portfolio" not in stocks.source_preferences

    pf = parse_routine_authoring(
        "every day at 8am send me a brief with my investment portfolio",
        brief_packages=brief_packages,
        router=router,
    )
    assert "portfolio" in pf.source_preferences
    assert "stocks" not in pf.source_preferences


@pytest.mark.minilm
def test_indexes_section_distinct_and_selectable(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """Market 'indexes' (Nifty/Sensex/S&P/Dow) is its own section, distinct from
    'stocks' (individual companies). Verified at the section-matching layer
    (match_brief_slots) so it's independent of the separate template-match
    threshold: a 'stocks' clause must not pull in indexes, an 'indexes' clause
    must not pull in stocks, and a combined clause selects all three."""

    mb = next(p for p in brief_packages if p.manifest.name == "morning-briefing")

    assert "indexes" not in match_brief_slots("the top 10 stocks", mb, router)
    idx = match_brief_slots("the market indexes in india and usa", mb, router)
    assert "indexes" in idx
    assert "stocks" not in idx
    combined = match_brief_slots(
        "top 10 trending stocks and indexes in india and usa and my portfolio holdings",
        mb,
        router,
    )
    assert {"stocks", "indexes", "portfolio"}.issubset(set(combined))


@pytest.mark.minilm
def test_parse_daily_repo_brief_routine(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    parsed = parse_routine_authoring(
        "Hey send me trending github repos at 10 am every day",
        brief_packages=brief_packages,
        router=router,
    )

    assert parsed.action == "draft"
    assert parsed.schedule == "daily:10:00"
    assert parsed.template == "daily-repo-brief"
    assert parsed.source_preferences == ("top_repos",)
    assert parsed.metadata["tool_callbacks"] == ["fetch_web_content"]


def test_one_shot_query_with_each_repo_is_not_a_routine(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    # issue 0019: "a brief summary of each repo" is a one-shot ask. Bare "each" must
    # NOT be read as a recurrence signal (it used to hijack this into a routine draft).
    parsed = parse_routine_authoring(
        "What are the top 10 git repositories today, ordered by stars, "
        "with a brief summary of each repo?",
        brief_packages=brief_packages,
        router=router,
    )
    assert parsed.action == "none"


def test_each_morning_still_triggers_routine(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    # Temporal "each <unit>" is still a recurrence signal.
    parsed = parse_routine_authoring(
        "send me trending github repos each morning at 8",
        brief_packages=brief_packages,
        router=router,
    )
    # "each <temporal>" is recognised as routine authoring (draft, or clarify if a
    # detail is still needed) — the point is it's NOT dropped to "none".
    assert parsed.action in {"draft", "clarify"}


@pytest.mark.minilm
def test_parse_morning_briefing_with_explicit_sections_drafts_routine(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    parsed = parse_routine_authoring(
        "Every weekday at 8:30 send me a morning briefing with reminders and active items",
        brief_packages=brief_packages,
        router=router,
    )

    assert parsed.action == "draft"
    assert parsed.schedule == "cron:30 8 * * 1-5"
    assert parsed.template == "morning-briefing"
    assert "reminders" in parsed.source_preferences
    assert "list_today_reminders" in parsed.metadata["tool_callbacks"]


def test_preview_request_defers_instead_of_authoring(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """A one-off "show me a sample of my daily brief" is a render request, not a
    routine — it must NOT draft a routine. It had been silently scheduling one,
    because "show"/"brief" tripped the verb gate and the "daily" in the brief's
    NAME was read as a daily schedule."""

    parsed = parse_routine_authoring(
        "can you show me a sample of my daily brief today",
        brief_packages=brief_packages,
        router=router,
    )

    assert parsed.action == "none"


def test_preview_with_recurrence_still_authors(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """A preview verb PLUS explicit recurrence ("...every morning") is still a
    routine — recurrence wins over the preview heuristic, so we don't regress
    genuine authoring."""

    parsed = parse_routine_authoring(
        "show me a sample of my morning briefing every day at 9 am with my reminders",
        brief_packages=brief_packages,
        router=router,
    )

    assert parsed.action != "none"


@pytest.mark.minilm
def test_parse_generic_morning_briefing_clarifies_sections(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """A bare 'morning briefing' without enumerated sections asks for
    clarification, listing the brief's actual slot keys."""

    parsed = parse_routine_authoring(
        "Every weekday at 8:30 send me a morning briefing",
        brief_packages=brief_packages,
        router=router,
    )

    assert parsed.action == "clarify"
    assert parsed.schedule == "cron:30 8 * * 1-5"
    assert parsed.template == "morning-briefing"
    assert "briefing_sections" in parsed.missing_slots


def test_parse_routine_clarifies_missing_template(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    parsed = parse_routine_authoring(
        "Every morning at 8 run my thing",
        brief_packages=brief_packages,
        router=router,
    )

    assert parsed.action == "clarify"
    assert "template" in parsed.missing_slots


def test_parse_approval_and_cancel_turns() -> None:
    """Approval / cancel signals short-circuit and need no registry."""

    assert parse_routine_authoring("approve it").action == "approve"
    assert parse_routine_authoring("cancel that").action == "cancel"


@pytest.mark.minilm
def test_origin_channel_used_as_delivery_default(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """When the user message doesn't name a delivery channel, fall back
    to the channel the user is messaging from."""

    parsed = parse_routine_authoring(
        "every day at 9 am send me a morning briefing with all sections",
        brief_packages=brief_packages,
        router=router,
        origin_channel="telegram",
    )

    assert parsed.action == "draft"
    assert parsed.delivery_channel == "telegram"


@pytest.mark.minilm
def test_explicit_delivery_channel_overrides_origin(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    parsed = parse_routine_authoring(
        "every day at 9 am send me a morning briefing with all sections, deliver to console",
        brief_packages=brief_packages,
        router=router,
        origin_channel="telegram",
    )

    assert parsed.action == "draft"
    assert parsed.delivery_channel == "console"


def test_extract_routine_delivery_and_content_refinements() -> None:
    refinements = extract_routine_refinements(
        "delivery is telegram, content should be 3 lines per news"
    )

    assert refinements == {
        "delivery_channel": "telegram",
        "content_lines_per_item": 3,
        "content_unit": "news",
        "content_style": "3 lines per news",
    }


def test_extract_content_style_requires_explicit_per_unit_form() -> None:
    """Bare ``<N> lines`` phrases like "in 5 lines or fewer" must not be
    treated as content-style refinements — they're natural-language
    constraints on the response, not routine config. Caught while
    smoke-testing the identity layer on 2026-05-19."""

    from iris_harness.services.routines import extract_content_style

    assert extract_content_style("in 5 lines or fewer") == {}
    assert extract_content_style("answer in 3 lines") == {}
    assert extract_content_style("about 7 lines please") == {}
    # Explicit per-unit form still resolves.
    style = extract_content_style("3 lines per news")
    assert style["content_lines_per_item"] == 3
    assert style["content_unit"] == "news"


def test_extract_routine_delivery_channel_ignores_bare_channel_mentions() -> None:
    """Bare mentions of channel words ("the web", "voice note") must not
    be treated as delivery refinements. Phrases like "don't search the
    web" used to trip the routine-authoring path and short-circuit the
    general handler before USER.md could reach the LLM. Caught while
    smoke-testing the new identity layer on 2026-05-19."""

    from iris_harness.services.routines import extract_routine_delivery_channel

    assert extract_routine_delivery_channel("Don't search the web") == ""
    assert extract_routine_delivery_channel("I left a voice note") == ""
    assert extract_routine_delivery_channel("opened a browser tab") == ""
    assert extract_routine_delivery_channel("the cli is great") == ""
    # Genuine delivery refinements still resolve.
    assert extract_routine_delivery_channel("deliver to telegram") == "telegram"
    assert extract_routine_delivery_channel("send via web") == "web"
    assert extract_routine_delivery_channel("notify on telegram") == "telegram"
    assert extract_routine_delivery_channel("over voice") == "voice"


# ---------------------------------------------------------------------------
# Phase A: routines bind to any addressable capability, not just briefs.
# ---------------------------------------------------------------------------


@pytest.mark.minilm
def test_routine_binds_to_zero_arg_non_brief_tool_and_drafts(
    candidate_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """A routine that asks for the overdue tasks every morning binds to an
    ``iris-tasks`` tool — no required args, on a non-brief core skill (it ships in
    every tree). It must draft immediately without asking for sections or args
    (no brief filter, no false clarification)."""

    parsed = parse_routine_authoring(
        "every morning at 8 send me my overdue tasks",
        candidate_packages=candidate_packages,
        router=router,
    )

    assert parsed.action == "draft", f"expected draft, got {parsed.action}: {parsed.reason}"
    assert parsed.schedule == "daily:08:00"
    # Template is the bound skill's manifest name (no `kind: brief`
    # filtering means non-brief skills can win the capability match).
    assert parsed.template == "iris-tasks"
    # No section-asking for non-brief routines.
    assert "briefing_sections" not in parsed.missing_slots
    # The bound tool name surfaces in metadata so the runtime knows
    # exactly which capability to invoke.
    assert "bound_tool" in parsed.metadata
    assert parsed.metadata["bound_tool"] in {
        t.name
        for t in next(
            p for p in candidate_packages if p.manifest.name == "iris-tasks"
        ).manifest.tools
    }


@pytest.mark.minilm
def test_routine_for_multi_arg_tool_clarifies_args(
    candidate_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """``web-fetch.fetch_web_content`` declares two required args
    (``type``, ``category``) via its manifest ``args:`` block. The
    routine must clarify rather than guess, and surface the full
    ``ToolArg`` shape so the next-turn UX can enumerate options."""

    parsed = parse_routine_authoring(
        "every day at 9 am fetch web content",
        candidate_packages=candidate_packages,
        router=router,
    )

    assert parsed.template == "web-fetch"
    assert parsed.action == "clarify"
    assert "tool_args" in parsed.missing_slots
    pending = parsed.metadata.get("pending_tool_args")
    assert isinstance(pending, list) and pending
    names = {entry["name"] for entry in pending}
    assert {"type", "category"}.issubset(names)
    # Optional limit (required=false, default=10) must NOT be in pending.
    assert "limit" not in names
    # Each pending arg carries the rich ToolArg shape — options / type
    # / examples are what the clarification UX renders.
    type_entry = next(entry for entry in pending if entry["name"] == "type")
    assert type_entry["type"] == "enum"
    assert type_entry["options"] == ["git", "news", "stocks"]


@pytest.mark.minilm
def test_brief_routine_path_unchanged_after_phase_a(
    candidate_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """Regression: composite morning briefing still binds to
    ``morning-briefing`` even though the candidate set now includes
    non-brief skills."""

    parsed = parse_routine_authoring(
        (
            "send me a morning briefing every day at 9 am, with reminders, "
            "top 10 git repositories of the day, top 10 ai news, top 10 global news, "
            "and top 10 stocks"
        ),
        candidate_packages=candidate_packages,
        router=router,
    )

    assert parsed.action == "draft"
    assert parsed.template == "morning-briefing"
    assert "top_repos" in parsed.source_preferences
    assert "fetch_web_content" in parsed.metadata["tool_callbacks"]


@pytest.mark.minilm
def test_legacy_brief_packages_kwarg_still_works(
    brief_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """The previous kwarg name is retained as a deprecated alias so
    callers mid-migration don't break."""

    parsed = parse_routine_authoring(
        "every day at 9 am send me a morning briefing with all sections",
        brief_packages=brief_packages,
        router=router,
    )

    assert parsed.action == "draft"
    assert parsed.template == "morning-briefing"


# ---------------------------------------------------------------------------
# Phase A-2 / Q-A3: user-named routine titles are LLM-extracted.
# ---------------------------------------------------------------------------


def test_user_named_routine_title_extracted_via_llm(
    candidate_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """When the user names the routine (e.g. 'my healthcheck'), the
    LLM caller's answer becomes the routine title — overriding the
    skill-derived default. The bound capability is still whatever the
    semantic router picked."""

    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "healthcheck"

    parsed = parse_routine_authoring(
        "schedule my healthcheck every morning at 8",
        candidate_packages=candidate_packages,
        router=router,
        llm_caller=fake_llm,
    )

    assert calls, "LLM caller should be exercised when authoring a routine"
    assert parsed.title == "healthcheck"
    assert parsed.metadata.get("user_named_title") == "healthcheck"


@pytest.mark.minilm
def test_user_unnamed_routine_falls_back_to_skill_title(
    candidate_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """When the LLM returns the empty string (user did not name the
    routine), the title falls back to the matched skill's title. The
    LLM caller is still exercised so the agent's stance stays
    consistent — LLM-first per Q-A3."""

    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return ""

    parsed = parse_routine_authoring(
        "every morning at 8 send me my overdue tasks",
        candidate_packages=candidate_packages,
        router=router,
        llm_caller=fake_llm,
    )

    assert calls, "LLM caller should still be exercised even when no name"
    # No name extracted → metadata must not record one.
    assert "user_named_title" not in parsed.metadata
    # Falls back to whatever skill the semantic router picked.
    # The existing zero-arg test asserts this routes to iris-tasks;
    # the title should mirror that skill's manifest name.
    assert parsed.title == parsed.template
    assert parsed.title


def test_user_named_title_without_llm_caller_degrades_cleanly(
    candidate_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """When no LLM caller is configured, title extraction is skipped
    and the parser still produces a usable routine — no crash, just
    the skill-derived title."""

    parsed = parse_routine_authoring(
        "schedule my healthcheck every morning at 8",
        candidate_packages=candidate_packages,
        router=router,
        llm_caller=None,
    )

    assert "user_named_title" not in parsed.metadata
    # Title falls back to whatever skill the router matched (or empty
    # if nothing matched). Either way: no crash.
    assert parsed.action in {"draft", "clarify"}


def test_user_named_title_llm_exception_degrades_cleanly(
    candidate_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """If the LLM caller raises, the parser logs and degrades — it
    must not bubble the exception out of routine authoring."""

    def failing_llm(prompt: str) -> str:
        raise RuntimeError("Tier-2 LLM unavailable")

    parsed = parse_routine_authoring(
        "schedule my healthcheck every morning at 8",
        candidate_packages=candidate_packages,
        router=router,
        llm_caller=failing_llm,
    )

    assert "user_named_title" not in parsed.metadata
    assert parsed.action in {"draft", "clarify"}


def test_extract_user_named_title_returns_none_without_caller() -> None:
    assert extract_user_named_title("schedule my healthcheck daily", None) is None


def test_extract_user_named_title_strips_quotes_and_punctuation() -> None:
    assert extract_user_named_title("x", lambda _: '"flashcards"') == "flashcards"
    assert extract_user_named_title("x", lambda _: "healthcheck.") == "healthcheck"
    assert extract_user_named_title("x", lambda _: "  appointments  ") == "appointments"


def test_extract_user_named_title_rejects_hedged_responses() -> None:
    """Hedged or explanatory LLM responses are treated as no-name so the
    agent can confirm with the user rather than commit to a wrong guess."""

    assert extract_user_named_title("x", lambda _: "") is None
    assert extract_user_named_title("x", lambda _: "none") is None
    assert extract_user_named_title("x", lambda _: "unknown") is None
    assert extract_user_named_title("x", lambda _: "the user did not name this") is None
    assert extract_user_named_title("x", lambda _: "name: healthcheck") is None


def test_extract_user_named_title_picks_first_line_when_multiline() -> None:
    """LLMs sometimes pad answers with rationale on the next line —
    take the first token-shaped line and treat the rest as commentary."""

    assert (
        extract_user_named_title("x", lambda _: "flashcards\n(the user said 'my flashcards')")
        == "flashcards"
    )


def test_extract_user_named_title_swallows_caller_exception() -> None:
    def failing(_: str) -> str:
        raise RuntimeError("boom")

    assert extract_user_named_title("x", failing) is None


# ---------------------------------------------------------------------------
# Phase A-2 / Q-A1: parse the user's reply to a tool-args clarification.
# ---------------------------------------------------------------------------


def _web_fetch_args() -> tuple[ToolArg, ...]:
    """Args matching the canonical web-fetch manifest."""

    return (
        ToolArg(
            name="type",
            description="Source family.",
            type="enum",
            options=("git", "news", "stocks"),
        ),
        ToolArg(
            name="category",
            description="Concrete source/category.",
            type="enum",
            options=(
                "git-repositories",
                "ai-news",
                "usa-news",
                "global-news",
                "stocks-trending",
            ),
        ),
        ToolArg(
            name="limit",
            description="Max items.",
            type="int",
            required=False,
            default=10,
            min=1,
            max=50,
        ),
    )


def test_parse_tool_arg_reply_full_llm_resolution() -> None:
    """Natural-language reply is mapped to all three arg values."""

    def fake_llm(_: str) -> str:
        return '{"type": "git", "category": "git-repositories", "limit": 10}'

    values, missing = parse_tool_arg_reply(
        "git, top repos, 10 of them",
        _web_fetch_args(),
        fake_llm,
    )

    assert values == {"type": "git", "category": "git-repositories", "limit": 10}
    assert missing == []


def test_parse_tool_arg_reply_partial_resolution_lists_missing() -> None:
    """LLM resolves only ``type``; the helper reports the other two as
    missing so the caller re-asks only what's still unresolved."""

    def fake_llm(_: str) -> str:
        return '{"type": "git", "missing": ["category", "limit"]}'

    values, missing = parse_tool_arg_reply(
        "the git one",
        _web_fetch_args(),
        fake_llm,
    )

    assert values == {"type": "git"}
    assert set(missing) == {"category", "limit"}


def test_parse_tool_arg_reply_includes_prior_answers_in_prompt() -> None:
    """Prior answers travel as context to the LLM so it doesn't re-ask
    them, but are NOT returned in ``values`` (the caller merges)."""

    captured: list[str] = []

    def fake_llm(prompt: str) -> str:
        captured.append(prompt)
        return '{"category": "git-repositories"}'

    values, missing = parse_tool_arg_reply(
        "the trending repositories one",
        (_web_fetch_args()[1],),  # only ask about category
        fake_llm,
        prior_answers={"type": "git"},
    )

    assert values == {"category": "git-repositories"}
    assert missing == []
    assert "Already answered" in captured[0]
    assert "type: git" in captured[0]


def test_parse_tool_arg_reply_llm_failure_falls_back_to_key_value() -> None:
    """LLM exception → deterministic ``key=value`` parser keeps the
    flow alive in degraded mode."""

    def failing(_: str) -> str:
        raise RuntimeError("Tier-2 down")

    values, missing = parse_tool_arg_reply(
        "type=git, category=git-repositories, limit=15",
        _web_fetch_args(),
        failing,
    )

    assert values == {"type": "git", "category": "git-repositories", "limit": 15}
    assert missing == []


def test_parse_tool_arg_reply_llm_garbage_falls_back() -> None:
    """LLM returns non-JSON text → fall back to deterministic parsing."""

    def garbage(_: str) -> str:
        return "I think the user meant git stuff"

    values, missing = parse_tool_arg_reply(
        "type: git, category: git-repositories",
        _web_fetch_args(),
        garbage,
    )

    assert values == {"type": "git", "category": "git-repositories"}
    assert "limit" in missing  # required=False so it's not asked here?
    # 'limit' is in pending_args but absent from reply → missing.
    assert set(missing) == {"limit"}


def test_parse_tool_arg_reply_positional_fallback_without_llm() -> None:
    """Reply with no key markers → match by position in order."""

    values, missing = parse_tool_arg_reply(
        "git, git-repositories, 5",
        _web_fetch_args(),
        llm_caller=None,
    )

    assert values == {"type": "git", "category": "git-repositories", "limit": 5}
    assert missing == []


def test_parse_tool_arg_reply_invalid_enum_marked_missing() -> None:
    """A value outside the declared options is rejected and the arg is
    listed as missing so the caller re-asks."""

    def fake_llm(_: str) -> str:
        return '{"type": "bitcoin", "category": "git-repositories"}'

    values, missing = parse_tool_arg_reply(
        "bitcoin, git-repositories",
        _web_fetch_args()[:2],
        fake_llm,
    )

    assert values == {"category": "git-repositories"}
    assert missing == ["type"]


def test_parse_tool_arg_reply_out_of_range_int_marked_missing() -> None:
    """Out-of-range int is rejected; arg goes back to missing."""

    def fake_llm(_: str) -> str:
        return '{"limit": 100}'

    values, missing = parse_tool_arg_reply(
        "100",
        (_web_fetch_args()[2],),
        fake_llm,
    )

    assert values == {}
    assert missing == ["limit"]


def test_parse_tool_arg_reply_empty_pending_returns_empty() -> None:
    assert parse_tool_arg_reply("anything", (), llm_caller=None) == ({}, [])


def test_parse_tool_arg_reply_enum_case_insensitive_match() -> None:
    """``GIT`` should map to the option ``git`` — users won't type the
    exact casing of every enum option."""

    def fake_llm(_: str) -> str:
        return '{"type": "GIT"}'

    values, missing = parse_tool_arg_reply("GIT", (_web_fetch_args()[0],), fake_llm)

    assert values == {"type": "git"}
    assert missing == []


def test_parse_tool_arg_reply_tolerates_markdown_fence() -> None:
    """LLMs sometimes wrap JSON in ```json fences — strip them."""

    def fenced_llm(_: str) -> str:
        return '```json\n{"type": "news"}\n```'

    values, missing = parse_tool_arg_reply(
        "news",
        (_web_fetch_args()[0],),
        fenced_llm,
    )

    assert values == {"type": "news"}
    assert missing == []


def test_parse_tool_arg_reply_bool_coercion() -> None:
    """Bool args accept yes/no/true/false/1/0 from string replies."""

    arg = ToolArg(name="enabled", description="On?", type="bool")

    def yes_llm(_: str) -> str:
        return '{"enabled": "yes"}'

    values, missing = parse_tool_arg_reply("yes", (arg,), yes_llm)
    assert values == {"enabled": True}
    assert missing == []


# ---------------------------------------------------------------------------
# Phase A-2 / Step 4: clarification message renders the rich ToolArg shape.
# ---------------------------------------------------------------------------


def test_format_tool_arg_prompt_line_enum_renders_options() -> None:
    arg = ToolArg(
        name="type",
        description="Source family.",
        type="enum",
        options=("git", "news", "stocks"),
    )

    line = format_tool_arg_prompt_line(arg)

    assert line.startswith("- **type**:")
    assert "`git`" in line and "`news`" in line and "`stocks`" in line
    assert "/" in line  # options separated by ' / '


def test_format_tool_arg_prompt_line_int_renders_range_and_default() -> None:
    arg = ToolArg(
        name="limit",
        description="Items.",
        type="int",
        required=False,
        default=10,
        min=1,
        max=50,
    )

    line = format_tool_arg_prompt_line(arg)

    assert "**limit**" in line
    assert "an integer" in line
    assert "1–50" in line
    assert "default 10" in line


def test_format_tool_arg_prompt_line_prompt_override_used_verbatim() -> None:
    """When the manifest supplies an explicit ``prompt``, use it raw and
    skip the per-type rendering. Lets skill authors override phrasing."""

    arg = ToolArg(
        name="symbol",
        description="Stock ticker.",
        type="string",
        prompt="which ticker (e.g. AAPL, TSLA)?",
    )

    assert format_tool_arg_prompt_line(arg) == "- **symbol**: which ticker (e.g. AAPL, TSLA)?"


def test_format_tool_arg_prompt_line_bool_renders_yes_no() -> None:
    arg = ToolArg(name="enabled", description="On?", type="bool", default=True)
    line = format_tool_arg_prompt_line(arg)
    assert "`yes`" in line and "`no`" in line
    assert "default `True`" in line


def test_format_tool_arg_prompt_line_string_shows_examples() -> None:
    arg = ToolArg(
        name="query",
        description="Search query.",
        type="string",
        examples=("trending repos", "ai-news"),
    )

    line = format_tool_arg_prompt_line(arg)

    assert "Search query." in line
    assert "trending repos" in line
    assert "ai-news" in line


@pytest.mark.minilm
def test_clarify_reason_for_tool_args_enumerates_all_choices(
    candidate_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """End-to-end: the parsed result's ``reason`` field already carries
    the rich rendering — bootstrap can render it as-is."""

    parsed = parse_routine_authoring(
        "every day at 9 am fetch web content",
        candidate_packages=candidate_packages,
        router=router,
    )

    assert parsed.action == "clarify"
    assert "tool_args" in parsed.missing_slots
    reason = parsed.reason
    assert "`web-fetch`" in reason
    assert "**type**" in reason and "`git`" in reason
    assert "**category**" in reason and "`git-repositories`" in reason
    # ``limit`` has required=False so it is NOT in the asked list.
    assert "**limit**" not in reason


# ---------------------------------------------------------------------------
# Phase B: resolve_routine_pick disambiguates refinement targets.
# ---------------------------------------------------------------------------


def _two_specs():
    a = create_routine_spec(
        title="Morning briefing",
        goal="g",
        schedule="daily:08:00",
        template="morning-briefing",
    )
    b = create_routine_spec(
        title="Daily repo brief",
        goal="g",
        schedule="daily:09:00",
        template="daily-repo-brief",
    )
    return a, b


def test_resolve_routine_pick_exact_id() -> None:
    a, b = _two_specs()
    assert resolve_routine_pick(b.id, (a, b)) is b


def test_resolve_routine_pick_position_keywords() -> None:
    a, b = _two_specs()
    assert resolve_routine_pick("first", (a, b)) is a
    assert resolve_routine_pick("the second one", (a, b)) is b
    assert resolve_routine_pick("2", (a, b)) is b
    assert resolve_routine_pick("#1", (a, b)) is a


def test_resolve_routine_pick_unique_title_substring() -> None:
    a, b = _two_specs()
    assert resolve_routine_pick("repo brief", (a, b)) is b
    assert resolve_routine_pick("MORNING", (a, b)) is a


def test_resolve_routine_pick_ambiguous_title_returns_none() -> None:
    a, b = _two_specs()
    # "brief" appears in both titles → ambiguous, helper bails.
    assert resolve_routine_pick("brief", (a, b)) is None


def test_resolve_routine_pick_empty_or_unknown_returns_none() -> None:
    a, b = _two_specs()
    assert resolve_routine_pick("", (a, b)) is None
    assert resolve_routine_pick("   ", (a, b)) is None
    assert resolve_routine_pick("nope", (a, b)) is None


def test_resolve_routine_pick_id_prefix_requires_min_length() -> None:
    a, b = _two_specs()
    # Routine ids share a timestamped prefix; a unique 8+ char prefix
    # of one resolves to that one.
    unique_prefix = a.id[:-8]  # everything except the random suffix
    assert resolve_routine_pick(unique_prefix, (a, b)) is a
    # A shared short prefix doesn't disambiguate, so the helper bails.
    assert resolve_routine_pick("rout", (a, b)) is None


# ---------------------------------------------------------------------------
# Phase B slice 3: detect_capability_swap_intent
# ---------------------------------------------------------------------------


@pytest.mark.minilm
def test_detect_capability_swap_intent_matches_explicit_switch(
    candidate_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """Verb-led message with a clear skill reference returns the package."""

    target = detect_capability_swap_intent(
        "switch to morning briefing instead",
        candidate_packages,
        router,
    )

    assert target is not None
    assert target.manifest.name == "morning-briefing"


def test_detect_capability_swap_intent_no_verb_returns_none(
    candidate_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """A skill name alone (no swap verb) doesn't trigger a swap."""

    target = detect_capability_swap_intent(
        "morning briefing",
        candidate_packages,
        router,
    )

    assert target is None


def test_detect_capability_swap_intent_no_semantic_match_returns_none(
    candidate_packages: tuple,
    router: SemanticSkillRouter,
) -> None:
    """A swap verb without an identifiable target doesn't trigger."""

    target = detect_capability_swap_intent(
        "switch it off",
        candidate_packages,
        router,
    )

    assert target is None


def test_detect_capability_swap_intent_empty_inputs_return_none() -> None:
    assert detect_capability_swap_intent("", (), None) is None
    assert detect_capability_swap_intent("switch to X", (), None) is None
