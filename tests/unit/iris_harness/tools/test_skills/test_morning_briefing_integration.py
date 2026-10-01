"""Integration test for the morning-briefing brief manifest after Phase 2
Track 2C extensions.

Verifies that the manifest loads cleanly, that every core skill listed in
``brief.uses`` resolves to a skill package on disk, and that every
``kind: tool`` slot on a core skill references a tool that skill actually
declares (the domain skills' half lives beside the domains: ``_DOMAIN_SKILLS``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.tools.skills.loader import load_skill_manifest, load_skill_package
from iris_harness.tools.skills.models import BriefToolSlot

REPO_ROOT = Path(__file__).resolve().parents[5]
MORNING_BRIEFING = REPO_ROOT / "config" / "skills" / "builtin" / "morning-briefing"


# The core's own skills the brief refers to: they ship wherever the brief does.
# Hardcoded for clarity over magic.
_SKILL_DIRS = {
    "iris-core": REPO_ROOT / "config" / "skills" / "builtin" / "iris-core",
    "iris-tasks": REPO_ROOT / "config" / "skills" / "builtin" / "iris-tasks",
    "web-fetch": REPO_ROOT / "config" / "skills" / "builtin" / "web-fetch",
}
# The domain plugins' skills it also uses. An installation without them renders the
# brief without their sections (a slot whose skill is not installed is absent, not
# failed: runtime/handlers/skill_brief.py). That each one is on disk and declares the
# tools its slots call is asserted beside the domains: tests/unit/iris_personal/plugins/
# test_personal_profile/test_morning_briefing_domain_skills.py.
_DOMAIN_SKILLS = frozenset(
    {
        "calendar-reminders",
        "calendar-events",
        "daily-plan",
        "email-followup",
        "email-triage",
        "finance-brief",
        "portfolio-brief",
    }
)


def test_morning_briefing_manifest_loads() -> None:
    manifest = load_skill_manifest(MORNING_BRIEFING)
    assert manifest.kind == "brief"
    assert manifest.brief is not None
    expected_uses = set(_SKILL_DIRS) | _DOMAIN_SKILLS
    assert set(manifest.brief.uses) == expected_uses


def test_morning_briefing_has_new_phase_2_slots() -> None:
    manifest = load_skill_manifest(MORNING_BRIEFING)
    assert manifest.brief is not None
    expected_new_slots = {
        "due_today",
        "open_tasks",
        "open_followups",
        "resolved_followups",
        "email_summary",
    }
    assert expected_new_slots.issubset(manifest.brief.slots)


@pytest.mark.parametrize("skill_name", list(_SKILL_DIRS))
def test_every_used_skill_has_a_manifest_on_disk(skill_name: str) -> None:
    skill_dir = _SKILL_DIRS[skill_name]
    assert (
        skill_dir / "manifest.yaml"
    ).exists(), f"morning-briefing references skill {skill_name!r} but no manifest at {skill_dir}"
    manifest = load_skill_manifest(skill_dir)
    assert manifest.name == skill_name


def test_every_tool_slot_references_a_declared_tool() -> None:
    brief = load_skill_manifest(MORNING_BRIEFING).brief
    assert brief is not None

    # Map skill_name -> set of declared tool names from each used skill's manifest.
    declared: dict[str, set[str]] = {}
    for skill_name, skill_dir in _SKILL_DIRS.items():
        manifest = load_skill_manifest(skill_dir)
        declared[skill_name] = {t.name for t in manifest.tools}

    misses: list[tuple[str, str, str]] = []
    for slot_name, slot in brief.slots.items():
        if isinstance(slot, BriefToolSlot):
            assert slot.skill in declared or slot.skill in _DOMAIN_SKILLS, slot.skill
            if slot.skill in declared and slot.tool not in declared[slot.skill]:
                misses.append((slot_name, slot.skill, slot.tool))
    assert not misses, f"slot tools not declared in their skill manifest: {misses}"


def test_iris_tasks_skill_loads_without_prerequisites_missing() -> None:
    """iris-tasks must not list data/tasks.db as a hard prereq — the tool
    auto-creates it on first use. Otherwise the brief renders empty slots
    on a fresh install before the user has created any tasks.
    """
    package = load_skill_package(REPO_ROOT, _SKILL_DIRS["iris-tasks"])
    assert package.missing_prerequisites == ()
    tool_names = {cls.model_fields["name"].default for cls in package.tool_classes}
    assert {"list_open_tasks", "list_due_today", "list_resolved_followups"} <= tool_names


# ─── PR 2 digest content (loop-proof plan D4, D17) ───────────────────────────


def _brief():
    brief = load_skill_manifest(MORNING_BRIEFING).brief
    assert brief is not None
    return brief


def test_focus_news_and_learned_yesterday_are_tool_slots() -> None:
    slots = _brief().slots
    expected = {
        "focus": ("email-triage", "email_focus"),
        "news_ai": ("web-fetch", "fetch_web_content"),
        "news_global": ("web-fetch", "fetch_web_content"),
        "news_local": ("web-fetch", "fetch_web_content"),
        "learned_yesterday": ("iris-core", "learned_yesterday"),
    }
    for name, (skill, tool) in expected.items():
        slot = slots[name]
        assert isinstance(slot, BriefToolSlot)
        assert (slot.skill, slot.tool) == (skill, tool)


@pytest.mark.parametrize("slot", ["news_ai", "news_global", "news_local"])
def test_news_is_topic_driven_and_marks_preferred_sources(slot: str) -> None:
    brief = _brief()
    news = brief.slots[slot]
    assert isinstance(news, BriefToolSlot)
    assert news.args["category"] == "digest-topics"
    # Each news section fetches its own group's topics (digest.yaml news_groups).
    assert news.args["news_group"] == slot
    # Plain headline; only the source name in brackets links to the story.
    assert news.item_template == "{title} ([{source}]({url}){preferred})"
    # The fixed per-category news sections, and the one combined section, are gone.
    assert not {"ai_news", "global_news", "news"} & set(brief.slots)


def test_every_news_section_is_a_news_group_of_the_shipped_digest() -> None:
    from iris_harness.services.digest.settings import load_defaults

    groups = load_defaults(REPO_ROOT / "config").settings.news_groups
    slots = {
        name
        for name, slot in _brief().slots.items()
        if isinstance(slot, BriefToolSlot) and slot.args.get("category") == "digest-topics"
    }
    assert slots == set(groups)


def test_a_news_line_links_only_the_source_on_every_channel() -> None:
    """Owner feedback on PR 2: the headline is text, the source in brackets is the link."""
    from iris_harness.runtime.handlers.brief_formats import to_html, to_telegram
    from iris_harness.runtime.handlers.skill_brief import _format_tool_output

    news = _brief().slots["news_global"]
    assert isinstance(news, BriefToolSlot)
    items = [
        {
            "title": "Fed holds rates <steady> & signals cuts",
            "url": "https://www.cbsnews.com/news/fed-rates/",
            "source": "cbsnews.com",
            "preferred": " ★",
        },
        {
            "title": "OpenAI ships a new model",
            "url": "https://techcrunch.com/2026/09/25/openai/",
            "source": "techcrunch.com",
            "preferred": "",
        },
    ]
    body = _format_tool_output(items, news.format, news.empty, news.item_template)
    assert body.splitlines() == [
        "- Fed holds rates <steady> & signals cuts "
        "([cbsnews.com](https://www.cbsnews.com/news/fed-rates/) ★)",
        "- OpenAI ships a new model ([techcrunch.com](https://techcrunch.com/2026/09/25/openai/))",
    ]

    telegram, _meta = to_telegram(body)
    assert telegram.splitlines() == [
        "• Fed holds rates &lt;steady&gt; &amp; signals cuts "
        '(<a href="https://www.cbsnews.com/news/fed-rates/">cbsnews.com</a> ★)',
        "• OpenAI ships a new model "
        '(<a href="https://techcrunch.com/2026/09/25/openai/">techcrunch.com</a>)',
    ]
    html = to_html(body)
    assert '(<a href="https://techcrunch.com/2026/09/25/openai/">techcrunch.com</a>)' in html
    assert html.count("<a ") == 2  # one link per line: the source, never the headline


def test_top_repos_keep_their_links() -> None:
    repos = _brief().slots["top_repos"]
    assert isinstance(repos, BriefToolSlot)
    assert repos.item_template is not None and repos.item_template.startswith("[{title}]({url})")


def test_focus_follows_the_inbox_summary_and_the_footer_closes_the_digest() -> None:
    blocks = [b.strip() for b in _brief().layout.strip().split("\n\n")]
    assert blocks[-1] == "{{learned_yesterday}}"
    summary = next(i for i, b in enumerate(blocks) if "{{email_summary}}" in b)
    assert blocks[summary + 1] == "{{focus}}"


def test_the_footer_is_never_blank() -> None:
    footer = _brief().slots["learned_yesterday"]
    assert isinstance(footer, BriefToolSlot)
    assert footer.empty == "learned yesterday: nothing"


def test_task_and_bill_lines_carry_no_ids_or_codes() -> None:
    """Owner feedback on PR 2: no [t1], (p0) or [credit_card] in the digest's lines."""
    from iris_harness.runtime.handlers.skill_brief import _format_tool_output

    brief = _brief()
    task = {
        "task_id": "t1",
        "title": "Renew car registration",
        "due": "2026-09-25",
        "priority": "0",
        "when": "due today 17:00",
        "flag": "",
        "from": "a@b.example",
    }
    lines = {}
    for name in ("due_today", "open_tasks", "resolved_followups"):
        slot = brief.slots[name]
        assert isinstance(slot, BriefToolSlot)
        lines[name] = _format_tool_output([task], slot.format, slot.empty, slot.item_template)
    assert lines["due_today"] == "- Renew car registration — due today 17:00"
    assert lines["open_tasks"] == "- Renew car registration"
    followups = brief.slots["open_followups"]
    assert isinstance(followups, BriefToolSlot) and "task_id" not in (followups.item_template or "")

    bills = brief.slots["bills_due"]
    assert isinstance(bills, BriefToolSlot)
    assert bills.count_if == "owed"
    row = {
        "label": "[credit_card] Discover Card",
        "name": "Discover Card",
        "amount": "USD 35.00",
        "when": "in 18 days",
        # the rest of the line, as brief_bills_due words it (PR 4: the pushes' words)
        "text": "USD 35.00 (in 18 days)",
        "owed": "yes",
    }
    body = _format_tool_output([row], bills.format, bills.empty, bills.item_template)
    assert body == "- Discover Card: USD 35.00 (in 18 days)"
    for text in (*lines.values(), body):
        assert "[t1]" not in text and "(p0)" not in text and "[credit_card]" not in text


# ─── the Today group: today's items only, overdue once (owner rule 2026-09-25) ───


def test_today_slots_ask_for_todays_items_only() -> None:
    slots = _brief().slots
    plan, due, open_tasks = slots["todays_plan"], slots["due_today"], slots["open_tasks"]
    assert isinstance(plan, BriefToolSlot)
    # Reminders have their own "Reminders today" section (D18: one item, one place).
    assert plan.args == {"include_overdue": False, "include_reminders": False}
    assert isinstance(due, BriefToolSlot) and due.args == {"include_overdue": False}
    assert isinstance(open_tasks, BriefToolSlot)
    assert open_tasks.args == {"skip_due_by_today": True}


def test_overdue_is_one_headed_line_in_the_today_group() -> None:
    from iris_harness.runtime.handlers.skill_brief import _brief_section, _format_tool_output
    from iris_harness.services.digest.settings import load_defaults

    brief = _brief()
    slot = brief.slots["overdue"]
    assert isinstance(slot, BriefToolSlot)
    assert (slot.skill, slot.tool, slot.format) == ("iris-tasks", "list_overdue", "text")
    today = load_defaults(REPO_ROOT / "config").settings.groups[0]
    assert today.id == "today" and "overdue" in today.sections

    text = _format_tool_output(
        "## Overdue (2)\n- Renew registration · Call bank", "text", slot.empty
    )
    section = _brief_section("overdue", slot, text, brief.layout, footer=False)
    assert (section.title, section.items, section.empty) == ("Overdue (2)", 1, False)
    assert section.text == "- Renew registration · Call bank"

    quiet = _brief_section(
        "overdue", slot, _format_tool_output("", "text", slot.empty), brief.layout, footer=False
    )
    assert quiet.empty and quiet.text == "Nothing overdue."


def test_missed_reminders_sit_before_todays_reminders_and_vanish_when_none() -> None:
    """D14/D18 (prototype's Digest tab): "Missed reminders (N)" only when something
    failed, then "Reminders today" read from the one store."""
    from iris_harness.runtime.handlers.brief_formats import grouped_markdown
    from iris_harness.runtime.handlers.skill_brief import _brief_section, _format_tool_output
    from iris_harness.services.digest.settings import load_defaults

    brief = _brief()
    missed, today = brief.slots["missed_reminders"], brief.slots["reminders"]
    assert isinstance(missed, BriefToolSlot) and isinstance(today, BriefToolSlot)
    assert (missed.skill, missed.tool, missed.format) == (
        "calendar-reminders",
        "list_missed_reminders",
        "text",
    )
    assert (today.skill, today.tool, today.item_template) == (
        "calendar-reminders",
        "list_today_reminders",
        "{line}",
    )
    group = load_defaults(REPO_ROOT / "config").settings.groups[0]
    order = list(group.sections)
    assert order.index("overdue") < order.index("missed_reminders") < order.index("reminders")

    text = _format_tool_output(
        "## Missed reminders (1)\n"
        "- Take out the recycling — due Mon 8:00 AM, couldn't be delivered",
        "text",
        missed.empty,
    )
    section = _brief_section("missed_reminders", missed, text, brief.layout, footer=False)
    assert (section.title, section.items, section.empty) == ("Missed reminders (1)", 1, False)

    lines = [{"line": "9:30 AM — Call the pharmacy"}, {"line": "6:00 PM — Swim (every Monday)"}]
    todays = _brief_section(
        "reminders",
        today,
        _format_tool_output(lines, "bullets", today.empty, today.item_template),
        brief.layout,
        footer=False,
    )
    assert todays.title == "Reminders today"
    assert todays.text == "- 9:30 AM — Call the pharmacy\n- 6:00 PM — Swim (every Monday)"

    none = _brief_section(
        "missed_reminders",
        missed,
        _format_tool_output("", "text", missed.empty),
        brief.layout,
        footer=False,
    )
    assert none.empty and none.text == ""
    web = grouped_markdown([(group, [none, todays])])
    assert "Missed" not in web  # folds away entirely: no "nothing missed" line
