"""Unit tests for the per-channel brief formatters."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from iris_harness.runtime.handlers.brief_formats import (
    TELEGRAM_CHUNK_LIMIT,
    TELEGRAM_RULE,
    BriefSection,
    chunk_telegram,
    failure_text,
    format_for_channel,
    format_messages_for_channel,
    grouped_markdown,
    grouped_push_headline,
    grouped_telegram,
    push_headline,
    strip_action_links,
    to_cli,
    to_html,
    to_markdown,
    to_telegram,
)
from iris_harness.services.digest.settings import DigestGroup

SAMPLE_BODY = (
    "Good morning. Briefing for Friday, May 15, 2026.\n"
    "\n"
    "## Due reminders\n"
    "- Buy milk — 2026-05-15 16:10 (4:10 PM)\n"
    "\n"
    "## Top GitHub repos\n"
    "- [octocat/Hello-World](https://github.com/octocat/Hello-World) — A friendly repo\n"
    "\n"
    "## Top stocks\n"
    "- AAPL.US APPLE INC — $302.26 (1.46%)\n"
)


def test_to_markdown_is_passthrough() -> None:
    assert to_markdown(SAMPLE_BODY) == SAMPLE_BODY


def test_to_cli_strips_markdown_markers() -> None:
    rendered = to_cli(SAMPLE_BODY)
    assert "##" not in rendered
    assert "Due reminders" in rendered
    assert "Buy milk" in rendered


def test_to_html_converts_headings_bullets_and_links() -> None:
    html = to_html(SAMPLE_BODY)
    assert "<h2>Due reminders</h2>" in html
    assert "<ul>" in html and "</ul>" in html
    assert "<li>Buy milk — 2026-05-15 16:10 (4:10 PM)</li>" in html
    assert '<a href="https://github.com/octocat/Hello-World">octocat/Hello-World</a>' in html
    assert "<p>Good morning. Briefing for Friday, May 15, 2026.</p>" in html


def test_to_html_escapes_entities() -> None:
    html = to_html("- AT&T <signal>")
    assert "AT&amp;T" in html
    assert "&lt;signal&gt;" in html


def test_to_telegram_uses_html_parse_mode_with_links() -> None:
    body, meta = to_telegram(SAMPLE_BODY)
    assert meta == {"parse_mode": "HTML"}
    assert "<b>Due reminders</b>" in body
    assert "<b>Top GitHub repos</b>" in body
    assert "• Buy milk — 2026-05-15 16:10 (4:10 PM)" in body
    assert '<a href="https://github.com/octocat/Hello-World">octocat/Hello-World</a>' in body
    # No leftover markdown headings or list dashes
    assert "##" not in body
    assert not body.startswith("- ")


def test_to_telegram_escapes_text_but_preserves_link_url() -> None:
    body, _ = to_telegram("- [AT&T news](https://example.com/a?b=1&c=2)")
    assert "AT&amp;T news" in body
    assert 'href="https://example.com/a?b=1&amp;c=2"' in body


def test_format_for_channel_routes_correctly() -> None:
    cli_body, cli_meta = format_for_channel("console", SAMPLE_BODY)
    assert "##" not in cli_body
    assert cli_meta == {}

    html_body, html_meta = format_for_channel("email", SAMPLE_BODY)
    assert "<h2>" in html_body
    assert html_meta == {}

    tg_body, tg_meta = format_for_channel("telegram", SAMPLE_BODY)
    assert tg_meta == {"parse_mode": "HTML"}
    assert "<b>" in tg_body

    md_body, md_meta = format_for_channel("markdown", SAMPLE_BODY)
    assert md_body == SAMPLE_BODY
    assert md_meta == {}


def test_format_for_channel_unknown_falls_back_to_markdown() -> None:
    body, meta = format_for_channel("nonexistent-channel", SAMPLE_BODY)
    assert body == SAMPLE_BODY
    assert meta == {}


# ---------------------------------------------------------------------------
# Loop-proof PR 2: iris: action links, Telegram chunks, push headline
# ---------------------------------------------------------------------------

FOCUS_BODY = (
    "## Inbox summary\n"
    "- Mo Harbor School · Harbor school on Saturday — personal\n"
    "- Fabrikam · Travel smarter — finance [👎](iris:not-useful/offers%40fabrikam.test)\n"
    "- [octocat/Hello-World](https://github.com/octocat/Hello-World) — a repo\n"
)


def test_strip_action_links_drops_the_thumbs_down_token_and_keeps_https_links() -> None:
    out = strip_action_links(FOCUS_BODY)
    assert "iris:" not in out
    assert "👎" not in out
    assert "- Fabrikam · Travel smarter — finance\n" in out
    assert "[octocat/Hello-World](https://github.com/octocat/Hello-World)" in out


def test_every_non_web_formatter_drops_action_links() -> None:
    for channel in ("telegram", "console", "email", "markdown", "web_push"):
        body, _ = format_for_channel(channel, FOCUS_BODY)
        assert "iris:" not in body, channel
        assert "👎" not in body, channel
    telegram, _ = format_for_channel("telegram", FOCUS_BODY)
    assert '<a href="https://github.com/octocat/Hello-World">' in telegram


def _section(title: str, lines: int, width: int = 60) -> str:
    return f"## {title}\n" + "\n".join(f"- {title} item {i} " + "x" * width for i in range(lines))


def _open_tags_balanced(chunk: str) -> bool:
    opened = re.findall(r"<(b|a|i|code)\b[^>]*>", chunk)
    closed = re.findall(r"</(b|a|i|code)>", chunk)
    return sorted(opened) == sorted(closed)


def test_a_short_digest_is_one_telegram_message() -> None:
    chunks = chunk_telegram(SAMPLE_BODY)
    assert len(chunks) == 1
    assert chunks[0] == to_telegram(SAMPLE_BODY)[0].strip("\n")


def test_telegram_chunks_break_at_section_boundaries_under_the_limit() -> None:
    body = "Good morning.\n\n" + "\n\n".join(
        _section(name, 30) for name in ("Bills due", "Todays events", "AI news", "Portfolio")
    )
    chunks = chunk_telegram(body, limit=4000)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 4000
        # Each chunk starts at a section (or the greeting), never mid-list.
        assert chunk.startswith("<b>") or chunk.startswith("Good morning.")
    # Order preserved: the owner's section order is the message order.
    joined = "\n".join(chunks)
    positions = [joined.index(f"<b>{t}</b>") for t in ("Bills due", "Todays events", "AI news")]
    assert positions == sorted(positions)
    # Nothing lost: every item made it into some chunk.
    assert joined.count("item ") == 4 * 30


def test_an_oversized_section_splits_at_line_boundaries() -> None:
    body = _section("Huge", 200)  # ~15k characters in one section
    chunks = chunk_telegram(body, limit=4000)
    assert len(chunks) >= 4
    for chunk in chunks:
        assert len(chunk) <= 4000
        for line in chunk.splitlines():
            assert line.startswith("• Huge item") or line == "<b>Huge</b>"


def test_an_oversized_line_never_splits_a_tag_or_an_entity() -> None:
    link = "[a &amp; b](https://example.com/?q=1&r=2)"
    body = "## Links\n- " + " ".join([link] * 400) + " & more <stuff>"
    chunks = chunk_telegram(body, limit=500)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 500
        assert _open_tags_balanced(chunk), chunk
        # No dangling entity or half tag at either end.
        assert not re.search(r"&[#\w]*$", chunk)
        assert not re.search(r"<[^>]*$", chunk)
        assert not re.match(r"^[^<]*>", chunk)


def test_format_messages_for_telegram_chunks_and_keeps_parse_mode() -> None:
    body = "\n\n".join(_section(f"S{i}", 40) for i in range(4))
    messages = format_messages_for_channel("telegram", body)
    assert len(messages) > 1
    assert all(meta == {"parse_mode": "HTML"} for _, meta in messages)
    assert all(len(text) <= 4000 for text, _ in messages)


def test_format_messages_for_other_channels_is_one_message() -> None:
    assert format_messages_for_channel("console", SAMPLE_BODY) == [
        format_for_channel("console", SAMPLE_BODY)
    ]


def test_push_headline_names_the_first_three_list_sections_in_order() -> None:
    headline = push_headline(
        [("Bills due", 3), ("Today's events", 0), ("Awaiting reply", 1), ("AI news", 5)]
    )
    assert headline == "Bills due: 3\nToday's events: 0\nAwaiting reply: 1"


def test_push_headline_flags_a_partial_digest() -> None:
    headline = push_headline([("Bills due", 2)], ["Portfolio", "AI news"])
    assert headline.splitlines() == ["Bills due: 2", "⚠ partial: Portfolio, AI news failed"]


def test_push_headline_without_list_sections_still_says_something() -> None:
    assert push_headline([]) == "Your digest is ready."


# ---------------------------------------------------------------------------
# The grouped digest (v5): web markdown, Telegram messages, push headline
# ---------------------------------------------------------------------------

TODAY = DigestGroup(
    id="today",
    title="Today",
    icon="☀️",
    sections=("todays_events", "due_today", "reminders", "open_tasks"),
    push="{todays_events} event · {due_today} due today · {open_tasks} tasks",
)
MONEY = DigestGroup(
    id="money",
    title="Money",
    icon="💳",
    sections=("bills_due", "unusual_spend"),
    push="{bills_due} bill due ({first:bills_due})",
)
INBOX = DigestGroup(
    id="inbox", title="Inbox", icon="📬", sections=("focus",), push="{focus} in Focus"
)
NEWS = DigestGroup(id="news", title="News", icon="📰", sections=("news_ai", "news_global"))


def _sec(name: str, title: str, lines: list[str], *, empty_text: str = "") -> BriefSection:
    if not lines:
        return BriefSection(name=name, title=title, text=empty_text, items=0, empty=True)
    return BriefSection(
        name=name, title=title, text="\n".join(f"- {line}" for line in lines), items=len(lines)
    )


def _groups(*, failed_global: bool = False, quiet: bool = False) -> list:
    events = [] if quiet else ["16:30–17:15 Parent-teacher meeting"]
    due = [] if quiet else ["Renew car registration — due today 17:00"]
    tasks = [] if quiet else ["Renew car registration", "Book dentist follow-up"]
    bills = [] if quiet else ["Discover Card: $35.00 min due Oct 13 (in 18 days)"]
    news_global = (
        BriefSection(name="news_global", title="Global", failed="search timed out")
        if failed_global
        else _sec(
            "news_global",
            "Global",
            ["Global bond selloff deepens ([reuters.com](https://r.example/1))"],
        )
    )
    return [
        (
            TODAY,
            [
                _sec("todays_events", "Events", events, empty_text="No events today."),
                _sec("due_today", "Due today", due, empty_text="Nothing due today."),
                _sec("reminders", "Reminders", [], empty_text="No reminders due."),
                _sec("open_tasks", "Open tasks", tasks, empty_text="No open tasks."),
            ],
        ),
        (
            MONEY,
            [
                _sec("bills_due", "Bills due", bills, empty_text="No bills due."),
                _sec("unusual_spend", "Unusual spend", [], empty_text="Spending looks normal."),
            ],
        ),
        (
            INBOX,
            [
                _sec(
                    "focus",
                    "Focus — personal, newest 5",
                    [
                        "Mo Harbor School · Saturday class [👎](iris:not-useful/mo%40school.org)",
                        "Northwind Bank · RM call <urgent> [👎](iris:not-useful/rm%40northwind.test)",
                    ],
                )
            ],
        ),
        (
            NEWS,
            [
                _sec(
                    "news_ai",
                    "AI / Tech",
                    ["AI moves at the speed of trust ([cnbc.com](https://cnbc.example/a?b=1&c=2))"],
                ),
                news_global,
            ],
        ),
    ]


class _TagBalance(HTMLParser):
    """Checks Telegram HTML nests: every opened tag closes, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.ok = True

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack.pop() != tag:
            self.ok = False


def _balanced(html: str) -> bool:
    parser = _TagBalance()
    parser.feed(html)
    parser.close()
    return parser.ok and not parser.stack


def test_grouped_markdown_heads_each_group_and_folds_the_empty_sections() -> None:
    body = grouped_markdown(
        _groups(),
        greeting="Good morning. Briefing for Friday, September 25, 2026.",
        closing="learned yesterday: nothing",
    )
    assert body.startswith("Good morning. Briefing for Friday")
    assert "## ☀️ Today\n\n### Events\n- 16:30–17:15 Parent-teacher meeting" in body
    assert "### Reminders" not in body  # empty: folded, not headed
    assert "*No reminders due.*" in body
    assert "*Spending looks normal.*" in body
    # The Focus 👎 action link stays in the stored copy; news keeps its source link.
    assert "[👎](iris:not-useful/mo%40school.org)" in body
    assert "([cnbc.com](https://cnbc.example/a?b=1&c=2))" in body
    # Groups in order, the footer last after a rule.
    order = [body.index(h) for h in ("## ☀️ Today", "## 💳 Money", "## 📬 Inbox", "## 📰 News")]
    assert order == sorted(order)
    assert body.endswith("---\n\nlearned yesterday: nothing")


def test_grouped_markdown_folds_every_empty_section_of_a_group_into_one_line() -> None:
    body = grouped_markdown(_groups(quiet=True))
    assert "*No events today · Nothing due today · No reminders due · No open tasks.*" in body
    assert "*No bills due · Spending looks normal.*" in body


def test_grouped_markdown_names_a_failed_section_under_its_group() -> None:
    body = grouped_markdown(_groups(failed_global=True))
    news = body[body.index("## 📰 News") :]
    assert "⚠ couldn't build: Global (search timed out) — everything else is current." in news
    assert "### Global" not in body


def test_grouped_markdown_empty_option_hide_and_show() -> None:
    hidden = grouped_markdown(_groups(), options={"empty": "hide"})
    assert "No reminders due" not in hidden
    shown = grouped_markdown(_groups(), options={"empty": "show"})
    assert "### Reminders\nNo reminders due." in shown


def test_grouped_telegram_sends_a_greeting_then_one_message_per_group() -> None:
    messages = grouped_telegram(
        _groups(), greeting="Good morning.", closing="learned yesterday: nothing"
    )
    bodies = [body for body, _ in messages]
    assert bodies[0] == "<b>Good morning.</b>"
    assert bodies[1].startswith("<b>☀️ TODAY</b>\n<blockquote><b>Events</b>\n• 16:30")
    assert all(meta["parse_mode"] == "HTML" for _, meta in messages)
    assert all(_balanced(body) for body in bodies)
    # Empty sections are left out, the footer closes the last message in italics.
    assert "Reminders" not in bodies[1]
    assert bodies[-1].endswith("<i>learned yesterday: nothing</i>")
    # iris: action links never reach Telegram; text is escaped, links become <a>.
    joined = "\n".join(bodies)
    assert "iris:" not in joined and "👎" not in joined
    assert "RM call &lt;urgent&gt;" in joined
    assert '(<a href="https://cnbc.example/a?b=1&amp;c=2">cnbc.com</a>)' in joined


def test_grouped_telegram_has_no_rule_by_default_but_one_can_be_set() -> None:
    """A fixed-length rule wrapped on the owner's phone (2026-09-25), so it is off by
    default; ``channels.telegram.rule`` can still set one."""
    assert TELEGRAM_RULE == ""
    bodies = [body for body, _ in grouped_telegram(_groups(), greeting="Hi")][1:]
    assert bodies and all(body.split("\n")[1].startswith("<blockquote") for body in bodies)
    custom = [body for body, _ in grouped_telegram(_groups(), options={"rule": "— <x> —"})]
    assert all(body.split("\n")[1] == "— &lt;x&gt; —" for body in custom)
    assert all(_balanced(body) for body in custom)
    bare = [body for body, _ in grouped_telegram(_groups(), options={"rule": ""})]
    assert bare[0].startswith("<b>☀️ TODAY</b>\n<blockquote>")


def test_shipped_telegram_rule_is_the_default() -> None:
    from iris_harness.services.digest.settings import load_defaults

    repo_config = Path(__file__).resolve().parents[5] / "config"
    telegram = load_defaults(repo_config).settings.channels["telegram"]
    assert telegram["rule"] == TELEGRAM_RULE


def test_grouped_telegram_skips_a_group_with_nothing_to_show() -> None:
    bodies = [body for body, _ in grouped_telegram(_groups(quiet=True), greeting="Hi")]
    assert not any("TODAY" in body or "MONEY" in body for body in bodies)
    assert any("📬 INBOX" in body for body in bodies)


def test_grouped_telegram_uses_expandable_cards_where_the_options_say() -> None:
    bodies = [body for body, _ in grouped_telegram(_groups(), options={"news": "expandable_card"})]
    news = next(body for body in bodies if "📰 NEWS" in body)
    assert news.count("<blockquote expandable><b>") == 2
    assert "<blockquote expandable>" not in "".join(b for b in bodies if b is not news)


def test_grouped_telegram_makes_a_failed_section_its_own_warning_card() -> None:
    bodies = [body for body, _ in grouped_telegram(_groups(failed_global=True))]
    news = next(body for body in bodies if "📰 NEWS" in body)
    assert (
        "<blockquote><b>⚠ Global</b>\ncouldn't build (search timed out) "
        "— everything else is current</blockquote>" in news
    )


def test_grouped_telegram_puts_the_buttons_on_the_last_message_only() -> None:
    keyboard = [[{"text": "📄 Full digest", "url": "https://iris.example/digest/abc"}]]
    messages = grouped_telegram(_groups(), greeting="Hi", inline_keyboard=keyboard)
    assert messages[-1][1]["inline_keyboard"] == keyboard
    assert all("inline_keyboard" not in meta for _, meta in messages[:-1])


def test_grouped_telegram_splits_an_oversized_group_into_valid_messages() -> None:
    big = [
        (
            NEWS,
            [
                _sec("news_ai", "AI / Tech", [f"STORY{i:03d} " + "x" * 90 for i in range(150)]),
                _sec("news_global", "Global", ["one more"]),
            ],
        )
    ]
    messages = grouped_telegram(big, options={"news": "expandable_card"})
    bodies = [body for body, _ in messages]
    assert len(bodies) > 1
    assert all(len(body) <= TELEGRAM_CHUNK_LIMIT for body in bodies)
    assert all(_balanced(body) for body in bodies)
    numbers = [int(n) for n in re.findall(r"STORY(\d{3})", "\n".join(bodies))]
    assert numbers == list(range(150))


def test_grouped_push_has_one_count_line_per_group_from_its_template() -> None:
    text = grouped_push_headline(_groups(), options={"lines": 3, "exclude": ["news"]})
    assert text == (
        "☀️ 1 event · 1 due today · 2 tasks\n"
        "💳 1 bill due (Discover Card: $35.00 min due Oct 13…)\n"
        "📬 2 in Focus"
    )


def test_grouped_push_drops_zero_parts_and_groups_and_caps_the_lines() -> None:
    assert grouped_push_headline(_groups(quiet=True)) == "📬 2 in Focus"
    assert grouped_push_headline(_groups(), options={"lines": 1}).count("\n") == 0
    assert grouped_push_headline(_groups(), options={"exclude": ["today", "money", "inbox"]}) == (
        "Your digest is ready."
    )


def test_grouped_push_flags_a_partial_digest() -> None:
    text = grouped_push_headline(_groups(failed_global=True), failed_titles=["Global"])
    assert text.endswith("\n⚠ partial: Global failed")


def test_failure_text_groups_titles_by_reason() -> None:
    assert failure_text([("A", "x"), ("B", "x"), ("C", "y")]) == (
        "⚠ couldn't build: A, B (x); C (y) — everything else is current."
    )
    assert failure_text([]) == ""


def test_to_html_renders_the_grouped_digest_rule_and_quiet_line() -> None:
    html = to_html(grouped_markdown(_groups(quiet=True), closing="learned yesterday: nothing"))
    assert "<h2>☀️ Today</h2>" in html
    assert "<p><em>No bills due · Spending looks normal.</em></p>" in html
    assert "<hr>\n<p>learned yesterday: nothing</p>" in html
