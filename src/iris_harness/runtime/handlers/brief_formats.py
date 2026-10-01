"""Per-channel formatters for the canonical Markdown brief body.

Each formatter takes the markdown body produced by `skill_brief` and returns
either a string (the channel-specific body) or a `(body, extra_metadata)`
tuple when the channel needs hints (e.g. Telegram's `parse_mode`).

The morning digest (digest v5) is rendered from its sections instead, grouped by
Settings -> Digest: :func:`grouped_markdown` (the stored web copy),
:func:`grouped_telegram` (one message per group, a card per section) and
:func:`grouped_push_headline` (one count line per group).
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iris_harness.services.digest.settings import DigestGroup


def to_markdown(body: str) -> str:
    """Markdown passthrough — file exports and markdown-aware channels."""
    return body


def to_cli(body: str) -> str:
    """Render the markdown body as aligned plain text for terminal output.

    Uses `rich` if available (already a dependency) to render headers with
    underlines and bullets with proper indentation. Falls back to the raw
    markdown if rich is somehow missing at runtime.
    """
    try:
        from rich.console import Console
        from rich.markdown import Markdown
    except ImportError:  # pragma: no cover — rich is a hard dependency
        return body

    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=88, no_color=True)
    console.print(Markdown(body))
    return buffer.getvalue().rstrip() + "\n"


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_BULLET_RE = re.compile(r"^[-*]\s+(.+)$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def _replace_markdown_links_with_html(text: str) -> str:
    """Convert ``[label](url)`` segments to ``<a href="url">label</a>``.

    Assumes ``text`` has already had its HTML special chars escaped — so the URL
    inside the markdown link is NOT re-escaped here, otherwise ``&amp;`` would
    become ``&amp;amp;``.
    """
    return _LINK_RE.sub(
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        text,
    )


def _escape_html_preserving_links(text: str) -> str:
    """Escape HTML special chars in text while preserving markdown link syntax."""
    escaped = escape(text)
    return _replace_markdown_links_with_html(escaped)


_RULE_RE = re.compile(r"^\s*-{3,}\s*$")
_EMPHASIS_LINE_RE = re.compile(r"^\*([^*]+)\*$")


def to_html(body: str) -> str:
    """Convert the limited markdown subset (headings, bullets, inline links) to HTML."""
    lines = body.splitlines()
    out: list[str] = []
    in_list = False
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            if in_list:
                out.append("</ul>")
                in_list = False
            level = len(heading.group(1))
            text = _escape_html_preserving_links(heading.group(2))
            out.append(f"<h{level}>{text}</h{level}>")
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            text = _escape_html_preserving_links(bullet.group(1))
            out.append(f"  <li>{text}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if _RULE_RE.match(line):
            out.append("<hr>")
            continue
        quiet = _EMPHASIS_LINE_RE.match(line)
        if quiet:
            # The grouped digest's folded "nothing" line (``*No bills due.*``).
            out.append(f"<p><em>{_escape_html_preserving_links(quiet.group(1))}</em></p>")
            continue
        out.append(f"<p>{_escape_html_preserving_links(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def to_telegram(body: str) -> tuple[str, dict[str, str]]:
    """Convert markdown to Telegram's HTML subset.

    Telegram's HTML parse_mode supports ``<b>``, ``<i>``, ``<a href>``, ``<code>``,
    ``<pre>`` — but no headings or lists. Render ``## X`` as ``<b>X</b>`` and bullets
    as ``• item`` (plain-text bullet). Preserves inline ``[text](url)`` links.
    """
    lines = body.splitlines()
    out: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            out.append("")
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            text = _escape_html_preserving_links(heading.group(2))
            out.append(f"<b>{text}</b>")
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            text = _escape_html_preserving_links(bullet.group(1))
            out.append(f"• {text}")
            continue
        out.append(_escape_html_preserving_links(line))
    return "\n".join(out), {"parse_mode": "HTML"}


#: Telegram rejects a message over 4096 characters; stay clear of it so a
#: formatting tweak never tips a chunk over (graph §7: "≤4000-char chunks").
TELEGRAM_CHUNK_LIMIT = 4000

# One token of Telegram HTML: a tag, an entity, or a run of plain text.
_HTML_TOKEN_RE = re.compile(r"<[^>]*>|&#?\w+;|[^<&]+|[<&]")
_TAG_NAME_RE = re.compile(r"^</?\s*([A-Za-z0-9]+)")


def _close_tags(stack: list[tuple[str, str]]) -> str:
    return "".join(f"</{name}>" for name, _ in reversed(stack))


def _split_html_line(line: str, limit: int, first_limit: int | None = None) -> list[str]:
    """Split one over-long line of Telegram HTML without breaking markup.

    Cuts only between tokens (never inside a tag or an entity); any element open
    at a cut is closed at the end of the piece and reopened at the start of the
    next, so every piece parses on its own. ``first_limit`` (when given) caps
    only the first piece, so it can fill the rest of a partly used message.
    """
    pieces: list[str] = []
    stack: list[tuple[str, str]] = []  # (tag name, opening tag text)
    current = ""
    cap = first_limit if first_limit is not None else limit

    def budget() -> int:
        return cap - len(_close_tags(stack))

    def flush() -> None:
        nonlocal current, cap
        pieces.append(current + _close_tags(stack))
        current = "".join(opening for _, opening in stack)
        cap = limit

    for token in _HTML_TOKEN_RE.findall(line):
        is_markup = token.startswith("<") and token.endswith(">") and len(token) > 1
        is_entity = token.startswith("&") and token.endswith(";") and len(token) > 1
        if is_markup or is_entity:
            match = _TAG_NAME_RE.match(token) if is_markup else None
            name = match.group(1).lower() if match else ""
            opening = is_markup and not token.startswith("</") and not token.endswith("/>")
            # An opening tag needs room for its closer and some content, or the
            # piece would end on an empty element.
            needed = len(token) + (len(name) + 4 if opening else 0)
            if len(current) + needed > budget() and current.strip():
                flush()
            current += token
            if is_markup:
                if token.startswith("</"):
                    if stack and stack[-1][0] == name:
                        stack.pop()
                elif name and opening:
                    stack.append((name, token))
            continue
        # Plain text: split anywhere, preferring a space near the cut.
        rest = token
        while rest:
            room = budget() - len(current)
            if len(rest) <= room:
                current += rest
                break
            if room <= 0:
                if current == "".join(opening for _, opening in stack):
                    room = 1  # reopened tags alone fill the budget: progress anyway
                else:
                    flush()
                    continue
            cut = rest.rfind(" ", 0, room + 1)
            if cut <= 0:
                cut = room
            current += rest[:cut]
            rest = rest[cut:].lstrip(" ")
            flush()
    if current.strip() and current != "".join(opening for _, opening in stack):
        pieces.append(current + _close_tags(stack))
    return pieces


def _pack_lines(lines: list[str], limit: int) -> list[str]:
    """Greedily join lines into chunks of at most ``limit`` characters."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if len(line) <= limit:
            parts = [line]
        else:
            # Fill what is left of the current message first (a heading never
            # goes out alone), unless too little is left to be worth it.
            room = limit - size - 1
            parts = _split_html_line(line, limit, room if current and room > limit // 4 else None)
        for part in parts:
            added = len(part) + (1 if current else 0)
            if current and size + added > limit:
                chunks.append("\n".join(current))
                current, size = [], 0
                added = len(part)
            current.append(part)
            size += added
    if current:
        chunks.append("\n".join(current))
    return chunks


def _markdown_sections(body: str) -> list[str]:
    """Split a markdown body into sections: a new one starts at each heading."""
    sections: list[list[str]] = [[]]
    for line in body.splitlines():
        if _HEADING_RE.match(line.rstrip()) and any(s.strip() for s in sections[-1]):
            sections.append([])
        sections[-1].append(line)
    return ["\n".join(lines).strip("\n") for lines in sections if any(s.strip() for s in lines)]


def chunk_telegram(body: str, limit: int = TELEGRAM_CHUNK_LIMIT) -> list[str]:
    """Render ``body`` for Telegram as messages of at most ``limit`` characters.

    Chunks break at section boundaries (a markdown heading), so a section is
    never split across two messages unless it alone is over the limit — then it
    breaks between lines, and a single over-long line breaks between tokens with
    its open tags closed and reopened. The section order is the owner's, so the
    actionable sections lead the first message (graph §7).
    """
    rendered = [to_telegram(section)[0].strip("\n") for section in _markdown_sections(body)]
    chunks: list[str] = []
    current = ""
    for section in rendered:
        if not section:
            continue
        if len(section) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_pack_lines(section.split("\n"), limit))
            continue
        candidate = f"{current}\n\n{section}" if current else section
        if len(candidate) > limit:
            chunks.append(current)
            current = section
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [chunk.strip("\n") for chunk in chunks if chunk.strip()]


#: ``[label](iris:verb/arg)`` — an in-app action (e.g. the Focus line's 👎). Only
#: the web view can act on it; every other rendering drops it entirely.
_ACTION_LINK_RE = re.compile(r"[ \t]*\[[^\]]*\]\(iris:[^)\s]*\)")


def strip_action_links(body: str) -> str:
    """Remove ``iris:`` action links (label included) from a markdown body."""
    return _ACTION_LINK_RE.sub("", body)


#: Sections a push headline names. A notification is a glance (graph §7:
#: "push: 3-line headline"); the detail is one tap away.
PUSH_HEADLINE_SECTIONS = 3


def push_headline(
    counts: list[tuple[str, int]] | tuple[tuple[str, int], ...],
    failed_titles: list[str] | tuple[str, ...] = (),
) -> str:
    """The web-push body: item counts of the first countable sections, in order.

    ``counts`` is ``(section title, item count)`` for every list section in the
    order the digest rendered them — the owner's section order, which puts the
    actionable sections first. A partial digest says so on its own line.
    """
    lines = [f"{title}: {count}" for title, count in list(counts)[:PUSH_HEADLINE_SECTIONS]]
    if not lines:
        lines = ["Your digest is ready."]
    if failed_titles:
        lines.append(f"⚠ partial: {', '.join(failed_titles)} failed")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The grouped digest (digest v5): one rendering per channel, built from sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BriefSection:
    """One rendered section of a brief, in render order (see ``render_brief_result``).

    ``text`` is the section's markdown without its heading (``title`` carries that).
    ``empty`` is True when the slot rendered its ``empty`` text or no items, and
    ``failed`` holds the reason when its slot raised (``text`` is then blank).
    ``footer`` marks the brief's own footer slots, which close every rendering.
    """

    name: str
    title: str
    text: str = ""
    items: int = 0
    empty: bool = False
    failed: str = ""
    footer: bool = False


#: ``(group, its sections in the group's order)`` — what the grouped renderers take.
#: The group is a ``DigestGroup`` (``services.digest.settings``): id, title, icon, push.
GroupedSections = Sequence[tuple["DigestGroup", Sequence[BriefSection]]]


def failure_text(failed: Sequence[tuple[str, str]]) -> str:
    """The "couldn't build" line for ``(title, reason)`` pairs, or ``""`` for none.

    Sections that failed for the same reason share it:
    ``⚠ couldn't build: Portfolio, AI news (timeout after 20 s) — everything else is current.``
    """
    if not failed:
        return ""
    by_reason: dict[str, list[str]] = {}
    for title, reason in failed:
        by_reason.setdefault(reason, []).append(title)
    groups = "; ".join(f"{', '.join(titles)} ({reason})" for reason, titles in by_reason.items())
    return f"⚠ couldn't build: {groups} — everything else is current."


def _group_heading(group: DigestGroup) -> str:
    return f"{group.icon} {group.title}".strip()


def _quiet_phrase(text: str) -> str:
    """A section's empty text as one phrase of the folded line ("No bills due")."""
    phrase = " ".join(line.strip() for line in text.splitlines() if line.strip())
    phrase = phrase.rstrip(" .")
    return phrase[:1].upper() + phrase[1:]


def grouped_markdown(
    groups: GroupedSections,
    *,
    greeting: str = "",
    header: str = "",
    closing: str = "",
    options: Mapping[str, Any] | None = None,
) -> str:
    """The stored (web) copy of a grouped digest, as markdown.

    ``## <icon> <Group>`` per group, ``### <Section>`` + its lines per section that has
    something. Empty sections fold into ONE quiet italic line per group
    (``options["empty"]``: ``fold``, the default; ``hide`` leaves them out; ``show``
    gives each its heading and empty text). A failed section is one ⚠ line under its
    group. ``closing`` (the brief's footer) goes last, after a rule.
    """
    empty_mode = str((options or {}).get("empty") or "fold")
    blocks: list[str] = [part.strip() for part in (header, greeting) if part and part.strip()]
    for group, sections in groups:
        blocks.append(f"## {_group_heading(group)}")
        quiet: list[str] = []
        failed: list[tuple[str, str]] = []
        for section in sections:
            if section.failed:
                failed.append((section.title, section.failed))
            elif section.empty and empty_mode != "show":
                if empty_mode == "fold" and section.text.strip():
                    quiet.append(_quiet_phrase(section.text))
            else:
                text = section.text.strip("\n")
                blocks.append(f"### {section.title}\n{text}" if text else f"### {section.title}")
        if quiet:
            blocks.append(f"*{' · '.join(quiet)}.*")
        line = failure_text(failed)
        if line:
            blocks.append(line)
    if closing.strip():
        blocks.append("---")
        blocks.append(closing.strip())
    return "\n\n".join(blocks)


def _telegram_lines(text: str) -> list[str]:
    """A section's markdown lines as Telegram HTML lines (bullets become •)."""
    out: list[str] = []
    for raw in strip_action_links(text).splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        heading = _HEADING_RE.match(line)
        bullet = _BULLET_RE.match(line)
        if heading:
            out.append(f"<b>{_escape_html_preserving_links(heading.group(2))}</b>")
        elif bullet:
            out.append(f"• {_escape_html_preserving_links(bullet.group(1))}")
        else:
            out.append(_escape_html_preserving_links(line))
    return out


def _telegram_cards(title: str, lines: list[str], *, opening: str, limit: int) -> list[str]:
    """One ``<blockquote>`` card with a bold title; split into several when too long.

    A card that alone is over ``limit`` breaks between lines (each piece keeps the
    title), and a single over-long line breaks between tokens (``_split_html_line``).
    """
    head = f"{opening}<b>{escape(title, quote=False)}</b>"
    close = "</blockquote>"
    budget = max(limit - len(head) - len(close) - 1, 1)
    pieces: list[list[str]] = []
    current: list[str] = []
    size = 0
    for line in lines:
        for part in [line] if len(line) <= budget else _split_html_line(line, budget):
            added = len(part) + 1
            if current and size + added > budget:
                pieces.append(current)
                current, size = [], 0
            current.append(part)
            size += added
    if current or not pieces:
        pieces.append(current)
    return [head + "".join(f"\n{line}" for line in piece) + close for piece in pieces]


def _pack_blocks(blocks: list[str], limit: int) -> list[str]:
    """Join blocks with newlines into messages of at most ``limit`` characters."""
    messages: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n{block}" if current else block
        if current and len(candidate) > limit:
            messages.append(current)
            current = block
        else:
            current = candidate
    if current:
        messages.append(current)
    return messages


#: An optional line under each Telegram group heading (``channels.telegram.rule`` in
#: digest.yaml). Off by default: a fixed-length rule wrapped onto a second line on the
#: owner's phone at its text size (2026-09-25), so each group reads heading → cards.
TELEGRAM_RULE = ""


def grouped_telegram(
    groups: GroupedSections,
    *,
    greeting: str = "",
    header: str = "",
    closing: str = "",
    options: Mapping[str, Any] | None = None,
    inline_keyboard: list[list[dict[str, str]]] | None = None,
    limit: int = TELEGRAM_CHUNK_LIMIT,
) -> list[tuple[str, dict[str, Any]]]:
    """A grouped digest as Telegram messages: a greeting, then one per group.

    Each group opens with its bold icon + title over a thin full-width rule
    (``options["rule"]``, default :data:`TELEGRAM_RULE`); each section is a ``<blockquote>``
    card with a bold title. A group or section id set to ``expandable_card`` in
    ``options`` (``news: expandable_card``) gets ``<blockquote expandable>`` cards.
    Empty sections are left out (``options["empty"]: show`` keeps them) and a group
    with nothing to show is not sent. A failed section is its own ⚠ card. The footer
    (``closing``) ends the last message in italics, and the last message carries
    ``inline_keyboard`` when given. ``iris:`` action links are dropped; every message
    stays within ``limit`` (a group too big for one message continues in the next).
    """
    opts = dict(options or {})
    show_empty = str(opts.get("empty") or "hide") == "show"
    default_style = str(opts.get("section") or "card")
    raw_rule = opts.get("rule", TELEGRAM_RULE)
    rule = escape(str(raw_rule).strip(), quote=False) if raw_rule is not None else ""
    messages: list[str] = []

    intro = [f"<b>{escape(greeting.strip(), quote=False)}</b>"] if greeting.strip() else []
    intro += _telegram_lines(header) if header.strip() else []
    if intro:
        messages.extend(_pack_blocks(intro, limit))

    for group, sections in groups:
        cards: list[str] = []
        for section in sections:
            style = str(opts.get(section.name) or opts.get(group.id) or default_style)
            opening = "<blockquote expandable>" if style == "expandable_card" else "<blockquote>"
            if section.failed:
                reason = f"couldn't build ({section.failed}) — everything else is current"
                cards.extend(
                    _telegram_cards(
                        f"⚠ {section.title}",
                        [escape(reason, quote=False)],
                        opening="<blockquote>",
                        limit=limit,
                    )
                )
                continue
            if section.empty and not show_empty:
                continue
            lines = _telegram_lines(section.text)
            if not lines:
                continue
            cards.extend(_telegram_cards(section.title, lines, opening=opening, limit=limit))
        if cards:
            head = f"<b>{escape(_group_heading(group).upper(), quote=False)}</b>"
            if rule:
                head = f"{head}\n{rule}"
            messages.extend(_pack_blocks([head, *cards], limit))

    if closing.strip():
        footer = "\n".join(f"<i>{line}</i>" for line in _telegram_lines(closing))
        if messages and len(messages[-1]) + 2 + len(footer) <= limit:
            messages[-1] = f"{messages[-1]}\n\n{footer}"
        elif footer:
            messages.extend(_pack_lines(footer.split("\n"), limit))

    out: list[tuple[str, dict[str, Any]]] = [(m, {"parse_mode": "HTML"}) for m in messages]
    if out and inline_keyboard:
        out[-1][1]["inline_keyboard"] = inline_keyboard
    return out


_PUSH_FIELD_RE = re.compile(r"\{(first:)?([a-z][a-z0-9_]*)\}")
#: How much of a section's first line a push template's ``{first:<section>}`` quotes.
PUSH_FIRST_MAX = 40


def _first_item(section: BriefSection) -> str:
    """The section's first line as plain text (links reduced to their label)."""
    for raw in strip_action_links(section.text).splitlines():
        bullet = _BULLET_RE.match(raw.strip())
        if bullet:
            text = _LINK_RE.sub(lambda m: m.group(1), bullet.group(1)).strip()
            if len(text) > PUSH_FIRST_MAX:
                cut = text[: PUSH_FIRST_MAX - 1]
                space = cut.rfind(" ")
                if space > PUSH_FIRST_MAX // 2:
                    cut = cut[:space]
                text = cut.rstrip(" ,;:(—-") + "…"
            return text
    return ""


def _fill_push_template(template: str, sections: Sequence[BriefSection]) -> str:
    """Fill ``{section}`` (item count) and ``{first:section}`` in a group's push line.

    The line is `` · ``-separated parts; a part whose every section counted nothing
    is dropped ("0 events" says nothing), and so is the line when nothing is left.
    """
    by_name = {s.name: s for s in sections if not s.failed}
    kept: list[str] = []
    for part in template.split(" · "):
        names = [m.group(2) for m in _PUSH_FIELD_RE.finditer(part)]
        if names and not any(by_name[n].items for n in names if n in by_name):
            continue

        def fill(match: re.Match[str]) -> str:
            section = by_name.get(match.group(2))
            if section is None:
                return "0" if not match.group(1) else ""
            return _first_item(section) if match.group(1) else str(section.items)

        kept.append(_PUSH_FIELD_RE.sub(fill, part).strip())
    return " · ".join(part for part in kept if part)


def grouped_push_headline(
    groups: GroupedSections,
    *,
    options: Mapping[str, Any] | None = None,
    failed_titles: Sequence[str] = (),
) -> str:
    """The push body of a grouped digest: one count line per group.

    Each group with a ``push`` template and something to count gives one line, its
    icon first, up to ``options["lines"]`` (3) lines; groups in ``options["exclude_groups"]``
    (digest.yaml's key; ``exclude`` also accepted) never appear. A partial digest says so on its own line.
    """
    opts = dict(options or {})
    raw_lines = opts.get("lines")
    max_lines = (
        raw_lines
        if isinstance(raw_lines, int) and not isinstance(raw_lines, bool) and raw_lines > 0
        else PUSH_HEADLINE_SECTIONS
    )
    exclude = {str(item) for item in opts.get("exclude_groups") or opts.get("exclude") or ()}
    lines: list[str] = []
    for group, sections in groups:
        if group.id in exclude or not group.push:
            continue
        text = _fill_push_template(group.push, sections)
        if text:
            lines.append(f"{group.icon} {text}".strip())
        if len(lines) >= max_lines:
            break
    if not lines:
        lines = ["Your digest is ready."]
    if failed_titles:
        lines.append(f"⚠ partial: {', '.join(failed_titles)} failed")
    return "\n".join(lines)


_CHANNEL_FORMATTERS: dict[str, object] = {
    "console": to_cli,
    "cli": to_cli,
    "markdown": to_markdown,
    "file": to_markdown,
    "html": to_html,
    "email": to_html,
    "web": to_html,
    "telegram": to_telegram,
}


def format_for_channel(channel: str, body: str) -> tuple[str, dict[str, str]]:
    """Dispatch a channel name to its formatter. Returns (body, extra_metadata)."""
    fmt = _CHANNEL_FORMATTERS.get(channel.lower(), to_markdown)
    # ``iris:`` action links only work in the web view, which renders the stored
    # markdown itself; anywhere else they are a dead link.
    result = fmt(strip_action_links(body))  # type: ignore[operator]
    if isinstance(result, tuple):
        formatted_body, extra = result
        return str(formatted_body), dict(extra)
    return str(result), {}


def format_messages_for_channel(channel: str, body: str) -> list[tuple[str, dict[str, str]]]:
    """Like :func:`format_for_channel`, but one entry per message to send.

    Telegram caps a message at 4096 characters, so its rendering is chunked at
    section boundaries (:func:`chunk_telegram`); every other channel gets one.
    """
    if channel.lower() == "telegram":
        chunks = chunk_telegram(strip_action_links(body))
        if chunks:
            return [(chunk, {"parse_mode": "HTML"}) for chunk in chunks]
    return [format_for_channel(channel, body)]


__all__ = [
    "PUSH_FIRST_MAX",
    "PUSH_HEADLINE_SECTIONS",
    "TELEGRAM_CHUNK_LIMIT",
    "TELEGRAM_RULE",
    "BriefSection",
    "GroupedSections",
    "chunk_telegram",
    "failure_text",
    "format_for_channel",
    "format_messages_for_channel",
    "grouped_markdown",
    "grouped_push_headline",
    "grouped_telegram",
    "push_headline",
    "strip_action_links",
    "to_cli",
    "to_html",
    "to_markdown",
    "to_telegram",
]
