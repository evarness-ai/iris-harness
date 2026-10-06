"""Generic `skill_brief` heartbeat handler.

Renders a brief skill's declarative layout by resolving each slot against
either a literal format string or a tool exposed by another loaded skill,
then dispatches the rendered body via the channel gateway.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from re import findall, match, sub
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

from iris_harness.foundation.clock import local_now
from iris_harness.foundation.public_url import PUBLIC_URL_ENV as _PUBLIC_URL_ENV
from iris_harness.foundation.public_url import public_base_url
from iris_harness.kernel.governance.external_content import (
    MARKER,
    REDACTION_NOTICE,
    add_redaction_notice,
)
from iris_harness.runtime.external_text import redact_external_text
from iris_harness.runtime.handlers.brief_formats import (
    BriefSection,
    failure_text,
    format_messages_for_channel,
    grouped_markdown,
    grouped_push_headline,
    grouped_telegram,
    push_headline,
)
from iris_harness.runtime.harness_services import SkillRegistryService
from iris_harness.services.channels import ChannelMessage
from iris_harness.services.channels.models import DeliveryStatus
from iris_harness.services.digests import FailedSection, StoredDigest, shared_digest_store
from iris_harness.services.heartbeat import HeartbeatDefinition, HeartbeatRun, HeartbeatStatus
from iris_harness.services.heartbeat.scheduler import HeartbeatHandler
from iris_harness.tools.skills.models import (
    BriefLiteralSlot,
    BriefSlot,
    BriefSpec,
    BriefToolSlot,
    SkillPackage,
    pick_greeting,
)

if TYPE_CHECKING:
    from iris_harness.runtime.bootstrap import IrisRuntime
    from iris_harness.services.digest.settings import DigestGroup, DigestSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlotContext:
    """Per-render context passed to every slot resolver."""

    now: datetime
    tool_index: dict[tuple[str, str], type[BaseTool]]
    greeting: str = ""
    daypart: str = ""
    #: ``(skill, tool)`` pairs whose manifest declares ``content: external``; a slot over one
    #: has its text passed through the external-content tripwire (see ``_resolve_tool_counted``).
    external_tools: frozenset[tuple[str, str]] = frozenset()


def _build_tool_index(
    skill_registry: SkillRegistryService,
) -> dict[tuple[str, str], type[BaseTool]]:
    """Index loadable tool classes by (skill_name, tool_name)."""
    index: dict[tuple[str, str], type[BaseTool]] = {}
    for package in skill_registry.list_packages(only_loadable=True):
        for tool_manifest, tool_class in zip(
            package.manifest.tools, package.tool_classes, strict=False
        ):
            index[(package.manifest.name, tool_manifest.name)] = tool_class
    return index


def _build_external_index(skill_registry: SkillRegistryService) -> frozenset[tuple[str, str]]:
    """``(skill, tool)`` of every loadable tool that declares ``content: external``."""
    return frozenset(
        (package.manifest.name, tool_manifest.name)
        for package in skill_registry.list_packages(only_loadable=True)
        for tool_manifest in package.manifest.tools
        if tool_manifest.content == "external"
    )


def build_slot_context(
    skill_registry: SkillRegistryService,
    *,
    now: datetime,
    greeting: str = "",
    daypart: str = "",
) -> SlotContext:
    """The one way to build a :class:`SlotContext`, so no render path misses the declarations."""
    return SlotContext(
        now=now,
        tool_index=_build_tool_index(skill_registry),
        greeting=greeting,
        daypart=daypart,
        external_tools=_build_external_index(skill_registry),
    )


def _resolve_literal(slot: BriefLiteralSlot, ctx: SlotContext) -> str:
    """Render a literal slot using the limited placeholder set."""
    return slot.value.format(
        today=ctx.now,
        weekday=ctx.now.strftime("%A"),
        date=ctx.now,
        greeting=ctx.greeting,
        daypart=ctx.daypart,
    )


def brief_greeting(spec: BriefSpec, now: datetime) -> tuple[str, str]:
    """``(greeting, daypart)`` for ``now`` from the brief's greeting table ("", "" if none)."""
    row = pick_greeting(spec.greetings, now.strftime("%H:%M"))
    return (row.greeting, row.daypart) if row is not None else ("", "")


def brief_subject(spec: BriefSpec, now: datetime) -> str:
    """The subject with ``{daypart}`` / ``{greeting}`` filled; as written if it can't be."""
    greeting, daypart = brief_greeting(spec, now)
    try:
        return " ".join(spec.subject.format(greeting=greeting, daypart=daypart).split())
    except (KeyError, IndexError, ValueError):
        return spec.subject


def _resolve_tool(
    slot: BriefToolSlot,
    ctx: SlotContext,
    uses: tuple[str, ...],
    *,
    line_cap: int | None = None,
) -> str:
    """Invoke a skill tool and format its result for inclusion in the layout."""
    return _resolve_tool_counted(slot, ctx, uses, line_cap=line_cap)[0]


def _resolve_tool_counted(
    slot: BriefToolSlot,
    ctx: SlotContext,
    uses: tuple[str, ...],
    *,
    line_cap: int | None = None,
) -> tuple[str, int | None]:
    """Invoke a skill tool: its formatted text, and how many of its items count.

    The count is ``None`` unless the slot sets ``count_if`` (then: the rendered dict
    items whose ``count_if`` value is truthy), so the caller counts bullets instead.
    """
    if slot.skill not in uses:
        raise ValueError(
            f"slot invokes skill {slot.skill!r} which is not in brief.uses {list(uses)!r}"
        )
    tool_class = ctx.tool_index.get((slot.skill, slot.tool))
    if tool_class is None:
        raise ValueError(f"tool not found: {slot.skill}.{slot.tool}")
    instance = tool_class()
    result = instance.invoke(slot.args)
    text = _format_tool_output(
        result, slot.format, slot.empty, slot.item_template, line_cap=line_cap
    )
    if (slot.skill, slot.tool) in ctx.external_tools:
        # Text a third party wrote (a subject line, a feed item) goes to the owner's digest,
        # Telegram, web push or chat from here: tripwire only, no envelope (owner channel).
        text = redact_external_text(text, skill=slot.skill, tool=slot.tool)
    return text, _counted_items(result, slot, line_cap=line_cap)


def _counted_items(value: Any, slot: BriefToolSlot, *, line_cap: int | None) -> int | None:
    """The items of a ``count_if`` bullets slot that count; ``None`` when it has none."""
    key = slot.count_if
    if not key or slot.format != "bullets" or not isinstance(value, (list, tuple)):
        return None
    items = list(value)
    if line_cap is not None and line_cap > 0:
        items = items[:line_cap]
    return sum(1 for item in items if isinstance(item, dict) and item and item.get(key))


def _format_tool_output(
    value: Any,
    fmt: str,
    empty: str,
    item_template: str | None = None,
    *,
    line_cap: int | None = None,
) -> str:
    """Render a tool's return value according to the slot's `format`.

    ``line_cap`` (when set) limits the number of bullet items rendered for a
    ``bullets`` slot — the per-routine ``content_lines_per_item`` / per-section
    cap. It is ignored for non-list formats.
    """
    if value is None or value == [] or value == "" or value == {}:
        return empty
    if fmt == "json":
        return json.dumps(value, indent=2, default=str)
    if fmt == "bullets":
        return _format_bullets(value, empty, item_template, line_cap=line_cap)
    text = str(value).strip()
    return text or empty


def _format_bullets(
    value: Any,
    empty: str,
    item_template: str | None = None,
    *,
    line_cap: int | None = None,
) -> str:
    """Render a list-like value as a bullet list, or fall back to text."""
    if isinstance(value, (list, tuple)):
        if not value:
            return empty
        if line_cap is not None and line_cap > 0:
            value = list(value)[:line_cap]
        lines: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if not item:
                    continue
                if item_template is not None:
                    try:
                        rendered = item_template.format(**item)
                    except KeyError as exc:
                        raise ValueError(
                            f"item_template references missing key {exc} in item {item!r}"
                        ) from exc
                    lines.append(f"- {rendered}")
                else:
                    first = next(iter(item.values()))
                    lines.append(f"- {first}")
            else:
                lines.append(f"- {item}")
        return "\n".join(lines) if lines else empty
    text = str(value).strip()
    return f"- {text}" if text else empty


def resolve_slot(
    slot: BriefSlot,
    ctx: SlotContext,
    uses: tuple[str, ...],
    *,
    line_cap: int | None = None,
) -> str:
    """Dispatch a single slot to its kind-specific resolver."""
    if isinstance(slot, BriefLiteralSlot):
        return _resolve_literal(slot, ctx)
    if isinstance(slot, BriefToolSlot):
        return _resolve_tool(slot, ctx, uses, line_cap=line_cap)
    raise TypeError(f"unsupported slot kind: {type(slot).__name__}")


_PLACEHOLDER_PATTERN = r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}"


def render_layout(layout: str, slots: dict[str, str]) -> str:
    """Substitute `{{name}}` placeholders with their rendered slot values."""

    def replace(match: Any) -> str:
        name = match.group(1)
        if name not in slots:
            raise KeyError(f"layout placeholder has no slot: {name}")
        return slots[name]

    return sub(_PLACEHOLDER_PATTERN, replace, layout)


def _find_brief_package(skill_registry: SkillRegistryService, skill_id: str) -> SkillPackage | None:
    """Locate a loaded brief skill by its manifest name."""
    for package in skill_registry.list_packages(only_loadable=True):
        if package.manifest.kind == "brief" and package.manifest.name == skill_id:
            return package
    return None


def _block_placeholders(block: str) -> list[str]:
    """Return the ``{{slot}}`` names referenced in a layout block, in order."""
    return findall(_PLACEHOLDER_PATTERN, block)


def _filter_layout(
    layout: str,
    slots: dict[str, BriefSlot],
    selected: tuple[str, ...],
    section_order: tuple[str, ...] | None = None,
) -> tuple[str, list[str]]:
    """Prune a brief layout to the user-selected tool sections.

    The layout is split into blank-line-separated blocks (each typically a
    ``## Header`` + a single ``{{slot}}``). A block is kept when it references
    no tool slot (greeting / literals / static headers) or when **every** tool
    slot it references is in ``selected``. Tool blocks are reordered to match
    ``section_order`` when given; leading non-tool blocks (the greeting) stay
    pinned to the top. Fail-open: a block whose placeholders can't be cleanly
    attributed is kept rather than dropped, so a scheduled run never loses the
    greeting or crashes.

    Returns ``(filtered_layout, needed_slot_names)`` where ``needed_slot_names``
    are exactly the slots referenced by the retained blocks (so the caller
    resolves — and invokes tools for — only those).
    """
    selected_set = set(selected)
    blocks = layout.split("\n\n")

    pinned: list[tuple[int, str]] = []  # leading literal/static blocks (greeting)
    tool_blocks: list[tuple[str, str]] = []  # (primary tool slot, block text)
    for index, block in enumerate(blocks):
        placeholders = _block_placeholders(block)
        tool_names = [name for name in placeholders if isinstance(slots.get(name), BriefToolSlot)]
        if not tool_names:
            pinned.append((index, block))
            continue
        if all(name in selected_set for name in tool_names):
            tool_blocks.append((tool_names[0], block))
        # else: drop the block entirely (unselected section)

    if section_order:
        order_index = {name: pos for pos, name in enumerate(section_order)}
        tool_blocks.sort(key=lambda pair: order_index.get(pair[0], len(order_index)))

    # Reassemble: static blocks (greeting/headers/footer) that appeared before
    # the first tool block lead; static blocks after it trail. Tool blocks sit
    # between, in selected/ordered order.
    first_tool_index = next(
        (
            i
            for i, b in enumerate(blocks)
            if any(isinstance(slots.get(n), BriefToolSlot) for n in _block_placeholders(b))
        ),
        len(blocks),
    )
    leading = [block for idx, block in pinned if idx < first_tool_index]
    trailing = [block for idx, block in pinned if idx >= first_tool_index]

    ordered = leading + [block for _, block in tool_blocks] + trailing
    filtered_layout = "\n\n".join(ordered)

    needed = [name for block in ordered for name in _block_placeholders(block) if name in slots]
    return filtered_layout, needed


def _slot_line_cap(
    name: str,
    slot: BriefSlot,
    content_lines_per_item: int | None,
    section_line_caps: dict[str, int] | None,
) -> int | None:
    """Resolve the bullet cap for one slot: per-section override beats global."""
    if section_line_caps and name in section_line_caps:
        return section_line_caps[name]
    return content_lines_per_item


@dataclass(frozen=True)
class BriefRender:
    """One render of a brief: the body plus what the renderings need from it.

    ``failed`` names every section whose slot raised — the body already carries
    the "couldn't build" line for them. ``counts`` is ``(section title, items)``
    for each list section that rendered, in body order, for the push headline.

    ``sections`` is every tool section in render order (failed ones included, the
    footer slots last-marked with ``footer``), so a channel can be rendered from the
    sections instead of by re-reading the markdown. ``greeting`` is the layout's
    leading text (before the first section), ``header`` the routine's header and
    ``closing`` the footer: the footer slots, then the routine's footer.
    """

    body: str
    failed: tuple[FailedSection, ...] = ()
    counts: tuple[tuple[str, int], ...] = ()
    sections: tuple[BriefSection, ...] = ()
    greeting: str = ""
    header: str = ""
    closing: str = ""


#: How much of an exception message a failure line quotes.
_REASON_MAX = 80


def _installed_skills(skill_registry: SkillRegistryService) -> frozenset[str]:
    """Every discovered skill package's name, loadable or not.

    A package that was discovered but failed to load is installed-and-broken: its
    slots fail loudly. Only a name with no package at all is absent.
    """
    return frozenset(p.manifest.name for p in skill_registry.list_packages())


def _short_reason(exc: BaseException) -> str:
    """First line of the exception message, or its type when there is none."""
    text = str(exc).strip()
    reason = text.splitlines()[0].strip() if text else type(exc).__name__
    if len(reason) > _REASON_MAX:
        reason = reason[: _REASON_MAX - 1].rstrip() + "…"
    return reason


def _section_title(layout: str, name: str) -> str:
    """The ``## Heading`` of the layout block that holds ``{{name}}``.

    Falls back to the slot name made readable, so a failure is always named.
    """
    for block in layout.split("\n\n"):
        if name not in _block_placeholders(block):
            continue
        for line in block.splitlines():
            heading = match(r"^#{1,6}\s+(.+)$", line.strip())
            if heading:
                return heading.group(1).strip()
    readable = name.replace("_", " ").strip()
    return readable[:1].upper() + readable[1:] if readable else name


def _drop_blocks(layout: str, names: set[str]) -> str:
    """Remove every layout block that references one of ``names``."""
    kept = [
        block
        for block in layout.split("\n\n")
        if not names.intersection(_block_placeholders(block))
    ]
    return "\n\n".join(kept)


def _split_footer(layout: str, footer_slots: set[str]) -> tuple[str, str]:
    """Split off the blocks that render a footer slot, keeping their order.

    The brief's own footer (``BriefSpec.footer_slots``) always closes it, after
    the "couldn't build" line.
    """
    if not footer_slots:
        return layout, ""
    body: list[str] = []
    tail: list[str] = []
    for block in layout.split("\n\n"):
        (tail if footer_slots.intersection(_block_placeholders(block)) else body).append(block)
    return "\n\n".join(body), "\n\n".join(tail)


def failure_line(failed: tuple[FailedSection, ...] | list[FailedSection]) -> str:
    """The one line a partial digest carries (prototype wording, D6).

    Sections that failed for the same reason share it:
    ``⚠ couldn't build: Portfolio, AI news (timeout after 20 s) — everything else is current.``
    """
    return failure_text([(item.title, item.reason) for item in failed])


def _bullet_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith("- "))


def _brief_section(
    name: str, slot: BriefToolSlot, text: str, layout: str, *, footer: bool
) -> BriefSection:
    """One rendered tool slot as a :class:`BriefSection`.

    A slot that writes its own heading (Focus: ``## Focus — …`` then bullets) is
    titled by it; any other by its layout heading. It is empty when it rendered its
    ``empty`` text, or no items where items were expected (a ``bullets`` slot, or a
    headed text slot with nothing under the heading).
    """
    lines = text.strip("\n").splitlines()
    own = match(r"^#{1,6}\s+(.+)$", lines[0].strip()) if lines else None
    if own is not None:
        title = own.group(1).strip()
        body = "\n".join(lines[1:]).strip("\n")
    else:
        title = _section_title(layout, name)
        body = "\n".join(lines)
    empty_lines = slot.empty.strip("\n").splitlines()
    if empty_lines and match(r"^#{1,6}\s+", empty_lines[0].strip()):
        empty_lines = empty_lines[1:]
    items = _bullet_count(body)
    empty = body.strip() == "\n".join(empty_lines).strip() or (
        items == 0 and (slot.format == "bullets" or own is not None)
    )
    return BriefSection(name=name, title=title, text=body, items=items, empty=empty, footer=footer)


def _leading_text(layout: str, slots: dict[str, BriefSlot], resolved: dict[str, str]) -> str:
    """The layout's blocks before its first tool section (the greeting), rendered."""
    lead: list[str] = []
    for block in layout.split("\n\n"):
        names = _block_placeholders(block)
        if any(isinstance(slots.get(n), BriefToolSlot) for n in names):
            break
        if all(n in resolved for n in names if n in slots):
            lead.append(render_layout(block, resolved))
    return "\n\n".join(part for part in lead if part.strip())


def render_brief_result(
    package: SkillPackage,
    skill_registry: SkillRegistryService,
    *,
    now: datetime | None = None,
    section_keys: tuple[str, ...] | None = None,
    content_lines_per_item: int | None = None,
    section_order: tuple[str, ...] | None = None,
    section_line_caps: dict[str, int] | None = None,
    header: str | None = None,
    footer: str | None = None,
) -> BriefRender:
    """Render a brief skill, surviving any one section's failure.

    The customization knobs are **opt-in**: when ``section_keys is None`` the
    full manifest brief is rendered (every slot, full layout) so the
    non-routine callers are unaffected. When ``section_keys`` is given, only
    those tool sections (plus always-on greeting/literals) are resolved and
    rendered — and crucially only their tools are invoked. ``content_lines_per_item``
    caps bullets per section (``section_line_caps`` overrides per section),
    ``section_order`` reorders the kept sections, and ``header`` / ``footer`` are
    prepended / appended as plain markdown.

    A slot that raises does not abort the brief (D6: a partial digest beats
    none). Its block is left out, the exception is logged, and one
    "couldn't build" line naming every failed section goes after the sections,
    before the footer (``footer_slots``, then ``footer``).

    A tool slot whose skill is not installed at all (no package of that name was
    discovered) is **absent**, not failed: its block is left out silently (logged at
    debug) and it is named nowhere. A brief ships with slots for skills an installation
    may not carry — the core's morning digest names the domain plugins' skills, which a
    core-only install does not have — and "couldn't build" there would be a false
    alarm on every run. A skill that IS installed but cannot serve the slot (it failed
    to load, lacks the tool, or the tool raised) still fails, as before.

    Raises ``ValueError`` if the package isn't a brief; a layout placeholder
    with no slot (a manifest bug, not a runtime failure) still raises.
    """
    if package.manifest.brief is None:
        raise ValueError(f"package {package.manifest.name!r} is not a brief skill")
    spec = package.manifest.brief
    at = now or local_now()
    greeting, daypart = brief_greeting(spec, at)
    ctx = build_slot_context(skill_registry, now=at, greeting=greeting, daypart=daypart)

    if section_keys is None:
        # Full-brief path (synthetic render tool, direct-local exec): every slot.
        layout, needed = spec.layout, list(spec.slots)
    else:
        layout, needed = _filter_layout(spec.layout, spec.slots, tuple(section_keys), section_order)

    installed = _installed_skills(skill_registry)
    absent: dict[str, str] = {}
    for name in needed:
        candidate = spec.slots.get(name)
        if isinstance(candidate, BriefToolSlot) and candidate.skill not in installed:
            absent[name] = candidate.skill
    for name, skill in absent.items():
        logger.debug(
            "brief %s: section %r left out: skill %r is not installed",
            package.manifest.name,
            name,
            skill,
        )
    if absent:
        layout = _drop_blocks(layout, set(absent))
        needed = [name for name in needed if name not in absent]

    resolved: dict[str, str] = {}
    # A ``count_if`` slot's own count of the items that count (else its bullets count).
    counted: dict[str, int] = {}
    failed: list[FailedSection] = []
    for name in dict.fromkeys(needed):
        line_cap = (
            None
            if section_keys is None
            else _slot_line_cap(name, spec.slots[name], content_lines_per_item, section_line_caps)
        )
        try:
            slot_spec = spec.slots[name]
            if isinstance(slot_spec, BriefToolSlot):
                text, count = _resolve_tool_counted(slot_spec, ctx, spec.uses, line_cap=line_cap)
                resolved[name] = text
                if count is not None:
                    counted[name] = count
            else:
                resolved[name] = resolve_slot(slot_spec, ctx, spec.uses, line_cap=line_cap)
        except Exception as exc:  # one section never sinks the brief
            logger.exception("brief %s: section %r failed to render", package.manifest.name, name)
            failed.append(
                FailedSection(
                    name=name,
                    title=_section_title(spec.layout, name),
                    reason=_short_reason(exc),
                )
            )

    footer_names = set(spec.footer_slots)
    failed_by_name = {item.name: item for item in failed}
    sections: list[BriefSection] = []
    for name in dict.fromkeys(_block_placeholders(layout)):
        slot = spec.slots.get(name)
        if not isinstance(slot, BriefToolSlot):
            continue
        if name in failed_by_name:
            item = failed_by_name[name]
            sections.append(
                BriefSection(
                    name=name, title=item.title, failed=item.reason, footer=name in footer_names
                )
            )
        elif name in resolved:
            section = _brief_section(
                name, slot, resolved[name], spec.layout, footer=name in footer_names
            )
            if name in counted:
                section = replace(section, items=counted[name])
            sections.append(section)

    if failed:
        layout = _drop_blocks(layout, {item.name for item in failed})
    layout, footer_layout = _split_footer(layout, footer_names)
    greeting = _leading_text(layout, spec.slots, resolved)
    body = render_layout(layout, resolved)

    counts: list[tuple[str, int]] = []
    for name in dict.fromkeys(_block_placeholders(layout)):
        slot = spec.slots.get(name)
        if not isinstance(slot, BriefToolSlot) or name not in resolved:
            continue
        items = counted.get(name, _bullet_count(resolved[name]))
        # A text slot (e.g. Focus, which writes its own heading) counts when it
        # renders bullets; a bullets slot always counts, zero included.
        if slot.format == "bullets" or items:
            counts.append((_section_title(spec.layout, name), items))

    if header:
        body = f"{header.strip()}\n\n{body}"
    line = failure_line(failed)
    if line:
        body = f"{body.rstrip()}\n\n{line}"
    closing_parts: list[str] = []
    if footer_layout:
        closing_parts.append(render_layout(footer_layout, resolved))
        body = f"{body.rstrip()}\n\n{closing_parts[-1]}"
    if footer:
        closing_parts.append(footer.strip())
        body = f"{body}\n\n{footer.strip()}"
    # A slot's text had a span cut out by the floor: say so once, in the body and in the
    # closing a composer reorders sections around (issue #139).
    if any(MARKER in text for text in resolved.values()):
        body = add_redaction_notice(body)
        closing_parts.append(REDACTION_NOTICE)
    return BriefRender(
        body=body,
        failed=tuple(failed),
        counts=tuple(counts),
        sections=tuple(sections),
        greeting=greeting,
        header=(header or "").strip(),
        closing="\n\n".join(part.strip() for part in closing_parts if part.strip()),
    )


def render_brief_package(
    package: SkillPackage,
    skill_registry: SkillRegistryService,
    *,
    now: datetime | None = None,
    section_keys: tuple[str, ...] | None = None,
    content_lines_per_item: int | None = None,
    section_order: tuple[str, ...] | None = None,
    section_line_caps: dict[str, int] | None = None,
    header: str | None = None,
    footer: str | None = None,
) -> str:
    """Render a brief skill into its layout text.

    Shared helper for the heartbeat handler, the deterministic chat
    short-circuit, and the synthetic ``render_<skill>`` tool that exposes
    briefs to the LLM tool surface. See :func:`render_brief_result` for the
    knobs and for how a failing section is handled.
    """
    return render_brief_result(
        package,
        skill_registry,
        now=now,
        section_keys=section_keys,
        content_lines_per_item=content_lines_per_item,
        section_order=section_order,
        section_line_caps=section_line_caps,
        header=header,
        footer=footer,
    ).body


def _render_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Translate routine-tick ``params`` into ``render_brief_package`` kwargs.

    An empty / absent ``source_preferences`` means "no section selection" and
    yields the full brief (``section_keys=None``); a non-empty list filters.
    """
    prefs = params.get("source_preferences")
    section_keys = tuple(prefs) if prefs else None
    order = params.get("section_order")
    caps = params.get("section_line_caps")
    cap = params.get("content_lines_per_item")
    return {
        "section_keys": section_keys,
        "content_lines_per_item": int(cap) if isinstance(cap, (int, float)) else None,
        "section_order": tuple(order) if order else None,
        "section_line_caps": dict(caps) if isinstance(caps, dict) and caps else None,
        "header": params.get("header") or None,
        "footer": params.get("footer") or None,
    }


def render_routine_body(runtime: IrisRuntime, routine: Any) -> str:
    """Render a routine's brief body for preview — no dispatch, no counters.

    Resolves the same skill_id + customization params the scheduled tick would
    use and renders the body via :func:`render_brief_package`. Raises
    ``ValueError`` when the routine's bound capability is not a previewable brief
    (e.g. a non-brief tool routine), so callers can surface a clear message.
    """
    from iris_harness.runtime.handlers.ticks import (  # sibling, keeps import order flat
        resolve_brief_skill_id,
        routine_render_params,
    )

    skill_id = resolve_brief_skill_id(runtime, routine.template) or str(
        routine.metadata.get("skill_id", "")
    )
    package = _find_brief_package(runtime.skill_registry, skill_id) if skill_id else None
    if package is None or package.manifest.brief is None:
        raise ValueError(
            f"routine {routine.title!r} is not a previewable brief "
            f"(template={routine.template!r}); use run instead"
        )
    params = routine_render_params(routine)
    rendered = render_brief_result(package, runtime.skill_registry, **_render_kwargs(params))
    layout = digest_layout(runtime, params, rendered)
    return grouped_body(rendered, layout) if layout is not None else rendered.body


#: The web-push channel gets a headline, not the body.
PUSH_CHANNEL = "web_push"

#: The Telegram channel gets the grouped digest as one message per group.
TELEGRAM_CHANNEL = "telegram"

#: The routine param that asks for the grouped digest rendering (digest v5). The
#: seeded ``morning-digest`` routine carries it (``services/routines/seeded.py``);
#: every other brief, and chat, keeps the flat rendering.
DIGEST_PARAM = "digest"

#: The absolute base URL the web console is reached at (``https://<host>``). Telegram's
#: URL buttons need one; without it the digest goes out with no buttons. The reader is
#: the foundation's, shared with the health alerts and the connection callbacks.
PUBLIC_URL_ENV = _PUBLIC_URL_ENV

#: Where Settings -> Digest lives in the web console (the Settings tab is the hash).
DIGEST_SETTINGS_PATH = "/settings#digest"

#: ``channel: all`` means every registered channel except these (D5, graph §7):
#: the console is a log sink on the server, not somewhere the owner reads.
_ALL_EXCLUDES = frozenset({"console"})

#: Where the web console shows a stored digest (webui route ``/digest/:id``).
DIGEST_WEB_PATH = "/digest"


def digest_url(stored: StoredDigest | None) -> str:
    """The in-app link to a stored digest, or to the latest when none was stored."""
    return f"{DIGEST_WEB_PATH}/{stored.id}" if stored is not None else DIGEST_WEB_PATH


def digest_buttons(
    stored: StoredDigest | None, options: dict[str, Any] | None = None
) -> list[list[dict[str, str]]]:
    """The URL buttons under the last Telegram message, one row; ``[]`` without a base URL.

    ``options["buttons"]`` (Settings -> Digest, ``channels.telegram``) names them in
    order: ``full_digest`` opens the stored web copy, ``settings`` Settings -> Digest.
    """
    base = public_base_url()
    if not base:
        return []
    names = (options or {}).get("buttons")
    if not isinstance(names, (list, tuple)):
        names = ("full_digest", "settings")
    known = {
        "full_digest": ("📄 Full digest", digest_url(stored)),
        "settings": ("⚙️ Digest settings", DIGEST_SETTINGS_PATH),
    }
    row = [
        {"text": known[name][0], "url": f"{base}{known[name][1]}"}
        for name in names
        if isinstance(name, str) and name in known
    ]
    return [row] if row else []


def _digest_settings(runtime: Any) -> DigestSettings | None:
    """Settings -> Digest (groups, channel options), or ``None`` without a data dir."""
    data_dir = getattr(runtime, "data_dir", None)
    if data_dir is None:
        return None
    from iris_harness.services.digest.settings import (  # only digests need it
        load_digest_settings,
    )

    try:
        return load_digest_settings(data_dir, getattr(runtime, "config_dir", None))
    except Exception:  # a flat digest beats none
        logger.exception("skill_brief: could not read the digest settings; rendering flat")
        return None


def _grouped(
    rendered: BriefRender, settings: DigestSettings
) -> list[tuple[DigestGroup, list[BriefSection]]]:
    """The rendered sections under Settings -> Digest's groups, in the groups' order."""
    from iris_harness.services.digest.settings import (  # see _digest_settings
        group_sections,
        news_group_title,
    )

    # A news group is titled by Settings -> Digest ("Local — St. Louis"), not by the
    # manifest's generic layout heading.
    by_name = {
        section.name: (
            replace(section, title=news_group_title(settings, section.name))
            if section.name in settings.news_groups
            else section
        )
        for section in rendered.sections
    }
    names = [section.name for section in rendered.sections if not section.footer]
    return [
        (group, [by_name[name] for name in members])
        for group, members in group_sections(settings, names)
    ]


@dataclass(frozen=True)
class DigestLayout:
    """The grouped digest in hand: the settings it was grouped by, and the groups."""

    settings: DigestSettings
    groups: list[tuple[DigestGroup, list[BriefSection]]]

    def channel(self, name: str) -> dict[str, Any]:
        """One channel's options (``channels.<name>`` in Settings -> Digest), or ``{}``."""
        value = self.settings.channels.get(name)
        return dict(value) if isinstance(value, dict) else {}


def digest_layout(
    runtime: Any, params: dict[str, Any], rendered: BriefRender
) -> DigestLayout | None:
    """The grouped layout when ``params`` asks for the digest and groups are configured.

    ``None`` (flat rendering) for every other brief, and for a digest whose settings
    name no groups.
    """
    if not params.get(DIGEST_PARAM):
        return None
    settings = _digest_settings(runtime)
    if settings is None or not settings.groups:
        return None
    return DigestLayout(settings=settings, groups=_grouped(rendered, settings))


def grouped_body(rendered: BriefRender, layout: DigestLayout) -> str:
    """The stored (web) copy of a grouped digest."""
    return grouped_markdown(
        layout.groups,
        greeting=rendered.greeting,
        header=rendered.header,
        closing=rendered.closing,
        options=layout.channel("web"),
    )


def _store_digest(
    heartbeat: str, skill_id: str, subject: str, rendered: BriefRender
) -> StoredDigest | None:
    """Keep the full web copy. A store failure never stops the delivery."""
    try:
        return shared_digest_store().save(
            rendered.body,
            heartbeat=heartbeat,
            skill_id=skill_id,
            subject=subject,
            failed_sections=rendered.failed,
        )
    except Exception:  # the digest still goes out
        logger.exception("skill_brief: could not store the digest for %s", skill_id)
        return None


def _send_all(
    runtime: IrisRuntime, target: str, messages: list[ChannelMessage], *, strict: bool
) -> tuple[HeartbeatStatus, str | None]:
    """Send ``messages`` to one channel in order; stop at the first failure.

    Telegram gets several (the chunks), in order. With ``strict`` off (a
    ``channel: all`` fan-out) a channel that skips — web push with no browser
    subscribed — is a note, not a failure: it has nobody to tell.
    """
    for message in messages:
        receipt = runtime.channels.send(target, message)
        if receipt.status is DeliveryStatus.SENT:
            continue
        if receipt.status is DeliveryStatus.SKIPPED and not strict:
            return HeartbeatStatus.SUCCESS, f"skipped ({receipt.error or 'nothing to send to'})"
        return HeartbeatStatus.FAILED, receipt.error or receipt.status.value
    return HeartbeatStatus.SUCCESS, None


def _build_skill_brief_handler(runtime: IrisRuntime) -> HeartbeatHandler:
    """Return a heartbeat handler that renders and dispatches a brief skill.

    The default channel is read at FIRE time, not here: chat surfaces are channel
    plugins (OSS plan M4.5) and mount after the heartbeats register, so the value
    at registration time is not yet the resolved one.
    """

    def handler(definition: HeartbeatDefinition) -> HeartbeatRun:
        channel = str(definition.params.get("channel") or runtime.default_channel)
        skill_id = str(definition.params.get("skill_id") or "")
        if not skill_id:
            return HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.FAILED,
                error="skill_brief requires a 'skill_id' parameter",
            )

        package = _find_brief_package(runtime.skill_registry, skill_id)
        if package is None or package.manifest.brief is None:
            return HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.FAILED,
                error=f"brief skill not found or not loadable: {skill_id}",
            )

        spec = package.manifest.brief
        try:
            rendered = render_brief_result(
                package, runtime.skill_registry, **_render_kwargs(definition.params)
            )
        except Exception as exc:  # surface as a failed run
            logger.exception("skill_brief render failed for %s", skill_id)
            return HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
        layout = digest_layout(runtime, definition.params, rendered)
        if layout is not None:
            # The digest (v5): the stored web copy is the grouped one, and every
            # channel below renders from the same groups.
            rendered = replace(rendered, body=grouped_body(rendered, layout))
        body = rendered.body
        subject = brief_subject(spec, local_now())
        stored = _store_digest(definition.name, skill_id, subject, rendered)

        recipient = "" if spec.recipient == "user" else spec.recipient
        base_metadata: dict[str, Any] = {"heartbeat": definition.name, "skill_id": skill_id}
        if stored is not None:
            base_metadata["digest_id"] = stored.id

        def _messages(target: str) -> list[ChannelMessage]:
            if target == PUSH_CHANNEL:
                # A notification is a glance: counts, not the digest. The tap
                # lands on the stored web copy (graph §7).
                failed_titles = (
                    [s.title for _g, members in layout.groups for s in members if s.failed]
                    if layout is not None
                    else [item.title for item in rendered.failed]
                )
                headline = (
                    grouped_push_headline(
                        layout.groups,
                        options=layout.channel(PUSH_CHANNEL),
                        failed_titles=failed_titles,
                    )
                    if layout is not None
                    else push_headline(rendered.counts, failed_titles)
                )
                return [
                    ChannelMessage(
                        recipient=recipient,
                        body=headline,
                        subject=subject,
                        metadata={
                            **base_metadata,
                            "url": digest_url(stored),
                            "tag": f"digest:{skill_id}",
                        },
                    )
                ]
            formatted: list[tuple[str, dict[str, Any]]]
            if target == TELEGRAM_CHANNEL and layout is not None:
                options = layout.channel(TELEGRAM_CHANNEL)
                formatted = grouped_telegram(
                    layout.groups,
                    greeting=rendered.greeting,
                    header=rendered.header,
                    closing=rendered.closing,
                    options=options,
                    inline_keyboard=digest_buttons(stored, options) or None,
                )
            else:
                formatted = [
                    (text, dict(extra)) for text, extra in format_messages_for_channel(target, body)
                ]
            return [
                ChannelMessage(
                    recipient=recipient,
                    body=formatted_body,
                    subject=subject,
                    metadata={**base_metadata, **extra_meta},
                )
                for formatted_body, extra_meta in formatted
            ]

        targets = (
            [name for name in runtime.channels.channels() if name not in _ALL_EXCLUDES]
            if channel == "all"
            else [channel]
        )
        errors: list[str] = []
        notes: list[str] = []
        any_failed = False
        for target in targets:
            status, error = _send_all(runtime, target, _messages(target), strict=channel != "all")
            if status is HeartbeatStatus.FAILED:
                any_failed = True
                if error:
                    errors.append(f"{target}: {error}" if channel == "all" else error)
            elif error:
                notes.append(f"{target}: {error}")
        if rendered.failed:
            # The run records which sections failed; the digest itself names them.
            notes.insert(0, "partial: " + failure_line(rendered.failed))
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.FAILED if any_failed else HeartbeatStatus.SUCCESS,
            output=body,
            error="; ".join(errors + notes),
        )

    return handler


__all__ = [
    "DIGEST_PARAM",
    "DIGEST_SETTINGS_PATH",
    "DIGEST_WEB_PATH",
    "PUBLIC_URL_ENV",
    "PUSH_CHANNEL",
    "TELEGRAM_CHANNEL",
    "BriefRender",
    "BriefSection",
    "DigestLayout",
    "SlotContext",
    "_build_skill_brief_handler",
    "_build_tool_index",
    "build_slot_context",
    "_find_brief_package",
    "digest_buttons",
    "digest_layout",
    "digest_url",
    "failure_line",
    "grouped_body",
    "public_base_url",
    "render_brief_package",
    "render_brief_result",
    "render_layout",
    "render_routine_body",
    "resolve_slot",
]
