"""Grounded analysis over the inbox: group every matching email by sender.

The loop's other email tools answer "find the email about X" — they return the top
15 hits, which is right for a lookup and wrong for an analysis. "Which credit cards
do I have?", "how many subscriptions am I paying for?", "which banks send me
statements?" need EVERY matching message, collapsed into the distinct things behind
them, so the model lists each card or subscription once instead of echoing subjects.

This module is the pure half — grouping and rendering. The ``analyze_inbox`` tool in
:mod:`iris_personal.email.agent_tools` runs the search and hands the hits here. No
model is involved: the groups are counted from the store, and the model only names
the distinct items it can see in them (2026-09-15 session, where a keyword shortcut
answered "list the credit card accounts I have" with a dump of dues subjects).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from iris_personal.email.store import SearchHit

#: How many sender groups one observation carries; the rest are counted, not listed.
MAX_GROUPS = 25
#: Distinct example subjects shown per sender.
MAX_SUBJECTS = 3

_NAME_RE = re.compile(r"^\s*\"?(?P<name>[^\"<]*?)\"?\s*<")
_ADDRESS_RE = re.compile(r"<?(?P<addr>[\w.+-]+@(?P<domain>[\w-]+(?:\.[\w-]+)+))>?")
# Long digit runs in a subject are account/card numbers; keep only the last four.
_ACCOUNT_NUMBER_RE = re.compile(r"(?<!\d)\d{8,18}(?!\d)")
# What makes two subjects "the same kind of mail": digits and dates removed.
_SUBJECT_SHAPE_RE = re.compile(r"[\d/,:.-]+")


@dataclass
class SenderGroup:
    """Every matching email from one sender."""

    key: str
    names: list[str] = field(default_factory=list)
    count: int = 0
    first: datetime | None = None
    last: datetime | None = None
    categories: dict[str, int] = field(default_factory=dict)
    subjects: list[str] = field(default_factory=list)
    _shapes: set[str] = field(default_factory=set)

    @property
    def label(self) -> str:
        name = self.names[0] if self.names else ""
        return f"{name} ({self.key})" if name and name.lower() != self.key else self.key


def sender_key(from_address: str, from_domain: str | None) -> tuple[str, str]:
    """``(key, display_name)`` for a sender: the domain it mails from, and its name."""
    name_match = _NAME_RE.match(from_address or "")
    name = name_match.group("name").strip() if name_match else ""
    domain = (from_domain or "").strip().lower()
    if not domain:
        addr = _ADDRESS_RE.search(from_address or "")
        domain = addr.group("domain").lower() if addr else (from_address or "").strip().lower()
    return domain or "(unknown sender)", name


def mask_account_numbers(text: str) -> str:
    return _ACCOUNT_NUMBER_RE.sub(lambda m: f"••{m.group(0)[-4:]}", text or "")


def group_by_sender(hits: Iterable[SearchHit]) -> list[SenderGroup]:
    """Collapse hits into sender groups, most emails first (then most recent)."""
    groups: dict[str, SenderGroup] = {}
    seen_ids: set[str] = set()
    for hit in sorted(hits, key=lambda h: h.received_at, reverse=True):
        if hit.id in seen_ids:  # the same message can match more than one account pass
            continue
        seen_ids.add(hit.id)
        key, name = sender_key(hit.from_address, hit.from_domain)
        group = groups.setdefault(key, SenderGroup(key=key))
        group.count += 1
        if name and name not in group.names:
            group.names.append(name)
        group.last = hit.received_at if group.last is None else max(group.last, hit.received_at)
        group.first = hit.received_at if group.first is None else min(group.first, hit.received_at)
        if hit.classified_category:
            group.categories[hit.classified_category] = (
                group.categories.get(hit.classified_category, 0) + 1
            )
        subject = " ".join(mask_account_numbers(hit.subject or "").split())
        shape = _SUBJECT_SHAPE_RE.sub("", subject.lower())
        if subject and shape not in group._shapes and len(group.subjects) < MAX_SUBJECTS:
            group._shapes.add(shape)
            group.subjects.append(subject[:120])
    return sorted(
        groups.values(),
        key=lambda g: (-g.count, -(g.last.timestamp() if g.last else 0.0)),
    )


def render_groups(query: str, groups: list[SenderGroup], *, broadened: bool = False) -> str:
    """The observation the loop reads: one block per sender, then how to use it."""
    total = sum(g.count for g in groups)
    if not groups:
        return f"No emails matched '{query}', so there is nothing to analyse."
    firsts = [g.first for g in groups if g.first is not None]
    lasts = [g.last for g in groups if g.last is not None]
    span = f", {min(firsts):%Y-%m-%d} to {max(lasts):%Y-%m-%d}" if firsts and lasts else ""
    match_note = "mentioning related terms" if broadened else "matching"
    lines = [
        f"Analysed ALL {total} email(s) {match_note} '{query}' from {len(groups)} "
        f"sender(s){span}:"
    ]
    for i, group in enumerate(groups[:MAX_GROUPS], start=1):
        when = ""
        if group.first and group.last:
            when = (
                f", {group.last:%Y-%m-%d}"
                if group.first.date() == group.last.date()
                else f", {group.first:%Y-%m-%d} to {group.last:%Y-%m-%d}"
            )
        cats = ""
        if group.categories:
            top = sorted(group.categories.items(), key=lambda kv: -kv[1])[:2]
            cats = " [" + ", ".join(c for c, _ in top) + "]"
        lines.append(f"{i}. {group.label}: {group.count} email(s){when}{cats}")
        for subject in group.subjects:
            lines.append(f"   - {subject}")
    if len(groups) > MAX_GROUPS:
        rest = groups[MAX_GROUPS:]
        lines.append(
            f"(+{len(rest)} more sender(s) with {sum(g.count for g in rest)} email(s) not listed)"
        )
    lines.append(
        "\nAnswer from these groups: name each distinct item the user asked about ONCE "
        "(one card, account, subscription or company per line), using the sender and "
        "subjects as evidence. Marketing or offer mail is not an account the user holds. "
        "Do not list individual emails."
    )
    return "\n".join(lines)


__all__ = [
    "MAX_GROUPS",
    "MAX_SUBJECTS",
    "SenderGroup",
    "group_by_sender",
    "mask_account_numbers",
    "render_groups",
    "sender_key",
]
