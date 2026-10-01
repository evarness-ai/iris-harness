"""Plain category names → the categories that exist in the owner's mail.

The model passes what the owner said ("promo", "credit cards", "Newsletters"); the
store keeps topic paths (ADR-0017: ``email/promotions``,
``email/shopping/deals-promotions/amazon``). Before the promo-classification work a
category was a raw ``LIKE`` prefix on the path, which always starts with the mailbox
root, so "promo" matched nothing and the tools reported "categorisation has not run"
over an inbox with 2,000 promotions in it (grill decision 5).

Resolution is by word prefix over the paths that exist — the stored paths, plus the
paths a provider maps its own buckets to — so the vocabulary is the owner's data, not
a list in code. A name with a ``/`` is a path and keeps the old prefix meaning.

A provider can also say which of its labels map to which path (Gmail's inbox tabs,
from the gmail plugin's ``vendor_categories.yaml``). A resolved path carries those
labels, so a promotion still matches after IRIS triage refiled it under a path of its
own (decision 3): 1,047 of the owner's promotions were in that state on 2026-09-22.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from iris_personal.email.store import CategoryFilter, EmailStore

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
# An asked word matches a path word when they share its first letters, up to this
# many: "promo" and "promotional" both match "promotions", "emails" matches "email";
# "shoe" does not match "shopping", and "newsletters" does not match "news".
_STEM = 5
# How many categories a miss lists.
_MENU_SIZE = 15

#: account id → {provider label: topic path}. Empty when the provider maps none.
LabelsFor = Callable[[str], Mapping[str, str]]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _path_words(path: str) -> list[str]:
    return _words(path.replace("/", " "))


def _word_matches(asked: str, word: str) -> bool:
    shared = 0
    for a, b in zip(asked, word, strict=False):
        if a != b:
            break
        shared += 1
    return shared >= min(len(asked), _STEM)


def resolve_paths(name: str, paths: Iterable[str]) -> list[str]:
    """The paths among ``paths`` that the category ``name`` refers to, sorted.

    Every word of the name that matches some path word must match a word of the
    path; words that match no path at all ("my") are ignored. A word every path
    shares ("emails", for the ``email/`` root) therefore narrows nothing, so "promo
    emails" means the promotions paths. A name with a ``/`` is a path prefix, as the
    tools always accepted.
    """
    candidates = sorted({p for p in paths if p})
    name = name.strip()
    if not name:
        return []
    if "/" in name:
        return [p for p in candidates if p.lower().startswith(name.lower())]
    words_of = {p: _path_words(p) for p in candidates}
    asked = [
        a
        for a in _words(name)
        if any(_word_matches(a, w) for words in words_of.values() for w in words)
    ]
    if not asked:
        return []
    return [
        p
        for p, words in words_of.items()
        if all(any(_word_matches(a, w) for w in words) for a in asked)
    ]


def provider_labels(account_id: str) -> Mapping[str, str]:
    """The registered mail provider's label → path table for the account, or {}."""
    from iris_personal.email.providers import mail_provider_for

    provider = mail_provider_for(account_id)
    table = getattr(provider, "category_labels", None) if provider is not None else None
    if table is None:
        return {}
    try:
        return dict(table())
    except Exception:  # provider plugin code; the path match still works
        logger.warning("category_labels() failed for %s", account_id, exc_info=True)
        return {}


@dataclass(frozen=True)
class ResolvedCategory:
    """What a category name means in one account: the paths it names and the
    provider labels that map to them."""

    name: str
    filter: CategoryFilter

    @property
    def found(self) -> bool:
        return bool(self.filter.paths or self.filter.labels)


def resolve_category(
    store: EmailStore, name: str, account_id: str, *, labels_for: LabelsFor = provider_labels
) -> ResolvedCategory:
    """Resolve ``name`` against the account's stored paths and its provider's labels."""
    labels = dict(labels_for(account_id))
    stored = store.category_counts(account_id)
    paths = resolve_paths(name, [*stored, *labels.values()])
    chosen = set(paths)
    return ResolvedCategory(
        name=name,
        filter=CategoryFilter(
            paths=tuple(paths),
            labels=tuple(sorted(label for label, path in labels.items() if path in chosen)),
        ),
    )


def category_menu(store: EmailStore) -> str:
    """The categories in use, with counts, for a name that matched none of them.

    Grouped two levels deep (``email/shopping`` with every shop under it) so the list
    names kinds of mail rather than 400 senders.
    """
    grouped: dict[str, int] = {}
    for path, n in store.category_counts().items():
        key = "/".join(path.split("/")[:2])
        grouped[key] = grouped.get(key, 0) + n
    if not grouped:
        return "No email has a category yet."
    top = sorted(grouped.items(), key=lambda kv: (-kv[1], kv[0]))[:_MENU_SIZE]
    shown = ", ".join(f"{path.split('/')[-1]} ({n})" for path, n in top)
    more = len(grouped) - len(top)
    return f"Categories in the mail: {shown}" + (f", and {more} more." if more else ".")


__all__ = [
    "LabelsFor",
    "ResolvedCategory",
    "category_menu",
    "provider_labels",
    "resolve_category",
    "resolve_paths",
]
