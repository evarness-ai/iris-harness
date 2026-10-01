"""Senders the memory Map should not draw: the ones whose mail is mostly promotions.

Every "check my email" chat lists the senders it read out, and the chat's summary keeps
them on its "Referenced:" line. A shop that mails daily is in every such summary, so it
recurred enough for the Map to draw it next to the people the owner knows (ADR-0119,
cleanup plan decision 9). The core cannot tell a shop from a friend; this plugin
classified every message, so it can.

A sender is left off when at least ``share`` of its stored mail falls in the configured
categories, and it has sent at least ``min_messages`` — one promotion from a friend is
not a pattern. The category names are resolved the way the email tools resolve them
(``iris_personal.email.categories``), so "promotions" also matches a promotion IRIS
triage refiled under a path of its own. Every number and name lives in
``memory_map.yaml`` beside this module.

The answer is cached for ``cache_seconds``: the Map is drawn per request, the answer
changes only when mail is classified.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parseaddr
from pathlib import Path
from typing import Any

import yaml

from iris_personal.email.categories import LabelsFor, provider_labels, resolve_category
from iris_personal.email.store import CategoryFilter, EmailStore

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).with_name("memory_map.yaml")


@dataclass(frozen=True)
class ExclusionRule:
    categories: tuple[str, ...]
    share: float
    min_messages: int
    cache_seconds: float


def load_rule(path: Path = CONFIG_PATH) -> ExclusionRule:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    senders = raw.get("promotional_senders") or {}
    return ExclusionRule(
        categories=tuple(str(c) for c in senders.get("categories") or ()),
        share=float(senders.get("share", 1.0)),
        min_messages=int(senders.get("min_messages", 1)),
        cache_seconds=float(senders.get("cache_seconds", 0)),
    )


def _category_filter(
    store: EmailStore, account_id: str, names: tuple[str, ...], labels_for: LabelsFor
) -> CategoryFilter:
    paths: set[str] = set()
    labels: set[str] = set()
    for name in names:
        resolved = resolve_category(store, name, account_id, labels_for=labels_for)
        paths.update(resolved.filter.paths)
        labels.update(resolved.filter.labels)
    return CategoryFilter(paths=tuple(sorted(paths)), labels=tuple(sorted(labels)))


def promotional_senders(
    store: EmailStore, rule: ExclusionRule, *, labels_for: LabelsFor = provider_labels
) -> list[str]:
    """Display names and domains whose mail is mostly in ``rule.categories``, sorted.

    Counted across every account: a name that is a shop in one mailbox and a
    correspondent in another is judged on all of its mail.
    """
    by_name: dict[str, list[int]] = {}
    by_domain: dict[str, list[int]] = {}
    for account_id in store.list_accounts():
        category = _category_filter(store, account_id, rule.categories, labels_for)
        for address, domain, total, matched in store.sender_category_counts(account_id, category):
            name = parseaddr(address)[0].strip() or parseaddr(address)[1].strip()
            for key, table in ((name, by_name), (domain.strip().lower(), by_domain)):
                if key:
                    counts = table.setdefault(key, [0, 0])
                    counts[0] += total
                    counts[1] += matched
    return sorted(
        key
        for table in (by_name, by_domain)
        for key, (total, matched) in table.items()
        if total >= rule.min_messages and matched >= rule.share * total
    )


def build_provider(
    store_factory: Callable[[], EmailStore] = EmailStore,
    *,
    rule_loader: Callable[[], ExclusionRule] = load_rule,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[], list[str]]:
    """The provider the plugin registers: ``promotional_senders``, cached."""
    cached: list[str] = []
    expires = [float("-inf")]

    def provider() -> list[str]:
        now = clock()
        if now < expires[0]:
            return cached
        rule = rule_loader()
        store = store_factory()
        names = promotional_senders(store, rule) if store.db_path.exists() else []
        cached[:] = names
        expires[0] = now + rule.cache_seconds
        logger.debug("memory map: %d promotional senders left off", len(names))
        return cached

    return provider


__all__ = ["CONFIG_PATH", "ExclusionRule", "build_provider", "load_rule", "promotional_senders"]
