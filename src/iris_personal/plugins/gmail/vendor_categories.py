"""Gmail's inbox tabs → IRIS topic paths, from the plugin's ``vendor_categories.yaml``.

One mapping function, used on both paths that need it: :func:`gmail_fetch._parse_message`
sets ``EmailMessage.vendor_category`` from a freshly-fetched message's ``labelIds``,
and ``iris email label-from-vendor`` re-derives it from the ``labels`` already stored
in email.db. The label names live in the YAML, not here (owner rule: plugin
vocabulary is config, not code).
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import yaml

#: The table shipped with the plugin.
SHIPPED_TABLE = Path(__file__).with_name("vendor_categories.yaml")


def _read_table(path: Path) -> dict[str, str | None]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tabs = raw.get("tabs") if isinstance(raw, dict) else None
    if not isinstance(tabs, dict) or not tabs:
        raise ValueError(f"gmail vendor categories: {path} must hold a non-empty `tabs` mapping")
    table: dict[str, str | None] = {}
    for label, path_value in tabs.items():
        if path_value is not None and (not isinstance(path_value, str) or not path_value.strip()):
            raise ValueError(f"gmail vendor categories: {label!r} must map to a topic path or null")
        table[str(label)] = path_value.strip() if isinstance(path_value, str) else None
    return table


@lru_cache(maxsize=4)
def load_table(path: Path = SHIPPED_TABLE) -> dict[str, str | None]:
    """The label → topic-path table, in file order. Cached per path."""
    return _read_table(path)


def vendor_category_for(labels: Iterable[str], *, table_path: Path = SHIPPED_TABLE) -> str | None:
    """The topic path for a Gmail message's labels, or None.

    Walks the table in file order so a message carrying two tab labels resolves the
    same way every time. A label mapped to null (Primary) contributes no path, and a
    label the table does not list is ignored.
    """
    present = set(labels)
    for label, path in load_table(table_path).items():
        if label in present and path:
            return path
    return None


__all__ = ["SHIPPED_TABLE", "load_table", "vendor_category_for"]
