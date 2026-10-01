"""Per-user brief preferences (ADR-0103 Layer 3), now a view on the digest settings.

Which tool sections the user wants in their brief, their order, and per-section
config. Read by the on-demand brief render path (the chat intercept) and written by
the ``configure_brief`` ReAct tool — this is what lets an action-escalated agent
actually *fulfil* "add the outstanding dues to my daily briefing" instead of only
reasoning about it.

One source of truth (loop-proof plan D4): these used to live in
``<data_dir>/brief_prefs.json``; they now live in the digest settings
(``services/digest/settings.py``: ``config/digest.yaml`` defaults, the owner's changes
in ``settings.db`` section ``digest``), which Settings -> Digest edits and the morning
digest reads. The first load moves the old JSON file in once; nothing writes it after.
This module keeps its API so ``configure_brief`` and the render path are unchanged:
``load_brief_prefs`` reads the digest settings, ``save_brief_prefs`` writes them.

A **default** prefs object (the owner never chose sections) means "render the full
manifest brief, every section, in manifest order" for the on-demand brief, exactly as
before; the morning digest itself follows ``digest.yaml``'s order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BriefPreferences:
    """A user's brief customization overlay over the static manifest.

    ``enabled_sections`` is the authoritative section selection: ``None`` means "all
    sections" (full brief); a tuple means "only these tool sections". ``section_order``
    reorders the kept sections (manifest order for any omitted). ``section_config``
    holds per-section knobs (e.g. ``{"bills_due": {"within_days": 7}}``) for render
    paths that consume them.
    """

    enabled_sections: tuple[str, ...] | None = None
    section_order: tuple[str, ...] = ()
    section_config: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        """True when nothing is customized — render the full brief as before."""
        return self.enabled_sections is None and not self.section_order and not self.section_config

    def with_section_added(self, section: str) -> BriefPreferences:
        """Return a copy with ``section`` enabled (and appended to the order).

        Adding to a default (all-sections) prefs starts an *explicit* selection seeded
        with the section, so "add dues" doesn't silently widen to every section — it
        expresses "I want the dues section" deterministically. Re-adding is a no-op.
        """
        current = self.enabled_sections
        if current is None:
            enabled: tuple[str, ...] = (section,)
        elif section in current:
            enabled = current
        else:
            enabled = (*current, section)
        order = (
            self.section_order if section in self.section_order else (*self.section_order, section)
        )
        return BriefPreferences(enabled, order, dict(self.section_config))

    def with_section_removed(self, section: str) -> BriefPreferences:
        """Return a copy with ``section`` disabled.

        From a default (all) selection, removing a section materializes an explicit
        "all-but-this" selection is ambiguous without the manifest, so removal only
        applies once an explicit selection exists; the caller passes the manifest
        section list to seed one when needed (see ``configure_brief``).
        """
        if self.enabled_sections is None:
            return self  # nothing explicitly selected yet; caller seeds from manifest
        enabled = tuple(s for s in self.enabled_sections if s != section)
        order = tuple(s for s in self.section_order if s != section)
        config = {k: v for k, v in self.section_config.items() if k != section}
        return BriefPreferences(enabled, order, config)

    def with_section_config(self, section: str, **knobs: Any) -> BriefPreferences:
        """Return a copy with per-section config merged in (e.g. within_days=7)."""
        merged = dict(self.section_config)
        merged[section] = {**merged.get(section, {}), **knobs}
        return BriefPreferences(self.enabled_sections, self.section_order, merged)


# The history's name for a change made through this API (the chat tool).
CHAT_ACTOR = "chat:configure_brief"


def load_brief_prefs(data_dir: Path) -> BriefPreferences:
    """The user's brief prefs from the digest settings; default (full brief) until the
    owner chooses sections. Never raises: the digest settings fall back to defaults."""
    from iris_harness.services.digest.settings import (
        load_defaults,
        load_digest_settings,
        read_saved,
        store_for,
    )

    settings = load_digest_settings(data_dir)  # runs the one-time brief_prefs.json move
    try:
        # Under today's section names (a saved ``news`` is the three news slots).
        saved = read_saved(store_for(data_dir), load_defaults())
    except Exception:  # noqa: BLE001 — a broken store must never break a brief render
        logger.warning("brief_prefs: digest settings unreadable; using full-brief default")
        return BriefPreferences()
    chosen = "sections" in saved or "sections_off" in saved
    # Only the sections whose knobs the owner set: digest.yaml's own caps (top 5
    # news, ...) reach the scheduled digest through its routine, and are not a
    # customization of the chat brief.
    owned = saved.get("section_config")
    owned_keys = set(owned) if isinstance(owned, dict) else set()
    return BriefPreferences(
        enabled_sections=settings.sections if chosen else None,
        section_order=settings.sections if chosen else (),
        section_config={k: dict(v) for k, v in settings.section_config.items() if k in owned_keys},
    )


def default_off_sections() -> tuple[str, ...]:
    """Sections ``config/digest.yaml`` leaves off until the owner turns them on
    (``sections_off``, e.g. top_repos): the full brief leaves them out too. Never
    raises (a broken file has none)."""
    try:
        from iris_harness.services.digest.settings import load_defaults

        return load_defaults().sections_off
    except Exception:  # noqa: BLE001 — a broken file must never break a brief render
        logger.warning("brief_prefs: digest defaults unreadable; no default-off sections")
        return ()


def save_brief_prefs(data_dir: Path, prefs: BriefPreferences) -> None:
    """Persist the user's brief prefs into the digest settings (history: chat)."""
    from iris_harness.services.digest import edits
    from iris_harness.services.digest.settings import load_defaults

    changes: dict[str, Any] = {"section_config": prefs.section_config}
    if prefs.enabled_sections is None:
        # Back to every section: the file's list, nothing off.
        changes["sections"] = list(load_defaults().settings.sections)
        changes["sections_off"] = []
    else:
        enabled = list(dict.fromkeys(prefs.enabled_sections))
        first = [s for s in dict.fromkeys(prefs.section_order) if s in enabled]
        changes["sections"] = first + [s for s in enabled if s not in first]
    edits.update(data_dir, changes, actor=CHAT_ACTOR, any_section=True)


__all__ = ["BriefPreferences", "default_off_sections", "load_brief_prefs", "save_brief_prefs"]
