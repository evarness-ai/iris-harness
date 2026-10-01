"""The owner's digest edits: Settings -> Digest and chat both land here (ADR-0120).

Same shape as the health watch's edits (``services/health/watch_edits.py``): the
change is validated, the fields that differ from ``config/digest.yaml`` are saved in
the settings store (section ``digest``, key ``config``), and editing a field back to
the file's value drops it, so a later change to the file reaches it. The digest reads
its settings when it runs, so an edit applies to the next digest with no restart.

Sections: sending ``sections`` alone means "exactly these, in this order": every other
section the digest knows is turned off. ``sections_off`` alone turns those off as
well and keeps the rest where they are. The app sends both.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from iris_harness.services.digest.settings import (
    EDITABLE,
    KEY,
    MIGRATION_KEY,
    SETTINGS_SECTION,
    DigestDefaults,
    DigestEditError,
    DigestSettings,
    apply_saved,
    diff_from_defaults,
    file_view,
    iris_timezone,
    known_sections,
    load_defaults,
    migrate_brief_prefs,
    news_group_title,
    read_saved,
    sections_off,
    store_for,
    validate,
    view,
)

if TYPE_CHECKING:
    from iris_harness.foundation.settings.store import SettingsStore


def _state(
    data_dir: Path, config_dir: Path | None
) -> tuple[DigestDefaults, SettingsStore, dict[str, Any]]:
    defaults = load_defaults(config_dir)
    store = store_for(data_dir)
    migrate_brief_prefs(data_dir, store, defaults)
    return defaults, store, read_saved(store, defaults)


def _universe(defaults: DigestDefaults, saved: dict[str, Any]) -> list[str]:
    """Every section the digest knows, shown ones first in their order, then off."""
    shown = list(apply_saved(defaults, saved).sections)
    off = [s for s in sections_off(defaults, saved) if s not in shown]
    rest = sorted(known_sections(defaults, saved) - set(shown) - set(off))
    return shown + off + rest


def payload(data_dir: Path, config_dir: Path | None = None) -> dict[str, Any]:
    """The Digest tab: every field now, the file's value, which differ, every section."""
    defaults, store, saved = _state(data_dir, config_dir)
    settings = apply_saved(defaults, saved)
    current = view(settings, sections_off(defaults, saved))
    from_file = file_view(defaults)
    return {
        "fields": current,
        "file": from_file,
        "changed": [k for k in EDITABLE if current[k] != from_file[k]],
        "all_sections": _universe(defaults, saved),
        "locked_sections": list(defaults.locked_sections),
        # Read-only: the file's layout, and each news slot's heading as it reads now.
        "groups": [
            {"id": g.id, "title": g.title, "icon": g.icon, "sections": list(g.sections)}
            for g in settings.groups
        ],
        "section_titles": {slot: news_group_title(settings, slot) for slot in settings.news_groups},
        "timezone": iris_timezone().key,
        "migration": store.get(SETTINGS_SECTION, MIGRATION_KEY),
    }


def update(
    data_dir: Path,
    changes: dict[str, Any],
    *,
    actor: str,
    config_dir: Path | None = None,
    any_section: bool = False,
) -> DigestSettings:
    """Save ``changes`` and return the settings in effect after them.

    ``any_section`` accepts a section name the digest does not know yet (chat adding a
    slot the brief skill has and ``digest.yaml`` does not); the app sends only known
    ones. Raises ``DigestEditError`` for a change that does not fit; nothing is saved.
    """
    if not changes:
        raise DigestEditError("send at least one field to change")
    defaults, store, saved = _state(data_dir, config_dir)
    known = None if any_section else known_sections(defaults, saved)
    clean = validate(changes, known_sections=known)
    unknown = [s for s in clean.get("news_groups") or {} if s not in defaults.settings.news_groups]
    if unknown:
        raise DigestEditError(f"news_groups: no news section {unknown[0]!r}")
    on, off = clean.get("sections"), clean.get("sections_off")
    if on is not None and off is not None and set(on) & set(off):
        raise DigestEditError("a section cannot be both shown and off")
    locked = [s for s in off or () if s in defaults.locked_sections]
    if locked:
        raise DigestEditError(f"{locked[0]} is always shown; it cannot be turned off")
    target = {**saved, **clean}
    if on is not None and off is None:
        target["sections_off"] = [
            s
            for s in _universe(defaults, saved)
            if s not in on and s not in defaults.locked_sections
        ]
    elif off is not None and on is None:
        before_off = list(saved.get("sections_off") or ())
        target["sections_off"] = before_off + [s for s in off if s not in before_off]
        shown = apply_saved(defaults, saved).sections
        target["sections"] = [s for s in shown if s not in off]
    diff = diff_from_defaults(defaults, target)
    before = apply_saved(defaults, saved)
    after = apply_saved(defaults, diff)
    if diff == saved:
        return after
    old = view(before, sections_off(defaults, saved))
    new = view(after, sections_off(defaults, diff))
    if diff:
        store.set(SETTINGS_SECTION, KEY, diff, old=old, new=new, actor=actor)
    else:
        store.clear(SETTINGS_SECTION, KEY, old=old, new=new, actor=actor)
    return after


def reset(data_dir: Path, *, actor: str, config_dir: Path | None = None) -> DigestSettings:
    """Back to ``digest.yaml``: every saved change dropped (kept in the history)."""
    defaults, store, saved = _state(data_dir, config_dir)
    if saved:
        store.clear(
            SETTINGS_SECTION,
            KEY,
            old=view(apply_saved(defaults, saved), sections_off(defaults, saved)),
            new=file_view(defaults),
            actor=actor,
        )
    return defaults.settings


__all__ = ["DigestEditError", "payload", "reset", "update"]
