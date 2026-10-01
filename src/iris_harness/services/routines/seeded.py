"""Core routines IRIS seeds itself: the ``morning-digest`` (ADR-0122 §3, loop-proof D4).

Every other routine is user-authored and runs only once the owner approves it. The
morning digest is the exception the loop proof needs: it exists on every install, is
scheduled from first boot, and is never auto-retired. Its *delivery* fields are not the
routine's own — they are derived from Settings → Digest on every tick, so a settings
change takes effect at the next tick without a restart:

- ``schedule``           ← ``daily:<settings.time>``
- ``metadata.timezone``  ← ``IRIS_TZ`` (the store reads wall-clock schedules in it)
- ``delivery_channel``   ← ``settings.channel``
- ``source_preferences`` + ``metadata.section_order`` ← ``settings.sections``
- ``metadata.section_line_caps`` ← per-section caps in ``settings.section_config``
- ``metadata.digest``    ← always ``True``: render it as the grouped digest (Settings →
  Digest ``groups`` and ``channels``), not as a flat brief

Everything else belongs to the owner and is never written after the seed: the
approval status (pausing or retiring it is respected), title, goal, header/footer,
run counters. Seeding only ever creates the row when it is absent.

Pure store logic — no runtime, no I/O beyond the ``RoutineStore`` passed in.
"""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from typing import Any

from iris_harness.services.digest.settings import DigestSettings

from .models import RoutineApprovalStatus, RoutineSpec
from .store import RoutineStore

MORNING_DIGEST_ROUTINE_ID = "morning-digest"
MORNING_DIGEST_TEMPLATE = "morning-briefing"
"""The brief skill manifest the digest renders (``config/skills/builtin/morning-briefing``)."""

# Keys in a section's ``section_config`` entry read as its bullet cap. ``line_cap``
# is the brief renderer's own word; ``max_items`` / ``limit`` are what a settings form
# or chat edit most naturally writes.
_SECTION_CAP_KEYS = ("line_cap", "max_items", "limit")

# The routine fires this many minutes before the Settings time, so the digest is
# built and delivered by the time the owner asked for (graph §7: 06:55 routine for
# a 07:00 digest).
DIGEST_LEAD_MINUTES = 5

# Metadata keys this module derives from settings — the only metadata it ever writes.
_DERIVED_METADATA_KEYS = ("timezone", "section_order", "section_line_caps", "digest")


def is_core_routine(spec: RoutineSpec) -> bool:
    """True for a routine IRIS seeds itself; such a routine is never proposed for retirement."""
    return spec.id == MORNING_DIGEST_ROUTINE_ID or bool(spec.metadata.get("core"))


def _zone_name(zone: tzinfo) -> str:
    return str(getattr(zone, "key", None) or zone.tzname(None) or "UTC")


def _daily_schedule(time_text: str) -> str:
    """``daily:HH:MM`` for a Settings delivery time, moved earlier by the lead."""
    hour_text, _, minute_text = time_text.strip().partition(":")
    hour, minute = int(hour_text), int(minute_text or "0")
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid digest time {time_text!r}")
    fire = (hour * 60 + minute - DIGEST_LEAD_MINUTES) % (24 * 60)
    return f"daily:{fire // 60:02d}:{fire % 60:02d}"


def _section_line_caps(settings: DigestSettings) -> dict[str, int]:
    caps: dict[str, int] = {}
    for section, config in settings.section_config.items():
        if not isinstance(config, dict):
            continue
        for key in _SECTION_CAP_KEYS:
            value = config.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                caps[section] = value
                break
    return caps


def digest_derived_fields(settings: DigestSettings, zone: tzinfo) -> dict[str, Any]:
    """The routine fields Settings → Digest owns, as a ``model_copy`` update.

    ``metadata`` here holds only the derived keys; :func:`sync_morning_digest` merges
    them over the row's own metadata. The schedule fires
    ``DIGEST_LEAD_MINUTES`` before ``time``. A malformed ``time`` keeps the default 07:00 so
    a bad setting can never silence the digest.
    """
    try:
        schedule = _daily_schedule(settings.time)
    except ValueError:
        schedule = _daily_schedule(DigestSettings().time)
    sections = tuple(settings.sections)
    return {
        "schedule": schedule,
        "delivery_channel": settings.channel or "all",
        "source_preferences": sections,
        "metadata": {
            "timezone": _zone_name(zone),
            "section_order": list(sections),
            "section_line_caps": _section_line_caps(settings),
            # The skill_brief handler renders this routine grouped, per channel.
            "digest": True,
        },
    }


def build_morning_digest_spec(settings: DigestSettings, zone: tzinfo) -> RoutineSpec:
    """The first-boot ``morning-digest`` row: scheduled, bound to the morning brief."""
    derived = digest_derived_fields(settings, zone)
    return RoutineSpec(
        id=MORNING_DIGEST_ROUTINE_ID,
        title="Morning digest",
        goal="Deliver the daily morning digest from Settings → Digest.",
        schedule=derived["schedule"],
        template=MORNING_DIGEST_TEMPLATE,
        source_preferences=derived["source_preferences"],
        delivery_channel=derived["delivery_channel"],
        approval_status=RoutineApprovalStatus.SCHEDULED,
        metadata={"core": True, "managed_by": "digest_settings", **derived["metadata"]},
    )


def seed_morning_digest(
    store: RoutineStore, settings: DigestSettings, zone: tzinfo
) -> RoutineSpec | None:
    """Create the ``morning-digest`` row if absent. Returns it when created, else ``None``.

    Idempotent across restarts: an existing row — edited, paused or retired by the
    owner — is never touched here.
    """
    if store.load(MORNING_DIGEST_ROUTINE_ID) is not None:
        return None
    return store.save(build_morning_digest_spec(settings, zone))


def sync_morning_digest(
    store: RoutineStore, settings: DigestSettings, zone: tzinfo
) -> RoutineSpec | None:
    """Bring the row's settings-derived fields in line with Settings → Digest.

    Writes only when something changed (so a quiet tick is a read), never touches the
    approval status or any owner-set field, and returns the current row (``None`` when
    it does not exist — seeding is :func:`seed_morning_digest`'s job, at boot).
    """
    spec = store.load(MORNING_DIGEST_ROUTINE_ID)
    if spec is None:
        return None
    derived = digest_derived_fields(settings, zone)
    metadata = {**spec.metadata, **derived.pop("metadata")}
    unchanged = all(getattr(spec, key) == value for key, value in derived.items()) and all(
        spec.metadata.get(key) == metadata[key] for key in _DERIVED_METADATA_KEYS
    )
    if unchanged:
        return spec
    return store.save(
        spec.model_copy(update={**derived, "metadata": metadata, "updated_at": datetime.now(UTC)})
    )


__all__ = [
    "MORNING_DIGEST_ROUTINE_ID",
    "MORNING_DIGEST_TEMPLATE",
    "build_morning_digest_spec",
    "digest_derived_fields",
    "is_core_routine",
    "seed_morning_digest",
    "sync_morning_digest",
]
