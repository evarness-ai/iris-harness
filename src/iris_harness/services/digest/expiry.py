"""Expiry: every dated item ends CLOSED or EXPIRED, and the digest shows neither.

The owner's rule (2026-09-25): a dated item either ends **closed** — done, paid,
dismissed by the owner — or **expired**, aged out automatically. The digest shows
only open, unexpired items. Nothing is deleted: an expired task stays in the store
with status ``expired`` and a ``closed_reason``.

One mechanism for every kind of dated item, its numbers in ``config/digest.yaml``
``expiry:`` (read into :class:`ExpiryPolicy` by ``settings.load_defaults``):

========================  =====================================================
``prep_after_event``      minutes after its meeting ENDS a prep task expires
``event_after_end``       minutes after it ends a calendar event leaves the digest
``task_overdue_days``     days a dated task shows as Overdue after its due day
``bill_ask_days``         days after its due date an unpaid bill is asked about
                          ("Did you pay?", one ask a day; PR 4)
``bill_overdue_digest_days``  days an unpaid bill then stays in the digest
``inbox_notice_days``     days an inbox notice (a dated email item) stays
``reminder_missed_digests``  digests a missed reminder is carried into
========================  =====================================================

**A plugin's own kinds** (core/SDK boundary plan PR 5, email slice step 4): a plugin
whose dated items age out by local days declares each kind under ``expiry:`` in its
``manifest.yaml`` -- ``{default, min, max, description}`` -- and reads the owner's value
with ``iris_harness.sdk.digest.expiry_days(key)``. The owner tunes it in the same
``digest.yaml`` ``expiry:`` block, by the same key. Declarations are read from every
INSTALLED plugin, mounted or not (the ontology's precedent): a key in ``digest.yaml`` must
stay valid while its plugin is switched off. The core names none of them; a key no
installed plugin declares is refused, loudly (``digest.yaml``: a warning and the
built-in policy; :func:`expiry_days`: ``KeyError``).

Two uses, belt and braces: :func:`sweep_expired_tasks` closes aged-out tasks on a
schedule (the meeting-prep heartbeat, and the routine tick just before the morning
digest renders), and every digest slot tool filters with :func:`is_expired` /
:func:`is_task_expired` so an item never shows between sweeps.

Reminders count digests, not days (graph §4/§10): a reminder that could not be
delivered is listed under "Missed reminders" in ``reminder_missed_digests`` digests,
then expires. :func:`sweep_missed_reminders` runs just before each real (recorded)
morning digest: it expires the rows already shown that many times and marks the rest
as shown in the digest about to render. A manual re-run neither counts nor expires.
A reminder that WAS delivered (``sent``) but the owner never acted on — no Done, no
Snooze (PR 3b) — is missed too, once its day is over: :func:`sweep_unacknowledged_reminders`
counts it into the same number of digests (listed "sent Mon 8:00 AM, not acknowledged")
and then expires it.

A bill counts both: unpaid, it is asked about for ``bill_ask_days`` after its due date,
then stays in the digest ``bill_overdue_digest_days`` more, then expires (owner, PR 4 —
due Oct 13 with 3 + 7 is asked Oct 14–16, in the digest to Oct 23, expired Oct 24).

Day-based kinds count the owner's local days (``IRIS_TZ``): a task due Sep 22 with
``task_overdue_days: 3`` is Overdue on Sep 23, 24 and 25 and expired from Sep 26.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from iris_harness.foundation.process_state import track_globals

if TYPE_CHECKING:
    from iris_harness.services.notifications.models import Reminder
    from iris_harness.services.notifications.store import ReminderStore
    from iris_harness.services.tasks.models import Task
    from iris_harness.services.tasks.store import TaskStore

logger = logging.getLogger(__name__)

ExpiryKind = Literal["prep", "event", "task", "bill", "inbox_notice", "reminder"]


@dataclass(frozen=True)
class ExpiryPolicy:
    """How long each kind of dated item lives past its date (``digest.yaml`` ``expiry``)."""

    prep_after_event: int = 0  # minutes after the meeting ends
    event_after_end: int = 0  # minutes after the event ends
    task_overdue_days: int = 3  # local days shown as Overdue after the due day
    bill_ask_days: int = 3  # local days after the due date an unpaid bill is asked about
    bill_overdue_digest_days: int = 7  # local days an unpaid bill then stays in the digest
    inbox_notice_days: int = 3  # local days an inbox notice stays past its date
    reminder_missed_digests: int = 1  # digests a missed reminder is carried into
    # The owner's values for the kinds installed plugins declare (``digest.yaml``); a
    # declared kind the file leaves out takes its declared default (:meth:`days`).
    plugin_kinds: Mapping[str, int] = field(default_factory=dict, hash=False)

    def days(self, key: str) -> int:
        """The owner's value for the plugin-declared kind ``key``.

        ``KeyError`` when no installed plugin declares it: a plugin reading a kind it
        never declared is a bug to see, not a default to guess.
        """
        if key in self.plugin_kinds:
            return self.plugin_kinds[key]
        declared = declared_expiry_kinds().get(key)
        if declared is None:
            raise KeyError(
                f"expiry: no installed plugin declares {key!r} under 'expiry:' in its "
                "manifest.yaml"
            )
        return declared.default


#: Allowed range per key (inclusive). Minutes up to a day; days up to a year.
EXPIRY_RANGES: dict[str, tuple[int, int]] = {
    "prep_after_event": (0, 1440),
    "event_after_end": (0, 1440),
    "task_overdue_days": (0, 365),
    "bill_ask_days": (0, 30),
    "bill_overdue_digest_days": (0, 365),
    "inbox_notice_days": (0, 365),
    "reminder_missed_digests": (0, 30),
}


class ExpiryKindDeclaration(BaseModel):
    """One ``expiry:`` entry of a plugin manifest: a kind of dated item the plugin ages
    out by local days, its default and the range the owner may set it to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default: int = Field(..., ge=0)
    min: int = Field(default=0, ge=0)
    max: int = Field(default=365, ge=0)
    description: str = ""

    @model_validator(mode="after")
    def _default_in_range(self) -> ExpiryKindDeclaration:
        if not self.min <= self.default <= self.max:
            raise ValueError(f"default {self.default} is not between {self.min} and {self.max}")
        return self


_KIND_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


def check_expiry_kind_name(name: str) -> str:
    """A declarable kind name: lowercase snake case, and not one of the core's keys."""
    if not name or not name[0].isalpha() or not set(name) <= _KIND_NAME_CHARS:
        raise ValueError(f"expiry: {name!r} is not a kind name (lowercase, digits, _)")
    if name in EXPIRY_RANGES:
        raise ValueError(f"expiry: {name!r} is the core's own key; a plugin may not declare it")
    return name


_declared_lock = threading.Lock()
_declared_cache: dict[str, dict[str, ExpiryKindDeclaration]] = {}


def declared_expiry_kinds() -> dict[str, ExpiryKindDeclaration]:
    """Every kind an installed plugin declares under ``expiry:``, by key.

    Read from the manifests on disk, mounted or not, once per ``IRIS_HOME`` per process
    (plugins are discovered at start-up too). A declaration that does not fit, or a key
    another plugin already declared, is skipped with a warning -- and the plugin that
    reads it gets the ``KeyError`` of an undeclared kind.
    """
    import yaml

    from iris_harness.foundation.plugin_dirs import (
        MANIFEST_FILENAME,
        installed_plugin_dirs,
        iris_home,
    )

    home = str(iris_home())
    with _declared_lock:
        cached = _declared_cache.get(home)
    if cached is not None:
        return cached
    found: dict[str, ExpiryKindDeclaration] = {}
    owner: dict[str, str] = {}
    for plugin, directory in installed_plugin_dirs():
        manifest = directory / MANIFEST_FILENAME
        try:
            raw = (yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}).get("expiry")
        except (OSError, yaml.YAMLError, AttributeError) as exc:
            logger.warning("plugin %r: unreadable %s (%s)", plugin, MANIFEST_FILENAME, exc)
            continue
        if not raw:
            continue
        if not isinstance(raw, Mapping):
            logger.warning("plugin %r: 'expiry:' must map a kind to its policy; skipped", plugin)
            continue
        for key, spec in raw.items():
            try:
                name = check_expiry_kind_name(str(key))
                declaration = ExpiryKindDeclaration.model_validate(spec)
            except (ValueError, ValidationError) as exc:
                logger.warning("plugin %r: expiry kind %r skipped (%s)", plugin, key, exc)
                continue
            if name in found:
                logger.warning(
                    "plugin %r: expiry kind %r is already declared by %r; skipped",
                    plugin,
                    name,
                    owner[name],
                )
                continue
            found[name] = declaration
            owner[name] = plugin
    with _declared_lock:
        _declared_cache[home] = found
    return found


def reset_declared_expiry_kinds() -> None:
    """Forget what the manifests declared (tests, after installing a plugin)."""
    with _declared_lock:
        _declared_cache.clear()


# The task store's producers → the kind of dated item their tasks are. A task from
# any other source is a plain ``task``.
_TASK_KINDS: dict[str, ExpiryKind] = {
    "calendar-prep": "prep",
    "finance-bills": "bill",
}

_REASONS: dict[str, str] = {
    "prep": "expired: the meeting is over",
    "event": "expired: the event is over",
    "task": "expired: overdue past {days} day(s)",
    "bill": "expired: bill overdue past {days} day(s)",
    "inbox_notice": "expired: notice older than {days} day(s)",
    "reminder": "expired: missed, shown in {digests} digest(s)",
}


#: ``closed_reason`` of a delivered reminder the owner never acted on (D18).
UNACKNOWLEDGED_REMINDER_REASON = "expired: delivered, not acknowledged"


def parse_expiry(raw: Any) -> ExpiryPolicy:
    """``digest.yaml``'s ``expiry`` mapping as a policy; ``ValueError`` when it does not fit.

    Keys it leaves out keep their defaults; an unknown key or a number out of range is
    an error (the caller warns and keeps the built-in policy).
    """
    if raw is None:
        return ExpiryPolicy()
    if not isinstance(raw, Mapping):
        raise ValueError("expiry must map a key to a whole number")
    declared = declared_expiry_kinds()
    values: dict[str, int] = {}
    plugin_kinds: dict[str, int] = {}
    for key, value in raw.items():
        if key in EXPIRY_RANGES:
            low, high = EXPIRY_RANGES[key]
        elif key in declared:
            low, high = declared[key].min, declared[key].max
        else:
            known = ", ".join([*EXPIRY_RANGES, *declared])
            raise ValueError(f"expiry: {key!r} is not a key (known: {known})")
        if isinstance(value, bool):
            raise ValueError(f"expiry: {key} must be a whole number")
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(f"expiry: {key} must be a whole number") from None
        if not low <= number <= high:
            raise ValueError(f"expiry: {key} must be between {low} and {high}")
        if key in EXPIRY_RANGES:
            values[str(key)] = number
        else:
            plugin_kinds[str(key)] = number
    return ExpiryPolicy(**values, plugin_kinds=plugin_kinds)


def expiry_view(policy: ExpiryPolicy) -> dict[str, int]:
    """The policy as plain JSON (for a read-only view): the core's keys, then every
    installed plugin's declared kind with the owner's value (or its default)."""
    view = {f.name: getattr(policy, f.name) for f in fields(policy) if f.name != "plugin_kinds"}
    for key in declared_expiry_kinds():
        view[key] = policy.days(key)
    return view


def load_expiry_policy(config_dir: Path | None = None) -> ExpiryPolicy:
    """The policy in ``digest.yaml``; the built-in one if the file cannot be read."""
    try:
        from iris_harness.services.digest.settings import load_defaults

        return load_defaults(config_dir).settings.expiry
    except Exception:  # expiry must never take a digest down
        logger.warning("expiry: digest.yaml unreadable; using the built-in policy", exc_info=True)
        return ExpiryPolicy()


def expiry_days(key: str, config_dir: Path | None = None) -> int:
    """The owner's value (``digest.yaml`` ``expiry:``) for a kind a plugin declares.

    ``KeyError`` when no installed plugin declares ``key`` under ``expiry:``.
    """
    return load_expiry_policy(config_dir).days(key)


def _zone(tz: tzinfo | None) -> tzinfo:
    if tz is not None:
        return tz
    from iris_harness.services.digest.settings import iris_timezone

    return iris_timezone()


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _grace_days(kind: ExpiryKind, policy: ExpiryPolicy) -> int:
    if kind == "task":
        return policy.task_overdue_days
    if kind == "bill":
        return policy.bill_ask_days + policy.bill_overdue_digest_days
    return policy.inbox_notice_days


def expires_at(
    kind: ExpiryKind,
    due_at: datetime | date | None,
    policy: ExpiryPolicy | None = None,
    *,
    end_at: datetime | None = None,
    tz: tzinfo | None = None,
) -> datetime | None:
    """The first instant the item counts as expired; ``None`` when it never does.

    ``prep`` / ``event``: ``end_at`` (the meeting's end; ``due_at`` — its start — when
    the end is unknown) plus the kind's minutes. The day kinds: the start of the local
    day after ``due_at``'s day plus the kind's days.
    """
    policy = policy or ExpiryPolicy()
    if kind == "reminder":
        return None  # counted in digests, not time: ``sweep_missed_reminders``
    if kind in ("prep", "event"):
        anchor = end_at or due_at
        if anchor is None:
            return None
        minutes = policy.prep_after_event if kind == "prep" else policy.event_after_end
        if not isinstance(anchor, datetime):
            # An all-day event given as a bare date ends when its local day does.
            anchor = datetime.combine(anchor + timedelta(days=1), time.min, tzinfo=_zone(tz))
        return _aware(anchor) + timedelta(minutes=minutes)
    if due_at is None:
        return None
    zone = _zone(tz)
    # A bare date (a bill's due date) is already the local day.
    due_day = _aware(due_at).astimezone(zone).date() if isinstance(due_at, datetime) else due_at
    last_day = due_day + timedelta(days=_grace_days(kind, policy))
    return datetime.combine(last_day + timedelta(days=1), time.min, tzinfo=zone)


def is_expired(
    kind: ExpiryKind,
    due_at: datetime | date | None,
    now: datetime,
    policy: ExpiryPolicy | None = None,
    *,
    end_at: datetime | None = None,
    tz: tzinfo | None = None,
) -> bool:
    """Whether a ``kind`` item dated ``due_at`` (ending ``end_at``) has expired at ``now``."""
    at = expires_at(kind, due_at, policy, end_at=end_at, tz=tz)
    return at is not None and _aware(now) >= at


def task_expiry_kind(task: Task) -> ExpiryKind | None:
    """The kind of dated item a task is; ``None`` when it never expires.

    Undated tasks never expire, nor do system-raised pending actions (a task with an
    ``action``: approvals keep their own lifecycle, ADR-0073/0118).
    """
    if task.due_at is None or task.action is not None:
        return None
    return _TASK_KINDS.get(task.source_kind or "", "task")


def is_task_expired(
    task: Task,
    now: datetime,
    policy: ExpiryPolicy | None = None,
    *,
    end_at: datetime | None = None,
    tz: tzinfo | None = None,
) -> bool:
    """Whether an open task has aged out (a closed one is never shown anyway)."""
    kind = task_expiry_kind(task)
    if kind is None:
        return False
    return is_expired(kind, task.due_at, now, policy, end_at=end_at, tz=tz)


def expiry_reason(kind: ExpiryKind, policy: ExpiryPolicy | None = None) -> str:
    """The ``closed_reason`` an expired item of ``kind`` gets."""
    policy = policy or ExpiryPolicy()
    if kind == "reminder":
        return _REASONS[kind].format(digests=policy.reminder_missed_digests)
    return _REASONS[kind].format(days=_grace_days(kind, policy))


def sweep_expired_tasks(
    store: TaskStore,
    *,
    now: datetime | None = None,
    policy: ExpiryPolicy | None = None,
    event_end: Callable[[Task], datetime | None] | None = None,
    tz: tzinfo | None = None,
) -> list[Task]:
    """Close every open/doing task that has aged out; returns the tasks it expired.

    Covers ANY past item, however old (the first run cleans up the backlog).
    ``event_end`` gives a prep task's meeting end (the calendar plugin passes one);
    without it a prep task expires at its meeting's start (its ``due_at``). Idempotent.
    """
    when = _aware(now or datetime.now(UTC))
    policy = policy or load_expiry_policy()
    expired: list[Task] = []
    for status in ("open", "doing"):
        for task in store.list(status=status, limit=100_000):
            kind = task_expiry_kind(task)
            if kind is None:
                continue
            end = event_end(task) if (kind == "prep" and event_end is not None) else None
            if not is_expired(kind, task.due_at, when, policy, end_at=end, tz=tz):
                continue
            expired.append(store.expire(task.id, expiry_reason(kind, policy)))
    if expired:
        logger.info("expiry: %d task(s) expired", len(expired))
    return expired


def sweep_missed_reminders(
    store: ReminderStore,
    *,
    now: datetime | None = None,
    policy: ExpiryPolicy | None = None,
) -> tuple[list[Reminder], list[str]]:
    """Age missed reminders by one digest; returns ``(expired, marked shown)``.

    Call it once per REAL morning digest, just before it renders (never on a re-run):
    a missed row already shown in ``reminder_missed_digests`` digests expires; every
    other missed row is marked shown in this one (it is about to be listed). With the
    default of 1, a reminder that failed on Monday is in Tuesday's digest and expired
    before Wednesday's. ``reminder_missed_digests: 0`` expires them unshown.
    """
    when = _aware(now or datetime.now(UTC))
    policy = policy or load_expiry_policy()
    limit = policy.reminder_missed_digests
    reason = expiry_reason("reminder", policy)
    expired: list[Reminder] = []
    shown: list[str] = []
    for reminder in store.list_missed(when):
        if reminder.missed_digests >= limit:
            expired.append(store.expire(reminder.id, reason))
        else:
            shown.append(reminder.id)
    if shown:
        store.mark_shown_in_digest(shown)
    if expired:
        logger.info("expiry: %d missed reminder(s) expired", len(expired))
    return expired, shown


def unacknowledged_cutoff(now: datetime | None = None, tz: tzinfo | None = None) -> datetime:
    """Start of the owner's local today: a delivered reminder due before it, still not
    answered, is "not acknowledged" (its day is over)."""
    when = _aware(now or datetime.now(UTC))
    zone = _zone(tz)
    return datetime.combine(when.astimezone(zone).date(), time.min, tzinfo=zone)


def sweep_unacknowledged_reminders(
    store: ReminderStore,
    *,
    now: datetime | None = None,
    policy: ExpiryPolicy | None = None,
    tz: tzinfo | None = None,
) -> list[Reminder]:
    """Age delivered, unanswered reminders by one digest; returns the rows it expired.

    Call it with :func:`sweep_missed_reminders`, just before each REAL morning digest.
    Every ``sent`` row due before the start of the owner's local today is one the owner
    never answered (Done / Snooze exist since PR 3b), so it is missed like a failed one:
    a row already shown in ``reminder_missed_digests`` digests expires, every other one
    is marked shown in the digest about to render ("sent Mon 8:00 AM, not
    acknowledged"). With the default of 1 it is listed once, then expired. A repeating
    series carries on: its next occurrence already exists (``ReminderStore.expire``
    makes sure).
    """
    when = _aware(now or datetime.now(UTC))
    policy = policy or load_expiry_policy()
    limit = policy.reminder_missed_digests
    expired: list[Reminder] = []
    shown: list[str] = []
    for reminder in store.list_sent_before(unacknowledged_cutoff(when, tz)):
        if reminder.missed_digests >= limit:
            expired.append(store.expire(reminder.id, UNACKNOWLEDGED_REMINDER_REASON, now=when))
        else:
            shown.append(reminder.id)
    if shown:
        store.mark_shown_in_digest(shown)
    if expired:
        logger.info("expiry: %d delivered, unacknowledged reminder(s) expired", len(expired))
    return expired


__all__ = [
    "EXPIRY_RANGES",
    "UNACKNOWLEDGED_REMINDER_REASON",
    "ExpiryKind",
    "ExpiryKindDeclaration",
    "ExpiryPolicy",
    "check_expiry_kind_name",
    "declared_expiry_kinds",
    "expires_at",
    "expiry_days",
    "expiry_reason",
    "expiry_view",
    "is_expired",
    "is_task_expired",
    "load_expiry_policy",
    "parse_expiry",
    "reset_declared_expiry_kinds",
    "sweep_expired_tasks",
    "sweep_missed_reminders",
    "sweep_unacknowledged_reminders",
    "task_expiry_kind",
    "unacknowledged_cutoff",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_declared_cache")
