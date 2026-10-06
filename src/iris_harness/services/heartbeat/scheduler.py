"""APScheduler-backed heartbeat scheduler."""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta, tzinfo
from typing import TYPE_CHECKING, Any, Protocol

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .models import HeartbeatDefinition, HeartbeatRun, HeartbeatStatus
from .schedule_text import validate_schedule

if TYPE_CHECKING:
    from iris_harness.foundation.settings import SettingsStore

    from .run_store import HeartbeatRunStore, StoredRun
    from .slots import JobStatus

logger = logging.getLogger(__name__)


class HeartbeatHandler(Protocol):
    """Callable that executes a heartbeat and returns the resulting run."""

    def __call__(self, definition: HeartbeatDefinition) -> HeartbeatRun: ...


HandlerFactory = Callable[[], HeartbeatHandler]


# How late (seconds) a missed interval run may still execute on wake before it is
# discarded. One hour comfortably covers a closed-laptop gap while bounding stale
# catch-up runs of slow periodic jobs.
_MISFIRE_GRACE_SECONDS = 3600
# The one-off job that re-runs a skipped heartbeat (``retry_skipped_after_minutes``).
RETRY_JOB_SUFFIX = "__retry"

# The settings-store section heartbeat overrides live in (ADR-0120). Each row is keyed by
# heartbeat name and holds only the fields that differ from the declared definition, so
# a later deploy that changes a field the owner never touched still takes effect.
SETTINGS_SECTION = "heartbeat"
_EDITABLE = ("schedule", "enabled")

# An interval faster than this keeps only its status changes (``record_runs`` unset).
_RECORD_EVERY_RUN_FROM_SECONDS = 300


def _iris_timezone() -> tzinfo:
    from iris_harness.services.digest.settings import iris_timezone

    return iris_timezone()


def records_every_run(definition: HeartbeatDefinition) -> bool:
    """Whether every run of ``definition`` is kept, or only its status changes.

    ``record_runs`` in heartbeats.yaml decides when set; otherwise every cron job and
    every interval of 5 minutes or more keeps every run, a faster tick only changes.
    """
    if definition.record_runs is not None:
        return definition.record_runs
    schedule = definition.schedule.strip()
    if not schedule.startswith("interval:"):
        return True
    try:
        return int(schedule.split(":", 1)[1]) >= _RECORD_EVERY_RUN_FROM_SECONDS
    except ValueError:
        return True


class HeartbeatScheduler:
    """Registers heartbeat definitions with APScheduler triggers.

    Handlers are looked up via the ``handler`` field on each definition.
    Pass ``scheduler=None`` to defer APScheduler instantiation (useful in tests).

    Three views of the definitions, because the app can change them (ADR-0120):

    * *declared* — as ``heartbeats.yaml`` or a plugin registered it, the default;
    * *effective* — declared plus the owner's saved override, disabled ones included;
      :meth:`all_definitions` lists these;
    * *scheduled* — the effective definitions that are on and have a handler;
      :meth:`list_definitions` lists these, as it always has.

    With ``settings`` given, overrides are read at registration (so they survive a
    restart and a fresh container) and :meth:`update` / :meth:`reset` save them.
    """

    def __init__(
        self,
        *,
        scheduler: BackgroundScheduler | None = None,
        handlers: dict[str, HeartbeatHandler] | None = None,
        settings: SettingsStore | None = None,
        platform: str | None = None,
        run_store: HeartbeatRunStore | None = None,
        timezone: tzinfo | None = None,
    ) -> None:
        self._scheduler = scheduler
        # Cron schedules are read in the owner's zone (IRIS_TZ), never the machine's:
        # the VM's clock is UTC unless compose sets TZ, and a Mac's is wherever it is.
        # Every cron trigger is built with this zone, so the APScheduler's own zone
        # (the machine's by default) never decides a wall-clock time.
        self._tz: tzinfo = timezone or _iris_timezone()
        # Kept runs (loop-proof D13): survive a restart; None keeps memory only (tests).
        self._run_store = run_store
        # The status last kept per fast tick, so only its changes are written.
        self._last_kept_status: dict[str, str] = {}
        # ``sys.platform`` unless a test pins another, so the platform lock is testable.
        self._platform = platform or sys.platform
        self._handlers: dict[str, HeartbeatHandler] = dict(handlers or {})
        # Why a plugin is not mounted here (None when it is): bound by the runtime once
        # plugins have mounted. Unbound, a missing handler cannot be explained and warns.
        self._plugin_gap: Callable[[str], str | None] | None = None
        self._runs: list[HeartbeatRun] = []
        self._definitions: dict[str, HeartbeatDefinition] = {}
        self._declared: dict[str, HeartbeatDefinition] = {}
        self._effective: dict[str, HeartbeatDefinition] = {}
        self._settings = settings
        # Updates come from API threads while APScheduler's own thread runs jobs; the
        # three maps change together under this lock.
        self._lock = threading.RLock()
        # Runs in progress per heartbeat name (any trigger), so a caller can wait for
        # one to finish instead of starting a second, concurrent run of the same job.
        self._running: dict[str, int] = {}
        self._idle = threading.Condition(threading.Lock())
        self._created_at = datetime.now(UTC)

    # ------------------------------------------------------------------
    # Handler registry
    # ------------------------------------------------------------------

    def register_handler(self, name: str, handler: HeartbeatHandler) -> None:
        self._handlers[name] = handler

    def has_handler(self, name: str) -> bool:
        return name in self._handlers

    def bind_plugin_gap(self, plugin_gap: Callable[[str], str | None]) -> None:
        """Say how to tell a plugin is not mounted: ``plugin_gap(name)`` returns why (a
        sentence) or ``None`` when ``name`` is mounted. Lets a definition whose handler
        belongs to an absent plugin be skipped quietly instead of warning."""
        self._plugin_gap = plugin_gap

    def _missing_plugin(self, definition: HeartbeatDefinition) -> str | None:
        """Why the plugin that owns ``definition`` is not mounted, or None when no plugin
        is named, it is mounted, or there is no way to tell (nothing bound)."""
        if not definition.plugin or self._plugin_gap is None:
            return None
        gap = self._plugin_gap(definition.plugin)
        return None if gap is None else f"plugin {definition.plugin} is not mounted: {gap}"

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def register(self, definition: HeartbeatDefinition) -> bool:
        """Register a definition, with the owner's saved override applied.

        Returns False when the effective definition is disabled or no handler is bound;
        it is still listed by :meth:`all_definitions`, so the app can turn it on.
        """
        with self._lock:
            self._declared[definition.name] = definition
            effective = self._with_override(definition)
            self._effective[definition.name] = effective
            if self._run_store is not None:
                self._run_store.note_job(definition.name)
            return self._apply(effective)

    def unavailable_reason(self, definition: HeartbeatDefinition) -> str | None:
        """Why ``definition`` cannot run on this harness, or None when it can.

        Two reasons: it declares ``platforms`` this machine is not, or no handler for it
        is registered. The app locks the switch and shows this sentence.
        """
        if definition.platforms and self._platform not in definition.platforms:
            return (
                f"needs {' or '.join(definition.platforms)}; "
                f"this harness runs on {self._platform}"
            )
        if definition.handler not in self._handlers:
            gap = self._missing_plugin(definition)
            if gap is not None:
                return gap
            return f"no handler {definition.handler!r} is registered on this harness"
        return None

    def _apply(self, definition: HeartbeatDefinition) -> bool:
        """Schedule ``definition`` if it is on and can run here, else unschedule it."""
        if not definition.enabled:
            logger.debug("heartbeat %s disabled; skipping registration", definition.name)
            self._unschedule(definition.name)
            return False
        if definition.platforms and self._platform not in definition.platforms:
            logger.info(
                "heartbeat %s is on but needs %s; not scheduling it on %s",
                definition.name,
                "/".join(definition.platforms),
                self._platform,
            )
            self._unschedule(definition.name)
            return False
        handler = self._handlers.get(definition.handler)
        if handler is None:
            gap = self._missing_plugin(definition)
            if gap is not None:
                # Not a fault: config/heartbeats.yaml declares the schedules of every
                # domain's jobs, and a handler arrives with its plugin, so on a harness
                # without that plugin the job is simply unavailable. That state is shown
                # (unavailable_reason, the Heartbeats screen) and register_all() logs one
                # summary line; a plugin that failed to load is reported by the plugin host.
                logger.debug("heartbeat %s not scheduled; %s", definition.name, gap)
            elif definition.plugin and self._plugin_gap is not None:
                # Its plugin is mounted and still did not register the handler: a bug.
                logger.warning(
                    "heartbeat %s: plugin %s is mounted but registered no handler %s; skipping",
                    definition.name,
                    definition.plugin,
                    definition.handler,
                )
            else:
                # No owner named (a typo), or no way to tell: a real fault.
                logger.warning(
                    "heartbeat %s references unknown handler %s; skipping",
                    definition.name,
                    definition.handler,
                )
            self._unschedule(definition.name)
            return False
        self._definitions[definition.name] = definition
        if self._scheduler is not None:
            trigger = self._build_trigger(definition.schedule)
            self._scheduler.add_job(
                func=lambda d=definition, h=handler: self._invoke(d, h, trigger="schedule"),
                trigger=trigger,
                id=definition.name,
                replace_existing=True,
                # On a laptop that sleeps, an interval tick scheduled while asleep
                # is "missed". With APScheduler's 1s default grace such misfires are
                # dropped and only the next boundary runs — so a due reminder fires
                # up to one interval late after wake. A generous grace + coalesce
                # makes the missed run execute once, immediately on resume, so
                # catch-up firing (notification_reminder_tick) is as prompt as the
                # machine being awake allows.
                misfire_grace_time=_MISFIRE_GRACE_SECONDS,
                coalesce=True,
            )
        return True

    def register_all(self, definitions: list[HeartbeatDefinition]) -> int:
        registered = sum(1 for d in definitions if self.register(d))
        with self._lock:
            effective = [self._effective.get(d.name, d) for d in definitions]
            by_plugin: dict[str, list[str]] = {}
            for d in effective:
                if (
                    d.enabled
                    and d.handler not in self._handlers
                    and not (d.platforms and self._platform not in d.platforms)
                    and self._missing_plugin(d) is not None
                ):
                    by_plugin.setdefault(d.plugin, []).append(d.name)
        if by_plugin:
            logger.info(
                "%d heartbeat(s) unavailable, plugin not mounted (%s): %s",
                sum(len(names) for names in by_plugin.values()),
                ", ".join(sorted(by_plugin)),
                ", ".join(sorted(n for names in by_plugin.values() for n in names)),
            )
        return registered

    def _unschedule(self, name: str) -> None:
        self._definitions.pop(name, None)
        if self._scheduler is None:
            return
        if self._scheduler.get_job(name) is not None:
            self._scheduler.remove_job(name)
        self._cancel_retry(name)

    def _with_override(self, definition: HeartbeatDefinition) -> HeartbeatDefinition:
        """``definition`` with the saved override applied, or unchanged.

        A saved value that no longer validates (hand-edited, or a rule tightened since)
        is ignored with a warning rather than failing startup: the declared schedule is
        a safe place to land, and the app still shows the heartbeat.
        """
        if self._settings is None:
            return definition
        try:
            saved = self._settings.get(SETTINGS_SECTION, definition.name)
        except Exception:  # a broken store must not stop the heartbeats
            logger.exception("heartbeat %s: could not read its saved override", definition.name)
            return definition
        if not isinstance(saved, dict):
            return definition
        try:
            schedule = (
                validate_schedule(str(saved["schedule"]))
                if "schedule" in saved
                else definition.schedule
            )
        except ValueError as exc:
            logger.warning(
                "heartbeat %s: ignoring saved schedule %r (%s)",
                definition.name,
                saved["schedule"],
                exc,
            )
            schedule = definition.schedule
        enabled = bool(saved["enabled"]) if "enabled" in saved else definition.enabled
        return replace(definition, schedule=schedule, enabled=enabled)

    # ------------------------------------------------------------------
    # Owner edits (ADR-0120)
    # ------------------------------------------------------------------

    def all_definitions(self) -> list[HeartbeatDefinition]:
        """Every registered heartbeat as it now stands, disabled ones included."""
        with self._lock:
            return list(self._effective.values())

    def declared(self, name: str) -> HeartbeatDefinition | None:
        """The definition as shipped (``heartbeats.yaml`` or its plugin), the default."""
        return self._declared.get(name)

    def next_run_at(self, name: str) -> datetime | None:
        """When the scheduler will next run ``name``; None when it is not scheduled."""
        if self._scheduler is None:
            return None
        job = self._scheduler.get_job(name)
        # A job added before the scheduler starts has no next_run_time attribute yet.
        return getattr(job, "next_run_time", None) if job is not None else None

    def timezone(self) -> str | None:
        """The zone a cron schedule is read in (IRIS_TZ unless one was given)."""
        return str(self._tz)

    def tz(self) -> tzinfo:
        """The zone object cron schedules are read in."""
        return self._tz

    def update(
        self,
        name: str,
        *,
        schedule: str | None = None,
        enabled: bool | None = None,
        actor: str,
    ) -> HeartbeatDefinition:
        """Change a heartbeat's schedule and/or on-off state, now and after restarts.

        Raises ``KeyError`` for an unknown heartbeat and ``ValueError`` for a schedule
        that does not validate or for turning on a heartbeat whose handler this harness
        does not have. Nothing is saved or rescheduled unless every check passes. An
        edit that lands back on the declared values clears the override instead.
        """
        with self._lock:
            declared = self._declared.get(name)
            if declared is None:
                raise KeyError(name)
            current = self._effective[name]
            candidate = replace(
                current,
                schedule=validate_schedule(schedule) if schedule is not None else current.schedule,
                enabled=current.enabled if enabled is None else bool(enabled),
            )
            reason = self.unavailable_reason(candidate) if candidate.enabled else None
            if reason is not None:
                raise ValueError(f"heartbeat {name!r} cannot run on this harness: {reason}")
            if candidate == current:
                return current
            self._save(declared, current, candidate, actor=actor)
            self._effective[name] = candidate
            self._apply(candidate)
            logger.info(
                "heartbeat %s changed by %s: %s",
                name,
                actor,
                _value(candidate),
            )
            return candidate

    def reset(self, name: str, *, actor: str) -> HeartbeatDefinition:
        """Drop the owner's override: the declared definition applies again."""
        with self._lock:
            declared = self._declared.get(name)
            if declared is None:
                raise KeyError(name)
            current = self._effective[name]
            if current == declared:
                return current
            self._save(declared, current, declared, actor=actor)
            self._effective[name] = declared
            self._apply(declared)
            logger.info("heartbeat %s reset by %s", name, actor)
            return declared

    def _save(
        self,
        declared: HeartbeatDefinition,
        current: HeartbeatDefinition,
        target: HeartbeatDefinition,
        *,
        actor: str,
    ) -> None:
        if self._settings is None:
            return
        diff = {
            f: getattr(target, f) for f in _EDITABLE if getattr(target, f) != getattr(declared, f)
        }
        if diff:
            self._settings.set(
                SETTINGS_SECTION,
                declared.name,
                diff,
                old=_value(current),
                new=_value(target),
                actor=actor,
            )
        else:
            self._settings.clear(
                SETTINGS_SECTION,
                declared.name,
                old=_value(current),
                new=_value(target),
                actor=actor,
            )

    # ------------------------------------------------------------------
    # Direct invocation (used for tests + manual runs)
    # ------------------------------------------------------------------

    def trigger_by_name(self, name: str, *, trigger: str = "manual") -> HeartbeatRun | None:
        """Trigger a registered heartbeat by name. Returns None if not found.

        ``trigger`` is what the kept run says started it (``manual`` for Run now).
        """
        definition = self._definitions.get(name)
        if definition is None:
            return None
        return self.trigger_now(definition, trigger=trigger)

    def list_definitions(self) -> list[HeartbeatDefinition]:
        return list(self._definitions.values())

    def trigger_now(
        self, definition: HeartbeatDefinition, *, trigger: str = "manual"
    ) -> HeartbeatRun:
        handler = self._handlers.get(definition.handler)
        if handler is None:
            run = HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.SKIPPED,
                finished_at=datetime.now(UTC),
                error=f"no handler registered for {definition.handler!r}",
                trigger=trigger,
            )
            self._runs.append(run)
            self._keep(definition, run)
            return run
        return self._invoke(definition, handler, trigger=trigger)

    def is_running(self, name: str) -> bool:
        """True while a run of ``name`` (scheduled, Run now or any other) is in progress."""
        with self._idle:
            return self._running.get(name, 0) > 0

    def wait_until_idle(self, name: str, timeout: float) -> bool:
        """Block until no run of ``name`` is in progress, at most ``timeout`` seconds.

        Returns True when it is idle (at once, or once the run finished), False when
        the time ran out first. Never interrupts the run.
        """
        with self._idle:
            return self._idle.wait_for(
                lambda: self._running.get(name, 0) == 0, timeout=max(timeout, 0.0)
            )

    def runs(self) -> list[HeartbeatRun]:
        """This process's runs, in memory (every tick, lost on restart)."""
        return list(self._runs)

    # ------------------------------------------------------------------
    # Kept runs (loop-proof D13)
    # ------------------------------------------------------------------

    @property
    def run_store(self) -> HeartbeatRunStore | None:
        return self._run_store

    def kept_runs(
        self,
        *,
        name: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[StoredRun]:
        """Kept runs, newest first; empty without a store or when it cannot be read."""
        if self._run_store is None:
            return []
        try:
            return self._run_store.recent(name=name, since=since, limit=limit)
        except Exception:  # a broken store must not break a read surface
            logger.warning("could not read kept heartbeat runs", exc_info=True)
            return []

    def job_status(
        self,
        name: str,
        *,
        grace: timedelta = timedelta(minutes=30),
        now: datetime | None = None,
    ) -> JobStatus | None:
        """Did ``name``'s most recent slot run (``slots.evaluate_job``)? None without a
        store."""
        if self._run_store is None:
            return None
        from .slots import evaluate_job

        with self._lock:
            definition = self._effective.get(name)
        unavailable = (
            self.unavailable_reason(definition)
            if definition is not None and definition.enabled
            else None
        )
        return evaluate_job(
            definition,
            self._run_store,
            now=now or datetime.now(UTC),
            tz=self._tz,
            grace=grace,
            unavailable=unavailable,
            name=name,
        )

    def created_at(self) -> datetime:
        """When this scheduler instance was created.

        Used by health diagnostics to decide whether a heartbeat has had enough
        time to run at least once.
        """

        return self._created_at

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._scheduler is not None and not self._scheduler.running:
            self._scheduler.start()

    def shutdown(self, *, wait: bool = False) -> None:
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _invoke(
        self,
        definition: HeartbeatDefinition,
        handler: HeartbeatHandler,
        *,
        trigger: str = "manual",
    ) -> HeartbeatRun:
        started = datetime.now(UTC)
        with self._idle:
            self._running[definition.name] = self._running.get(definition.name, 0) + 1
        try:
            run = handler(definition)
        except Exception as exc:  # noqa: BLE001 — handlers must not crash the scheduler
            run = HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.FAILED,
                started_at=started,
                finished_at=datetime.now(UTC),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            with self._idle:
                left = self._running.get(definition.name, 1) - 1
                if left > 0:
                    self._running[definition.name] = left
                else:
                    self._running.pop(definition.name, None)
                self._idle.notify_all()
        if run.finished_at is None:
            run.finished_at = datetime.now(UTC)
        # A handler that builds its run at the end stamps started_at late; the scheduler
        # knows when it really began, which is what "ran at/after the slot" compares.
        if run.started_at > started:
            run.started_at = started
        run.trigger = trigger
        self._runs.append(run)
        self._keep(definition, run)
        if run.status == HeartbeatStatus.SKIPPED:
            self._schedule_retry(definition)
        else:
            self._cancel_retry(definition.name)  # it got through: no retry is owed
        return run

    def _cancel_retry(self, name: str) -> None:
        job_id = f"{name}{RETRY_JOB_SUFFIX}"
        if self._scheduler is not None and self._scheduler.get_job(job_id) is not None:
            self._scheduler.remove_job(job_id)

    def _schedule_retry(self, definition: HeartbeatDefinition) -> None:
        """One more run ``retry_skipped_after_minutes`` from now, unless the job's next
        scheduled slot comes first. A retry that skips again schedules the next one."""
        minutes = definition.retry_skipped_after_minutes
        if not minutes or self._scheduler is None:
            return
        at = datetime.now(UTC) + timedelta(minutes=minutes)
        job = self._scheduler.get_job(definition.name)
        next_slot = getattr(job, "next_run_time", None) if job is not None else None
        if next_slot is not None and next_slot <= at:
            return
        name = definition.name

        def retry() -> None:
            # Read the job as it is then: turned off or changed since the skip.
            current = self._definitions.get(name)
            handler = self._handlers.get(current.handler) if current is not None else None
            if current is None or handler is None or not current.enabled:
                return
            self._invoke(current, handler, trigger="retry")

        self._scheduler.add_job(
            func=retry,
            trigger=DateTrigger(run_date=at),
            id=f"{name}{RETRY_JOB_SUFFIX}",
            replace_existing=True,
            misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        )
        logger.info("heartbeat %s skipped; retrying at %s", name, at.isoformat())

    def _keep(self, definition: HeartbeatDefinition, run: HeartbeatRun) -> None:
        """Write ``run`` to the run store when its policy says so. Never raises."""
        if self._run_store is None:
            return
        try:
            if not records_every_run(definition):
                if self._last_kept_status.get(definition.name) == run.status.value:
                    return
            slot = self._slot_for(definition, run)
            if self._run_store.record(run, slot=slot):
                self._last_kept_status[definition.name] = run.status.value
        except Exception:  # keeping the record must never fail the job
            logger.warning("could not keep heartbeat run %s", definition.name, exc_info=True)

    def _slot_for(self, definition: HeartbeatDefinition, run: HeartbeatRun) -> datetime | None:
        """The cron slot a run served: the latest scheduled time at or before its start."""
        from .slots import recent_slots

        if definition.schedule.strip().startswith("interval:"):
            return None
        found = recent_slots(definition.schedule, run.started_at, self._tz, count=1)
        return found[-1] if found else None

    def _build_trigger(self, schedule: str) -> CronTrigger | IntervalTrigger:
        schedule = schedule.strip()
        if schedule.startswith("interval:"):
            try:
                seconds = int(schedule.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"invalid interval schedule {schedule!r}") from exc
            return IntervalTrigger(seconds=seconds)
        return CronTrigger.from_crontab(schedule, timezone=self._tz)


def _value(definition: HeartbeatDefinition) -> dict[str, Any]:
    """The editable part of a definition, as the history records it."""
    return {field: getattr(definition, field) for field in _EDITABLE}
