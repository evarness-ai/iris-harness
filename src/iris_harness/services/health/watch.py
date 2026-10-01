"""The health watch — notice, repair, and tell the owner (ADR-0116).

ADR-0069 built the snapshot and left it pull-only: a revoked Gmail token showed as a
red row on a screen nobody was looking at. The watch runs after every
``health_tick`` refresh and walks each red check through one incident:

1. **Confirm** — red for ``confirm_ticks`` samples in a row (a service mid-restart
   is not an incident).
2. **Repair** — the registered repairers get up to ``max_attempts`` tries, one per
   tick, each verified by the next snapshot rather than by the repairer's say-so.
3. **Engage** — fixes ran out (or one said retrying cannot help): every configured
   channel gets one notice with what was tried and the exact fix command. It is
   repeated only every ``renotify_hours`` while the check stays red.
4. **Close** — the check is green (or gone): the incident is resolved, and a
   "working again" notice follows if the owner had been told.

**Resume.** The watch runs inside ``health_tick``, so it cannot see a gap while it is in
one. When its own passes stop for ``resume_gap_minutes`` (a laptop asleep, a stalled
process), every heartbeat reads as overdue on the first pass back and the network may not
be up yet. That pass resets the red streaks and holds new incidents for
``resume_grace_minutes`` so the scheduler and network can catch up. A real fault is still
red after the grace and opens an incident as usual.

**Startup.** A deploy or restart starts every container at once, so the first passes of a
new process see siblings (the channel gateway above all) still coming up. The first pass
after process start opens the same kind of hold for ``startup_grace_minutes`` (0 = off in
code; 5 in the shipped configs). Nothing is opened inside a hold, so a restart's
transient red never becomes an incident and never counts toward "the Nth time in 7 days".

Config: ``config/health_watch.yaml``; kill-switch ``IRIS_HEALTH_WATCH_ENABLED=0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris_harness.foundation.process_state import track_globals

if TYPE_CHECKING:
    from iris_harness.foundation.settings.store import SettingsStore

import logging
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from iris_harness.foundation.clock import utc_now
from iris_harness.foundation.paths import config_dir as resolve_config_dir
from iris_harness.foundation.public_url import absolute_console_url
from iris_harness.services.health.incidents import (
    NEEDS_USER,
    REPAIRING,
    RESOLVED,
    SELF_HEALED,
    USER_FIXED,
    Incident,
    IncidentStore,
)
from iris_harness.services.health.models import HealthCheck, HealthSnapshot, HealthState
from iris_harness.services.health.repair import (
    Repairer,
    RepairOutcome,
    registered_repairers,
    run_repairs,
)

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WatchConfig:
    """``config/health_watch.yaml``, with the built-in defaults below."""

    enabled: bool = True
    confirm_ticks: int = 2
    max_attempts: int = 2
    channels: tuple[str, ...] | None = None  # None = every registered channel
    renotify_hours: float = 12.0
    notify_self_healed: bool = False
    notify_recovered: bool = True
    heartbeat_retry: bool = True
    skip_heartbeats: tuple[str, ...] = ("health_tick",)
    restart: dict[str, tuple[str, ...]] = field(default_factory=dict)
    never_restart: tuple[str, ...] = ("iris_api",)
    diagnose_credentials: bool = True
    recurring_days: int = 7
    ignore: tuple[str, ...] = ()
    resume_gap_minutes: float = 5.0
    resume_grace_minutes: float = 10.0
    startup_grace_minutes: float = 0.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> WatchConfig:
        notify = raw.get("notify") or {}
        repair = raw.get("repair") or {}
        channels = notify.get("channels", "all")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            confirm_ticks=max(1, int(raw.get("confirm_ticks", 2))),
            max_attempts=max(0, int(repair.get("max_attempts", 2))),
            channels=None if channels in (None, "all") else tuple(str(c) for c in channels),
            renotify_hours=float(notify.get("renotify_hours", 12)),
            notify_self_healed=bool(notify.get("self_healed", False)),
            notify_recovered=bool(notify.get("recovered", True)),
            heartbeat_retry=bool(repair.get("heartbeat_retry", True)),
            skip_heartbeats=tuple(repair.get("skip_heartbeats") or ("health_tick",)),
            restart={
                str(k): tuple(str(a) for a in v)
                for k, v in (repair.get("restart") or {}).items()
                if v
            },
            never_restart=tuple(repair.get("never_restart") or ("iris_api",)),
            diagnose_credentials=bool(raw.get("diagnose_credentials", True)),
            recurring_days=max(1, int(raw.get("recurring_days", 7))),
            ignore=tuple(str(t) for t in raw.get("ignore") or ()),
            resume_gap_minutes=max(1.0, float(raw.get("resume_gap_minutes", 5))),
            resume_grace_minutes=max(0.0, float(raw.get("resume_grace_minutes", 10))),
            startup_grace_minutes=max(0.0, float(raw.get("startup_grace_minutes", 0))),
        )


def load_watch_config(
    config_dir: Path | None = None,
    *,
    settings: SettingsStore | None = None,
    with_edits: bool = True,
) -> WatchConfig:
    """Read ``health_watch.yaml`` from ``config_dir`` (or ``foundation.paths.config_dir()``),
    with the owner's saved edits laid over it (ADR-0120) unless ``with_edits`` is off —
    the file alone is what an edit is compared with and a reset returns to."""
    base = config_dir or resolve_config_dir()
    path = base / "health_watch.yaml"
    raw: dict[str, Any] = {}
    if path.exists():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s — using built-in watch defaults", path, exc)
    config = WatchConfig.from_mapping(raw)
    if with_edits:
        from iris_harness.foundation.settings.store import (
            SettingsStore as _Store,
        )
        from iris_harness.services.health.watch_edits import apply_saved

        config = apply_saved(config, settings or _Store())
    env = os.environ.get("IRIS_HEALTH_WATCH_ENABLED", "").strip().lower()
    if env:
        config = WatchConfig(**{**config.__dict__, "enabled": env in _TRUTHY})
    return config


class Notifier(Protocol):
    """Sends one notice. ``url`` is the console page that fixes what the notice is
    about (a relative path), when the check named one; a channel that can open a
    page on a tap (web push) opens it."""

    def __call__(
        self,
        subject: str,
        body: str,
        channels: Sequence[str] | None,
        url: str | None = None,
    ) -> None: ...


def _label(item: HealthCheck | Incident) -> str:
    return f"{item.target} ({item.subject})" if item.subject else item.target


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# An error that already names its own fix: "… — run `iris auth gmail login --user a@b`
# to reconnect." (CredentialRevokedError and friends). A command with a <placeholder>
# is a template, not a fix, and is ignored.
_EMBEDDED_FIX = re.compile(r"\brun `([^`<>]+)`", re.IGNORECASE)


def _tried_text(incident: Incident) -> str:
    """What the repairs did, identical attempts folded ("re-run x ×2 (failed)"), and a
    failure's detail left out when it only repeats what the first line already says."""
    folded: dict[tuple[str, bool, str], int] = {}
    for r in incident.repairs:
        key = (str(r.get("tried", "")), bool(r.get("ok")), str(r.get("detail") or ""))
        folded[key] = folded.get(key, 0) + 1
    parts = []
    for (tried, ok, detail), count in folded.items():
        times = f" ×{count}" if count > 1 else ""
        why = f": {detail}" if detail and not ok and detail not in incident.detail else ""
        parts.append(f"{tried}{times} ({'ran' if ok else 'failed'}{why})")
    return "; ".join(parts)


def _fix_lines(incident: Incident) -> list[str]:
    """The command to run. When the failure names its own fix (a revoked token's
    re-login), that comes first and the check's generic action becomes the retry."""
    embedded = _EMBEDDED_FIX.search(incident.detail)
    fix = embedded.group(1).strip() if embedded else None
    if fix and fix != incident.action:
        lines = [f"To fix it, run: {fix}"]
        if incident.action:
            lines.append(f"Then retry with: {incident.action}")
        return lines
    if incident.action:
        return [f"To fix it, run: {incident.action}"]
    return ["`iris status` shows the details."]


class HealthWatcher:
    """Walks red checks through incidents. Thread-safe; one per process."""

    def __init__(
        self,
        *,
        store: IncidentStore,
        config: WatchConfig,
        notify: Notifier | None = None,
        repairers: Sequence[Repairer] = (),
        diagnose: Callable[[], None] | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self._notify = notify
        self._core_repairers = list(repairers)
        self._diagnose = diagnose
        self._streaks: dict[str, int] = {}
        self._last_pass: datetime | None = None
        self._grace_until: datetime | None = None
        self._lock = Lock()

    # ── the pass ─────────────────────────────────────────────────────────
    def observe(self, snapshot: HealthSnapshot, *, now: datetime | None = None) -> list[str]:
        """Advance every incident one step; returns a line per thing that happened."""
        if not self.config.enabled:
            return []
        when = now or utc_now()
        with self._lock:
            return self._observe(snapshot, when)

    def _watched(self, check: HealthCheck) -> bool:
        return check.state is HealthState.RED and not (
            check.target in self.config.ignore or check.key in self.config.ignore
        )

    def _resumed(self, now: datetime) -> list[str]:
        """Start the grace window on the process's first pass (``startup_grace_minutes``)
        or when this pass follows a gap in the watch's own passes."""
        last, self._last_pass = self._last_pass, now
        if last is None:
            # The first pass of this process: siblings started with it may still be
            # coming up (a deploy restarts every container at once).
            if self.config.startup_grace_minutes <= 0:
                return []
            self._streaks = {}
            self._grace_until = now + timedelta(minutes=self.config.startup_grace_minutes)
            return [f"started: holding new incidents for {self.config.startup_grace_minutes:g} min"]
        gap = timedelta(minutes=self.config.resume_gap_minutes)
        if now - last < gap:
            return []
        self._streaks = {}
        self._grace_until = now + timedelta(minutes=self.config.resume_grace_minutes)
        return [f"resumed after {int((now - last).total_seconds())}s gap: holding new incidents"]

    def _in_grace(self, now: datetime) -> bool:
        return self._grace_until is not None and now < self._grace_until

    def _observe(self, snapshot: HealthSnapshot, now: datetime) -> list[str]:
        events = self._resumed(now)
        reds = {c.key: c for c in snapshot.checks if self._watched(c)}
        self._streaks = {key: self._streaks.get(key, 0) + 1 for key in reds}
        open_by_key = {i.key: i for i in self.store.open_incidents()}

        for key, incident in open_by_key.items():
            if key not in reds:
                events.append(self._resolve(incident, now))

        if self._in_grace(now):
            # Just back from a gap: overdue heartbeats and a network still coming up are
            # the gap, not a fault. Close what went green; open and repair nothing yet.
            self._streaks = {}
            return events

        opened_any = False
        for key, check in reds.items():
            existing = open_by_key.get(key)
            if existing is None:
                if self._streaks[key] < self.config.confirm_ticks:
                    continue
                incident = self.store.open(
                    key=key,
                    target=check.target,
                    subject=check.subject,
                    kind=check.kind.value,
                    detail=check.detail,
                    action=check.action,
                    now=now,
                )
                opened_any = True
                events.append(f"opened {key}: {check.detail}")
            else:
                incident = existing
                incident.detail = check.detail
                incident.action = check.action or incident.action
            events.extend(self._step(incident, check, now))

        if opened_any and self.config.diagnose_credentials and self._diagnose is not None:
            # One live credential probe per pass that opened something: a failed sweep
            # is most often a revoked token, and the probe turns that into its own row
            # with the exact re-auth command (read on the next pass).
            try:
                self._diagnose()
            except Exception:
                logger.warning("health watch: credential diagnosis failed", exc_info=True)
        return events

    def _step(self, incident: Incident, check: HealthCheck, now: datetime) -> list[str]:
        events: list[str] = []
        if incident.state == REPAIRING:
            outcome: RepairOutcome | None = None
            if incident.attempts < self.config.max_attempts:
                outcome = run_repairs(check, self._core_repairers + registered_repairers())
            if outcome is not None:
                incident.attempts += 1
                incident.repairs.append({"at": now.isoformat(), **outcome.as_dict()})
                events.append(f"repair {incident.key}: {outcome.tried} → ok={outcome.ok}")
            if outcome is None or outcome.final:
                incident.state = NEEDS_USER
        if incident.state == NEEDS_USER:
            events.extend(self._engage(incident, now, fix_url=check.fix_url))
        self.store.save(incident, now=now)
        return events

    def _engage(
        self, incident: Incident, now: datetime, *, fix_url: str | None = None
    ) -> list[str]:
        if incident.notified_at is not None:
            last = datetime.fromisoformat(incident.notified_at)
            if now - last < timedelta(hours=self.config.renotify_hours):
                return []
        self._send(
            "IRIS needs your help",
            self._needs_user_text(incident, now, fix_url=fix_url),
            url=fix_url,
        )
        incident.notified_at = now.isoformat()
        incident.notify_count += 1
        return [f"notified {incident.key} (#{incident.notify_count})"]

    def _resolve(self, incident: Incident, now: datetime) -> str:
        told = incident.notify_count > 0
        incident.resolution = USER_FIXED if incident.state == NEEDS_USER else SELF_HEALED
        incident.state = RESOLVED
        incident.resolved_at = now.isoformat()
        if told and self.config.notify_recovered:
            self._send("IRIS is back", f"{_label(incident)} is working again.")
        elif incident.resolution == SELF_HEALED and self.config.notify_self_healed:
            tried = "; ".join(r.get("tried", "") for r in incident.repairs) or "no action"
            self._send("IRIS fixed itself", f"{_label(incident)} failed and recovered ({tried}).")
        self.store.save(incident, now=now)
        return f"resolved {incident.key}: {incident.resolution}"

    # ── wording ──────────────────────────────────────────────────────────
    def _needs_user_text(
        self, incident: Incident, now: datetime, *, fix_url: str | None = None
    ) -> str:
        lines = [f"{_label(incident)} is not working: {incident.detail}"]
        if incident.repairs:
            lines.append(f"I tried: {_tried_text(incident)} — it is still failing.")
        else:
            lines.append("I have no automatic fix for this one.")
        lines.extend(_fix_lines(incident))
        if fix_url:
            # Beside the command, not instead of it: the command is the Mac path, the
            # page works from the phone the alert arrived on.
            lines.append(f"Or fix it in the app: {absolute_console_url(fix_url)}")
        seen = self.store.count_since(incident.key, days=self.config.recurring_days, now=now)
        if seen > 1:
            lines.append(f"This is the {_ordinal(seen)} time in {self.config.recurring_days} days.")
        return "\n".join(lines)

    def _send(self, subject: str, body: str, *, url: str | None = None) -> None:
        if self._notify is None:
            logger.warning("health watch (no notifier): %s — %s", subject, body)
            return
        try:
            self._notify(subject, body, self.config.channels, url=url)
        except Exception:  # a failed send must not stop the watch
            logger.warning("health watch: notify failed", exc_info=True)


_installed: HealthWatcher | None = None
_install_lock = Lock()


def install_watcher(watcher: HealthWatcher | None) -> None:
    """Publish the process's watcher (the API reads incidents through it)."""
    global _installed
    with _install_lock:
        _installed = watcher


def current_watcher() -> HealthWatcher | None:
    with _install_lock:
        return _installed


def broadcast_notifier(gateway: Any) -> Notifier:
    """Send a notice to every registered channel (or the configured subset)."""
    from iris_harness.services.channels.models import ChannelMessage

    def notify(
        subject: str, body: str, channels: Sequence[str] | None, url: str | None = None
    ) -> None:
        registered = list(gateway.channels())
        targets = [c for c in (channels or registered) if c in registered]
        metadata: dict[str, Any] = {"health": True}
        if url:
            metadata["url"] = url  # web push opens it on a tap; the body carries it too
        message = ChannelMessage(recipient="", body=body, subject=subject, metadata=metadata)
        for receipt in gateway.broadcast(message, channels=targets):
            status = getattr(getattr(receipt, "status", None), "value", "")
            if status != "sent":
                logger.warning(
                    "health notice not sent on %s: %s",
                    getattr(receipt, "channel", "?"),
                    getattr(receipt, "error", "") or status,
                )

    return notify


def build_watcher(
    *,
    config_dir: Path,
    data_dir: Path,
    channels: Any,
    heartbeats: Any,
    diagnostics_provider: Callable[[], list[Any]] | None = None,
    repo_root: Path | None = None,
) -> HealthWatcher:
    """The process's watcher over the runtime's channels and heartbeat scheduler."""
    from iris_harness.services.health.repair import (
        heartbeat_retry_repairer,
        service_restart_repairer,
    )
    from iris_harness.services.health.service import refresh

    config = load_watch_config(config_dir)
    repairers: list[Repairer] = []
    if config.heartbeat_retry:
        repairers.append(heartbeat_retry_repairer(heartbeats, skip=config.skip_heartbeats))
    if config.restart:
        if repo_root is None:
            from iris_harness.foundation.paths import repo_root as _repo_root

            repo_root = _repo_root()
        repairers.append(
            service_restart_repairer(config.restart, cwd=repo_root, never=config.never_restart)
        )

    def diagnose() -> None:
        diagnostics = diagnostics_provider() if diagnostics_provider else None
        refresh(net_probe=True, heartbeat_diagnostics=diagnostics)

    return HealthWatcher(
        store=IncidentStore(Path(data_dir) / "health.db"),
        config=config,
        notify=broadcast_notifier(channels) if channels is not None else None,
        repairers=repairers,
        diagnose=diagnose,
    )


__all__ = [
    "HealthWatcher",
    "build_watcher",
    "WatchConfig",
    "broadcast_notifier",
    "current_watcher",
    "install_watcher",
    "load_watch_config",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_installed")
