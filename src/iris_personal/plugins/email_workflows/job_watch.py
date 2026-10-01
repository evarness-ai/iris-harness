"""Hidden mail: ``judge_reachable`` and the digest's "Email jobs" line (loop-proof D13).

A ``waiting`` email is invisible to every IRIS surface until the judge (the
``email_judge`` heartbeat, on the Mac's Ollama) has judged it. When the Mac sleeps or the
judge job stops, mail — bills included — stays hidden, so the owner is told:

* **judge_reachable** (a health check, paged by the health watch on red):
  red when mail waits and either the last judge run reported the Mac unreachable or a
  probe of the judge tier's Ollama fails once the oldest waiting email is past
  ``probe_red_after_minutes``; yellow while the wait is young, or when the Mac answers
  but the oldest waiting email is older than one judge cycle; green otherwise. The probe
  (``GET <root>/api/version``, 3 s) is only made while mail waits.
* **Email jobs** (a digest footer line, over yesterday): ``Email jobs: sweep 3/3 · judge
  3/3 · 0 waiting``.

Both read the core's kept runs (``heartbeat_runs.db``) and this plugin's judgments
table; every word is in ``job_watch.yaml``.

**The judge run contract** (what the ``email_judge`` handler returns): a ``HeartbeatRun``
with ``output`` = one human line (``judged 12 · waiting 0``) and ``result`` =
``{"judged": int, "waiting": int, "oldest_waiting": iso | None, "unsure": int,
"errors": int, "unreachable": bool}``; status ``success`` when the model answered (even
with nothing to judge), ``skipped`` with ``unreachable: true`` and ``error`` saying why
when it could not reach the Mac, ``failed`` on a crash. Only ``result["unreachable"]`` is
read here; the counts are shown on the Heartbeats screen through ``output``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any

import yaml

from iris_harness.sdk.health import CheckKind, HealthCheck, HealthState
from iris_harness.sdk.heartbeat import HeartbeatRunHistory, clock, heartbeat_runs, tally

from .judgments import JudgmentStore

logger = logging.getLogger(__name__)

SHIPPED_CONFIG = Path(__file__).with_name("job_watch.yaml")
OVERRIDE_RELATIVE_PATH = Path("email") / "job_watch.yaml"
TARGET = "judge_reachable"


@dataclass(frozen=True)
class JobWatchConfig:
    judge_job: str = "email_judge"
    tier: str = "email_judge"
    probe_path: str = "/api/version"
    probe_timeout_seconds: float = 3.0
    probe_red_after: timedelta = timedelta(minutes=30)
    stale_waiting: timedelta = timedelta(hours=7)
    words: dict[str, str] = field(default_factory=dict)
    footer: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_dir: Path | None = None) -> JobWatchConfig:
        raw: dict[str, Any] = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8")) or {}
        if config_dir is not None:
            override = Path(config_dir) / OVERRIDE_RELATIVE_PATH
            if override.is_file():
                raw.update(yaml.safe_load(override.read_text(encoding="utf-8")) or {})
        return cls(
            judge_job=str(raw.get("judge_job") or "email_judge"),
            tier=str(raw.get("tier") or "email_judge"),
            probe_path=str(raw.get("probe_path") or "/api/version"),
            probe_timeout_seconds=float(raw.get("probe_timeout_seconds", 3)),
            probe_red_after=timedelta(minutes=float(raw.get("probe_red_after_minutes", 30))),
            stale_waiting=timedelta(hours=float(raw.get("stale_waiting_hours", 7))),
            words={str(k): str(v) for k, v in (raw.get("words") or {}).items()},
            footer=dict(raw.get("footer") or {}),
        )


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _waiting(judgments: JudgmentStore) -> tuple[int, datetime | None]:
    """How many emails wait, and since when the oldest has. (0, None) with no table."""
    if not judgments.db_path.exists():
        return 0, None
    judgments.ensure_schema()
    count = judgments.count_waiting()
    oldest = judgments.waiting(limit=1)
    return count, (_parse(oldest[0].created_at) if oldest else None)


def judge_reachable_checks(
    *,
    runs: HeartbeatRunHistory,
    judgments: JudgmentStore,
    probe: Callable[[], bool],
    config: JobWatchConfig,
    tz: tzinfo,
    now: datetime | None = None,
) -> list[HealthCheck]:
    """The one ``judge_reachable`` row."""
    now = now or datetime.now(UTC)
    words = config.words
    job = config.judge_job
    count, oldest = _waiting(judgments)
    if count == 0:
        return [_check(HealthState.GREEN, words.get("idle", "nothing waiting"), job)]

    last = runs.last(config.judge_job)
    last_unreachable = bool(last is not None and last.result.get("unreachable"))
    reachable = probe()
    age = now - oldest if oldest is not None else timedelta(0)
    fill = {
        "waiting": count,
        "oldest": clock(oldest, now, tz),
        "unreachable": words.get("unreachable", "unreachable"),
        "last_run": clock(last.started_at if last else None, now, tz),
    }

    if last_unreachable:
        key = "red" if not reachable else "red_last_run"
        return [_check(HealthState.RED, _say(words, key, fill), job)]
    if not reachable:
        state = HealthState.RED if age > config.probe_red_after else HealthState.YELLOW
        key = "red" if state is HealthState.RED else "young"
        return [_check(state, _say(words, key, fill), job)]
    if age > config.stale_waiting:
        return [_check(HealthState.YELLOW, _say(words, "stale", fill), job)]
    return [_check(HealthState.GREEN, _say(words, "green", fill), job)]


def _say(words: dict[str, str], key: str, fill: dict[str, Any]) -> str:
    template = words.get(key) or "{waiting} waiting"
    try:
        return template.format(**fill)
    except (KeyError, IndexError, ValueError):
        return template


def _check(state: HealthState, detail: str, job: str) -> HealthCheck:
    return HealthCheck(
        target=TARGET,
        kind=CheckKind.SERVICE,
        state=state,
        detail=detail,
        endpoint="/heartbeat",
        action=None if state is HealthState.GREEN else f"iris heartbeats trigger {job}",
    )


def ollama_probe(
    root_url: Callable[[], str | None], path: str, timeout: float
) -> Callable[[], bool]:
    """One GET of ``<root><path>``; True on any 2xx. The URL is read per call."""

    def probe() -> bool:
        import httpx

        from iris_harness.sdk.logging import log_egress

        root = root_url()
        if not root:
            return False
        url = f"{root.rstrip('/')}{path}"
        try:
            log_egress(
                destination=httpx.URL(url).netloc.decode(),
                method="GET",
                kind="service",
                purpose="health-probe",
            )
            return 200 <= httpx.get(url, timeout=timeout).status_code < 300
        except Exception:  # noqa: BLE001 — any transport error means unreachable
            return False

    return probe


def tier_root(tier_router: Any, tier: str) -> Callable[[], str | None]:
    """The judge tier's provider root, looked up per call (a tier edit applies)."""

    def root() -> str | None:
        from iris_harness.sdk.llm import provider_root_url

        found = tier_router.get_tier_by_name(tier) if tier_router is not None else None
        return provider_root_url(found.provider if found is not None else "ollama")

    return root


def email_jobs_line(
    *,
    runs: HeartbeatRunHistory,
    judgments: JudgmentStore,
    schedules: Callable[[], dict[str, str]],
    config: JobWatchConfig,
    tz: tzinfo,
    start: datetime,
    end: datetime,
) -> str | None:
    """``Email jobs: sweep 3/3 · judge 3/3 · 0 waiting`` for ``[start, end)``."""
    footer = config.footer
    labels: dict[str, str] = {str(k): str(v) for k, v in (footer.get("jobs") or {}).items()}
    if not labels:
        return None
    current = schedules()
    parts: list[str] = []
    for job, label in labels.items():
        schedule = current.get(job)
        if schedule is None:
            continue  # not scheduled here: nothing to count
        counted = tally(schedule, runs, job, start, end, tz)
        piece = str(footer.get("job", "{label} {ran}/{expected}")).format(
            label=label, ran=counted.ran, expected=counted.expected
        )
        if counted.missed:
            times = ", ".join(clock(s, s, tz) for s in counted.missed)
            piece += str(footer.get("missed", " ({times} missed)")).format(times=times)
        parts.append(piece)
    if not parts:
        return None  # none of the jobs is scheduled on this harness
    count, _ = _waiting(judgments)
    waiting = str(footer.get("waiting", "{waiting} waiting")).format(waiting=count)
    last = runs.last(config.judge_job)
    if last is not None and last.result.get("unreachable"):
        waiting += str(footer.get("unreachable", " ({unreachable})")).format(
            unreachable=config.words.get("unreachable", "unreachable")
        )
    parts.append(waiting)
    return str(footer.get("prefix", "")) + str(footer.get("separator", " · ")).join(parts)


def register(api: Any) -> None:
    """Register ``judge_reachable`` and the "Email jobs" footer line for this data dir."""
    from iris_harness.sdk.health import register_check_provider

    services = getattr(api, "services", None)
    if services is None or getattr(services, "data_dir", None) is None:
        return  # a host with no data dir keeps no runs to watch
    data_dir = Path(services.data_dir)
    config_dir = getattr(services, "config_dir", None)
    heartbeats = getattr(services, "heartbeats", None)
    runs = heartbeat_runs(data_dir)

    def _judgments() -> JudgmentStore:
        return JudgmentStore(db_path=data_dir / "email.db")

    def _tz() -> tzinfo:
        if heartbeats is not None and hasattr(heartbeats, "tz"):
            return heartbeats.tz()  # type: ignore[no-any-return]
        from iris_harness.sdk.time import iris_timezone

        return iris_timezone()

    def _checks() -> list[HealthCheck]:
        config = JobWatchConfig.load(config_dir)
        probe = ollama_probe(
            tier_root(getattr(services, "tier_router", None), config.tier),
            config.probe_path,
            config.probe_timeout_seconds,
        )
        return judge_reachable_checks(
            runs=runs, judgments=_judgments(), probe=probe, config=config, tz=_tz()
        )

    def _schedules() -> dict[str, str]:
        if heartbeats is None:
            return {}
        return {d.name: d.schedule for d in heartbeats.all_definitions() if d.enabled}

    def _line(start: datetime, end: datetime) -> str | None:
        return email_jobs_line(
            runs=runs,
            judgments=_judgments(),
            schedules=_schedules,
            config=JobWatchConfig.load(config_dir),
            tz=_tz(),
            start=start,
            end=end,
        )

    register_check_provider(TARGET, _checks)
    api.register_footer_line("email_jobs", _line)


__all__ = [
    "JobWatchConfig",
    "TARGET",
    "email_jobs_line",
    "judge_reachable_checks",
    "ollama_probe",
    "register",
    "tier_root",
]
