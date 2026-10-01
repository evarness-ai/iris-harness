"""The health watch (ADR-0116): confirm → repair → engage the owner → close."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.services.health import repair as repair_mod
from iris_harness.services.health.incidents import IncidentStore
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthSnapshot, HealthState
from iris_harness.services.health.repair import RepairOutcome
from iris_harness.services.health.watch import (
    HealthWatcher,
    WatchConfig,
    load_watch_config,
)

T0 = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
MIN = timedelta(minutes=1)


@pytest.fixture(autouse=True)
def _no_plugin_repairers():
    repair_mod.clear_repairers()
    yield
    repair_mod.clear_repairers()


def _snap(*checks: HealthCheck) -> HealthSnapshot:
    return HealthSnapshot(checks=checks, sampled_at=T0.isoformat())


def _gmail(state: HealthState, who: str = "a@b.com") -> HealthCheck:
    return HealthCheck(
        "Gmail",
        CheckKind.CREDENTIAL,
        state,
        f"{who}: token revoked — re-authenticate" if state is HealthState.RED else "ok",
        action=f"iris auth gmail login --user {who}",
        subject=who,
    )


def _sweep(state: HealthState = HealthState.RED) -> HealthCheck:
    return HealthCheck(
        "heartbeat:email_sweep",
        CheckKind.SERVICE,
        state,
        "last run failed: No Gmail credentials",
        action="iris heartbeats trigger email_sweep",
    )


class _Outbox:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, Sequence[str] | None]] = []
        self.urls: list[str | None] = []

    def __call__(
        self, subject: str, body: str, channels: Sequence[str] | None, url: str | None = None
    ) -> None:
        self.sent.append((subject, body, channels))
        self.urls.append(url)


def _watcher(
    tmp_path: Path,
    *,
    repairers: Sequence[repair_mod.Repairer] = (),
    diagnose=None,
    **config: object,
) -> tuple[HealthWatcher, _Outbox]:
    outbox = _Outbox()
    watcher = HealthWatcher(
        store=IncidentStore(tmp_path / "health.db"),
        config=WatchConfig(**config),  # type: ignore[arg-type]
        notify=outbox,
        repairers=repairers,
        diagnose=diagnose,
    )
    return watcher, outbox


def _repairer(outcome: RepairOutcome, calls: list[str]) -> repair_mod.Repairer:
    def repair(check: HealthCheck) -> RepairOutcome | None:
        calls.append(check.key)
        return outcome

    return repair


# ── confirm ──────────────────────────────────────────────────────────────


def test_one_red_sample_is_not_an_incident(tmp_path: Path) -> None:
    calls: list[str] = []
    watcher, outbox = _watcher(tmp_path, repairers=[_repairer(RepairOutcome("x", True), calls)])

    assert watcher.observe(_snap(_sweep()), now=T0) == []
    assert watcher.store.open_incidents() == []
    assert calls == [] and outbox.sent == []


def test_a_streak_broken_by_green_starts_over(tmp_path: Path) -> None:
    watcher, _ = _watcher(tmp_path)
    watcher.observe(_snap(_sweep()), now=T0)
    watcher.observe(_snap(), now=T0 + MIN)
    watcher.observe(_snap(_sweep()), now=T0 + 2 * MIN)
    assert watcher.store.open_incidents() == []


# ── repair ───────────────────────────────────────────────────────────────


def test_confirmed_red_opens_an_incident_and_repairs(tmp_path: Path) -> None:
    calls: list[str] = []
    watcher, outbox = _watcher(
        tmp_path, repairers=[_repairer(RepairOutcome("re-run email_sweep", True), calls)]
    )
    watcher.observe(_snap(_sweep()), now=T0)
    events = watcher.observe(_snap(_sweep()), now=T0 + MIN)

    [incident] = watcher.store.open_incidents()
    assert incident.state == "repairing"
    assert incident.attempts == 1
    assert incident.repairs[0]["tried"] == "re-run email_sweep"
    assert calls == ["heartbeat:email_sweep"]
    assert any(e.startswith("opened heartbeat:email_sweep") for e in events)
    assert outbox.sent == []  # still trying — the owner is not bothered yet


def test_repair_that_works_closes_quietly_as_self_healed(tmp_path: Path) -> None:
    watcher, outbox = _watcher(
        tmp_path, repairers=[_repairer(RepairOutcome("re-run email_sweep", True), [])]
    )
    watcher.observe(_snap(_sweep()), now=T0)
    watcher.observe(_snap(_sweep()), now=T0 + MIN)
    watcher.observe(_snap(), now=T0 + 2 * MIN)  # diagnostic gone = healthy

    [incident] = watcher.store.recent()
    assert incident.resolution == "self_healed"
    assert incident.resolved_at is not None
    assert outbox.sent == []  # self_healed notices are off by default


def test_self_healed_notice_when_asked_for(tmp_path: Path) -> None:
    watcher, outbox = _watcher(
        tmp_path,
        repairers=[_repairer(RepairOutcome("re-run email_sweep", True), [])],
        notify_self_healed=True,
    )
    watcher.observe(_snap(_sweep()), now=T0)
    watcher.observe(_snap(_sweep()), now=T0 + MIN)
    watcher.observe(_snap(), now=T0 + 2 * MIN)
    assert [s for s, _, _ in outbox.sent] == ["IRIS fixed itself"]
    assert "re-run email_sweep" in outbox.sent[0][1]


# ── engage ───────────────────────────────────────────────────────────────


def test_repairs_run_out_then_the_owner_is_told_once(tmp_path: Path) -> None:
    calls: list[str] = []
    watcher, outbox = _watcher(
        tmp_path,
        repairers=[_repairer(RepairOutcome("re-run email_sweep", False, "still failing"), calls)],
        max_attempts=2,
    )
    for i in range(6):
        watcher.observe(_snap(_sweep()), now=T0 + i * MIN)

    assert len(calls) == 2  # max_attempts, one per tick
    [incident] = watcher.store.open_incidents()
    assert incident.state == "needs_user"
    assert len(outbox.sent) == 1
    subject, body, channels = outbox.sent[0]
    assert subject == "IRIS needs your help"
    assert "heartbeat:email_sweep is not working" in body
    assert "re-run email_sweep ×2 (failed: still failing)" in body  # folded, not repeated
    assert "iris heartbeats trigger email_sweep" in body
    assert channels is None  # every channel


def test_still_broken_is_repeated_only_after_renotify_hours(tmp_path: Path) -> None:
    # The hour-long jumps stand in for steady ticks, not a gap the watch should resume from.
    watcher, outbox = _watcher(tmp_path, renotify_hours=12, resume_gap_minutes=1e6)
    watcher.observe(_snap(_sweep()), now=T0)
    watcher.observe(_snap(_sweep()), now=T0 + MIN)  # no repairer → told now
    watcher.observe(_snap(_sweep()), now=T0 + timedelta(hours=11))
    assert len(outbox.sent) == 1
    watcher.observe(_snap(_sweep()), now=T0 + timedelta(hours=12, minutes=2))
    assert len(outbox.sent) == 2
    [incident] = watcher.store.open_incidents()
    assert incident.notify_count == 2


def test_no_repairer_asks_the_owner_straight_away(tmp_path: Path) -> None:
    watcher, outbox = _watcher(tmp_path)
    watcher.observe(_snap(_sweep()), now=T0)
    watcher.observe(_snap(_sweep()), now=T0 + MIN)
    assert len(outbox.sent) == 1
    assert "I have no automatic fix" in outbox.sent[0][1]


def test_a_final_outcome_skips_the_remaining_attempts(tmp_path: Path) -> None:
    calls: list[str] = []
    refused = RepairOutcome("token refresh", False, "revoked", final=True)
    watcher, outbox = _watcher(tmp_path, repairers=[_repairer(refused, calls)], max_attempts=3)
    watcher.observe(_snap(_gmail(HealthState.RED)), now=T0)
    watcher.observe(_snap(_gmail(HealthState.RED)), now=T0 + MIN)
    watcher.observe(_snap(_gmail(HealthState.RED)), now=T0 + 2 * MIN)

    assert calls == ["Gmail:a@b.com"]
    assert len(outbox.sent) == 1
    body = outbox.sent[0][1]
    assert body.startswith("Gmail (a@b.com) is not working")
    assert "To fix it, run: iris auth gmail login --user a@b.com" in body


def test_the_notice_goes_to_the_configured_channels(tmp_path: Path) -> None:
    watcher, outbox = _watcher(tmp_path, channels=("telegram",))
    watcher.observe(_snap(_sweep()), now=T0)
    watcher.observe(_snap(_sweep()), now=T0 + MIN)
    assert outbox.sent[0][2] == ("telegram",)


# ── close ────────────────────────────────────────────────────────────────


def test_fixed_after_the_owner_was_told_says_so(tmp_path: Path) -> None:
    watcher, outbox = _watcher(tmp_path)
    watcher.observe(_snap(_gmail(HealthState.RED)), now=T0)
    watcher.observe(_snap(_gmail(HealthState.RED)), now=T0 + MIN)
    watcher.observe(_snap(_gmail(HealthState.GREEN)), now=T0 + 5 * MIN)

    assert [s for s, _, _ in outbox.sent] == ["IRIS needs your help", "IRIS is back"]
    assert outbox.sent[1][1] == "Gmail (a@b.com) is working again."
    [incident] = watcher.store.recent()
    assert incident.resolution == "user_fixed"


def test_one_accounts_green_row_does_not_close_anothers_incident(tmp_path: Path) -> None:
    watcher, _ = _watcher(tmp_path)
    both = _snap(_gmail(HealthState.RED, "a@b.com"), _gmail(HealthState.GREEN, "c@d.com"))
    watcher.observe(both, now=T0)
    watcher.observe(both, now=T0 + MIN)
    watcher.observe(both, now=T0 + 2 * MIN)
    assert [i.key for i in watcher.store.open_incidents()] == ["Gmail:a@b.com"]


# ── memory across restarts, repeats, switches ────────────────────────────


def test_a_restarted_watcher_does_not_re_alert(tmp_path: Path) -> None:
    first, outbox1 = _watcher(tmp_path)
    first.observe(_snap(_sweep()), now=T0)
    first.observe(_snap(_sweep()), now=T0 + MIN)
    assert len(outbox1.sent) == 1

    second, outbox2 = _watcher(tmp_path)  # same health.db, fresh process
    for i in range(3):
        second.observe(_snap(_sweep()), now=T0 + (2 + i) * MIN)
    assert outbox2.sent == []


def test_a_repeat_problem_is_counted(tmp_path: Path) -> None:
    watcher, outbox = _watcher(tmp_path, resume_gap_minutes=1e6)  # the day jump is not a gap
    for day in range(2):
        start = T0 + timedelta(days=day)
        watcher.observe(_snap(_sweep()), now=start)
        watcher.observe(_snap(_sweep()), now=start + MIN)
        watcher.observe(_snap(), now=start + 2 * MIN)
    notices = [b for s, b, _ in outbox.sent if s == "IRIS needs your help"]
    assert "time in" not in notices[0]
    assert "This is the 2nd time in 7 days." in notices[1]


def test_ignored_targets_are_never_acted_on(tmp_path: Path) -> None:
    watcher, outbox = _watcher(tmp_path, ignore=("heartbeat:email_sweep",))
    for i in range(4):
        watcher.observe(_snap(_sweep()), now=T0 + i * MIN)
    assert watcher.store.recent() == [] and outbox.sent == []


def test_disabled_watch_does_nothing(tmp_path: Path) -> None:
    watcher, outbox = _watcher(tmp_path, enabled=False)
    for i in range(4):
        assert watcher.observe(_snap(_sweep()), now=T0 + i * MIN) == []
    assert watcher.store.recent() == [] and outbox.sent == []


def test_grey_and_yellow_are_not_incidents(tmp_path: Path) -> None:
    watcher, _ = _watcher(tmp_path)
    quiet = _snap(
        HealthCheck("Drive", CheckKind.CREDENTIAL, HealthState.GREY, "not connected"),
        HealthCheck("host", CheckKind.HARDWARE, HealthState.YELLOW, "low free RAM"),
    )
    for i in range(3):
        watcher.observe(quiet, now=T0 + i * MIN)
    assert watcher.store.recent() == []


def test_opening_an_incident_diagnoses_credentials_once(tmp_path: Path) -> None:
    calls: list[int] = []
    watcher, _ = _watcher(tmp_path, diagnose=lambda: calls.append(1))
    for i in range(4):
        watcher.observe(_snap(_sweep()), now=T0 + i * MIN)
    assert calls == [1]


def test_a_failing_notifier_does_not_stop_the_watch(tmp_path: Path) -> None:
    def broken(
        subject: str, body: str, channels: Sequence[str] | None, url: str | None = None
    ) -> None:
        raise RuntimeError("telegram down")

    watcher = HealthWatcher(
        store=IncidentStore(tmp_path / "health.db"), config=WatchConfig(), notify=broken
    )
    watcher.observe(_snap(_sweep()), now=T0)
    watcher.observe(_snap(_sweep()), now=T0 + MIN)
    [incident] = watcher.store.open_incidents()
    assert incident.state == "needs_user"


def test_plugin_repairers_are_consulted_after_core_ones(tmp_path: Path) -> None:
    calls: list[str] = []
    repair_mod.register_repairer(
        "gmail_credentials", _repairer(RepairOutcome("token refresh", True), calls)
    )
    watcher, _ = _watcher(tmp_path)
    watcher.observe(_snap(_gmail(HealthState.RED)), now=T0)
    watcher.observe(_snap(_gmail(HealthState.RED)), now=T0 + MIN)
    assert calls == ["Gmail:a@b.com"]


# ── config ───────────────────────────────────────────────────────────────


def test_config_file_and_env_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "health_watch.yaml").write_text(
        "confirm_ticks: 3\n"
        "notify: {channels: [telegram, web], renotify_hours: 2}\n"
        "repair: {max_attempts: 1, restart: {governor: [x, --restart=governor]}}\n"
        "ignore: [Drive]\n"
    )
    monkeypatch.delenv("IRIS_HEALTH_WATCH_ENABLED", raising=False)
    config = load_watch_config(tmp_path)
    assert config.enabled is True
    assert config.confirm_ticks == 3
    assert config.channels == ("telegram", "web")
    assert config.renotify_hours == 2
    assert config.max_attempts == 1
    assert config.restart == {"governor": ("x", "--restart=governor")}
    assert config.never_restart == ("iris_api",)
    assert config.ignore == ("Drive",)

    monkeypatch.setenv("IRIS_HEALTH_WATCH_ENABLED", "0")
    assert load_watch_config(tmp_path).enabled is False


def test_missing_config_uses_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_HEALTH_WATCH_ENABLED", raising=False)
    config = load_watch_config(tmp_path)
    assert config == WatchConfig()
    assert config.channels is None  # every channel


def test_the_shipped_config_parses_and_never_restarts_the_api() -> None:
    repo_config = Path(__file__).resolve().parents[5] / "config"
    config = load_watch_config(repo_config)
    assert config.channels is None
    assert "iris_api" in config.never_restart
    assert "iris_api" not in config.restart
    assert config.restart["governor"] == ("scripts/start_iris.sh", "--restart=governor")


# ── notice wording, from the revoked-Gmail simulation (2026-09-19) ─────────


def _revoked_sweep() -> HealthCheck:
    return HealthCheck(
        "heartbeat:email_sweep",
        CheckKind.SERVICE,
        HealthState.RED,
        "last run failed: gmail:a@b.com: Gmail access for a@b.com was revoked or has "
        "expired — run `iris auth gmail login --user a@b.com` to reconnect.",
        action="iris heartbeats trigger email_sweep",
    )


def test_a_failure_that_names_its_own_fix_leads_with_it(tmp_path: Path) -> None:
    same_error = _revoked_sweep().detail.removeprefix("last run failed: ")
    watcher, outbox = _watcher(
        tmp_path,
        repairers=[_repairer(RepairOutcome("re-run email_sweep", False, same_error), [])],
    )
    for i in range(4):
        watcher.observe(_snap(_revoked_sweep()), now=T0 + i * MIN)

    body = outbox.sent[0][1]
    assert "To fix it, run: iris auth gmail login --user a@b.com" in body
    assert "Then retry with: iris heartbeats trigger email_sweep" in body
    # the repair's error repeats the first line, so it is not printed again
    assert "I tried: re-run email_sweep ×2 (failed) — it is still failing." in body
    assert body.count("revoked") == 1


def test_a_template_command_is_not_taken_as_the_fix(tmp_path: Path) -> None:
    check = HealthCheck(
        "heartbeat:email_sweep",
        CheckKind.SERVICE,
        HealthState.RED,
        "No Gmail credentials. Run `iris auth gmail login --user <address>` first.",
        action="iris heartbeats trigger email_sweep",
    )
    watcher, outbox = _watcher(tmp_path)
    watcher.observe(_snap(check), now=T0)
    watcher.observe(_snap(check), now=T0 + MIN)
    body = outbox.sent[0][1]
    assert "To fix it, run: iris heartbeats trigger email_sweep" in body
    assert "Then retry" not in body


def test_different_repair_outcomes_are_listed_separately(tmp_path: Path) -> None:
    outcomes = iter(
        [RepairOutcome("restart governor", True), RepairOutcome("restart governor", False, "boom")]
    )
    watcher, outbox = _watcher(tmp_path, repairers=[lambda check: next(outcomes)])
    for i in range(4):
        watcher.observe(
            _snap(HealthCheck("governor", CheckKind.SERVICE, HealthState.RED, "down")),
            now=T0 + i * MIN,
        )
    assert "I tried: restart governor (ran); restart governor (failed: boom)" in outbox.sent[0][1]


# ── resume after a gap ───────────────────────────────────────────────────


def _overdue(seconds: int) -> HealthCheck:
    return HealthCheck(
        "heartbeat:health_tick",
        CheckKind.SERVICE,
        HealthState.RED,
        f"overdue by {seconds}s; cadence is every 60s",
        action="iris heartbeats trigger health_tick",
    )


def test_a_sleeping_host_does_not_page_the_owner(tmp_path: Path) -> None:
    # 2026-09-24: the Mac slept and woke briefly (DarkWake) at 03:07 and 03:13. Each
    # short wake ran one pass that saw health_tick overdue; the two built a streak and
    # the owner was told "health_tick is not working" at 03:13, then "working again".
    watcher, outbox = _watcher(tmp_path)
    watcher.observe(_snap(), now=T0)
    watcher.observe(_snap(_overdue(1140)), now=T0 + 20 * MIN)  # DarkWake 1
    watcher.observe(_snap(_overdue(1500)), now=T0 + 26 * MIN)  # DarkWake 2
    watcher.observe(_snap(), now=T0 + 27 * MIN)

    assert outbox.sent == []
    assert watcher.store.open_incidents() == []


def test_a_network_still_coming_up_after_wake_is_not_reported(tmp_path: Path) -> None:
    # 2026-09-24 09:23: lid opened, email_sweep failed on DNS for a few minutes.
    calls: list[str] = []
    watcher, outbox = _watcher(
        tmp_path, repairers=[_repairer(RepairOutcome("re-run email_sweep", False), calls)]
    )
    watcher.observe(_snap(), now=T0)
    for minute in range(90, 96):
        watcher.observe(_snap(_sweep()), now=T0 + minute * MIN)
    watcher.observe(_snap(), now=T0 + 97 * MIN)

    assert outbox.sent == [] and calls == []
    assert watcher.store.open_incidents() == []


def test_a_fault_still_red_after_the_grace_is_reported(tmp_path: Path) -> None:
    watcher, outbox = _watcher(tmp_path)
    watcher.observe(_snap(), now=T0)
    for minute in range(60, 73):  # back at +60; grace ends at +70
        watcher.observe(_snap(_sweep()), now=T0 + minute * MIN)

    [incident] = watcher.store.open_incidents()
    assert incident.opened_at == (T0 + 71 * MIN).isoformat()  # grace + confirm_ticks
    assert len(outbox.sent) == 1


def test_an_open_incident_that_clears_during_the_grace_is_closed(tmp_path: Path) -> None:
    watcher, outbox = _watcher(tmp_path)
    watcher.observe(_snap(_sweep()), now=T0)
    watcher.observe(_snap(_sweep()), now=T0 + MIN)
    assert len(outbox.sent) == 1  # no repairer: the owner is asked straight away

    watcher.observe(_snap(), now=T0 + 60 * MIN)

    assert watcher.store.open_incidents() == []
    assert outbox.sent[-1][1].endswith("is working again.")


def test_regular_passes_never_start_a_grace(tmp_path: Path) -> None:
    watcher, outbox = _watcher(tmp_path, resume_gap_minutes=5.0)
    for minute in range(0, 20, 4):  # a slow but steady tick: every 4 minutes
        watcher.observe(_snap(_sweep()), now=T0 + minute * MIN)
    assert len(watcher.store.open_incidents()) == 1


def test_the_shipped_config_sets_the_resume_window() -> None:
    config = load_watch_config(Path(__file__).resolve().parents[5] / "config")
    assert (config.resume_gap_minutes, config.resume_grace_minutes) == (5, 10)


# ── starting up (loop-proof PR 3a) ───────────────────────────────────────


def _gateway(state: HealthState = HealthState.RED) -> HealthCheck:
    return HealthCheck(
        "channel_gateway",
        CheckKind.SERVICE,
        state,
        "unreachable" if state is HealthState.RED else "ok",
    )


def test_a_restart_does_not_page_a_sibling_still_starting(tmp_path: Path) -> None:
    # A deploy restarts every container at once; the new API's first passes see the
    # channel gateway still coming up and, before the grace, paged "unreachable".
    watcher, outbox = _watcher(tmp_path, startup_grace_minutes=5.0)
    for minute in range(3):
        watcher.observe(_snap(_gateway()), now=T0 + minute * MIN)
    watcher.observe(_snap(_gateway(HealthState.GREEN)), now=T0 + 3 * MIN)

    assert outbox.sent == []
    assert watcher.store.recent() == []


def test_a_restart_red_never_counts_toward_repeats(tmp_path: Path) -> None:
    store_path = tmp_path
    for day in range(2):  # two deploys, each with a gateway still starting
        watcher, _ = _watcher(store_path, startup_grace_minutes=5.0)
        start = T0 + timedelta(days=day)
        watcher.observe(_snap(_gateway()), now=start)
        watcher.observe(_snap(_gateway()), now=start + MIN)
        watcher.observe(_snap(), now=start + 2 * MIN)

    watcher, outbox = _watcher(store_path, startup_grace_minutes=5.0)
    start = T0 + timedelta(days=3)
    for minute in range(0, 8):  # a real outage that outlasts the grace
        watcher.observe(_snap(_gateway()), now=start + minute * MIN)

    [notice] = [b for s, b, _ in outbox.sent if s == "IRIS needs your help"]
    assert "time in" not in notice, "restart reds are not earlier occurrences"


def test_a_fault_still_red_after_the_startup_grace_is_reported(tmp_path: Path) -> None:
    watcher, outbox = _watcher(tmp_path, startup_grace_minutes=5.0)
    for minute in range(8):
        watcher.observe(_snap(_gateway()), now=T0 + minute * MIN)

    [incident] = watcher.store.open_incidents()
    assert incident.opened_at == (T0 + 6 * MIN).isoformat()  # grace ends +5, confirm 2
    assert len(outbox.sent) == 1


def test_no_startup_grace_by_default(tmp_path: Path) -> None:
    watcher, _ = _watcher(tmp_path)
    watcher.observe(_snap(_gateway()), now=T0)
    watcher.observe(_snap(_gateway()), now=T0 + MIN)
    assert len(watcher.store.open_incidents()) == 1


def test_the_shipped_config_holds_incidents_after_a_restart() -> None:
    root = Path(__file__).resolve().parents[5]
    config = load_watch_config(root / "config", with_edits=False)
    assert config.startup_grace_minutes == 5


# ── the page that fixes it (Reconnect Google, the alert link) ──────────────


def _revoked_with_fix() -> HealthCheck:
    from dataclasses import replace

    return replace(_gmail(HealthState.RED), fix_url="/settings#connections")


def test_a_row_that_names_its_fix_page_links_it_beside_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_PUBLIC_URL", "https://iris-vm.example.ts.net/")
    watcher, outbox = _watcher(tmp_path)
    watcher.observe(_snap(_revoked_with_fix()), now=T0)
    watcher.observe(_snap(_revoked_with_fix()), now=T0 + MIN)

    body = outbox.sent[0][1]
    assert "To fix it, run: iris auth gmail login --user a@b.com" in body  # the Mac path stays
    assert body.endswith(
        "Or fix it in the app: https://iris-vm.example.ts.net/settings#connections"
    )
    assert outbox.urls == ["/settings#connections"]  # a push opens it on a tap


def test_without_a_public_url_the_link_is_the_console_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_PUBLIC_URL", raising=False)
    watcher, outbox = _watcher(tmp_path)
    watcher.observe(_snap(_revoked_with_fix()), now=T0)
    watcher.observe(_snap(_revoked_with_fix()), now=T0 + MIN)
    assert outbox.sent[0][1].endswith("Or fix it in the app: /settings#connections")


def test_a_row_without_a_fix_page_gets_no_link(tmp_path: Path) -> None:
    watcher, outbox = _watcher(tmp_path)
    watcher.observe(_snap(_gmail(HealthState.RED)), now=T0)
    watcher.observe(_snap(_gmail(HealthState.RED)), now=T0 + MIN)
    assert "in the app" not in outbox.sent[0][1]
    assert outbox.urls == [None]


def test_the_broadcast_notice_carries_the_fix_page_for_web_push() -> None:
    from iris_harness.services.health.watch import broadcast_notifier

    sent: list[dict[str, object]] = []

    class _Gateway:
        def channels(self) -> list[str]:
            return ["web_push"]

        def broadcast(self, message, channels):  # type: ignore[no-untyped-def]
            sent.append(dict(message.metadata))
            return []

    notify = broadcast_notifier(_Gateway())
    notify("IRIS needs your help", "body", None, url="/settings#connections")
    notify("IRIS is back", "body", None)
    assert sent == [{"health": True, "url": "/settings#connections"}, {"health": True}]
