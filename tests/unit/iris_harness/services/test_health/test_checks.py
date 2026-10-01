"""Slice 1 of System Health (ADR-0069): service + hardware checks, the snapshot,
and the alert projection. IO is injected so nothing here touches the network."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from iris_harness.services.health import build_snapshot, render_text
from iris_harness.services.health.checks import (
    _classify_status,
    disk_check,
    hardware_check,
    service_checks,
    uptime_check,
)
from iris_harness.services.health.models import CheckKind, HealthSnapshot, HealthState, alerts


@dataclass(frozen=True)
class _FakeHost:
    """Stand-in for iris_harness.services.system.status.HostStatus."""

    ram_free_gb: float = 16.0
    ram_total_gb: float = 32.0
    cpu_percent: float = 10.0
    thermal_throttled: bool = False
    cpu_speed_limit: int = 100

    def summary(self) -> str:
        return f"RAM {self.ram_free_gb:.1f}/{self.ram_total_gb:.1f} GB free, CPU {self.cpu_percent:.0f}%"


# ── status classification (pure) ────────────────────────────────────────────


def test_classify_status_maps_codes_to_states() -> None:
    assert _classify_status(200)[0] is HealthState.GREEN
    assert _classify_status(204)[0] is HealthState.GREEN
    assert _classify_status(503)[0] is HealthState.YELLOW
    assert _classify_status(404)[0] is HealthState.YELLOW
    assert _classify_status(None)[0] is HealthState.RED
    assert "unreachable" in _classify_status(None)[1]


# ── service checks (injected prober) ────────────────────────────────────────


def test_service_checks_use_injected_prober_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # All services answer 200 → all green; every check is a SERVICE with a URL.
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    checks = service_checks(host="127.0.0.1", prober=lambda _url: 200)
    assert checks, "expected at least one service target"
    assert all(c.kind is CheckKind.SERVICE for c in checks)
    assert all(c.state is HealthState.GREEN for c in checks)
    assert all(c.endpoint and c.endpoint.startswith("http://127.0.0.1:") for c in checks)
    names = {c.target for c in checks}
    assert {"governor", "iris_api", "ollama"} <= names


def test_required_service_down_is_red_with_its_start_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IRIS_GOVERNANCE_EVALUATOR_MODE", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    by_name = {c.target: c for c in service_checks(prober=lambda _url: None)}
    for name in ("governor", "iris_api", "ollama"):
        assert by_name[name].state is HealthState.RED, name
        assert by_name[name].action, f"{name} alert must carry its start command"


def test_wildcard_bind_host_is_probed_on_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """``iris serve --host 0.0.0.0`` binds every interface, but nothing answers
    a probe sent to 0.0.0.0 itself — it has to ask loopback instead."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    checks = service_checks(host="0.0.0.0", prober=lambda _url: 200)  # noqa: S104
    assert all(c.endpoint and c.endpoint.startswith("http://127.0.0.1:") for c in checks)


def test_wildcard_bind_host_from_env_is_probed_on_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("IRIS_API_HOST", "0.0.0.0")  # noqa: S104
    checks = service_checks(prober=lambda _url: 200)
    assert all(c.endpoint and c.endpoint.startswith("http://127.0.0.1:") for c in checks)


# ── a remote model backend (OLLAMA_BASE_URL) ────────────────────────────────


def test_ollama_probe_follows_ollama_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cloud harness runs no models: it reaches the Mac's Ollama over the
    tailnet. Probing our own loopback reported it down forever."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.0.2.10:11434")
    probed: list[str] = []

    def _prober(url: str) -> int:
        probed.append(url)
        return 200

    by_name = {c.target: c for c in service_checks(host="127.0.0.1", prober=_prober)}
    assert by_name["ollama"].endpoint == "http://192.0.2.10:11434/api/tags"
    assert "http://192.0.2.10:11434/api/tags" in probed
    assert by_name["ollama"].state is HealthState.GREEN
    # Only the model backend moves; the IRIS services stay on this host.
    assert by_name["governor"].endpoint is not None
    assert by_name["governor"].endpoint.startswith("http://127.0.0.1:")


def test_ollama_base_url_trailing_v1_and_slash_are_trimmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same variable feeds the OpenAI-compat client, where it may carry
    ``/v1``; the health path hangs off the root."""
    for raw in ("http://mac:11434/v1", "http://mac:11434/", "http://mac:11434/v1/"):
        monkeypatch.setenv("OLLAMA_BASE_URL", raw)
        by_name = {c.target: c for c in service_checks(prober=lambda _url: 200)}
        assert by_name["ollama"].endpoint == "http://mac:11434/api/tags", raw


def test_blank_ollama_base_url_falls_back_to_this_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "   ")
    by_name = {c.target: c for c in service_checks(host="127.0.0.1", prober=lambda _url: 200)}
    assert by_name["ollama"].endpoint == "http://127.0.0.1:11434/api/tags"


def test_remote_ollama_down_names_the_box_in_its_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare "unreachable" reads as advice for *this* machine, which is not
    where the backend runs. The command itself stays pasteable, because the
    Action Center offers it as a copy_command."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.0.2.10:11434")
    by_name = {c.target: c for c in service_checks(prober=lambda _url: None)}
    ollama = by_name["ollama"]
    assert ollama.detail == "unreachable at 192.0.2.10:11434"
    assert ollama.action == "ollama serve"
    # A local service's detail is untouched.
    assert by_name["governor"].detail == "unreachable"


def test_remote_ollama_green_keeps_a_plain_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.0.2.10:11434")
    by_name = {c.target: c for c in service_checks(prober=lambda _url: 200)}
    assert by_name["ollama"].detail == "HTTP 200"


def test_local_ollama_down_keeps_the_plain_start_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    by_name = {c.target: c for c in service_checks(prober=lambda _url: None)}
    assert by_name["ollama"].action == "ollama serve"
    assert by_name["ollama"].detail == "unreachable"


def test_opt_in_service_not_asked_for_is_grey_not_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evaluator sidecar and the channel gateway are opt-in: the harness runs
    without them, so an unreachable one is informational unless the configuration
    asked for it. This is what turned `iris system status` red on a default box."""
    monkeypatch.delenv("IRIS_GOVERNANCE_EVALUATOR_MODE", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("LM_STUDIO_BASE_URL", raising=False)
    by_name = {c.target: c for c in service_checks(prober=lambda _url: None)}
    for name in ("evaluator", "channel_gateway", "llm_proxy"):
        assert by_name[name].state is HealthState.GREY, name
        assert "optional" in by_name[name].detail
        assert by_name[name].action, f"{name} still says how to start it"
    assert by_name["evaluator"].action == "iris evaluator start"


def test_opt_in_service_asked_for_and_down_is_red(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_EVALUATOR_MODE", "remote")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    by_name = {c.target: c for c in service_checks(prober=lambda _url: None)}
    assert by_name["evaluator"].state is HealthState.RED
    assert by_name["channel_gateway"].state is HealthState.RED


def test_healthy_service_carries_no_action() -> None:
    checks = service_checks(prober=lambda _url: 200)
    assert all(c.action is None for c in checks)


# ── hardware check (injected host status) ───────────────────────────────────


def test_hardware_check_green_when_nominal() -> None:
    check = hardware_check(host_status_fn=lambda: _FakeHost())
    assert check.kind is CheckKind.HARDWARE
    assert check.state is HealthState.GREEN


def test_hardware_check_yellow_on_thermal_throttle() -> None:
    check = hardware_check(host_status_fn=lambda: _FakeHost(thermal_throttled=True))
    assert check.state is HealthState.YELLOW
    assert "thermal" in check.detail


def test_hardware_check_yellow_on_low_ram() -> None:
    check = hardware_check(host_status_fn=lambda: _FakeHost(ram_free_gb=1.5))
    assert check.state is HealthState.YELLOW
    assert "RAM" in check.detail


# ── snapshot + projection ───────────────────────────────────────────────────


def _steady_machine() -> dict[str, Any]:
    """A half-full disk and a day of uptime, for snapshot tests about something else.

    Without these, build_snapshot reads the runner's real disk and its own start time:
    on a machine 95% full (the red threshold) the disk check turned red and three
    unrelated assertions failed (found 2026-09-26 on a nearly full dev Mac).
    """
    return {
        "disk_check_fn": lambda: disk_check(path=Path("/"), usage_fn=_usage(50.0)),
        "uptime_check_fn": lambda: uptime_check(started_at=0.0, now_fn=lambda: 86400.0),
    }


def test_build_snapshot_combines_services_and_hardware() -> None:
    snap = build_snapshot(
        service_prober=lambda _url: 200,
        host_status_fn=lambda: _FakeHost(),
        credential_checker=lambda: [],
        **_steady_machine(),
    )
    assert isinstance(snap, HealthSnapshot)
    kinds = {c.kind for c in snap.checks}
    assert kinds == {CheckKind.SERVICE, CheckKind.HARDWARE}
    assert snap.sampled_at  # ISO timestamp set
    assert snap.worst() is HealthState.GREEN


def test_worst_and_alerts_track_red_services() -> None:
    snap = build_snapshot(
        service_prober=lambda url: None if "8080" in url else 200,  # governor down
        host_status_fn=lambda: _FakeHost(),
        credential_checker=lambda: [],
        **_steady_machine(),
    )
    assert snap.worst() is HealthState.RED
    active = alerts(snap)
    assert [c.target for c in active] == ["governor"]


def test_alerts_excludes_yellow_and_grey() -> None:
    # Thermal throttle is yellow, not an alert; healthy services produce none.
    snap = build_snapshot(
        service_prober=lambda _url: 200,
        host_status_fn=lambda: _FakeHost(thermal_throttled=True),
        credential_checker=lambda: [],
        **_steady_machine(),
    )
    assert snap.worst() is HealthState.YELLOW
    assert alerts(snap) == []


def test_render_text_is_agent_readable() -> None:
    snap = build_snapshot(
        service_prober=lambda url: None if "8080" in url else 200,
        host_status_fn=lambda: _FakeHost(),
        credential_checker=lambda: [],
        **_steady_machine(),
    )
    text = render_text(snap)
    assert "System health [red]" in text
    assert "governor [red] unreachable" in text
    assert "Needs attention:" in text


def test_render_text_no_alerts_when_all_green() -> None:
    snap = build_snapshot(
        service_prober=lambda _url: 200,
        host_status_fn=lambda: _FakeHost(),
        credential_checker=lambda: [],
        **_steady_machine(),
    )
    assert "No active alerts." in render_text(snap)


# ── disk + uptime (plan decision 29) ────────────────────────────────────────
#
# Nothing watched either before Track 2 PR 5. Both read the real machine, so
# both take their IO as an argument and every test here injects it: a threshold
# test that depended on the disk of whoever ran it would pass or fail by luck.


@dataclass(frozen=True)
class _FakeUsage:
    """Stand-in for shutil.disk_usage's named tuple."""

    total: int
    used: int
    free: int


def _usage(pct: float) -> Callable[[str], _FakeUsage]:
    total = 29_000_000_000  # the trial VM's disk
    used = int(total * pct / 100)
    return lambda _path: _FakeUsage(total=total, used=used, free=total - used)


@pytest.mark.parametrize(
    ("pct", "expected"),
    [
        (10.0, HealthState.GREEN),
        (84.9, HealthState.GREEN),
        (85.0, HealthState.YELLOW),  # boundary: yellow is inclusive
        (94.9, HealthState.YELLOW),
        (95.0, HealthState.RED),  # boundary: red is inclusive
        (99.9, HealthState.RED),
    ],
)
def test_disk_check_states_by_fullness(pct: float, expected: HealthState) -> None:
    check = disk_check(path=Path("/"), usage_fn=_usage(pct))
    assert check.state is expected
    assert check.kind is CheckKind.HARDWARE
    assert check.target == "disk"


def test_disk_check_reports_the_numbers_not_just_a_colour() -> None:
    check = disk_check(path=Path("/"), usage_fn=_usage(50.0))
    # A bare "disk: green" tells the owner nothing they can act on.
    assert "14.5 GB" in check.detail
    assert "29.0 GB" in check.detail
    assert "50%" in check.detail


def test_disk_check_offers_the_fix_only_when_there_is_a_problem() -> None:
    assert disk_check(path=Path("/"), usage_fn=_usage(50.0)).action is None
    assert disk_check(path=Path("/"), usage_fn=_usage(90.0)).action == "iris housekeeping run"
    assert disk_check(path=Path("/"), usage_fn=_usage(99.0)).action == "iris housekeeping run"


def test_disk_check_is_grey_not_red_when_the_volume_cannot_be_read() -> None:
    def boom(_path: str) -> _FakeUsage:
        raise OSError("no such device")

    check = disk_check(path=Path("/"), usage_fn=boom)
    # Grey is "not measured". Red would page the owner about a working disk.
    assert check.state is HealthState.GREY
    assert "no such device" in check.detail


def test_disk_check_is_grey_when_the_volume_reports_no_size() -> None:
    check = disk_check(path=Path("/"), usage_fn=lambda _p: _FakeUsage(total=0, used=0, free=0))
    assert check.state is HealthState.GREY


def test_disk_check_walks_up_to_a_path_that_exists(tmp_path: Path) -> None:
    # The data dir may not exist yet on a fresh checkout; the volume it would
    # land on is what is being measured, so the check must not give up.
    seen: list[str] = []

    def record(path: str) -> _FakeUsage:
        seen.append(path)
        return _FakeUsage(total=100, used=10, free=90)

    disk_check(path=tmp_path / "not" / "created" / "yet", usage_fn=record)
    assert seen == [str(tmp_path)]


def test_uptime_check_is_yellow_while_the_process_is_new() -> None:
    check = uptime_check(started_at=0.0, now_fn=lambda: 30.0)
    assert check.state is HealthState.YELLOW
    assert check.detail == "restarted just now"


def test_uptime_check_says_how_long_ago_once_it_is_worth_saying() -> None:
    check = uptime_check(started_at=0.0, now_fn=lambda: 300.0)
    assert check.state is HealthState.YELLOW
    assert check.detail == "restarted 5m ago"


def test_uptime_check_goes_green_past_the_window() -> None:
    check = uptime_check(started_at=0.0, now_fn=lambda: 11 * 60.0)
    assert check.state is HealthState.GREEN
    assert check.detail == "up 11m"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(11 * 60.0, "up 11m"), (3 * 3600.0 + 840, "up 3h 14m"), (4 * 86400.0 + 7200, "up 4d 2h")],
)
def test_uptime_check_formats_two_units(seconds: float, expected: str) -> None:
    assert uptime_check(started_at=0.0, now_fn=lambda: seconds).detail == expected


def test_uptime_check_survives_a_clock_that_went_backwards() -> None:
    # NTP correcting a VM's drift must not produce a negative uptime.
    check = uptime_check(started_at=100.0, now_fn=lambda: 40.0)
    assert check.state is HealthState.YELLOW
    assert "-" not in check.detail


def test_build_snapshot_includes_disk_and_uptime() -> None:
    snap = build_snapshot(
        service_prober=lambda _url: 200,
        host_status_fn=lambda: _FakeHost(),
        credential_checker=lambda: [],
        disk_check_fn=lambda: disk_check(path=Path("/"), usage_fn=_usage(50.0)),
        uptime_check_fn=lambda: uptime_check(started_at=0.0, now_fn=lambda: 86400.0),
    )
    targets = {c.target for c in snap.checks}
    assert {"disk", "uptime", "host"} <= targets
    assert snap.worst() is HealthState.GREEN


def test_a_full_disk_alone_makes_the_snapshot_red() -> None:
    snap = build_snapshot(
        service_prober=lambda _url: 200,
        host_status_fn=lambda: _FakeHost(),
        credential_checker=lambda: [],
        disk_check_fn=lambda: disk_check(path=Path("/"), usage_fn=_usage(96.0)),
        uptime_check_fn=lambda: uptime_check(started_at=0.0, now_fn=lambda: 86400.0),
    )
    assert snap.worst() is HealthState.RED
    # It must reach the alert projection too: a red the owner is never told
    # about is the same as a green.
    assert any(c.target == "disk" and "almost full" in c.detail for c in alerts(snap))


# ── the LLM failover proxy (owner's call on decision 29's "containers") ─────
#
# `proxy` and `litellm` are compose services that nothing probed: only the
# governor, evaluator, iris_api, channel_gateway and ollama were checked. They
# are alternatives on port 4000, so one probe covers whichever is serving —
# which answers "is that container up" without mounting a docker socket into
# the app, and mounting one would hand it root-equivalent control of the host.


def test_llm_proxy_is_grey_on_a_mac_pointing_at_lm_studio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LM Studio and the proxy share LM_STUDIO_BASE_URL and are told apart by
    # port. A laptop must not grow a red row for a proxy it never runs.
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.delenv("IRIS_PROXY_PORT", raising=False)
    by_name = {c.target: c for c in service_checks(prober=lambda _url: None)}
    assert by_name["llm_proxy"].state is HealthState.GREY
    assert "optional" in by_name["llm_proxy"].detail


def test_llm_proxy_is_red_with_its_fix_when_the_deployment_wants_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:4000/v1")
    monkeypatch.delenv("IRIS_PROXY_PORT", raising=False)
    by_name = {c.target: c for c in service_checks(prober=lambda _url: None)}
    assert by_name["llm_proxy"].state is HealthState.RED
    assert by_name["llm_proxy"].action == "iris-compose up -d proxy"


def test_llm_proxy_probes_its_own_loopback_not_the_configured_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It shares the governor's network namespace, so it is never remote.

    Following LM_STUDIO_BASE_URL the way ollama follows OLLAMA_BASE_URL would
    make a Mac probe LM Studio's /health, get a 404 and report YELLOW — and the
    not-enabled downgrade only rescues RED.
    """
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://a-mac.tailnet.ts.net:4000/v1")
    seen: list[str] = []

    def record(url: str) -> int:
        seen.append(url)
        return 200

    by_name = {c.target: c for c in service_checks(prober=record)}
    assert by_name["llm_proxy"].endpoint == "http://127.0.0.1:4000/health"
    assert "a-mac.tailnet.ts.net" not in " ".join(seen)


def test_llm_proxy_is_green_when_it_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:4000/v1")
    by_name = {c.target: c for c in service_checks(prober=lambda _url: 200)}
    assert by_name["llm_proxy"].state is HealthState.GREEN
    assert by_name["llm_proxy"].action is None


def test_llm_proxy_honours_a_moved_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_PROXY_PORT", "4100")
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:4100/v1")
    by_name = {c.target: c for c in service_checks(prober=lambda _url: None)}
    assert by_name["llm_proxy"].state is HealthState.RED
    assert by_name["llm_proxy"].endpoint == "http://127.0.0.1:4100/health"


@pytest.mark.parametrize("base_url", ["", "   ", "not a url", "http://host-with-no-port/v1"])
def test_llm_proxy_unrequested_on_junk_configuration(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    # A malformed variable must read as "no proxy asked for", never as a crash
    # in the middle of building a health snapshot.
    monkeypatch.setenv("LM_STUDIO_BASE_URL", base_url)
    monkeypatch.delenv("IRIS_PROXY_PORT", raising=False)
    by_name = {c.target: c for c in service_checks(prober=lambda _url: None)}
    assert by_name["llm_proxy"].state is HealthState.GREY


def test_llm_proxy_unrequested_when_the_port_env_is_junk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_PROXY_PORT", "four thousand")
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:4000/v1")
    by_name = {c.target: c for c in service_checks(prober=lambda _url: None)}
    assert by_name["llm_proxy"].state is HealthState.GREY
