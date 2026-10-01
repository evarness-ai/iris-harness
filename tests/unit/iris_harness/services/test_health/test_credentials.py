"""Slice 2 of System Health (ADR-0069): the core's credential checks and the seam a
plugin's credential rows arrive on. Local-only -- the env is injected and the
registered checks are fakes, so nothing touches Keychain, the account DB, or the
network. The core names no provider (email slice step 4): the Google rows are the
plugins' own, tested in ``tests/unit/iris_personal/test_connections/``."""

from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest

from iris_harness.services.health import credentials
from iris_harness.services.health.credentials import (
    clear_credential_checks,
    cloud_key_checks,
    credential_checks,
    register_credential_check,
)
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthState


@pytest.fixture(autouse=True)
def _no_registered_checks(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(credentials, "audit_key_checks", lambda: [])
    clear_credential_checks()
    yield
    clear_credential_checks()


def _row(target: str, state: HealthState = HealthState.GREEN) -> HealthCheck:
    return HealthCheck(target, CheckKind.CREDENTIAL, state, f"{target}: connected")


# ── cloud key presence (no validation) ──────────────────────────────────────


def test_cloud_key_present_is_green_absent_is_grey() -> None:
    checks = cloud_key_checks(env={"ANTHROPIC_API_KEY": "placeholder"})
    by_target = {c.target: c for c in checks}
    assert by_target["Anthropic"].state is HealthState.GREEN
    assert by_target["OpenRouter"].state is HealthState.GREY
    assert all(c.kind is CheckKind.CREDENTIAL for c in checks)


def test_github_token_alias_counts_as_present() -> None:
    checks = cloud_key_checks(env={"GITHUB_PAT_CODING_AGENT": "placeholder"})
    assert {c.target: c.state for c in checks}["GitHub"] is HealthState.GREEN


# ── registered checks: after the core's own, in registration order ──────────


def test_registered_checks_follow_the_core_rows_in_registration_order() -> None:
    register_credential_check("first", lambda net_probe: [_row("Mail")])
    register_credential_check("second", lambda net_probe: [_row("Diary")])
    targets = [c.target for c in credential_checks(env={})]
    assert targets == ["Anthropic", "OpenRouter", "GitHub", "Mail", "Diary"]


def test_a_check_is_handed_the_refreshs_probe_choice() -> None:
    seen: list[bool] = []

    def check(net_probe: bool) -> list[HealthCheck]:
        seen.append(net_probe)
        return []

    register_credential_check("probe", check)
    credential_checks(env={}, net_probe=False)
    credential_checks(env={}, net_probe=True)
    assert seen == [False, True]


def test_registering_a_key_again_replaces_it() -> None:
    register_credential_check("mail", lambda net_probe: [_row("Mail")])
    register_credential_check("mail", lambda net_probe: [_row("Mail", HealthState.RED)])
    rows = [c for c in credential_checks(env={}) if c.target == "Mail"]
    assert [c.state for c in rows] == [HealthState.RED]


def test_a_failing_check_is_a_yellow_row_not_a_missing_one() -> None:
    """Mutation: a broken check used to vanish from System Health with a log line."""

    def broken(net_probe: bool) -> list[HealthCheck]:
        raise RuntimeError("keychain locked")

    register_credential_check("mail_credentials", broken)
    register_credential_check("diary", lambda net_probe: [_row("Diary")])
    rows = {c.target: c for c in credential_checks(env={})}
    assert rows["mail_credentials"].state is HealthState.YELLOW
    assert rows["mail_credentials"].kind is CheckKind.CREDENTIAL
    assert "keychain locked" in rows["mail_credentials"].detail
    assert rows["Diary"].state is HealthState.GREEN  # the others still report


def test_the_core_names_no_provider() -> None:
    source = inspect.getsource(credentials)
    for name in ("Gmail", "gmail", "Calendar", "gcalendar", "Drive", "gdrive", "Google"):
        assert name not in source, name
