"""Slice 2 of System Health (ADR-0069): the Google OAuth credential rows. Local-only --
provider status() functions are injected, so nothing touches Keychain, the account DB,
or the network. The rows are the Google connections' (email slice step 4); the core
keeps only the seam a plugin registers them on (``api.register_credential_check``)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from iris_harness.sdk.health import CheckKind, HealthState
from iris_personal.connections.google import google_credential_checks, reset_probe_cache

_NOW = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_probe_cache() -> None:
    reset_probe_cache()


@dataclass(frozen=True)
class _Acct:
    address: str


@dataclass(frozen=True)
class _Status:
    email_account: _Acct
    has_keychain_token: bool
    token_expiry: datetime | None
    refresh_token_present: bool


def _provider(label: str, cli: str, accounts: list[_Status]):
    return (label, cli, lambda: accounts)


def _one(accounts: list[_Status]):
    checks = google_credential_checks([_provider("Gmail", "gmail", accounts)], now=_NOW)
    assert len(checks) == 1
    return checks[0]


# ── not configured → grey, never alerts, offers connect ─────────────────────


def test_not_configured_is_grey_with_connect_action() -> None:
    check = _one([])
    assert check.kind is CheckKind.CREDENTIAL
    assert check.state is HealthState.GREY
    assert check.action == "iris auth gmail login"
    assert "not connected" in check.detail


# ── token presence / refresh capability ─────────────────────────────────────


def test_no_stored_token_is_red_with_user_scoped_remediation() -> None:
    check = _one(
        [
            _Status(
                _Acct("a@b.com"),
                has_keychain_token=False,
                token_expiry=None,
                refresh_token_present=False,
            )
        ]
    )
    assert check.state is HealthState.RED
    assert check.action == "iris auth gmail login --user a@b.com"


def test_refresh_token_present_is_green_even_when_access_expired() -> None:
    expired = _NOW - timedelta(hours=1)
    check = _one([_Status(_Acct("a@b.com"), True, expired, refresh_token_present=True)])
    assert check.state is HealthState.GREEN
    assert check.action is None
    assert "auto-refresh" in check.detail


def test_no_refresh_token_and_expired_is_red() -> None:
    check = _one(
        [_Status(_Acct("a@b.com"), True, _NOW - timedelta(days=1), refresh_token_present=False)]
    )
    assert check.state is HealthState.RED
    assert check.action == "iris auth gmail login --user a@b.com"


def test_no_refresh_token_expiring_soon_is_yellow() -> None:
    check = _one(
        [_Status(_Acct("a@b.com"), True, _NOW + timedelta(days=3), refresh_token_present=False)]
    )
    assert check.state is HealthState.YELLOW
    assert check.action is not None


def test_no_refresh_token_far_future_is_green() -> None:
    check = _one(
        [_Status(_Acct("a@b.com"), True, _NOW + timedelta(days=90), refresh_token_present=False)]
    )
    assert check.state is HealthState.GREEN
    assert check.action is None


def test_no_refresh_token_unknown_expiry_is_yellow() -> None:
    check = _one([_Status(_Acct("a@b.com"), True, None, refresh_token_present=False)])
    assert check.state is HealthState.YELLOW


def test_naive_expiry_does_not_crash() -> None:
    # token_expiry without tzinfo must be treated as UTC, not raise.
    naive = datetime(2026, 1, 1, 0, 0)  # intentionally naive
    check = _one([_Status(_Acct("a@b.com"), True, naive, refresh_token_present=False)])
    assert check.state is HealthState.RED  # 2026-01-01 < _NOW


def test_provider_status_failure_degrades_to_yellow_not_crash() -> None:
    def _boom() -> list[_Status]:
        raise RuntimeError("keychain locked")

    checks = google_credential_checks([("Gmail", "gmail", _boom)], now=_NOW)
    assert checks[0].state is HealthState.YELLOW
    assert "status unavailable" in checks[0].detail


# ── net probe (opt-in revocation detection) ─────────────────────────────────

# A token that looks healthy locally (refresh token + far-future expiry).
_HEALTHY = _Status(_Acct("a@b.com"), True, _NOW + timedelta(days=90), refresh_token_present=True)


def _provider4(loader):  # 4-tuple: (label, cli, status_fn, load_credentials)
    return ("Gmail", "gmail", lambda: [_HEALTHY], loader)


def _probe(loader, *, net_probe: bool = True):
    checks = google_credential_checks([_provider4(loader)], now=_NOW, net_probe=net_probe)
    assert len(checks) == 1
    return checks[0]


def test_net_probe_revoked_token_goes_red() -> None:
    # load_credentials returns None => refresh failed => revoked.
    check = _probe(lambda _addr: None)
    assert check.state is HealthState.RED
    assert "revoked" in check.detail
    assert check.action == "iris auth gmail login --user a@b.com"


def test_net_probe_valid_token_stays_green() -> None:
    check = _probe(lambda _addr: object())  # refresh ok
    assert check.state is HealthState.GREEN


def test_net_probe_transient_error_does_not_override() -> None:
    def _boom(_addr: str) -> object:
        raise OSError("network down")

    check = _probe(_boom)  # unknown verdict => keep local (green)
    assert check.state is HealthState.GREEN


def test_net_probe_off_never_calls_loader() -> None:
    calls: list[str] = []

    def _loader(addr: str) -> object | None:
        calls.append(addr)
        return None  # would be "revoked" IF probed

    check = _probe(_loader, net_probe=False)
    assert check.state is HealthState.GREEN  # slice-2 behavior, no probe
    assert calls == []


def test_net_probe_result_is_cached_within_interval() -> None:
    calls: list[str] = []

    def _loader(addr: str) -> object:
        calls.append(addr)
        return object()

    provider = _provider4(_loader)
    google_credential_checks([provider], now=_NOW, net_probe=True)
    google_credential_checks([provider], now=_NOW, net_probe=True)
    assert len(calls) == 1  # second call served from the 1-hour cache


# ── a revocation found by the watch's diagnosis sticks (ADR-0116) ───────────


def test_a_recent_revoked_verdict_holds_on_local_ticks() -> None:
    """The watch probes once when an incident opens; the 60s ticks that follow run
    without net_probe and must not flip the account back to green."""
    _probe(lambda _addr: None)  # the diagnosis: revoked

    def must_not_call(_addr: str) -> object:
        raise AssertionError("a local tick never touches the network")

    check = _probe(must_not_call, net_probe=False)
    assert check.state is HealthState.RED
    assert check.subject == "a@b.com"
    assert check.key == "Gmail:a@b.com"


def test_a_new_token_voids_the_old_revoked_verdict() -> None:
    _probe(lambda _addr: None)  # revoked on the old token
    relogged = _Status(
        _Acct("a@b.com"), True, _NOW + timedelta(days=120), refresh_token_present=True
    )
    provider = ("Gmail", "gmail", lambda: [relogged], lambda _addr: None)
    [check] = google_credential_checks([provider], now=_NOW, net_probe=False)
    assert check.state is HealthState.GREEN  # the owner re-logged in; stale verdict ignored


def test_a_revoked_verdict_expires_after_the_probe_interval() -> None:
    _probe(lambda _addr: None)
    later = google_credential_checks(
        [_provider4(lambda _addr: None)], now=_NOW + timedelta(hours=2), net_probe=False
    )
    assert later[0].state is HealthState.GREEN


def test_net_probe_reads_the_typed_revoked_error_as_revoked() -> None:
    """Loaders raise CredentialRevokedError on a refused refresh; the probe must not
    mistake it for a network blip ("unknown" would leave a revoked token green)."""
    from iris_harness.sdk.vault import CredentialRevokedError

    def revoked(addr: str) -> object:
        raise CredentialRevokedError("Gmail", addr, f"iris auth gmail login --user {addr}")

    check = _probe(revoked)
    assert check.state is HealthState.RED
    assert check.action == "iris auth gmail login --user a@b.com"
