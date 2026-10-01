"""Tests for paired devices — store + service (ADR-0117)."""

from __future__ import annotations

import os
import re
import sqlite3
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.devices import (
    PAIRING_CODE_TTL,
    PAIRING_MAX_ATTEMPTS,
    TOKEN_PREFIX,
    DeviceNotFoundError,
    DeviceService,
    DeviceStore,
    PairingRefusedError,
)
from iris_harness.kernel.governance.devices.store import default_devices_db_path

T0 = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


class _Ledger:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **row: Any) -> int:
        self.rows.append(row)
        return len(self.rows)

    def decisions(self) -> list[str]:
        return [r["decision"] for r in self.rows]


@pytest.fixture
def ledger() -> _Ledger:
    return _Ledger()


@pytest.fixture
def service(tmp_path: Path, ledger: _Ledger) -> DeviceService:
    return DeviceService(store=DeviceStore(db_path=tmp_path / "devices.db"), ledger=ledger)


def _pair(service: DeviceService, *, scope: str = "control", name: str = "iPhone", now=T0):
    code = service.start_pairing(scope=scope, actor="service", now=now)
    return service.claim(code=code.code, name=name, kind="browser", now=now)


# ── the store file ───────────────────────────────────────────────────────────


def test_db_created_with_0o600_perms(tmp_path: Path) -> None:
    db_path = tmp_path / "devices.db"
    DeviceStore(db_path=db_path)
    assert stat.S_IMODE(os.stat(db_path).st_mode) == 0o600


def test_default_path_sits_beside_the_approvals_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_HOME", str(tmp_path))
    assert default_devices_db_path() == tmp_path / "governance" / "devices.db"
    assert DeviceStore().db_path == tmp_path / "governance" / "devices.db"


def test_neither_token_nor_code_is_stored_in_the_clear(
    tmp_path: Path, service: DeviceService
) -> None:
    code = service.start_pairing(scope="read", actor="service", now=T0)
    paired = service.claim(code=code.code, name="iPad", kind="app", now=T0)
    raw = (tmp_path / "devices.db").read_bytes()
    wal = tmp_path / "devices.db-wal"
    raw += wal.read_bytes() if wal.exists() else b""
    assert paired.token.encode() not in raw
    assert paired.token.removeprefix(TOKEN_PREFIX).encode() not in raw
    assert code.code.replace("-", "").encode() not in raw


# ── pairing ──────────────────────────────────────────────────────────────────


def test_pairing_code_shape_and_expiry(service: DeviceService) -> None:
    code = service.start_pairing(scope="read", actor="service", now=T0)
    assert re.fullmatch(r"[A-HJKMNP-TV-Z2-9]{4}-[A-HJKMNP-TV-Z2-9]{4}", code.code)
    assert code.scope == "read"
    assert code.expires_at == (T0 + PAIRING_CODE_TTL).isoformat()
    assert timedelta(minutes=5) == PAIRING_CODE_TTL


def test_claim_returns_a_token_that_verifies_with_the_codes_scope(service: DeviceService) -> None:
    paired = _pair(service, scope="read", name="  Robin's   iPhone ")
    assert paired.token.startswith(TOKEN_PREFIX)
    assert len(paired.token) >= len(TOKEN_PREFIX) + 43  # 32 bytes, urlsafe base64
    assert paired.device.name == "Robin's iPhone"
    assert paired.device.kind == "browser"
    assert paired.device.scope == "read"
    assert service.verify(paired.token) == (paired.device.device_id, "read")


def test_tokens_are_unique_per_device(service: DeviceService) -> None:
    assert _pair(service).token != _pair(service).token


def test_code_is_typed_forgivingly(service: DeviceService) -> None:
    code = service.start_pairing(scope="control", actor="service", now=T0)
    typed = f"  {code.code.lower().replace('-', ' ')} "
    assert service.claim(code=typed, name="Mac", kind="browser", now=T0).device.scope == "control"


def test_code_is_single_use(service: DeviceService) -> None:
    code = service.start_pairing(scope="control", actor="service", now=T0)
    service.claim(code=code.code, name="first", kind="browser", now=T0)
    with pytest.raises(PairingRefusedError):
        service.claim(code=code.code, name="second", kind="browser", now=T0)
    assert [d.name for d in service.list_devices()] == ["first"]


def test_code_expires(service: DeviceService) -> None:
    code = service.start_pairing(scope="control", actor="service", now=T0)
    just_inside = T0 + PAIRING_CODE_TTL - timedelta(seconds=1)
    at_expiry = T0 + PAIRING_CODE_TTL
    with pytest.raises(PairingRefusedError):
        service.claim(code=code.code, name="late", kind="browser", now=at_expiry)
    # Exactly at expiry is too late; a second before is not.
    assert service.claim(code=code.code, name="ok", kind="browser", now=just_inside)


def test_wrong_guesses_void_the_live_code(service: DeviceService, ledger: _Ledger) -> None:
    code = service.start_pairing(scope="control", actor="service", now=T0)
    for _ in range(PAIRING_MAX_ATTEMPTS - 1):
        with pytest.raises(PairingRefusedError):
            service.claim(code="AAAA-AAAA", name="guess", kind="browser", now=T0)
    assert "pairing_code_voided" not in ledger.decisions()
    with pytest.raises(PairingRefusedError):
        service.claim(code="AAAA-AAAA", name="guess", kind="browser", now=T0)
    assert ledger.decisions().count("pairing_code_voided") == 1
    # The right code no longer pairs, and further guesses do not re-report it.
    with pytest.raises(PairingRefusedError):
        service.claim(code=code.code, name="owner", kind="browser", now=T0)
    assert ledger.decisions().count("pairing_code_voided") == 1
    assert service.list_devices() == []


def test_one_short_of_the_limit_still_pairs(service: DeviceService) -> None:
    code = service.start_pairing(scope="control", actor="service", now=T0)
    for _ in range(PAIRING_MAX_ATTEMPTS - 1):
        with pytest.raises(PairingRefusedError):
            service.claim(code="AAAA-AAAA", name="guess", kind="browser", now=T0)
    assert service.claim(code=code.code, name="owner", kind="browser", now=T0)


def test_a_new_code_starts_with_a_clean_count(service: DeviceService) -> None:
    service.start_pairing(scope="control", actor="service", now=T0)
    for _ in range(PAIRING_MAX_ATTEMPTS):
        with pytest.raises(PairingRefusedError):
            service.claim(code="AAAA-AAAA", name="guess", kind="browser", now=T0)
    fresh = service.start_pairing(scope="control", actor="service", now=T0)
    assert service.claim(code=fresh.code, name="owner", kind="browser", now=T0)


def test_starting_a_pairing_sweeps_dead_codes(tmp_path: Path, service: DeviceService) -> None:
    service.start_pairing(scope="read", actor="service", now=T0)
    used = service.start_pairing(scope="read", actor="service", now=T0)
    service.claim(code=used.code, name="x", kind="app", now=T0)
    service.start_pairing(scope="read", actor="service", now=T0 + PAIRING_CODE_TTL)
    with sqlite3.connect(tmp_path / "devices.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM pairing_codes").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "", "kind": "browser"}, "device name"),
        ({"name": "   ", "kind": "browser"}, "device name"),
        ({"name": "x" * 65, "kind": "browser"}, "device name"),
        ({"name": "bad\x00name", "kind": "browser"}, "device name"),
        ({"name": "ok", "kind": "toaster"}, "kind"),
    ],
)
def test_claim_validates_before_spending_the_code(
    service: DeviceService, kwargs: dict[str, str], match: str
) -> None:
    code = service.start_pairing(scope="control", actor="service", now=T0)
    with pytest.raises(ValueError, match=match) as excinfo:
        service.claim(code=code.code, now=T0, **kwargs)
    assert not isinstance(excinfo.value, PairingRefusedError)
    assert service.claim(code=code.code, name="x" * 64, kind="browser", now=T0)


def test_start_pairing_rejects_an_unknown_scope(service: DeviceService) -> None:
    with pytest.raises(ValueError, match="scope"):
        service.start_pairing(scope="admin", actor="service", now=T0)


# ── verification ─────────────────────────────────────────────────────────────


def test_verify_refuses_what_it_did_not_mint(service: DeviceService) -> None:
    paired = _pair(service)
    assert service.verify("") is None
    assert service.verify("not-a-device-token") is None
    assert service.verify(TOKEN_PREFIX + "x" * 43) is None
    assert service.verify(paired.token[:-1]) is None
    assert service.verify(paired.token.removeprefix(TOKEN_PREFIX)) is None


def test_revoked_token_stops_verifying(service: DeviceService) -> None:
    keep, lose = _pair(service, name="keep"), _pair(service, name="lose")
    revoked = service.revoke(lose.device.device_id, actor="service", now=T0)
    assert revoked.revoked and revoked.revoked_at == T0.isoformat()
    assert service.verify(lose.token) is None
    assert service.verify(keep.token) == (keep.device.device_id, "control")


def test_revoke_unknown_device_raises(service: DeviceService) -> None:
    with pytest.raises(DeviceNotFoundError):
        service.revoke("no-such-device", actor="service")


def test_last_seen_moves_at_most_once_a_minute(service: DeviceService) -> None:
    paired = _pair(service)
    device_id = paired.device.device_id
    assert service.get(device_id).last_seen_at is None  # type: ignore[union-attr]
    service.verify(paired.token, now=T0)
    service.verify(paired.token, now=T0 + timedelta(seconds=59))
    assert service.get(device_id).last_seen_at == T0.isoformat()  # type: ignore[union-attr]
    later = T0 + timedelta(seconds=61)
    service.verify(paired.token, now=later)
    assert service.get(device_id).last_seen_at == later.isoformat()  # type: ignore[union-attr]


def test_list_keeps_revoked_devices_as_history(service: DeviceService) -> None:
    first = _pair(service, name="first", now=T0)
    _pair(service, name="second", now=T0 + timedelta(seconds=1))
    service.revoke(first.device.device_id, actor="service")
    listed = service.list_devices()
    assert [(d.name, d.revoked) for d in listed] == [("first", True), ("second", False)]
    assert not hasattr(listed[0], "token_hash")


# ── the ledger ───────────────────────────────────────────────────────────────


def test_pair_and_revoke_reach_the_ledger_without_secrets(
    service: DeviceService, ledger: _Ledger
) -> None:
    code = service.start_pairing(scope="read", actor="device:abc", now=T0)
    paired = service.claim(code=code.code, name="iPhone", kind="app", now=T0)
    service.revoke(paired.device.device_id, actor="service")
    service.revoke(paired.device.device_id, actor="service")  # no-op: no second row

    assert ledger.decisions() == ["pairing_started", "paired", "revoked"]
    started, claimed, revoked = ledger.rows
    assert started["payload"]["actor"] == "device:abc"
    assert claimed["run_id"] == f"device:{paired.device.device_id}"
    assert claimed["payload"] == {
        "device_id": paired.device.device_id,
        "name": "iPhone",
        "kind": "app",
        "scope": "read",
    }
    assert revoked["severity"] == "warn" and revoked["payload"]["actor"] == "service"
    assert {r["agent_type"] for r in ledger.rows} == {"devices"}
    assert {r["hook_point"] for r in ledger.rows} == {"device_pairing"}
    everything = repr(ledger.rows)
    assert paired.token not in everything
    assert code.code not in everything and code.code.replace("-", "") not in everything


def test_default_ledger_is_the_governance_audit_log(tmp_path: Path) -> None:
    service = DeviceService(store=DeviceStore(db_path=tmp_path / "devices.db"))
    assert isinstance(service._ledger, AuditLog)


def test_pairing_lands_in_a_real_audit_log(tmp_path: Path) -> None:
    audit = AuditLog(db_path=tmp_path / "audit.db")
    service = DeviceService(store=DeviceStore(db_path=tmp_path / "devices.db"), ledger=audit)
    paired = _pair(service)
    rows = audit.query(run_id=f"device:{paired.device.device_id}")
    assert [r.decision for r in rows] == ["paired"]
    assert rows[0].plugin == "DeviceService"
