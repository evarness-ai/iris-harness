"""Paired devices — the capability, once, for every surface (ADR-0117).

Until now every client of ``iris_api`` held the same ``IRIS_AUTH_SECRET``. That is
fine for services on one machine and wrong for a phone: a lost phone could only be
locked out by rotating the secret every service shares, and the console could not
tell one client from another. A paired device holds its own token instead, scoped
``read`` or ``control``, revocable on its own.

The same rule as ``approvals/service.py``: the decision and its audit trail live
here, and the API routes (``server/iris_api/device_routes.py``) and the CLI
(``iris device``, which calls those routes) are thin callers. Nothing else mints,
verifies or revokes a token.

What is stored, and why that is enough:

- A **device token** is 32 random bytes. The store keeps its SHA-256. A slow KDF buys
  nothing against 256 bits of entropy and would cost every request a hash round.
- A **pairing code** is short enough to type, so its digest *can* be brute-forced
  from a stolen DB. It is single-use, dead in five minutes, and voided by five wrong
  guesses; someone who can read the DB within that window can read everything else
  in the data dir too.

Neither the token nor the code is ever logged or put in a ledger payload.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast

from iris_harness.foundation.auth import Scope
from iris_harness.kernel.governance.devices.store import DeviceRow, DeviceStore
from iris_harness.kernel.governance.devices.throttle import ClaimThrottle

__all__ = [
    "PAIRING_CODE_TTL",
    "PAIRING_MAX_ATTEMPTS",
    "TOKEN_PREFIX",
    "DeviceService",
    "Ledger",
    "PairedDevice",
    "PairingCode",
    "PairingRefusedError",
    "PairingThrottledError",
]

TOKEN_PREFIX = (
    "irisd_"  # noqa: S105 — a marker, not a credential: it makes a leaked token greppable
)
PAIRING_CODE_TTL = timedelta(minutes=5)
PAIRING_MAX_ATTEMPTS = 5
_LAST_SEEN_RESOLUTION = timedelta(seconds=60)
_MAX_NAME_LENGTH = 64

# No 0/O, 1/I/L or U: the code is read off one screen and typed into another.
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"
_CODE_LENGTH = 8

_SCOPES: frozenset[str] = frozenset({"read", "control"})
_KINDS: frozenset[str] = frozenset({"app", "browser"})


class PairingRefusedError(ValueError):
    """The code is wrong, expired, already used, or voided. Deliberately one error:
    telling a guesser *which* would tell them whether a code is live."""


class PairingThrottledError(Exception):
    """Too many failed claims lately: this one was refused without looking at its
    code. Deliberately NOT a :class:`PairingRefusedError` — a surface must not answer
    it with the "wrong code" reply, because the code was never checked."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("too many pairing attempts")
        self.retry_after = retry_after


class Ledger(Protocol):
    """The slice of ``AuditLog`` this module writes to."""

    def record(  # AuditLog.record's own signature
        self,
        *,
        run_id: str,
        step_id: int | None,
        agent_type: str,
        hook_point: str,
        plugin: str,
        decision: str,
        severity: str,
        reason: str,
        payload: dict[str, Any] | None = ...,
    ) -> int: ...


@dataclass(frozen=True)
class PairingCode:
    """Shown to the owner once. ``code`` is formatted for reading (``ABCD-EFGH``)."""

    code: str
    scope: Scope
    expires_at: str


@dataclass(frozen=True)
class PairedDevice:
    """The result of a claim. ``token`` exists only here: it is not stored and
    cannot be shown again."""

    device: DeviceRow
    token: str


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _normalise_code(code: str) -> str:
    return "".join(ch for ch in code.upper() if ch.isalnum())


def _clean_name(name: str) -> str:
    cleaned = " ".join(name.split())
    if not cleaned or len(cleaned) > _MAX_NAME_LENGTH or not cleaned.isprintable():
        raise ValueError(f"device name must be 1-{_MAX_NAME_LENGTH} printable characters")
    return cleaned


class DeviceService:
    """Pair, verify, list and revoke devices."""

    def __init__(
        self,
        store: DeviceStore | None = None,
        ledger: Ledger | None = None,
        throttle: ClaimThrottle | None = None,
    ) -> None:
        self._store = store or DeviceStore()
        self._ledger = ledger if ledger is not None else _default_ledger()
        # The failed-claim window lives in this object, so a process that accepts
        # claims must hold ONE service (iris_api does) rather than build one per call.
        self._throttle = throttle if throttle is not None else ClaimThrottle()

    # ------------------------------------------------------------------
    # Pairing
    # ------------------------------------------------------------------

    def start_pairing(self, *, scope: str, actor: str, now: datetime | None = None) -> PairingCode:
        """Mint a pairing code for a device of ``scope``. ``actor`` is who asked
        (``service`` or ``device:<id>``) and goes to the ledger."""
        if scope not in _SCOPES:
            raise ValueError(f"scope must be one of {sorted(_SCOPES)}")
        now = now or datetime.now(UTC)
        raw = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))
        expires_at = self._store.add_pairing_code(
            code_hash=_digest(raw), scope=scope, now=now, ttl=PAIRING_CODE_TTL
        )
        self._record(
            "pairing_started",
            subject="pairing",
            severity="info",
            reason=f"pairing code issued by {actor} for a {scope} device",
            payload={"actor": actor, "scope": scope, "expires_at": expires_at},
        )
        half = _CODE_LENGTH // 2
        return PairingCode(
            code=f"{raw[:half]}-{raw[half:]}", scope=cast(Scope, scope), expires_at=expires_at
        )

    def claim(
        self, *, code: str, name: str, kind: str, now: datetime | None = None
    ) -> PairedDevice:
        """Trade a live pairing code for a device token.

        Raises :class:`PairingRefusedError` for any code that does not pair, and
        charges the failure against the live codes (see ``record_failed_claim``) and
        against the global window (``throttle.py``). While that window is full, raises
        :class:`PairingThrottledError` instead — first, before anything about the
        request is examined, so a throttled claim tells the caller nothing and costs
        the owner's live code nothing.
        """
        retry_after = self._throttle.retry_after()
        if retry_after is not None:
            raise PairingThrottledError(retry_after)
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {sorted(_KINDS)}")
        cleaned = _clean_name(name)
        now = now or datetime.now(UTC)
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        device = self._store.claim(
            code_hash=_digest(_normalise_code(code)),
            token_hash=_digest(token),
            name=cleaned,
            kind=kind,
            now=now,
            max_attempts=PAIRING_MAX_ATTEMPTS,
        )
        if device is None:
            voided = self._store.record_failed_claim(now=now, max_attempts=PAIRING_MAX_ATTEMPTS)
            if voided:
                self._record(
                    "pairing_code_voided",
                    subject="pairing",
                    severity="warn",
                    reason=f"{voided} pairing code(s) voided after "
                    f"{PAIRING_MAX_ATTEMPTS} failed claims",
                    payload={"voided": voided},
                )
            if self._throttle.record_failure():
                window_s = int(self._throttle.window.total_seconds())
                self._record(
                    "pairing_throttled",
                    subject="pairing",
                    severity="warn",
                    reason=f"claims paused: {self._throttle.limit} failed claims "
                    f"within {window_s}s",
                    payload={"limit": self._throttle.limit, "window_seconds": window_s},
                )
            raise PairingRefusedError("pairing code is not valid")
        self._record(
            "paired",
            subject=device.device_id,
            severity="info",
            reason=f"device paired: {device.name} ({device.kind}, {device.scope})",
            payload={
                "device_id": device.device_id,
                "name": device.name,
                "kind": device.kind,
                "scope": device.scope,
            },
        )
        return PairedDevice(device=device, token=token)

    # ------------------------------------------------------------------
    # Verification — the ``DeviceVerifier`` the auth middleware is handed
    # ------------------------------------------------------------------

    def verify(self, token: str, *, now: datetime | None = None) -> tuple[str, Scope] | None:
        """``(device_id, scope)`` for a live device token, else ``None``."""
        if not token.startswith(TOKEN_PREFIX):
            return None
        # Looked up by digest: the digest of an attacker-chosen token says nothing
        # about a stored one, so there is no secret-dependent comparison to time.
        device = self._store.find_live(_digest(token))
        if device is None:
            return None
        self._store.touch(
            device.device_id, now=now or datetime.now(UTC), min_interval=_LAST_SEEN_RESOLUTION
        )
        return device.device_id, cast(Scope, device.scope)

    # ------------------------------------------------------------------
    # Listing and revoking
    # ------------------------------------------------------------------

    def list_devices(self) -> list[DeviceRow]:
        return self._store.list_devices()

    def get(self, device_id: str) -> DeviceRow | None:
        return self._store.get(device_id)

    def revoke(self, device_id: str, *, actor: str, now: datetime | None = None) -> DeviceRow:
        """Revoke a device; its token stops working on the next request. Raises
        ``DeviceNotFoundError`` for an unknown ID. Revoking twice is a no-op."""
        device, changed = self._store.revoke(device_id, now=now or datetime.now(UTC))
        if changed:
            self._record(
                "revoked",
                subject=device.device_id,
                severity="warn",
                reason=f"device revoked by {actor}: {device.name}",
                payload={"device_id": device.device_id, "name": device.name, "actor": actor},
            )
        return device

    # ------------------------------------------------------------------

    def _record(
        self, decision: str, *, subject: str, severity: str, reason: str, payload: dict[str, Any]
    ) -> None:
        # The audit log (an append-only row per event, no caller key), not the side-effect
        # ledger: there is no key to collide, so nothing here needs ``exclusive`` (#102).
        self._ledger.record(
            run_id=f"device:{subject}",
            step_id=None,
            agent_type="devices",
            hook_point="device_pairing",
            plugin="DeviceService",
            decision=decision,
            severity=severity,
            reason=reason,
            payload=payload,
        )


def _default_ledger() -> Ledger:
    from iris_harness.kernel.governance.audit import AuditLog

    return AuditLog()
