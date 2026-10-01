"""Keyed audit digests (``kernel/governance/audit/digest.py``).

An audit row carries fingerprints of a call's arguments and result, never their text. The
fingerprint is an HMAC-SHA256 under a key derived (HKDF) from the vault master key, so a
short value (an email, a phone number) cannot be recovered by hashing guesses. The key is
resolved lazily, on the first governed call, never when the kernel is built (a macOS
Keychain dialog at process start hangs the process). With no key, no governed call runs.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import keyring
import pytest
from cryptography.fernet import Fernet

from iris_harness.foundation.capability_fields import canonical_json
from iris_harness.kernel.governance import build_default_kernel
from iris_harness.kernel.governance.audit.digest import (
    AUDIT_DIGEST_INFO,
    NO_AUDIT_KEY_MESSAGE,
    AuditKeyUnavailable,
    audit_digester,
    audit_key_status,
    derive_audit_key,
    digester_for,
)
from iris_harness.kernel.governance.kernel import _AUDITED_PAYLOAD_KEYS
from iris_harness.kernel.governance.vault import keys as vault_keys
from iris_harness.kernel.governance.wiring import kernel_from_env
from iris_harness.services.health.credentials import audit_key_checks
from iris_harness.services.health.models import HealthState

VALUE = {"to": "someone@example.test", "n": 1}


def _key() -> bytes:
    return Fernet.generate_key()


# ------------------------------------------------------------------ the digest itself
def test_the_digest_is_deterministic_for_one_key() -> None:
    key = _key()
    assert digester_for(key).digest(VALUE) == digester_for(key).digest(dict(VALUE))
    assert digester_for(key).digest(VALUE) != digester_for(key).digest({**VALUE, "n": 2})


def test_the_digest_is_keyed_not_a_plain_hash() -> None:
    plain = hashlib.sha256(canonical_json(VALUE).encode("utf-8")).hexdigest()
    one, two = digester_for(_key()), digester_for(_key())
    assert one.digest(VALUE) != two.digest(VALUE)
    assert one.digest(VALUE) not in plain and two.digest(VALUE) not in plain
    assert "someone" not in one.digest(VALUE)


def test_the_audit_key_is_not_the_master_key_and_info_separates_keys() -> None:
    master = _key()
    import base64

    raw = base64.urlsafe_b64decode(master)
    audit = derive_audit_key(master)
    assert audit != raw and len(audit) == 32
    assert derive_audit_key(master, info=b"iris/something-else/v1") != audit
    assert digester_for(master, info=b"iris/something-else/v1").digest(VALUE) != digester_for(
        master
    ).digest(VALUE)
    assert AUDIT_DIGEST_INFO == b"iris/audit-digest/v1"


def test_digest_alg_names_the_key_and_changes_with_it() -> None:
    master = _key()
    one = digester_for(master)
    assert one.alg.startswith("hmac-sha256/v1/") and one.alg == digester_for(master).alg
    assert digester_for(_key()).key_id != one.key_id
    assert one.args_fields(VALUE) == {"args_digest": one.digest(VALUE), "digest_alg": one.alg}
    assert one.result_fields("x") == {"result_digest": one.digest("x"), "digest_alg": one.alg}
    # The key id is a one-way fingerprint of the derived key, never the key.
    assert one.key_id not in master.decode() and "digest_alg" in _AUDITED_PAYLOAD_KEYS
    assert master.decode() not in repr(one)


def test_a_key_that_is_not_a_fernet_key_is_no_key() -> None:
    with pytest.raises(AuditKeyUnavailable, match="not a Fernet key"):
        digester_for(b"not-a-fernet-key")


# ------------------------------------------------------------------ lazy resolution
def test_building_the_kernel_never_resolves_the_master_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    real = vault_keys.resolve_master_key
    monkeypatch.setattr(vault_keys, "resolve_master_key", lambda: calls.append(1) or real())

    assert kernel_from_env() is not None
    build_default_kernel(audit_log=None)
    assert calls == []
    assert audit_key_status()[0] == "unresolved"

    first = audit_digester()
    assert audit_digester() is first and calls == [1]  # resolved once, then cached


def test_resolution_is_once_per_process_across_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = vault_keys.resolve_master_key
    monkeypatch.setattr(vault_keys, "resolve_master_key", lambda: calls.append(1) or real())
    seen: list[object] = []
    threads = [threading.Thread(target=lambda: seen.append(audit_digester())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [1] and len({id(d) for d in seen}) == 1


def test_the_key_is_the_one_in_the_keyring() -> None:
    master = keyring.get_password("iris-vault", "master-key")
    assert master is not None
    assert audit_digester().alg == digester_for(master.encode()).alg


def test_the_env_key_wins_over_the_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    env_key = _key()
    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", env_key.decode())
    assert audit_digester().alg == digester_for(env_key).alg


@pytest.mark.usefixtures("no_vault_master_key")
def test_a_failure_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AuditKeyUnavailable) as err:
        audit_digester()
    assert NO_AUDIT_KEY_MESSAGE in str(err.value)
    assert audit_key_status()[0] == "unavailable"

    # The owner sets a key: the next call picks it up, no restart.
    env_key = _key()
    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", env_key.decode())
    assert audit_digester().alg == digester_for(env_key).alg
    assert audit_key_status() == ("ready", digester_for(env_key).alg)


@pytest.mark.usefixtures("no_vault_master_key")
def test_a_keyring_that_raises_is_refused_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def get_password(self, service: str, username: str) -> str | None:
            raise RuntimeError("no backend")

    monkeypatch.setattr(vault_keys, "keyring", Broken())
    with pytest.raises(AuditKeyUnavailable):
        audit_digester()


# ------------------------------------------------------------------ the health row
def test_the_health_row_reads_the_state_without_resolving_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vault_keys, "resolve_master_key", lambda: pytest.fail("a health check resolved the key")
    )
    assert audit_key_checks() == []


def test_the_health_row_is_green_once_resolved() -> None:
    alg = audit_digester().alg
    (row,) = audit_key_checks()
    assert row.state == HealthState.GREEN and row.detail == alg


@pytest.mark.usefixtures("no_vault_master_key")
def test_the_health_row_is_red_when_the_key_is_missing() -> None:
    with pytest.raises(AuditKeyUnavailable):
        audit_digester()
    (row,) = audit_key_checks()
    assert row.state == HealthState.RED and NO_AUDIT_KEY_MESSAGE in row.detail


# ------------------------------------------------------------------ what a row holds
def test_audit_rows_carry_keyed_digests_and_their_alg(tmp_path: Path) -> None:
    """Every governed tool call's rows carry the keyed digest and ``digest_alg``."""
    from iris_harness.agent.agentic_core import ToolSpec
    from iris_harness.agent.tool_runner import GovernedToolRunner, ToolCall
    from iris_harness.kernel.governance.audit.log import AuditLog

    audit = AuditLog(tmp_path / "audit.db")
    kernel = build_default_kernel(audit_log=audit)
    tool = ToolSpec(
        name="look_up",
        description="d",
        call=lambda args: f"found {args['q']}-result-text",
        effect="read",
    )
    runner = GovernedToolRunner(kernel=kernel, agent_type="chat")
    outcome = runner.execute(tool, {"q": "someone@example.test"}, ToolCall())
    assert outcome.status == "ran"

    rows = [json.loads(r.payload_json) for r in audit.query() if "look_up" in r.payload_json]
    digester = audit_digester()
    pre = [p for p in rows if "args_digest" in p]
    post = [p for p in rows if "result_digest" in p]
    assert pre and post
    assert {p["args_digest"] for p in pre} == {digester.digest({"q": "someone@example.test"})}
    assert {p["result_digest"] for p in post} == {digester.digest(outcome.text)}
    assert all(p["digest_alg"] == digester.alg for p in pre + post)
    blob = " ".join(r.payload_json for r in audit.query())
    assert "someone@example.test" not in blob and "result-text" not in blob


def test_the_module_keeps_no_unkeyed_digest() -> None:
    """``foundation.capability_fields.digest`` (an unkeyed SHA-256) is gone."""
    from iris_harness.foundation import capability_fields

    assert not hasattr(capability_fields, "digest")
