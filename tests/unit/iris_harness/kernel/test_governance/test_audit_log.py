from __future__ import annotations

import os
import stat
from datetime import UTC, datetime
from pathlib import Path

from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.kernel.governance.audit import AuditLog


def _audit(tmp_path: Path) -> AuditLog:
    return AuditLog(db_path=tmp_path / "audit.db")


class _AllowHook:
    name: str = "stub_allow"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 10

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="stub", audit_metadata={"k": "v"})


class _DenyHook:
    name: str = "stub_deny"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 20

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(
            outcome="deny",
            reason="stub deny",
            severity="critical",
            audit_metadata={"why": "test"},
        )


class _RaiseHook:
    name: str = "stub_raise"
    hook_point: HookPoint = HookPoint.PRE_LLM_CALL
    priority: int = 5

    async def __call__(self, ctx: HookContext) -> HookDecision:
        raise RuntimeError("boom")


def _ctx() -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="run-1",
        agent_type="chat",
        step_id=2,
        payload={"prompt": "hello"},
    )


def test_init_creates_schema_and_chmod_0o600(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    AuditLog(db_path=db_path)
    mode = stat.S_IMODE(os.stat(db_path).st_mode)
    assert mode == 0o600


def test_record_writes_row_and_returns_id(tmp_path: Path) -> None:
    log = _audit(tmp_path)
    row_id = log.record(
        run_id="r1",
        step_id=1,
        agent_type="chat",
        hook_point="pre_llm_call",
        plugin="egress_gate",
        decision="allow",
        severity="info",
        reason="ok",
        classification="public",
        tier="tier_3",
        payload={"x": 1},
    )
    assert row_id > 0
    rows = log.query(run_id="r1")
    assert len(rows) == 1
    assert rows[0].plugin == "egress_gate"
    assert rows[0].decision == "allow"
    assert rows[0].classification == "public"
    assert '"x": 1' in rows[0].payload_json


def test_query_by_decision_and_time_window(tmp_path: Path) -> None:
    log = _audit(tmp_path)
    base = datetime(2026, 5, 17, 12, 0, tzinfo=UTC)
    for i in range(5):
        log.record(
            run_id="r",
            step_id=i,
            agent_type="chat",
            hook_point="pre_llm_call",
            plugin="p",
            decision="allow" if i % 2 == 0 else "deny",
            severity="info",
            reason="ok",
            ts=base.replace(second=i),
        )

    denies = log.query(decision="deny")
    assert [r.step_id for r in denies] == [1, 3]

    in_window = log.query(
        since=base.replace(second=1),
        until=base.replace(second=3),
    )
    assert [r.step_id for r in in_window] == [1, 2, 3]


def test_query_ordered_by_ts_asc(tmp_path: Path) -> None:
    log = _audit(tmp_path)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    log.record(
        run_id="r",
        step_id=2,
        agent_type="chat",
        hook_point="pre_llm_call",
        plugin="p",
        decision="allow",
        severity="info",
        reason="ok",
        ts=base.replace(second=2),
    )
    log.record(
        run_id="r",
        step_id=1,
        agent_type="chat",
        hook_point="pre_llm_call",
        plugin="p",
        decision="allow",
        severity="info",
        reason="ok",
        ts=base.replace(second=1),
    )
    rows = log.query(run_id="r")
    assert [r.step_id for r in rows] == [1, 2]


async def test_kernel_records_one_row_per_hook(tmp_path: Path) -> None:
    log = _audit(tmp_path)
    kernel = GovernanceKernel(audit_log=log)
    kernel.register(_AllowHook())
    kernel.register(_DenyHook())
    kernel.init_lock()

    decision, _ = await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())
    assert decision.outcome == "deny"

    rows = log.query(run_id="run-1")
    plugins_in_order = [r.plugin for r in rows]
    # Both hooks fired (allow ran first; deny then short-circuits any
    # subsequent hooks but the deny itself is the second row).
    assert plugins_in_order == ["stub_allow", "stub_deny"]
    assert rows[1].decision == "deny"
    assert rows[1].severity == "critical"


async def test_kernel_records_hook_exception_as_deny_row(tmp_path: Path) -> None:
    log = _audit(tmp_path)
    kernel = GovernanceKernel(audit_log=log)
    kernel.register(_RaiseHook())
    kernel.init_lock()

    decision, _ = await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())
    assert decision.outcome == "deny"
    assert decision.severity == "error"

    rows = log.query(run_id="run-1")
    assert len(rows) == 1
    assert rows[0].plugin == "stub_raise"
    assert rows[0].decision == "deny"
    assert rows[0].severity == "error"
    assert "raised" in rows[0].reason


async def test_kernel_without_audit_log_does_not_crash(tmp_path: Path) -> None:
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(_AllowHook())
    kernel.init_lock()
    decision, _ = await kernel.fire(HookPoint.PRE_LLM_CALL, _ctx())
    assert decision.outcome == "allow"


def test_audit_write_failure_does_not_break_kernel(tmp_path: Path) -> None:
    """If the audit DB is unwriteable, the kernel still fires."""

    class _BrokenAudit:
        def record(self, **kwargs: object) -> int:
            raise OSError("disk full")

    import asyncio

    kernel = GovernanceKernel(audit_log=_BrokenAudit())  # type: ignore[arg-type]
    kernel.register(_AllowHook())
    kernel.init_lock()
    decision, _ = asyncio.run(kernel.fire(HookPoint.PRE_LLM_CALL, _ctx()))
    assert decision.outcome == "allow"


async def test_audit_payload_includes_whitelisted_context_keys(tmp_path: Path) -> None:
    """Phase 1 instrumentation: model/provider from ctx.payload land in the
    audit row payload so per-model queries work — but the prompt never does."""
    import json

    log = _audit(tmp_path)
    kernel = GovernanceKernel(audit_log=log)
    kernel.register(_AllowHook())
    kernel.init_lock()

    ctx = HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="run-payload",
        agent_type="chat",
        tier="tier_2",
        payload={
            "prompt": "personal data must not be audited",
            "model": "qwen2.5:7b-instruct",
            "provider": "ollama",
        },
    )
    await kernel.fire(HookPoint.PRE_LLM_CALL, ctx)

    rows = log.query(run_id="run-payload")
    assert len(rows) == 1
    payload = json.loads(rows[0].payload_json)
    assert payload["model"] == "qwen2.5:7b-instruct"
    assert payload["provider"] == "ollama"
    # hook-supplied audit_metadata is preserved
    assert payload["k"] == "v"
    # the prompt is never copied into the audit store
    assert "prompt" not in payload
    assert rows[0].tier == "tier_2"


async def test_audit_metadata_wins_over_context_payload_on_key_clash(tmp_path: Path) -> None:
    """A hook's explicit audit_metadata must not be silently overwritten."""
    import json

    class _MetadataHook:
        name: str = "stub_meta"
        hook_point: HookPoint = HookPoint.PRE_LLM_CALL
        priority: int = 10

        async def __call__(self, ctx: HookContext) -> HookDecision:
            return HookDecision(
                outcome="allow",
                reason="stub",
                audit_metadata={"model": "hook-supplied"},
            )

    log = _audit(tmp_path)
    kernel = GovernanceKernel(audit_log=log)
    kernel.register(_MetadataHook())
    kernel.init_lock()

    ctx = HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="run-clash",
        agent_type="chat",
        payload={"model": "context-supplied"},
    )
    await kernel.fire(HookPoint.PRE_LLM_CALL, ctx)

    rows = log.query(run_id="run-clash")
    payload = json.loads(rows[0].payload_json)
    assert payload["model"] == "hook-supplied"


def test_exotic_payload_does_not_crash_write(tmp_path: Path) -> None:
    log = _audit(tmp_path)

    class _Weird:
        def __repr__(self) -> str:
            return "<weird>"

    log.record(
        run_id="r",
        step_id=0,
        agent_type="chat",
        hook_point="pre_llm_call",
        plugin="p",
        decision="allow",
        severity="info",
        reason="ok",
        payload={"obj": _Weird()},
    )
    rows = log.query(run_id="r")
    assert len(rows) == 1
    # default=str on the encoder coerces the unknown object.
    assert "<weird>" in rows[0].payload_json


def test_default_db_path_honors_iris_home_and_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Regression: a bare AuditLog() must not write to the developer's real
    # ~/.local/share/iris/audit.db during tests. The default follows IRIS_HOME
    # (set to a temp dir by conftest before imports) and an explicit override.
    from iris_harness.kernel.governance.audit import log as audit_mod

    monkeypatch.delenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", raising=False)
    monkeypatch.setenv("IRIS_HOME", "/tmp/iris-home-xyz")
    assert audit_mod._default_audit_db_path() == Path("/tmp/iris-home-xyz/governance/audit.db")

    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", "/tmp/explicit-audit.db")
    assert audit_mod._default_audit_db_path() == Path("/tmp/explicit-audit.db")

    monkeypatch.delenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", raising=False)
    monkeypatch.delenv("IRIS_HOME", raising=False)
    assert audit_mod._default_audit_db_path() == (
        Path.home() / ".local" / "share" / "iris" / "audit.db"
    )
