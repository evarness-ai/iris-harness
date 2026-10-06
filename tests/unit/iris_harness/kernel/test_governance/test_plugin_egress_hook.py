"""``plugin_egress``: the PRE/POST_EGRESS hooks enforce the compiled policy and write rows (#103)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance import GovernanceKernel, build_default_kernel
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.hooks.types import HookContext, HookPoint
from iris_harness.kernel.governance.plugin_egress import (
    HostRule,
    PluginEgress,
    PluginEgressPolicy,
    register_egress_policy,
)


@pytest.fixture()
def kernel(tmp_path: Path) -> Iterator[tuple[GovernanceKernel, AuditLog]]:
    audit = AuditLog(db_path=tmp_path / "audit.db")
    register_egress_policy(
        PluginEgressPolicy(
            {
                "weather": PluginEgress(hosts=(HostRule("api.open-meteo.com"),)),
                "quiet": PluginEgress(),
            }
        )
    )
    try:
        yield build_default_kernel(audit_log=audit), audit
    finally:
        register_egress_policy(None)


def _ctx(
    point: HookPoint, plugin: str, host: str, classification: Any = None, **extra: Any
) -> HookContext:
    return HookContext(
        hook_point=point,
        run_id="r1",
        agent_type="chat",
        classification=classification,
        payload={
            "tool_name": "forecast",
            "tool_plugin": plugin,
            "egress": {
                "plugin": plugin,
                "scheme": "https",
                "host": host,
                "port": 443,
                "method": "GET",
                **extra,
            },
        },
        metadata={"caller": "model:chat"},
    )


async def test_a_declared_host_is_allowed_and_the_row_names_it(
    kernel: tuple[GovernanceKernel, AuditLog],
) -> None:
    k, audit = kernel
    decision, _ = await k.fire(
        HookPoint.PRE_EGRESS, _ctx(HookPoint.PRE_EGRESS, "weather", "api.open-meteo.com")
    )
    assert decision.outcome == "allow"
    [row] = [r for r in audit.query() if r.hook_point == "pre_egress"]
    assert (row.plugin, row.decision) == ("plugin_egress", "allow")
    assert '"host": "api.open-meteo.com"' in row.payload_json
    assert '"caller": "model:chat"' in row.payload_json
    assert '"tool_plugin": "weather"' in row.payload_json


@pytest.mark.parametrize(
    "plugin, host, why",
    [
        ("weather", "evil.example", "not in plugin 'weather'"),
        ("quiet", "api.open-meteo.com", "declares no egress"),
        ("ghost", "api.open-meteo.com", "no mounted manifest"),
    ],
)
async def test_everything_else_is_denied_with_the_host_in_the_row(
    kernel: tuple[GovernanceKernel, AuditLog], plugin: str, host: str, why: str
) -> None:
    k, audit = kernel
    decision, _ = await k.fire(HookPoint.PRE_EGRESS, _ctx(HookPoint.PRE_EGRESS, plugin, host))
    assert decision.outcome == "deny" and why in decision.reason
    [row] = [r for r in audit.query() if r.hook_point == "pre_egress"]
    assert row.decision == "deny" and f'"host": "{host}"' in row.payload_json


async def test_a_run_holding_more_than_the_host_receives_is_denied(
    kernel: tuple[GovernanceKernel, AuditLog],
) -> None:
    k, _ = kernel
    ctx = _ctx(HookPoint.PRE_EGRESS, "weather", "api.open-meteo.com", classification="personal")
    decision, _ = await k.fire(HookPoint.PRE_EGRESS, ctx)
    assert decision.outcome == "deny" and "personal" in decision.reason


async def test_with_no_policy_registered_everything_is_denied(
    kernel: tuple[GovernanceKernel, AuditLog],
) -> None:
    k, _ = kernel
    register_egress_policy(None)
    decision, _ = await k.fire(
        HookPoint.PRE_EGRESS, _ctx(HookPoint.PRE_EGRESS, "weather", "api.open-meteo.com")
    )
    assert decision.outcome == "deny" and "no egress policy" in decision.reason


async def test_the_outcome_row_carries_status_bytes_and_duration_or_the_error_class(
    kernel: tuple[GovernanceKernel, AuditLog],
) -> None:
    k, audit = kernel
    ok = _ctx(
        HookPoint.POST_EGRESS,
        "weather",
        "api.open-meteo.com",
        status=200,
        bytes_in=42,
        bytes_out=0,
        duration_ms=7,
    )
    failed = _ctx(HookPoint.POST_EGRESS, "weather", "api.open-meteo.com", error="ConnectError")
    await k.fire(HookPoint.POST_EGRESS, ok)
    await k.fire(HookPoint.POST_EGRESS, failed)
    rows = [r for r in audit.query() if r.hook_point == "post_egress"]
    assert [r.severity for r in rows] == ["info", "warn"]
    assert '"bytes_in": 42' in rows[0].payload_json and '"status": 200' in rows[0].payload_json
    assert "ConnectError" in rows[1].payload_json


def test_the_hooks_are_registered_at_their_points(
    kernel: tuple[GovernanceKernel, AuditLog],
) -> None:
    k, _ = kernel
    assert "plugin_egress" in k.hook_names(HookPoint.PRE_EGRESS)
    assert "plugin_egress_outcome" in k.hook_names(HookPoint.POST_EGRESS)
