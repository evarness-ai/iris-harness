"""A capability call mints its own call id, apart from its run id (#134, stage 1).

Sync, async and stream methods all pass through ``GovernedToolRunner.execute_call`` /
``aexecute_call`` / ``aexecute_stream``. Each call used to be keyed ``tool_call_id =
run_id``; it now carries a minted ULID that is not the run id, on every audit row of the
call (a stream's items and its end are one call), and the old metadata key has the same value.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from iris_harness.foundation.ids import is_ulid
from iris_harness.kernel.governance import GovernanceKernel, HookPoint
from iris_harness.kernel.governance.audit.log import AuditLog
from iris_harness.kernel.governance.plugins.caller_policy import CallerPolicyHook
from iris_harness.kernel.governance.plugins.capability_redaction import CapabilityRedactionHook
from iris_harness.kernel.governance.plugins.tool_policy import ToolPolicyHook

from .test_capability_calls import Spy, _isolation, _registry  # noqa: F401  (fixture)


def _kernel(tmp_path: Path, *extra: Any) -> tuple[GovernanceKernel, AuditLog]:
    audit = AuditLog(db_path=tmp_path / "audit.db")
    kernel = GovernanceKernel(audit_log=audit)
    for hook in (CallerPolicyHook(), ToolPolicyHook(), CapabilityRedactionHook(), *extra):
        kernel.register(hook)
    kernel.init_lock()
    return kernel, audit


def _by_run(audit: AuditLog) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in audit.query():
        if row.hook_point in ("pre_tool_use", "post_tool_use"):
            out.setdefault(row.run_id, []).append(json.loads(row.payload_json))
    return out


def _check(audit: AuditLog, pre: Spy, post: Spy, calls: int) -> None:
    runs = _by_run(audit)
    assert len(runs) == calls
    seen: set[str] = set()
    for run_id, payloads in runs.items():
        ids = {p.get("call_id") for p in payloads}
        assert len(ids) == 1, (run_id, ids)  # one id on every row of the call
        (call_id,) = ids
        assert is_ulid(call_id) and call_id != run_id  # minted, and not the run id
        seen.add(call_id)
    assert len(seen) == calls
    for ctx in (*pre.seen, *post.seen):
        # The old metadata key: same value as the new one.
        assert ctx.metadata["tool_call_id"] == ctx.metadata["call_id"] in seen


def test_a_sync_capability_call_mints_a_call_id(tmp_path: Path) -> None:
    pre, post = Spy(HookPoint.PRE_TOOL_USE), Spy(HookPoint.POST_TOOL_USE)
    kernel, audit = _kernel(tmp_path, pre, post)
    registry, _ = _registry(tmp_path, kernel)
    inbox = registry.resolve_capability("fin", "test.inbox")
    inbox.search("a")
    inbox.search("b")
    _check(audit, pre, post, calls=2)


def test_an_async_capability_call_mints_a_call_id(tmp_path: Path) -> None:
    pre, post = Spy(HookPoint.PRE_TOOL_USE), Spy(HookPoint.POST_TOOL_USE)
    kernel, audit = _kernel(tmp_path, pre, post)
    registry, _ = _registry(tmp_path, kernel)
    inbox = registry.resolve_capability("fin", "test.inbox")
    asyncio.run(inbox.asearch("a"))
    asyncio.run(inbox.asearch("b"))
    _check(audit, pre, post, calls=2)


def test_a_sync_stream_is_one_call_with_one_call_id(tmp_path: Path) -> None:
    pre, post = Spy(HookPoint.PRE_TOOL_USE), Spy(HookPoint.POST_TOOL_USE)
    kernel, audit = _kernel(tmp_path, pre, post)
    registry, _ = _registry(tmp_path, kernel)
    list(registry.resolve_capability("fin", "test.inbox").stream("a"))
    # One PRE, two items and the end as POST rows: all one call.
    assert len([c for c in post.seen if "stream_item" in c.payload]) == 2
    _check(audit, pre, post, calls=1)


def test_an_async_stream_is_one_call_with_one_call_id(tmp_path: Path) -> None:
    pre, post = Spy(HookPoint.PRE_TOOL_USE), Spy(HookPoint.POST_TOOL_USE)
    kernel, audit = _kernel(tmp_path, pre, post)
    registry, _ = _registry(tmp_path, kernel)
    inbox = registry.resolve_capability("fin", "test.inbox")

    async def drain() -> None:
        async for _ in inbox.astream("a"):
            pass

    asyncio.run(drain())
    assert len([c for c in post.seen if c.payload.get("stream_end")]) == 1
    _check(audit, pre, post, calls=1)


def test_a_denied_capability_call_still_names_its_call(tmp_path: Path) -> None:
    from iris_harness.foundation.capabilities import CapabilityDenied

    pre, post = Spy(HookPoint.PRE_TOOL_USE), Spy(HookPoint.POST_TOOL_USE)
    kernel, audit = _kernel(tmp_path, pre, post)
    registry, provider = _registry(tmp_path, kernel)
    with pytest.raises(CapabilityDenied):
        registry.resolve_capability("fin", "test.inbox").label(1, "x")  # confirm once: held
    assert provider.calls == []
    runs = _by_run(audit)
    assert len(runs) == 1
    [payloads] = runs.values()
    assert payloads and all(is_ulid(p.get("call_id")) for p in payloads)


def test_a_request_made_inside_a_capability_call_names_that_call_as_its_parent(
    tmp_path: Path,
) -> None:
    """#103 + #134: the egress scope a capability method runs in carries the call's minted
    id, so a governed HTTP request the provider makes has ``parent_call_id`` = that id."""
    from iris_harness.kernel.governance.plugin_egress import current_egress_scope

    pre, post = Spy(HookPoint.PRE_TOOL_USE), Spy(HookPoint.POST_TOOL_USE)
    kernel, _ = _kernel(tmp_path, pre, post)
    registry, provider = _registry(tmp_path, kernel)
    seen: list[str | None] = []
    real_search, real_asearch = provider.search, provider.asearch

    def search(query: str) -> Any:
        scope = current_egress_scope()
        seen.append(scope.tool_call_id if scope else None)
        return real_search(query)

    async def asearch(query: str) -> Any:
        scope = current_egress_scope()
        seen.append(scope.tool_call_id if scope else None)
        return await real_asearch(query)

    provider.search, provider.asearch = search, asearch  # type: ignore[method-assign]
    inbox = registry.resolve_capability("fin", "test.inbox")
    inbox.search("a")
    asyncio.run(inbox.asearch("b"))
    call_ids = [c.metadata["call_id"] for c in pre.seen]
    assert len(call_ids) == 2 and seen == call_ids and all(is_ulid(c) for c in seen)
