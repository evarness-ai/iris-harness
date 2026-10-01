"""Regression: NetworkEgress blocks exfiltration of identity/vault secrets (exp-007 GAP-14).

The kernel's per-persona domain allowlist is coding-persona-scoped, so chat-agent
network tools were ungoverned. This guard denies — for ANY agent — a network tool whose
args carry a secret-shaped literal from the identity docs, closing the data-exfil vector
at PreToolUse (defense-in-depth beneath the response-side curator guard).
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance.hooks.types import HookContext, HookPoint
from iris_harness.kernel.governance.plugins.network_egress import NetworkEgress

SECRET = "CANARY_SOUL_SECRET_DIRECTIVE_d41f8a27"


@pytest.fixture()
def egress(owner_identity_documents):
    owner_identity_documents([SECRET])
    return NetworkEgress(policy=None)


def _ctx(tool: str, args: dict, agent_type: str = "chat") -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="rt-test",
        agent_type=agent_type,
        route=f"tool/{tool}",
        payload={"tool_name": tool, "args": args},
    )


@pytest.mark.asyncio
async def test_denies_secret_in_chat_network_call(egress):
    ctx = _ctx("web_fetch", {"url": f"http://127.0.0.1:9/leak?data={SECRET}"})
    decision = await egress(ctx)
    assert decision.outcome == "deny"
    assert "secret" in decision.reason.lower()


@pytest.mark.asyncio
async def test_denies_secret_for_coding_agent_too(egress):
    ctx = _ctx("http_request", {"body": f"token={SECRET}"}, agent_type="coding")
    assert (await egress(ctx)).outcome == "deny"


@pytest.mark.asyncio
async def test_allows_benign_chat_search(egress):
    # No secret in args -> the secret-egress guard does not deny (chat short-circuits to allow).
    ctx = _ctx("research", {"query": "iris local agent framework"})
    assert (await egress(ctx)).outcome == "allow"


@pytest.mark.asyncio
async def test_non_network_tool_unaffected(egress):
    ctx = _ctx("read_file", {"path": f"/notes/{SECRET}.txt"})
    assert (await egress(ctx)).outcome == "allow"
