"""PR 3 leaves the egress guard and the response check as they were (ADR-0125).

The table decides ``deny`` for an email at egress and ``mask`` for a name in an answer to
someone else, but nothing reads those cells until PR 4 (shadow first). What moves in PR 3
is one thing: the owner's confirmed ``blog`` and ``website`` are ``link``, and egress has
always refused a link.
"""

from __future__ import annotations

from typing import Any

import pytest

from iris_harness.kernel.governance.hooks.response_payload import pre_response_payload
from iris_harness.kernel.governance.hooks.types import HookContext, HookPoint
from iris_harness.kernel.governance.plugins.network_egress import NetworkEgress
from iris_harness.kernel.governance.plugins.response_safety import (
    ResponseSafetyHook,
    check_response,
)

SECRET = "CANARY_SOUL_SECRET_DIRECTIVE_d41f8a27"
BLOG = "https://blog.robin.example/2026"
PII = {
    "name": "Robin Example",
    "first name": "Robin",
    "email": "owner.canary@example.com",
    "phone": "+1 555 0100 0199",
    "address": "1 Example Street, Springfield",
    "handle": "@robin-gh",
}


@pytest.fixture(autouse=True)
def _corpus(owner_identity_seam: Any) -> None:
    seam = owner_identity_seam
    seam.register_identity_text_provider(lambda: [f"my key: {SECRET}"])
    seam.register_owner_identity_source(
        "facts",
        lambda: {
            "name": ["Robin Example", "Robin"],
            "email": [PII["email"]],
            "phone": [PII["phone"]],
            "address": [PII["address"]],
            "handle": ["robin-gh"],
            "link": [BLOG],  # a confirmed blog fact, mapped by identity.yaml
        },
    )
    NetworkEgress._reset_identity_cache()


def _egress_ctx(tool: str, query: str) -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r",
        agent_type="chat",
        route=f"tool/{tool}",
        payload={"tool_name": tool, "args": {"query": query}},
    )


@pytest.mark.parametrize("kind", sorted(PII))
@pytest.mark.parametrize("tool", ["research", "web_fetch", "github_create_issue"])
async def test_egress_still_lets_the_new_kinds_through(kind: str, tool: str) -> None:
    decision = await NetworkEgress(policy=None)(_egress_ctx(tool, f"about {PII[kind]}"))
    assert decision.outcome == "allow", kind


@pytest.mark.parametrize("tool", ["research", "web_fetch", "github_create_issue", "mcp_x"])
async def test_egress_refuses_the_owners_confirmed_blog(tool: str) -> None:
    """The one intended change: a confirmed blog/website is a link, and links never leave."""
    decision = await NetworkEgress(policy=None)(_egress_ctx(tool, f"summarise {BLOG}"))
    assert decision.outcome == "deny"


async def test_egress_still_refuses_a_secret() -> None:
    decision = await NetworkEgress(policy=None)(_egress_ctx("research", f"k={SECRET}"))
    assert decision.outcome == "deny"


@pytest.mark.parametrize("kind", sorted(PII))
def test_the_response_check_still_passes_the_new_kinds(kind: str) -> None:
    assert check_response(f"Your {kind} is {PII[kind]}.").verdict == "pass"


def test_the_response_check_passes_the_blog_and_halts_a_secret() -> None:
    assert check_response(f"Your blog is {BLOG}").verdict == "pass"
    assert check_response(f"Your key is {SECRET}").verdict == "halt"


@pytest.mark.parametrize("audience", ["owner", "other"])
async def test_no_guard_reads_the_audience_yet(audience: Any) -> None:
    text = " ".join(PII.values())
    ctx = HookContext(
        hook_point=HookPoint.PRE_RESPONSE,
        run_id="r",
        agent_type="chat",
        payload=pre_response_payload(text, audience=audience),
    )
    assert (await ResponseSafetyHook()(ctx)).outcome == "allow"
