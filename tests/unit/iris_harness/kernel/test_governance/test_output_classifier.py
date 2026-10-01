"""PostToolUse output classification — a tool result raises the run's label (design §6.5).

The regression these pin: "wondering how my day looks like ?" was classified `public`,
`daily_plan` returned the user's inbox, and every later egress_gate decision in that run
was still made against `public` with personal data in the prompt. The label was derived
once from the question and cached; `_governance_pre_llm` only classified when it was
`None`. The gate answered correctly for the inputs it had — the inputs were stale.

Worse, nothing fired PostToolUse at all, so the side-effect ledger and the G2 injection
guard were registered and never ran.
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance.hooks.types import HookContext, HookPoint
from iris_harness.kernel.governance.plugins.output_classifier import (
    OutputClassifierHook,
    more_restrictive,
)

_EMAILS = "inbox: jordan1.kp@example.com, jordankpatel@example.com"


def _ctx(result: str, *, classification=None, tool: str = "daily_plan"):  # type: ignore[no-untyped-def]
    return HookContext(
        hook_point=HookPoint.POST_TOOL_USE,
        run_id="r1",
        agent_type="chat",
        step_id=0,
        classification=classification,
        payload={"tool_name": tool, "result": result},
    )


# ── the ordering ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("public", "personal", "personal"),
        ("personal", "public", "personal"),
        ("personal", "secret", "secret"),
        ("internal", "public", "internal"),
        ("secret", "secret", "secret"),
    ],
)
def test_the_stricter_label_wins(left: str, right: str, expected: str) -> None:
    assert more_restrictive(left, right) == expected  # type: ignore[arg-type]


def test_nothing_said_is_not_the_same_as_public() -> None:
    """None means the run has no label yet, not that it is safe to send anywhere."""
    assert more_restrictive(None, "personal") == "personal"
    assert more_restrictive("personal", None) == "personal"
    assert more_restrictive(None, None) is None


# ── the hook ──────────────────────────────────────────────────────────────────


async def test_a_personal_tool_result_raises_a_public_run() -> None:
    decision = await OutputClassifierHook()(_ctx(_EMAILS, classification="public"))
    assert decision.outcome == "allow"
    assert decision.set_classification == "personal"
    assert decision.severity == "warn"
    assert decision.audit_metadata["from"] == "public"
    assert decision.audit_metadata["to"] == "personal"
    assert decision.audit_metadata["tool_name"] == "daily_plan"


async def test_a_harmless_result_leaves_the_run_where_it_was() -> None:
    decision = await OutputClassifierHook()(
        _ctx("Today was mostly uneventful.", classification="public")
    )
    assert decision.set_classification is None
    assert "stays public" in decision.reason


async def test_the_label_can_never_be_talked_back_down() -> None:
    """The prompt keeps the personal content for the rest of the run, so a later
    innocuous tool result must not restore `public`."""
    decision = await OutputClassifierHook()(
        _ctx("Today was mostly uneventful.", classification="personal")
    )
    assert decision.set_classification is None
    assert "stays personal" in decision.reason


async def test_an_empty_result_says_nothing() -> None:
    decision = await OutputClassifierHook()(_ctx("", classification="public"))
    assert decision.set_classification is None
    assert "no text" in decision.reason


async def test_an_unlabelled_run_takes_the_result_s_label() -> None:
    decision = await OutputClassifierHook()(_ctx(_EMAILS, classification=None))
    assert decision.set_classification == "personal"


# ── the kernel applies it, and the loop threads it forward ────────────────────


def test_the_kernel_rebinds_the_classification_at_post_tool_use() -> None:
    from iris_harness.kernel.governance.kernel import GovernanceKernel

    kernel = GovernanceKernel()
    kernel.register(OutputClassifierHook())
    kernel.init_lock()
    _decision, final_ctx = kernel.fire_sync(
        HookPoint.POST_TOOL_USE, _ctx(_EMAILS, classification="public")
    )
    assert final_ctx.classification == "personal"


def test_the_hook_is_registered_by_the_governance_build() -> None:
    """A plugin nobody registers is the state PostToolUse was already in."""
    import inspect

    from iris_harness.kernel.governance import wiring

    source = inspect.getsource(wiring)
    assert "OutputClassifierHook()" in source
