"""The turn's label is the floor under EVERY model call in the turn, not only the loop's.

``kernel/governance/turn_label.apply_turn_floor`` is the one rule. The loop applied the
turn's label before each of its own model calls; every other model call in the turn --
the shared governed client (curator judges, narration, synthesis) and the callers that
fire ``PRE_LLM_CALL`` themselves (task planner, intent router, conversation compactor,
entity extractor) -- classified its own prompt afresh and could egress under a lower
label than the data the turn holds. Each is driven here through its real entry point.

The one exemption is declared: the cloud search-synthesis client carries stripped public
content, so personal reads as internal for it -- but ``secret`` is never lowered.
Outside a turn nothing changes: the floor is the call's own label.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

import iris_harness.llm.client as llm_client
from iris_harness.kernel.governance import (
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.kernel.governance.plugins.egress_gate import EgressGate
from iris_harness.kernel.governance.turn_label import (
    apply_turn_floor,
    lift_turn_label,
    turn_label_scope,
)
from iris_harness.llm.client import CodingLLMClient, CodingLLMInvocationError


class _PromptLabel:
    """PRE_CLASSIFY: every prompt reads as ``public`` -- tamer than the turn."""

    name = "prompt_label"
    hook_point = HookPoint.PRE_CLASSIFY
    priority = 10

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="t", set_classification="public")


class _Egress:
    """PRE_LLM_CALL: records the label each model call was governed as."""

    name = "egress_spy"
    hook_point = HookPoint.PRE_LLM_CALL
    priority = 99

    def __init__(self, priority: int = 99) -> None:
        self.priority = priority
        self.labels: list[str | None] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.labels.append(ctx.classification)
        return HookDecision(outcome="allow", reason="spy")


def _kernel(*hooks: Any) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=None)
    for hook in hooks:
        kernel.register(hook)
    kernel.init_lock()
    return kernel


class _Reply:
    content = "TASK: t1 | answer | system | none\nINTENT: general\nAGENT: system"
    tool_calls: list[Any] = []
    additional_kwargs: dict[str, Any] = {}
    usage_metadata = None
    response_metadata: dict[str, Any] = {}


class _Model:
    def invoke(self, messages: object, **_kw: Any) -> _Reply:
        return _Reply()


class _TierRouter:
    """Every intent on the local tier-1 model."""

    def get_llm_config(self, intent: str) -> Any:
        return llm_client.PROVIDER_DEFAULTS["ollama"].model_copy(update={"tier_name": "tier1"})


def _stub_llm(prompt: str) -> str:
    return _Reply.content


@pytest.fixture()
def egress(monkeypatch: pytest.MonkeyPatch) -> _Egress:
    """One kernel for every governed caller; no model is reached."""
    spy = _Egress()
    kernel = _kernel(_PromptLabel(), spy)
    for module in (
        "iris_harness.llm.client",
        "iris_harness.agent.task_planner",
        "iris_harness.agent.intent_router",
        "iris_harness.memory.compactor",
        "iris_harness.memory.knowledge.entity_extractor",
    ):
        monkeypatch.setattr(f"{module}.kernel_from_env", lambda: kernel)
    monkeypatch.setattr(llm_client, "_default_model_factory", lambda **_kw: _Model())
    monkeypatch.setenv("IRIS_CURATOR_FAITHFULNESS_LLM", "1")
    return spy


# ------------------------------------------------------------ the real entry points
def _planner() -> None:
    from iris_harness.agent.task_planner import TaskPlanner

    TaskPlanner(llm_call=_stub_llm).plan("plan my week", is_multi_step=True)


def _router() -> None:
    from iris_harness.agent.intent_router import LLMClassifier

    LLMClassifier(_stub_llm).classify("what is on today")


def _compactor() -> None:
    from iris_harness.memory.compactor import ConversationCompactor, ConversationTurn

    turns = [ConversationTurn(role="user", content=f"turn {i} " * 20) for i in range(6)]
    ConversationCompactor(compaction_threshold=2, keep_recent=1, llm_call=_stub_llm).compact(
        turns, force=True
    )


def _entity_extractor() -> None:
    from iris_harness.memory.knowledge.entity_extractor import EntityExtractor

    EntityExtractor(llm_call=_stub_llm).extract("a note about the quarterly review " * 4)


def _curator_judge() -> None:
    from iris_harness.runtime.judges import build_curator_faithfulness_client

    judge = build_curator_faithfulness_client(
        tier_router=_TierRouter(), llm_call=None  # type: ignore[arg-type]
    )
    assert judge is not None
    asyncio.run(judge.judge(query="q", response="r"))


def _narrate() -> None:
    from iris_harness.llm.narrate import make_narrative_llm_call

    call = make_narrative_llm_call(_TierRouter())  # type: ignore[arg-type]
    assert call is not None
    call("summarise these grounded rows")


ENTRY_POINTS: dict[str, Callable[[], None]] = {
    "task_planner": _planner,
    "intent_router": _router,
    "compactor": _compactor,
    "entity_extractor": _entity_extractor,
    "curator_judge": _curator_judge,
    "narrate": _narrate,
}


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_every_model_call_in_a_turn_is_governed_at_least_as_the_turn(
    egress: _Egress, name: str
) -> None:
    with turn_label_scope():
        lift_turn_label("personal")
        ENTRY_POINTS[name]()
    assert egress.labels, f"{name} made no governed model call"
    assert set(egress.labels) == {"personal"}, (name, egress.labels)


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_outside_a_turn_the_call_keeps_its_own_label(egress: _Egress, name: str) -> None:
    ENTRY_POINTS[name]()
    assert egress.labels and set(egress.labels) == {"public"}, (name, egress.labels)


def test_the_floor_never_lowers_a_stricter_prompt() -> None:
    with turn_label_scope():
        lift_turn_label("internal")
        assert apply_turn_floor("secret") == "secret"
        assert apply_turn_floor("public") == "internal"


# ------------------------------------------------------------ the gate it feeds
def _cloud_client(kernel: GovernanceKernel, **kw: Any) -> CodingLLMClient:
    """A client governance reads as cloud (tier_3); no credentials needed to build it."""
    return CodingLLMClient(
        llm_client.PROVIDER_DEFAULTS["ollama"],
        governance_kernel=kernel,
        governance_target_tier="tier_3",
        model_factory=lambda **_kw: _Model(),
        **kw,
    )


def test_a_personal_turn_cannot_reach_a_cloud_tier_through_a_tame_prompt() -> None:
    """The failure mode the floor closes: a narrator/judge prompt that classifies public
    was sent to a cloud tier while the turn held personal data."""
    kernel = _kernel(_PromptLabel(), EgressGate())
    client = _cloud_client(kernel)
    decision, _ = client._governance_pre_llm(prompt_text="tame", run_id="r")
    assert decision is not None and decision.outcome == "allow"  # outside a turn: public
    with turn_label_scope():
        lift_turn_label("personal")
        with pytest.raises(CodingLLMInvocationError, match="blocked by governance"):
            client.invoke(system_prompt="", user_prompt="tame")


# ------------------------------------------------------------ the one declared exemption
@pytest.fixture()
def synthesis(monkeypatch: pytest.MonkeyPatch) -> Callable[[GovernanceKernel], Any]:
    """The real cloud search-synthesis client, as ``handlers/react.py`` builds it."""
    from iris_harness.runtime.handlers import react

    monkeypatch.setenv("IRIS_SEARCH_SYNTHESIS_PROVIDER", "copilot")
    monkeypatch.setenv("IRIS_ENABLE_COPILOT_BACKEND", "1")
    monkeypatch.setattr(
        react, "config_from_profile", lambda _name: llm_client.PROVIDER_DEFAULTS["copilot"]
    )

    def build(kernel: GovernanceKernel) -> Any:
        monkeypatch.setattr(llm_client, "kernel_from_env", lambda: kernel)
        client = react._build_search_synthesis_client()
        assert client is not None
        return client

    return build


def test_synthesis_keeps_its_downgrade_under_a_personal_turn(synthesis: Any) -> None:
    spy = _Egress()
    client = synthesis(_kernel(_PromptLabel(), EgressGate(), spy))
    with turn_label_scope():
        lift_turn_label("personal")
        decision, _ = client._governance_pre_llm(prompt_text="public web pages", run_id="r")
    assert decision.outcome == "allow"
    assert spy.labels == ["internal"]


def test_synthesis_never_lowers_a_secret_turn(synthesis: Any) -> None:
    spy = _Egress(priority=20)  # ahead of the egress gate, which stops the chain
    client = synthesis(_kernel(_PromptLabel(), spy, EgressGate()))
    with turn_label_scope():
        lift_turn_label("secret")
        decision, _ = client._governance_pre_llm(prompt_text="public web pages", run_id="r")
    assert decision.outcome == "deny"
    assert spy.labels == ["secret"]


def test_the_exemption_is_only_for_a_declared_client() -> None:
    spy = _Egress()
    client = _cloud_client(_kernel(_PromptLabel(), spy))
    with turn_label_scope():
        lift_turn_label("personal")
        client._governance_pre_llm(prompt_text="public web pages", run_id="r")
    assert spy.labels == ["personal"]
    with turn_label_scope():
        lift_turn_label("personal")
        assert apply_turn_floor("public", stripped_public_content=True) == "internal"
        assert apply_turn_floor("public") == "personal"
