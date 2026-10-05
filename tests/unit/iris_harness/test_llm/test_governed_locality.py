"""A model call is governed by where it RUNS, never by what its tier is called.

The egress gate decides by the call's target tier, and ``tier_3`` means the prompt leaves
the owner's machines. The target used to be read off the tier's name, so the local LM
Studio model configured as ``tier3`` governed as cloud: a personal turn needed approval
for a call that never left the Mac. It is now the provider's declared locality
(``providers:`` in ``llm_tiers.yaml``, ``llm/locality.py``): cloud or undeclared is
``tier_3``; a local tier is ``tier_1``/``tier_2`` by size.

The four components that take a bare ``llm_call`` (task planner, intent router,
conversation compactor, entity extractor) fired their own ``PRE_LLM_CALL`` under a
hardcoded ``tier_1``. Handed the governed prompt call production hands them, they fire
nothing: its client governs the call once, at the real tier. Handed an opaque callable
they cannot know where it goes, and govern it as leaving the machine.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

import iris_harness.llm.client as llm_client
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.plugins.egress_gate import EgressGate
from iris_harness.kernel.governance.turn_label import lift_turn_label, turn_label_scope
from iris_harness.llm.client import CodingLLMClient, CodingLLMConfig, GovernedPromptCall
from iris_harness.llm.locality import parse_provider_localities
from iris_harness.llm.tier_router import TierRouter, governance_tier_for_intent

SHIPPED = Path("config/llm_tiers.yaml")


class _PromptLabel:
    name = "prompt_label"
    hook_point = HookPoint.PRE_CLASSIFY
    priority = 10

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="t", set_classification="public")


class _Spy:
    """PRE_LLM_CALL: which caller governed a call, at which tier."""

    name = "spy"
    hook_point = HookPoint.PRE_LLM_CALL
    priority = 99

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.calls.append((ctx.agent_type, ctx.tier))
        return HookDecision(outcome="allow", reason="spy")


def _kernel(*hooks: Any) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=None)
    for hook in (_PromptLabel(), *hooks):
        kernel.register(hook)
    kernel.init_lock()
    return kernel


def _without_provider(text: str, provider: str) -> str:
    """The YAML with ``provider``'s ``providers:`` entry removed (the mutation)."""
    lines = text.splitlines()
    out: list[str] = []
    skip = False
    for line in lines:
        if line == f"  {provider}:":
            skip = True
            continue
        if skip and line.startswith("    "):
            continue
        skip = False
        out.append(line)
    return "\n".join(out) + "\n"


def _router_from(tmp_path: Path, text: str) -> TierRouter:
    path = tmp_path / "llm_tiers.yaml"
    path.write_text(text)
    return TierRouter.load_from_yaml(path)


# ------------------------------------------------------------ where each tier governs
def test_the_local_model_named_tier3_governs_as_local() -> None:
    router = TierRouter.load_from_yaml(SHIPPED)
    assert router.get_tier_by_name("tier3").provider == "lmstudio"  # type: ignore[union-attr]

    assert governance_tier_for_intent(router, "complex_task") == "tier_2"
    cfg = cast(CodingLLMConfig, router.get_llm_config("complex_task"))
    assert cfg.governance_tier == "tier_2"
    assert CodingLLMClient(CodingLLMConfig(**vars(cfg)))._governance_target_tier == "tier_2"


def test_no_shipped_tier_governs_as_cloud() -> None:
    """The shipped tiers all run on the owner's machines (decision 13)."""
    router = TierRouter.load_from_yaml(SHIPPED)
    labels = {name: router.governance_tier_for_tier(name) for name in router._tiers}
    assert "tier_3" not in labels.values(), labels


def test_the_shipped_providers_declare_their_locality() -> None:
    import yaml

    mac = parse_provider_localities(yaml.safe_load(SHIPPED.read_text())["providers"])
    assert mac["ollama"] == mac["lmstudio"] == "local"
    assert {mac[p] for p in ("copilot", "github", "openrouter", "anthropic")} == {"cloud"}


def test_a_cloud_provider_governs_as_cloud_under_any_tier_name(tmp_path: Path) -> None:
    """The reverse hazard: a cloud model configured under a small tier's name."""
    text = SHIPPED.read_text().replace(
        '    provider: "ollama"\n    # Tier 1 evaluation log:',
        '    provider: "copilot"\n    # Tier 1 evaluation log:',
    )
    router = _router_from(tmp_path, text)
    assert router.get_tier_by_name("tier1").provider == "copilot"  # type: ignore[union-attr]
    assert router.governance_tier_for_tier("tier1") == "tier_3"
    assert governance_tier_for_intent(router, "general") == "tier_3"


# ------------------------------------------------------------ the YAML declaration drives it
def test_removing_the_declaration_governs_the_provider_as_cloud(tmp_path: Path) -> None:
    router = _router_from(tmp_path, _without_provider(SHIPPED.read_text(), "lmstudio"))

    assert router.governance_tier_for_tier("tier3") == "tier_3"
    assert governance_tier_for_intent(router, "complex_task") == "tier_3"


def test_declaring_the_provider_cloud_governs_it_as_cloud(tmp_path: Path) -> None:
    text = SHIPPED.read_text().replace(
        "  lmstudio:\n    runs: local", "  lmstudio:\n    runs: cloud"
    )
    router = _router_from(tmp_path, text)

    assert router.governance_tier_for_tier("tier3") == "tier_3"
    assert router.governance_tier_for_tier("tier2") == "tier_2"  # ollama is untouched


def test_a_config_dir_without_the_block_inherits_the_shipped_declarations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An owner's config dir from before the block existed must not turn every local
    model into a cloud one; a block that IS present stays authoritative."""
    from iris_harness.llm.locality import declared_localities, provider_locality

    (tmp_path / "llm_tiers.yaml").write_text("tiers: {}\n")
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(tmp_path))
    assert provider_locality("lmstudio") == "local"
    assert provider_locality("copilot") == "cloud"

    (tmp_path / "llm_tiers.yaml").write_text("providers:\n  ollama: {runs: local}\ntiers: {}\n")
    assert declared_localities() == {"ollama": "local"}
    assert provider_locality("lmstudio") == "cloud"


def test_a_router_with_no_tiers_labels_the_last_resort_tier_it_runs_on(tmp_path: Path) -> None:
    """No file (or no ``fallback``): the call runs on the inline last-resort tier (local
    ollama), so it is labelled local -- not cloud because there was no tier to read."""
    router = TierRouter.load_from_yaml(tmp_path / "missing.yaml")

    assert router.get_tier("communication").provider == "ollama"
    assert governance_tier_for_intent(router, "communication") == "tier_1"
    assert cast(CodingLLMConfig, router.get_llm_config("communication")).governance_tier == (
        "tier_1"
    )


def test_a_misspelt_locality_fails_closed() -> None:
    assert parse_provider_localities(
        {"lmstudio": {"runs": "lcoal"}, "ollama": {"runs": "local"}}
    ) == {"ollama": "local"}


# ------------------------------------------------------------ the gate it feeds
def _governed(cfg: CodingLLMConfig, kernel: GovernanceKernel) -> str:
    client = CodingLLMClient(cfg, governance_kernel=kernel)
    with turn_label_scope():
        lift_turn_label("personal")
        decision, _ = client._governance_pre_llm(prompt_text="statement text", run_id="r")
    assert decision is not None
    return decision.outcome


def test_a_personal_turn_reaches_the_local_tier3_without_approval() -> None:
    router = TierRouter.load_from_yaml(SHIPPED)
    cfg = cast(CodingLLMConfig, router.get_llm_config("complex_task"))

    assert _governed(cfg, _kernel(EgressGate())) == "allow"


def test_a_personal_turn_needs_approval_for_a_cloud_provider(tmp_path: Path) -> None:
    text = SHIPPED.read_text().replace(
        '    provider: "lmstudio"\n    model: "qwen/qwen3.6-35b-a3b"',
        '    provider: "anthropic"\n    model: "claude-haiku-4-5"',
    )
    router = _router_from(tmp_path, text)
    cfg = cast(CodingLLMConfig, router.get_llm_config("complex_task"))

    assert cfg.governance_tier == "tier_3"
    assert _governed(cfg, _kernel(EgressGate())) == "require_approval"


# ------------------------------------------------------------ the four bare-callable callers
def _planner(llm: Callable[[str], str]) -> None:
    from iris_harness.agent.task_planner import TaskPlanner

    TaskPlanner(llm_call=llm).plan("plan my week", is_multi_step=True)


def _router_call(llm: Callable[[str], str]) -> None:
    from iris_harness.agent.intent_router import LLMClassifier

    LLMClassifier(llm).classify("what is on today")


def _compactor(llm: Callable[[str], str]) -> None:
    from iris_harness.memory.compactor import ConversationCompactor, ConversationTurn

    turns = [ConversationTurn(role="user", content=f"turn {i} " * 20) for i in range(6)]
    ConversationCompactor(compaction_threshold=2, keep_recent=1, llm_call=llm).compact(
        turns, force=True
    )


def _extractor(llm: Callable[[str], str]) -> None:
    from iris_harness.memory.knowledge.entity_extractor import EntityExtractor

    EntityExtractor(llm_call=llm).extract("a note about the quarterly review " * 4)


CALLERS: dict[str, tuple[Callable[[Callable[[str], str]], None], str]] = {
    "task_planner": (_planner, "task_planner"),
    "intent_router": (_router_call, "intent_router"),
    "compactor": (_compactor, "memory_compactor"),
    "entity_extractor": (_extractor, "entity_extractor"),
}


class _Reply:
    content = "TASK: t1 | answer | system | none\nINTENT: general\nAGENT: system"
    tool_calls: list[Any] = []
    additional_kwargs: dict[str, Any] = {}
    usage_metadata = None
    response_metadata: dict[str, Any] = {}


class _Model:
    invocations = 0

    def invoke(self, messages: object, **_kw: Any) -> _Reply:
        _Model.invocations += 1
        return _Reply()


@pytest.fixture()
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    monkeypatch.setattr(_Model, "invocations", 0)
    spy = _Spy()
    kernel = _kernel(spy)
    for module in (
        "iris_harness.llm.client",
        "iris_harness.agent.task_planner",
        "iris_harness.agent.intent_router",
        "iris_harness.memory.compactor",
        "iris_harness.memory.knowledge.entity_extractor",
    ):
        monkeypatch.setattr(f"{module}.kernel_from_env", lambda: kernel)
    monkeypatch.setattr(llm_client, "_default_model_factory", lambda **_kw: _Model())
    return spy


@pytest.mark.parametrize("name", sorted(CALLERS))
def test_a_governed_prompt_call_is_governed_once_at_the_tier_it_goes_to(
    spy: _Spy, name: str
) -> None:
    router = TierRouter.load_from_yaml(SHIPPED)
    entry, own_agent_type = CALLERS[name]
    call = GovernedPromptCall(
        lambda: cast(CodingLLMConfig, router.get_llm_config("complex_task")),
        agent_type=own_agent_type,
    )

    entry(call)

    assert _Model.invocations, f"{name} made no model call"
    # One governed PRE_LLM_CALL per model call (the client's; the caller no longer fires
    # a second), at the tier the call goes to (tier3, local), not a hardcoded tier_1.
    assert spy.calls == [(own_agent_type, "tier_2")] * _Model.invocations


@pytest.mark.parametrize("name", sorted(CALLERS))
def test_an_opaque_callable_is_governed_as_leaving_the_machine(spy: _Spy, name: str) -> None:
    entry, own_agent_type = CALLERS[name]

    entry(lambda _prompt: _Reply.content)

    assert spy.calls == [(own_agent_type, "tier_3")] * len(spy.calls) and spy.calls
