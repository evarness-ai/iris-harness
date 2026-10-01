"""A secret turn's search step runs on the local tier, never on the cloud synthesis client.

Cloud search synthesis (``runtime/handlers/react.py``) is the search loop's model when
``IRIS_SEARCH_SYNTHESIS_PROVIDER=copilot``. Since the turn's label floors every model call
(#743), a turn labelled ``secret`` has the egress gate refuse that cloud client -- and the
refusal used to propagate out of the loop and fail the turn. The step now runs on the
local tier the loop would use without synthesis, and the ledger says so. Only the egress
gate's refusal moves the step: any other refusal, and any other error, stands.

Driven through the real handler (both the sync and the streaming path), the real
``AgenticCore`` loop, the real ``EgressGate`` and a real ``TierRouter``; only the model
transport is a stub, and it records every provider it was asked to reach.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import iris_harness.llm.client as llm_client
from iris_harness.agent.agent_executor import AgentTask
from iris_harness.foundation.paths import repo_root
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.hooks.types import DataClassification
from iris_harness.kernel.governance.plugins.egress_gate import EgressGate
from iris_harness.kernel.governance.turn_label import lift_turn_label, turn_label_scope
from iris_harness.llm.tier_router import TierRouter
from iris_harness.runtime.handlers import react
from iris_harness.tools.skills.registry import SkillRegistry

_LOCAL_TIERS = """
providers:
  ollama: {runs: local}
  copilot: {runs: cloud}
tiers:
  tier1:
    name: "Fast"
    provider: "ollama"
    model: "local-small"
    max_tokens: 256
    num_ctx: 4096
    temperature: 0.1
    timeout_seconds: 10
    use_for: [search, general]
"""


class _PromptLabel:
    """PRE_CLASSIFY: every prompt reads as ``public`` -- the turn's label is the floor."""

    name = "prompt_label"
    hook_point = HookPoint.PRE_CLASSIFY
    priority = 10

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="t", set_classification="public")


class _Refuse:
    """PRE_LLM_CALL: a non-egress refusal (a prompt guard, a cost limit) of cloud calls."""

    name = "other_guard"
    hook_point = HookPoint.PRE_LLM_CALL
    priority = 5

    async def __call__(self, ctx: HookContext) -> HookDecision:
        if ctx.tier == "tier_3":
            return HookDecision(outcome="deny", reason="other guard says no")
        return HookDecision(outcome="allow", reason="ok")


class _Reply:
    content = "Thought: done.\nFinal Answer: answered"
    tool_calls: list[Any] = []
    additional_kwargs: dict[str, Any] = {}
    usage_metadata = None
    response_metadata: dict[str, Any] = {}


class _Model:
    def __init__(self, provider: str) -> None:
        self._provider = provider

    def invoke(self, messages: object, **_kw: Any) -> _Reply:
        reply = _Reply()
        reply.content = f"Thought: done.\nFinal Answer: answered by {self._provider}"
        return reply


class _EmptyRegistry(SkillRegistry):
    def __init__(self) -> None:
        super().__init__(repo_root=Path("/nonexistent"))

    def discover(self) -> tuple[Any, ...]:
        return ()


@pytest.fixture()
def dialed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Cloud synthesis on; every provider the model transport is asked to reach."""
    calls: list[str] = []

    def factory(**kwargs: Any) -> _Model:
        calls.append(str(kwargs.get("provider")))
        return _Model(str(kwargs.get("provider")))

    monkeypatch.setattr(llm_client, "_default_model_factory", factory)
    monkeypatch.setenv("IRIS_SEARCH_SYNTHESIS_PROVIDER", "copilot")
    monkeypatch.setenv("IRIS_ENABLE_COPILOT_BACKEND", "1")
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    # The tool menu stays small here; keep the loop on the search intent regardless.
    monkeypatch.setenv("IRIS_REACT_TIER_BUMP_TOOL_COUNT", "1000")
    # The Copilot profile, minus its device-flow auth: the provider is what governance reads.
    monkeypatch.setattr(
        react,
        "config_from_profile",
        lambda _name: llm_client.PROVIDER_DEFAULTS["copilot"].model_copy(
            update={"auth_mode": "static", "api_key_env": None}
        ),
    )
    monkeypatch.setattr(react, "_SEARCH_SYNTHESIS_CLIENT", None)
    monkeypatch.setattr(react, "_SEARCH_SYNTHESIS_INIT", False)
    return calls


def _use_kernel(monkeypatch: pytest.MonkeyPatch, *hooks: Any) -> None:
    kernel = GovernanceKernel(audit_log=None)
    for hook in (_PromptLabel(), *hooks):
        kernel.register(hook)
    kernel.init_lock()
    monkeypatch.setattr("iris_harness.kernel.governance.kernel_from_env", lambda: kernel)
    monkeypatch.setattr(llm_client, "kernel_from_env", lambda: kernel)


def _router(tmp_path: Path, text: str = _LOCAL_TIERS) -> TierRouter:
    path = tmp_path / "llm_tiers.yaml"
    path.write_text(text)
    return TierRouter.load_from_yaml(path)


def _run(router: TierRouter, label: DataClassification, *, stream: bool) -> str:
    handler, stream_handler = react._make_react_handler(router, _EmptyRegistry())
    task = AgentTask(query="latest release notes", agent_type="system", params={"intent": "search"})
    with turn_label_scope():
        lift_turn_label(label)
        if not stream:
            result = handler(task)
            return str(result[0] if isinstance(result, tuple) else result)
        chunks: Iterator[Any] = stream_handler(task)
        return "".join(c for c in chunks if isinstance(c, str))


def _fallback_rows(tmp_path: Path) -> list[Any]:
    rows = AuditLog(db_path=tmp_path / "audit.db").query()
    return [r for r in rows if r.plugin == "search_synthesis_fallback"]


@pytest.mark.parametrize("stream", [False, True], ids=["chat", "chat_stream"])
def test_a_secret_turn_search_runs_on_the_local_tier(
    dialed: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stream: bool
) -> None:
    _use_kernel(monkeypatch, EgressGate())

    answer = _run(_router(tmp_path), "secret", stream=stream)

    assert "answered by ollama" in answer
    assert "copilot" not in dialed  # the cloud transport is never reached
    assert set(dialed) == {"ollama"}
    rows = _fallback_rows(tmp_path)
    assert len(rows) == len(dialed)  # one per step moved to the local tier
    assert json.loads(rows[0].payload_json)["refused_by"] == "egress_gate"
    assert rows[0].tier == "tier_1"


def test_a_personal_turn_search_still_uses_cloud_synthesis(
    dialed: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The stripped-public-content exemption is intact: personal reads as internal."""
    _use_kernel(monkeypatch, EgressGate())

    answer = _run(_router(tmp_path), "personal", stream=False)

    assert "answered by copilot" in answer
    assert set(dialed) == {"copilot"}
    assert _fallback_rows(tmp_path) == []


def _run_expecting_failure(router: TierRouter) -> str:
    """Run the search turn; the loop may surface the error or absorb it into its answer."""
    try:
        return _run(router, "secret", stream=False)
    except Exception as exc:  # noqa: BLE001 - what was dialed is what matters here
        return f"raised: {exc}"


def test_another_hooks_refusal_is_not_routed_around(
    dialed: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only the egress gate's refusal moves the step; a prompt guard's (say) stands."""
    _use_kernel(monkeypatch, _Refuse(), EgressGate())

    answer = _run_expecting_failure(_router(tmp_path))

    assert "other guard says no" in answer
    assert dialed == []  # neither the cloud nor the local model
    assert _fallback_rows(tmp_path) == []


def test_a_transport_error_is_not_a_fallback(
    dialed: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An error that is not a governance refusal propagates as before: no local retry."""
    _use_kernel(monkeypatch)  # no egress gate: the cloud call is attempted

    def broken(**kwargs: Any) -> _Model:
        dialed.append(str(kwargs.get("provider")))
        raise ConnectionError("synthesis provider down")

    monkeypatch.setattr(llm_client, "_default_model_factory", broken)

    answer = _run_expecting_failure(_router(tmp_path))

    assert "synthesis provider down" in answer  # the same error, not a masked one
    assert dialed and set(dialed) == {"copilot"}
    assert _fallback_rows(tmp_path) == []


@pytest.mark.parametrize("stream", [False, True], ids=["chat", "chat_stream"])
def test_on_the_shipped_tiers_a_secret_search_runs_on_the_local_tier(
    dialed: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stream: bool
) -> None:
    """``config/llm_tiers.yaml`` puts search on tier2, a local model (governed tier_2), and
    secret is local only (``config/governance/egress.yaml``: any local tier, never tier_3).
    The loop's own gate lets the step through; the cloud synthesis client is refused, so
    the step runs on tier2 and the fallback is audited. The cloud transport is never
    dialed, and the answer is the local model's."""
    _use_kernel(monkeypatch, EgressGate())
    router = TierRouter.load_from_yaml(repo_root() / "config" / "llm_tiers.yaml")
    assert router.governance_tier_for_tier(router._resolve_tier_name("search")) == "tier_2"
    local_provider = router.get_tier("search").provider

    answer = _run(router, "secret", stream=stream)

    assert f"answered by {local_provider}" in answer
    assert "copilot" not in dialed  # the cloud transport is never reached
    assert dialed and set(dialed) == {local_provider}
    rows = _fallback_rows(tmp_path)
    assert len(rows) == len(dialed)  # one per step moved to the local tier
    assert json.loads(rows[0].payload_json)["refused_by"] == "egress_gate"
    assert rows[0].tier == "tier_2"
