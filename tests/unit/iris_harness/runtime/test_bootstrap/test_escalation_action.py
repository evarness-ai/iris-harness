"""Live escalation action loop (ADR-0068 L3).

Drives ``EscalationActions.maybe_escalate`` / ``_resolve_escalation_target`` with a
lightweight stand-in for the runtime host so the action path is testable without a full
runtime — local-only egress guard, governor veto, bounded re-execution, and the
escalation outcome signals.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from iris_harness.agent.agent_executor import AgentResult
from iris_harness.agent.escalation import EscalationConfig
from iris_harness.agent.response_curator import CuratedResponse
from iris_harness.agent.task_planner import TaskPlan
from iris_harness.runtime.escalation_actions import EscalationActions


def _result(name: str, **metadata: Any) -> AgentResult:
    meta: dict[str, object] = {"effects_executed": []}
    meta.update(metadata)
    return AgentResult(agent_type="general", output=name, success=True, metadata=meta)


# A turn that reports it changed nothing — the only kind escalation may act on.
_R1 = _result("r1")
_R2 = _result("r2")


class _Tier:
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model


class _TierRouter:
    def __init__(self, tiers: dict[str, _Tier], *, governor: Any = None) -> None:
        self._tiers = tiers
        self.governor = governor

    def get_tier_by_name(self, name: str) -> _Tier | None:
        return self._tiers.get(name)

    def model_for_intent(self, intent: str) -> str:
        return "m-t1"

    def tier_name_for_model(self, model: str) -> str | None:
        for n, t in self._tiers.items():
            if t.model == model:
                return n
        return None


class _RecordingCollector:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_metric(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _Curator:
    def __init__(self, cfg: EscalationConfig, next_curated: CuratedResponse) -> None:
        self._cfg = cfg
        self._next = next_curated
        self.curate_calls = 0

    @property
    def escalation_config(self) -> EscalationConfig:
        return self._cfg

    def curate(self, results: Any, **_kw: Any) -> CuratedResponse:
        self.curate_calls += 1
        return self._next


def _curated(
    action: str | None,
    *,
    tokens: int = 10,
    has_errors: bool = False,
    confidence: float = 0.9,
    diagnosis: str = "capability_gap",
    question: str | None = None,
    tool_hint: str | None = None,
    target_tier: str = "tier2",
) -> CuratedResponse:
    signals = []
    if action is not None:
        verdict: dict[str, Any] = {
            "action": action,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "target_tier": target_tier,
        }
        if question is not None:
            verdict["question"] = question
        if tool_hint is not None:
            verdict["tool_hint"] = tool_hint
        signals.append({"name": "escalation", "metadata": {"escalation": verdict}})
    return CuratedResponse(
        text="answer",
        has_errors=has_errors,
        metadata={"total_tokens": tokens, "judge_bundle": {"signals": signals}},
    )


_LOCAL_TIERS = {"tier1": _Tier("ollama", "m-t1"), "tier2": _Tier("ollama", "m-t2")}


def _runtime(
    cfg: EscalationConfig,
    *,
    tiers: dict[str, _Tier],
    governor: Any = None,
    next_curated: CuratedResponse | None = None,
    execute_result: Any = None,
) -> Any:
    collector = _RecordingCollector()
    tr = _TierRouter(tiers, governor=governor)
    ns = SimpleNamespace(
        tier_router=tr,
        response_curator=_Curator(cfg, next_curated or _curated("accept")),
        signal_collector=collector,
        _execute_plan=lambda *a, **k: (execute_result if execute_result is not None else [_R2]),
    )
    return ns


# --- _resolve_escalation_target ---


def test_resolve_picks_local_tier() -> None:
    ns = _runtime(EscalationConfig(enabled=True, mode="enforce"), tiers=_LOCAL_TIERS)
    target = EscalationActions(ns)._resolve_escalation_target(
        {"target_tier": "tier2"}, ns.response_curator.escalation_config, current_model="m-t1"
    )
    assert target == ("tier2", "m-t2")


def test_resolve_blocks_cloud_tier_egress_invariant() -> None:
    tiers = {"tier1": _Tier("ollama", "m-t1"), "tier3": _Tier("openrouter", "claude-x")}
    ns = _runtime(EscalationConfig(enabled=True, mode="enforce"), tiers=tiers)
    target = EscalationActions(ns)._resolve_escalation_target(
        {"target_tier": "tier3"}, ns.response_curator.escalation_config, current_model="m-t1"
    )
    assert target is None  # never escalate personal content to cloud (v1 local-only)


def test_resolve_skips_same_tier() -> None:
    ns = _runtime(EscalationConfig(enabled=True, mode="enforce"), tiers=_LOCAL_TIERS)
    target = EscalationActions(ns)._resolve_escalation_target(
        {"target_tier": "tier2"}, ns.response_curator.escalation_config, current_model="m-t2"
    )
    assert target is None  # already on tier2


_CLOUD_TIERS = {
    "tier1": _Tier("ollama", "m-t1"),
    "tier3": _Tier("openrouter", "claude-x"),
}


def test_resolve_allows_cloud_when_enabled_and_eligible() -> None:
    cfg = EscalationConfig(enabled=True, mode="enforce", allow_cloud=True)
    ns = _runtime(cfg, tiers=_CLOUD_TIERS)
    target = EscalationActions(ns)._resolve_escalation_target(
        {"target_tier": "tier3"}, cfg, current_model="m-t1", egress_eligible=True
    )
    assert target == ("tier3", "claude-x")


def test_resolve_blocks_cloud_when_not_eligible() -> None:
    cfg = EscalationConfig(enabled=True, mode="enforce", allow_cloud=True)
    ns = _runtime(cfg, tiers=_CLOUD_TIERS)
    target = EscalationActions(ns)._resolve_escalation_target(
        {"target_tier": "tier3"}, cfg, current_model="m-t1", egress_eligible=False
    )
    assert target is None  # personal/secret content stays on-box


def test_resolve_blocks_cloud_when_allow_cloud_off() -> None:
    cfg = EscalationConfig(enabled=True, mode="enforce", allow_cloud=False)
    ns = _runtime(cfg, tiers=_CLOUD_TIERS)
    target = EscalationActions(ns)._resolve_escalation_target(
        {"target_tier": "tier3"}, cfg, current_model="m-t1", egress_eligible=True
    )
    assert target is None  # cloud escalation disabled by default


def test_egress_ok_false_when_cloud_disabled() -> None:
    from iris_harness.agent.response_curator import CuratedResponse

    cfg = EscalationConfig(enabled=True, mode="enforce", allow_cloud=False)
    ns = SimpleNamespace()
    ok = EscalationActions(ns)._escalation_egress_ok(
        cfg, message="anything", curated=CuratedResponse(text="x"), memory_ctx=None
    )
    assert ok is False  # never classify when cloud is off


def test_egress_ok_true_for_benign_content() -> None:
    from iris_harness.agent.response_curator import CuratedResponse

    cfg = EscalationConfig(enabled=True, mode="enforce", allow_cloud=True)
    ns = SimpleNamespace()
    ok = EscalationActions(ns)._escalation_egress_ok(
        cfg,
        message="what time is it in tokyo",
        curated=CuratedResponse(text="It is 9am in Tokyo."),
        memory_ctx=None,
    )
    assert ok is True  # benign public content is cloud-eligible


def test_resolve_respects_governor_veto() -> None:
    governor = SimpleNamespace(recommend_tier_name=lambda name: "tier1")  # downshift veto
    ns = _runtime(
        EscalationConfig(enabled=True, mode="enforce"), tiers=_LOCAL_TIERS, governor=governor
    )
    target = EscalationActions(ns)._resolve_escalation_target(
        {"target_tier": "tier2"}, ns.response_curator.escalation_config, current_model="m-t1"
    )
    assert target is None


# --- _maybe_escalate ---


_PLAN = TaskPlan(query="hard question", tasks=[])
_INTENT = SimpleNamespace(intent="general", is_multi_step=False, agent_type="general")


def _call_maybe_escalate(
    ns: Any, curated: CuratedResponse, *, results: list[AgentResult] | None = None
) -> Any:
    return EscalationActions(ns).maybe_escalate(
        results=results if results is not None else [_R1],
        curated=curated,
        plan=_PLAN,
        intent_result=_INTENT,
        memory_ctx=object(),
        session_id="s1",
        message="hard question",
        preferred_model=None,
        provider_profile=None,
        strict=False,
        history_text=(),
    )


def test_shadow_mode_does_not_act() -> None:
    ns = _runtime(EscalationConfig(enabled=True, mode="shadow"), tiers=_LOCAL_TIERS)
    results, curated, to, _clarify = _call_maybe_escalate(ns, _curated("escalate"))
    assert to is None
    assert results == [_R1]  # untouched
    assert ns.response_curator.curate_calls == 0
    assert ns.signal_collector.calls == []


def test_enforce_escalates_and_records() -> None:
    ns = _runtime(
        EscalationConfig(enabled=True, mode="enforce", max_escalations=1),
        tiers=_LOCAL_TIERS,
        next_curated=_curated("accept"),  # escalated answer is accepted
    )
    results, curated, to, _clarify = _call_maybe_escalate(ns, _curated("escalate"))
    assert to == "tier2"
    assert results == [_R2]  # re-executed at the escalated tier
    metrics = {c["metric_name"] for c in ns.signal_collector.calls}
    assert "escalation_decision" in metrics
    assert "escalation_cost" in metrics
    decision = next(
        c for c in ns.signal_collector.calls if c["metric_name"] == "escalation_decision"
    )
    assert decision["metadata"]["from_tier"] == "tier1"
    assert decision["metadata"]["to_tier"] == "tier2"


def test_accept_verdict_does_not_escalate() -> None:
    ns = _runtime(EscalationConfig(enabled=True, mode="enforce"), tiers=_LOCAL_TIERS)
    results, curated, to, _clarify = _call_maybe_escalate(ns, _curated("accept"))
    assert to is None
    assert ns.response_curator.curate_calls == 0


def test_bounded_by_max_escalations() -> None:
    # Even if every re-curation still says "escalate", we stop at max_escalations.
    ns = _runtime(
        EscalationConfig(enabled=True, mode="enforce", max_escalations=1),
        tiers=_LOCAL_TIERS,
        next_curated=_curated("escalate"),  # keeps asking to escalate
    )
    _results, _curated_out, to, _clarify = _call_maybe_escalate(ns, _curated("escalate"))
    assert to == "tier2"
    assert ns.response_curator.curate_calls == 1  # exactly one hop, not a loop


# --- reroute (grounding_gap) ---


def test_reroute_reruns_same_tier_and_records() -> None:
    ns = _runtime(
        EscalationConfig(enabled=True, mode="enforce", max_escalations=1),
        tiers=_LOCAL_TIERS,
        next_curated=_curated("accept"),
    )
    results, _curated_out, to, _clarify = _call_maybe_escalate(
        ns, _curated("reroute", diagnosis="grounding_gap", tool_hint="research")
    )
    assert results == [_R2]  # re-executed
    assert to is None  # tier unchanged (reroute is same-tier)
    assert ns.response_curator.curate_calls == 1
    decision = next(
        c for c in ns.signal_collector.calls if c["metric_name"] == "escalation_decision"
    )
    assert decision["metadata"]["action"] == "reroute"


# --- clarify (ambiguity) ---


def test_clarify_returns_question_without_rerunning() -> None:
    ns = _runtime(EscalationConfig(enabled=True, mode="enforce"), tiers=_LOCAL_TIERS)
    _results, _curated_out, to, clarify = _call_maybe_escalate(
        ns,
        _curated(
            "clarify",
            diagnosis="ambiguity",
            confidence=0.9,
            question="Which database did you mean?",
        ),
    )
    assert clarify == "Which database did you mean?"
    assert to is None
    assert ns.response_curator.curate_calls == 0  # clarify never re-executes
    decision = next(
        c for c in ns.signal_collector.calls if c["metric_name"] == "escalation_decision"
    )
    assert decision["metadata"]["action"] == "clarify"


def test_clarify_below_confidence_floor_does_not_ask() -> None:
    ns = _runtime(
        EscalationConfig(enabled=True, mode="enforce", clarify_confidence_floor=0.7),
        tiers=_LOCAL_TIERS,
    )
    _results, _curated_out, to, clarify = _call_maybe_escalate(
        ns,
        _curated("clarify", diagnosis="ambiguity", confidence=0.5, question="Which one?"),
    )
    assert clarify is None  # below floor → don't over-ask
    assert ns.signal_collector.calls == []


# --- action confidence floor (escalate / reroute) ---


def test_escalate_below_action_floor_keeps_the_answer() -> None:
    ns = _runtime(
        EscalationConfig(enabled=True, mode="enforce", action_confidence_floor=0.7),
        tiers=_LOCAL_TIERS,
    )
    results, _curated_out, to, clarify = _call_maybe_escalate(
        ns, _curated("escalate", diagnosis="capability_gap", confidence=0.0)
    )
    assert results == [_R1]  # not re-executed
    assert to is None
    assert clarify is None
    assert ns.response_curator.curate_calls == 0
    assert ns.signal_collector.calls == []


def test_reroute_below_action_floor_keeps_the_answer() -> None:
    ns = _runtime(
        EscalationConfig(enabled=True, mode="enforce", action_confidence_floor=0.7),
        tiers=_LOCAL_TIERS,
    )
    results, _curated_out, _to, _clarify = _call_maybe_escalate(
        ns,
        _curated("reroute", diagnosis="grounding_gap", confidence=0.5, tool_hint="research"),
    )
    assert results == [_R1]
    assert ns.response_curator.curate_calls == 0


def test_escalate_at_action_floor_acts() -> None:
    ns = _runtime(
        EscalationConfig(enabled=True, mode="enforce", action_confidence_floor=0.7),
        tiers=_LOCAL_TIERS,
        next_curated=_curated("accept"),
    )
    results, _curated_out, to, _clarify = _call_maybe_escalate(
        ns, _curated("escalate", confidence=0.7)
    )
    assert results == [_R2]
    assert to == "tier2"


# --- a turn that changed something is never re-run or replaced (ADR-0118 decision 5) ---


def _enforce(**kw: Any) -> Any:
    return _runtime(
        EscalationConfig(enabled=True, mode="enforce", max_escalations=2, **kw),
        tiers=_LOCAL_TIERS,
        next_curated=_curated("accept"),
    )


def test_escalate_after_a_write_keeps_the_answer() -> None:
    ns = _enforce()
    wrote = _result("r1", effects_executed=["write"])
    results, _curated_out, to, _clarify = _call_maybe_escalate(
        ns, _curated("escalate"), results=[wrote]
    )
    assert results == [wrote]  # not re-run: the reminder is not created twice
    assert to is None
    assert ns.response_curator.curate_calls == 0
    assert ns.signal_collector.calls == []


def test_clarify_after_a_write_does_not_hide_it() -> None:
    ns = _enforce()
    _results, _curated_out, _to, clarify = _call_maybe_escalate(
        ns,
        _curated("clarify", diagnosis="ambiguity", question="Which one?"),
        results=[_result("r1", effects_executed=["destructive"])],
    )
    assert clarify is None  # the answer saying what was done stays


def test_a_result_that_does_not_report_its_effects_is_not_re_run() -> None:
    # Fail-closed: a plugin handler can write outside the loop, and a loop that
    # wrote and then fell back to a deterministic answer loses its report.
    ns = _enforce()
    silent = AgentResult(agent_type="finance", output="r1", success=True, metadata={})
    results, _curated_out, _to, _clarify = _call_maybe_escalate(
        ns, _curated("escalate"), results=[silent]
    )
    assert results == [silent]
    assert ns.response_curator.curate_calls == 0


def test_a_resumed_run_is_never_escalated() -> None:
    ns = _enforce()
    resumed = _result("r1", resumed=True)
    results, _curated_out, _to, _clarify = _call_maybe_escalate(
        ns, _curated("escalate"), results=[resumed]
    )
    assert results == [resumed]
    assert ns.response_curator.curate_calls == 0


def test_one_writing_result_in_a_multi_task_turn_holds_the_whole_turn() -> None:
    ns = _enforce()
    wrote = _result("r1b", effects_executed=["write"])
    results, _curated_out, _to, _clarify = _call_maybe_escalate(
        ns, _curated("escalate"), results=[_R1, wrote]
    )
    assert results == [_R1, wrote]
    assert ns.response_curator.curate_calls == 0


def _two_hop(execute_result: list[AgentResult]) -> Any:
    return _runtime(
        EscalationConfig(enabled=True, mode="enforce", max_escalations=2),
        tiers={**_LOCAL_TIERS, "tier3": _Tier("ollama", "m-t3")},
        # After hop 1 the answer is on tier2 and the judge still wants more: tier3.
        next_curated=_curated("escalate", target_tier="tier3"),
        execute_result=execute_result,
    )


def test_a_clean_re_run_may_take_the_second_hop() -> None:
    ns = _two_hop([_R2])
    _results, _curated_out, to, _clarify = _call_maybe_escalate(ns, _curated("escalate"))
    assert to == "tier3"
    assert ns.response_curator.curate_calls == 2


def test_a_re_run_that_wrote_stops_the_next_hop() -> None:
    wrote = _result("r2", effects_executed=["write"])
    ns = _two_hop([wrote])
    results, _curated_out, to, _clarify = _call_maybe_escalate(ns, _curated("escalate"))
    assert results == [wrote]
    assert to == "tier2"  # the first hop happened; the second did not
    assert ns.response_curator.curate_calls == 1


def test_a_turn_waiting_on_an_approval_is_never_re_run() -> None:
    # Re-running it would raise a second approval for the same request (ADR-0118).
    ns = _enforce()
    waiting = _result("r1", pending_approval_id="a-1")
    results, _curated_out, _to, clarify = _call_maybe_escalate(
        ns, _curated("escalate"), results=[waiting]
    )
    assert results == [waiting]
    assert clarify is None
    assert ns.response_curator.curate_calls == 0
