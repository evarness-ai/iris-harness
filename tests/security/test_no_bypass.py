"""Mandatory-passage CI gate for governance-sensitive call sites.

The unified governance design requires every LLM call and tool invocation
to pass through ``GovernanceKernel`` hooks. This suite is strict: every
known direct call site must be in the governed baseline and newly-added
call sites fail CI until reviewed and explicitly categorized.

**Two things this gate used to claim and not deliver**, found by mutating the source
and checking whether the assertions noticed (2026-09-12, after the same exercise on
``test_governance_isolation.py::test_invariant_2`` found the same class of gap):

1. *It could not count.* Call sites were collected into a **set** keyed by
   ``file:scope:expression``, so a second ``self._llm(...)`` added inside an
   already-governed method produced a label that was already present and changed
   nothing. An ungoverned LLM call could be added to the middle of ``_loop`` and the
   gate stayed green. Sites are counted now, and a count that changes fails until
   reviewed — which is what "newly-added call sites fail CI" was always supposed to
   mean.

2. *Line order is not control flow.* "The hook is governed" was
   ``any(hook_line < llm_line)``, which a hook inside a branch the call is not in
   satisfies just as well as a hook that actually runs first.
   :func:`_hook_dominates` now requires the hook to be a statement of a block that
   encloses the call, so a hook the call can skip past no longer counts.

What it still cannot see, stated plainly rather than implied: it reasons about one
function's syntax. A hook called through a helper, behind a flag read at runtime, or in
a sibling method is invisible to it — those paths are covered by the runtime tests in
``tests/unit/test_governance/``, not here.
"""

from __future__ import annotations

import ast
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Every governed root: the harness, and — when present in the private tree — the domains
# (src/iris_personal since M6.1b) and the coding agent (src/iris_code since M6.1a). Their
# LLM calls are governed too, and a root left out of this list is a scan that silently
# covers less, which is the failure mode M6.1a's skip-count rule was written for.
SRC_ROOTS = [
    root
    for root in (
        PROJECT_ROOT / "src" / "iris_harness",
        PROJECT_ROOT / "src" / "iris_personal",
        PROJECT_ROOT / "src" / "iris_code",
    )
    if root.is_dir()
]


def _module_source(dotted: str) -> Path:
    """Where a module's source file is — asked of the module, not spelled out here.

    A literal path is a second, silent copy of the package layout, and it goes stale
    the moment a package moves: three M6.2 layers in a row failed this file with
    ``FileNotFoundError`` on a path that had simply been renamed, which reads like a
    governance regression and is not one. ``find_spec`` resolves the origin without
    executing the module, so the constant follows the tree.
    """
    spec = importlib.util.find_spec(dotted)
    if spec is None or spec.origin is None:  # pragma: no cover - a typo in the name
        raise AssertionError(f"cannot locate the source of {dotted}")
    return Path(spec.origin)


AGENTIC_CORE_PATH = _module_source("iris_harness.agent.agentic_core")
TOOL_RUNNER_PATH = _module_source("iris_harness.agent.tool_runner")
LLM_CLIENT_PATH = _module_source("iris_harness.llm.client")
INTENT_ROUTER_PATH = _module_source("iris_harness.agent.intent_router")
TASK_PLANNER_PATH = _module_source("iris_harness.agent.task_planner")
ENTITY_EXTRACTOR_PATH = _module_source("iris_harness.memory.knowledge.entity_extractor")
COMPACTOR_PATH = PROJECT_ROOT / "src" / "iris_harness" / "memory" / "compactor.py"

GOVERNED_DIRECT_LLM_CALLS = frozenset(
    {
        "src/iris_code/agent.py:CodingAgent._run_multi_turn_developer_loop:"
        "self.llm_client.invoke_turn",
        "src/iris_code/agent.py:CodingAgent._run_single_turn_developer:self.llm_client.invoke",
        "src/iris_code/agent.py:CodingAgent.run_architect_review:self.llm_client.invoke",
        "src/iris_code/execution_engine.py:PipelineExecutionEngine._invoke_llm:"
        "self.coding_agent.llm_client.invoke",
        "src/iris_code/persona_invoke.py:_default_llm_executor._invoke:client.invoke",
        "src/iris_harness/agent/agentic_core.py:AgenticCore._loop:self._llm",
        "src/iris_harness/agent/agentic_core.py:AgenticCore.run_stream:self._llm",
        # De-dup forced-synthesis pass; fires _governance_pre_llm before self._llm.
        "src/iris_harness/agent/agentic_core.py:AgenticCore._force_final_synthesis:self._llm",
        "src/iris_harness/agent/intent_router.py:LLMClassifier._invoke_with_governance:self._llm",
        "src/iris_harness/agent/task_planner.py:TaskPlanner._invoke_with_governance:self._llm",
        "src/iris_harness/memory/knowledge/entity_extractor.py:EntityExtractor._invoke_with_governance:self._llm",
        "src/iris_harness/memory/compactor.py:ConversationCompactor._invoke_with_governance:self._llm",
        "src/iris_harness/memory/fact_extractor.py:extract_facts_with_llm:client.invoke",
        # The governed prompt call the four wrappers above are handed in production (the
        # rolling session summary's, via bootstrap._tier_llm): CodingLLMClient built
        # without governance_handled_upstream, so invoke -> invoke_turn fires
        # _governance_pre_llm at the tier the call goes to. The wrappers fire nothing of
        # their own around it -- they did, under a hardcoded tier_1, twice per call.
        "src/iris_harness/llm/client.py:GovernedPromptCall.__call__:client.invoke",
        # Governed internally: CodingLLMClient.invoke -> invoke_turn fires
        # _governance_pre_llm (kernel defaults to kernel_from_env()).
        # Moved from bootstrap.py to the routine-authoring mixin (Phase 2 slice 5), then
        # the mixin became the RoutineAuthoring collaborator (OSS plan M5.7 track C
        # slice 10); same governed CodingLLMClient.invoke path, just relocated.
        "src/iris_harness/runtime/routine_authoring.py:"
        "RoutineAuthoring.routine_authoring_llm_caller.call:client.invoke",
        "src/iris_harness/runtime/judges.py:build_curator_faithfulness_client:client.invoke",
        # exp-007 leak-judge: same governed path as the faithfulness judge
        # (CodingLLMClient.invoke -> invoke_turn fires _governance_pre_llm).
        "src/iris_harness/runtime/judges.py:build_curator_leak_client:client.invoke",
        # Phase 5 grounding judge: same governed path as the faithfulness judge
        # (CodingLLMClient.invoke -> invoke_turn fires _governance_pre_llm).
        "src/iris_harness/runtime/judges.py:build_curator_grounding_client:client.invoke",
        # ADR-0068 escalation judge (shadow, L2): same governed path as the other
        # curator judges (CodingLLMClient.invoke -> invoke_turn fires governance).
        "src/iris_harness/runtime/judges.py:build_curator_escalation_client:client.invoke",
        # §9.2 governance judge (post-run safety review): governed CodingLLMClient.invoke
        # (governance_agent_type="chat"; no governance_handled_upstream, so PRE_LLM fires
        # on the trace). Built on the SAME routing intent as the run it reviews, so a
        # Mac-only run is judged Mac-only; opt-in via IRIS_GOVERNANCE_JUDGE_ENABLED.
        "src/iris_harness/runtime/governance_judge.py:"
        "build_governance_judge.invoke_for._invoke:client.invoke",
        # §4.2 user-correction outcome judge: governed CodingLLMClient.invoke
        # (governance_agent_type="chat"), opt-in + pre-filtered, off by default.
        "src/iris_harness/runtime/turn_capture.py:_build_correction_detector.detect:client.invoke",
        # Crystallizer skill synthesizer: governed CodingLLMClient.invoke
        # (governance_agent_type="chat"), opt-in via IRIS_SKILL_SYNTHESIS, off by default.
        "src/iris_harness/runtime/learning_builders.py:_build_skill_synthesizer:client.invoke",
        # Hybrid file-op parse fallback (move/delete by type): governed
        # CodingLLMClient.invoke (fires _governance_pre_llm via invoke_turn) on a LOCAL
        # tier, gated by IRIS_FILEOP_LLM_PARSE. Extracts structured params only; the
        # grounded file action stays deterministic (propose -> approve -> governed move).
        # Moved out of IrisRuntime with the filemanager plugin (OSS plan M2.4); same
        # call, same governance path, new home.
        "src/iris_personal/plugins/file_organizer/handlers.py:FileManagerHandlers._llm_parse_file_op:client.invoke",
        # ADR-0069 #4 learning analyst (slice 2): governed CodingLLMClient.invoke
        # (governance_agent_type="chat") on a LOCAL tier, opt-in via
        # IRIS_LEARNING_ANALYST, off by default. Interprets internal telemetry only.
        # This and the miners' builder below moved out of bootstrap with the learning
        # controls (OSS plan M5.7 track C slice 9); same call, same governance path.
        "src/iris_harness/runtime/learning_controls.py:_build_learning_analyst.invoke:client.invoke",
        # Digital-twin miners (behavior miner L1 + intention rollup L3): both build their
        # governed CodingLLMClient.invoke (governance_agent_type="chat") on a LOCAL tier via
        # the shared _build_learning_invoke (ADR-0080). Opt-in via IRIS_BEHAVIOR_MINER /
        # IRIS_INTENTION_ROLLUP, off by default; personal transcript/activity stays on-box;
        # both are propose-only. Preview (preview_*) reuses the same governed client.
        "src/iris_harness/runtime/learning_controls.py:_build_learning_invoke.invoke:client.invoke",
        "src/iris_harness/runtime/client_config.py:_llm_router_from_config.invoke:client.invoke",
        "src/iris_harness/plugins_builtin/code_exec/handler.py:"
        "_make_code_exec_handler._run_loop_gen:client.invoke_stream",
        "src/iris_harness/runtime/handlers/general_invoke.py:"
        "make_general_invoke._invoke_with_tools_native:client.invoke_turn",
        "src/iris_harness/runtime/handlers/general_invoke.py:"
        "make_general_invoke._invoke_with_tools_react:client.invoke_turn",
        "src/iris_harness/runtime/handlers/general.py:_make_general_handler.stream_handler:client.invoke_stream",
        "src/iris_harness/runtime/handlers/react.py:_make_react_handler._llm_call:client.invoke",
        # (The email agent's own ReAct loop, `make_email_react_handler._llm`, is gone:
        # email turns run on _make_react_handler's loop above, which the plugin claims
        # with api.register_loop_intent -- core/SDK boundary plan, email slice step 5.)
        # Narrative layer for the deterministic-first agents (email digest, daily
        # plan). CodingLLMClient built without governance_handled_upstream, so
        # invoke -> invoke_turn fires _governance_pre_llm per call. Summarises only
        # the already-assembled grounded digest/plan (no new facts).
        "src/iris_harness/llm/narrate.py:make_narrative_llm_call._call:client.invoke",
        # Finance F2 statement extraction. Tier-3 LOCAL only (ADR-0002/0033):
        # CodingLLMClient is built without governance_handled_upstream, so its
        # invoke -> invoke_turn fires _governance_pre_llm per call. Statement
        # content never reaches a cloud model.
        "src/iris_personal/plugins/finance_workflows/extract.py:Tier3LocalExtractor.extract:client.invoke",
        # RAG R3 document Q&A synthesis (ADR-0049). Same pattern: CodingLLMClient
        # built without governance_handled_upstream, so invoke fires governance
        # per call. Summarises only retrieved local-document passages; the
        # answer is grounded in the user's own files (no cloud, no outside data).
        "src/iris_harness/services/rag/qa.py:default_llm_call._call:client.invoke",
        # The email judge (issue #31, owner decision B): one governed
        # CodingLLMClient.invoke_json per email (governance_agent_type="chat", no
        # governance_handled_upstream), so each attempt's invoke_turn fires
        # _governance_pre_llm. It replaced a raw POST to Ollama that fired no hooks.
        "src/iris_personal/plugins/email_workflows/judge.py:make_tier_llm.call:llm_client.invoke_json",
    }
)

KNOWN_DIRECT_LLM_CALLS = GOVERNED_DIRECT_LLM_CALLS


# Labels whose method legitimately holds more than one direct call site. Everything
# absent from here is expected exactly once, so adding a second call to a governed method
# fails until someone writes down why — which is the whole point of a reviewed baseline,
# and was silently impossible while the sites lived in a set.
EXPECTED_CALL_COUNTS: dict[str, int] = {
    # Each of the four `_invoke_with_governance` wrappers calls `self._llm` twice: the
    # governed path, and the direct one for when the kernel is absent or the callable is a
    # GovernedPromptCall (whose client fires the hooks at the tier it goes to).
    "src/iris_harness/agent/intent_router.py:LLMClassifier._invoke_with_governance:self._llm": 2,
    "src/iris_harness/agent/task_planner.py:TaskPlanner._invoke_with_governance:self._llm": 2,
    "src/iris_harness/memory/knowledge/entity_extractor.py:"
    "EntityExtractor._invoke_with_governance:self._llm": 2,
    "src/iris_harness/memory/compactor.py:"
    "ConversationCompactor._invoke_with_governance:self._llm": 2,
}


# The governed handles a code caller reaches tools through (``BoundTools``, a plugin's
# ``api.tools``): ``<handle>.call(name, args)`` runs the tool through the runner. Any other
# ``<x>.call(...)`` is a ToolSpec's own function called directly, whatever the variable is
# named -- the email fallback's ``read_email.call`` was one, and keying on the literal
# ``tool.call`` did not see it (email slice step 5 review).
_GOVERNED_TOOL_HANDLES = frozenset({"tools", "self._tools", "api.tools"})


def _is_direct_tool_call(func: ast.AST, call_expression: str) -> bool:
    if not (isinstance(func, ast.Attribute) and func.attr == "call"):
        return False
    return call_expression.removesuffix(".call") not in _GOVERNED_TOOL_HANDLES


class _DirectCallVisitor(ast.NodeVisitor):
    def __init__(self, *, source: str, relative_path: str) -> None:
        self.source = source
        self.relative_path = relative_path
        self.scope: list[str] = []
        # Counters, not sets: a second ungoverned call inside an already-listed method
        # produces a label that is already present, and a set cannot tell the difference.
        self.llm_calls: Counter[str] = Counter()
        self.tool_calls: Counter[str] = Counter()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        call_expression = _call_expression(self.source, node.func)
        if _is_direct_llm_call(self.source, node.func):
            self.llm_calls[self._label(call_expression)] += 1
        if _is_direct_tool_call(node.func, call_expression):
            self.tool_calls[self._label(call_expression)] += 1
        self.generic_visit(node)

    def _label(self, call_expression: str) -> str:
        scope = ".".join(self.scope) if self.scope else "<module>"
        return f"{self.relative_path}:{scope}:{call_expression}"


def test_both_react_loops_fire_the_post_step_evaluator() -> None:
    """ADR-0107. `run_stream` shipped without PostStep while `_loop` had it, so the
    governance posture differed by which loop answered — and `run_stream` answers the
    first task of every turn. Pinned structurally so a third loop, or a refactor that
    drops the call, cannot land silently."""
    tree, source = _parse(AGENTIC_CORE_PATH)

    for method_name in ("_loop", "run_stream"):
        method = _method(tree, "AgenticCore", method_name)
        # Presence only, deliberately: PostStep fires *after* the step it judges, so
        # there is no "before the call" ordering to check. What a missing call looks
        # like is a loop with no halt point at all, which is what this catches.
        assert _call_lines(
            source, method, "self._fire_post_step"
        ), f"{method_name} must fire the PostStep evaluator (ADR-0107)"


def test_agentic_core_llm_calls_go_through_pre_llm_hook() -> None:
    tree, source = _parse(AGENTIC_CORE_PATH)

    # _loop is the shared ReAct body shared by run() and resume_from_checkpoint();
    # run_stream still has its own loop.
    for method_name in ("_loop", "run_stream"):
        method = _method(tree, "AgenticCore", method_name)
        llm_lines = _call_lines(source, method, "self._llm")
        hook_lines = _call_lines(source, method, "self._governance_pre_llm")

        assert hook_lines, f"expected a governance pre-LLM hook in {method_name}"
        # Exactly one. Each loop drives the model from a single place, and pinning the
        # count is what makes a *second*, ungoverned call fail — the dominance check
        # below cannot: an earlier hook satisfies it for any number of later calls.
        assert len(llm_lines) == 1, (
            f"{method_name} has {len(llm_lines)} direct self._llm call sites, expected 1. "
            f"A new one must be routed through _governance_pre_llm and this count updated "
            f"deliberately: {llm_lines}"
        )
        assert _hook_dominates(
            method,
            hook_expression="self._governance_pre_llm",
            call_expression="self._llm",
            source=source,
        ), (
            f"self._llm in {method_name} must be preceded by a _governance_pre_llm that "
            f"actually runs — not one sitting in a branch the call skips"
        )


def test_every_tool_call_goes_through_the_governed_runner() -> None:
    """One governed tool path, whoever calls (docs/architecture/plugin-capabilities.md).

    The runner's only ``tool.call`` is preceded by its ``PRE_TOOL_USE`` step on every path
    that reaches it; that step really fires the kernel at ``PRE_TOOL_USE``; and the loop
    reaches tools only through the runner — it makes no ``tool.call`` of its own.
    """
    tree, source = _parse(TOOL_RUNNER_PATH)
    execute = _method(tree, "GovernedToolRunner", "execute")
    assert len(_call_lines(source, execute, "tool.call")) == 1
    assert _call_lines(source, execute, "self.pre"), "expected the PRE_TOOL_USE step"
    assert _hook_dominates(
        execute, hook_expression="self.pre", call_expression="tool.call", source=source
    ), "tool.call must be preceded by a self.pre that actually runs"

    pre = _method(tree, "GovernedToolRunner", "pre")
    fires = [
        call
        for call in ast.walk(pre)
        if isinstance(call, ast.Call)
        and _call_expression(source, call.func) == "self._kernel.fire_sync"
        and call.args
        and _call_expression(source, call.args[0]) == "HookPoint.PRE_TOOL_USE"
    ]
    assert fires, "the runner's pre step must fire the kernel at HookPoint.PRE_TOOL_USE"

    core_tree, core_source = _parse(AGENTIC_CORE_PATH)
    loop = _method(core_tree, "AgenticCore", "_execute_tool")
    assert _call_lines(core_source, loop, "tool.call") == [], "the loop must not call a tool"
    assert _call_lines(
        core_source, loop, "self._tool_runner().execute"
    ), "the loop must reach tools through the governed runner"


PLUGIN_REGISTRY_PATH = _module_source("iris_harness.runtime.plugin_host.registry")


def _function(tree: ast.Module, class_name: str, name: str) -> ast.AST:
    """A method, sync or async (``_method`` only finds sync ones)."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    if member.name == name:
                        return member
    raise AssertionError(f"method not found: {class_name}.{name}")


def _fires(source: str, node: ast.AST, expression: str, point: str) -> bool:
    return any(
        isinstance(call, ast.Call)
        and _call_expression(source, call.func) == expression
        and call.args
        and _call_expression(source, call.args[0]) == point
        for call in ast.walk(node)
    )


def test_every_capability_call_goes_through_the_runner_and_the_kernel() -> None:
    """Facade -> runner -> kernel (docs/architecture/plugin-capabilities.md §4).

    A consumer's facade reaches a provider only through ``_ProviderGuard.invoke``, which
    hands the guarded provider method to the governed runner and never calls it itself.
    Each runner entry point calls the provider only after its ``PRE_TOOL_USE`` step, that
    step really fires the kernel at ``PRE_TOOL_USE`` (sync and async), and the ``POST``
    steps fire ``POST_TOOL_USE`` and return what the kernel's final context says.
    """
    registry_tree, registry_source = _parse(PLUGIN_REGISTRY_PATH)
    invoke = _function(registry_tree, "_ProviderGuard", "invoke")
    assert (
        _call_lines(registry_source, invoke, "provider_call") == []
    ), "the guard must not call the provider itself"
    reached = {
        entry
        for entry in ("runner.execute_call", "runner.aexecute_call", "runner.aexecute_stream")
        if _call_lines(registry_source, invoke, entry)
    }
    assert reached == {
        "runner.execute_call",
        "runner.aexecute_call",
        "runner.aexecute_stream",
    }, "every method shape must go through the governed runner"

    tree, source = _parse(TOOL_RUNNER_PATH)
    for entry, pre in (
        ("execute_call", "self.capability_pre"),
        ("aexecute_call", "self.capability_apre"),
        ("aexecute_stream", "self.capability_apre"),
    ):
        method = _function(tree, "GovernedToolRunner", entry)
        assert len(_call_lines(source, method, "provider_call")) == 1, entry
        assert _hook_dominates(
            method, hook_expression=pre, call_expression="provider_call", source=source
        ), f"{entry}: the provider must run only after {pre}"
    assert _fires(
        source,
        _function(tree, "GovernedToolRunner", "capability_pre"),
        "self._kernel.fire_sync",
        "HookPoint.PRE_TOOL_USE",
    )
    assert _fires(
        source,
        _function(tree, "GovernedToolRunner", "capability_apre"),
        "self._kernel.fire",
        "HookPoint.PRE_TOOL_USE",
    )
    for post, fire in (
        ("capability_post", "self._kernel.fire_sync"),
        ("capability_apost", "self._kernel.fire"),
    ):
        method = _function(tree, "GovernedToolRunner", post)
        assert _fires(source, method, fire, "HookPoint.POST_TOOL_USE"), post
        assert _call_lines(
            source, method, "self._redacted"
        ), f"{post} must hand back the kernel's redacted result, not the provider's"


def test_coding_developer_tool_dispatches_go_through_pre_tool_hook() -> None:
    """Story 12.gov-4.1: every developer-loop tool dispatch branch in the
    coding agent must call ``_governance_pre_tool`` before the dispatch.

    The coding agent doesn't use ``tool.call`` — its two governed dispatch
    branches inside ``_execute_developer_tool_calls`` are
    ``self._execute_local_tool_call`` (builtin tools) and
    ``self._execute_extension_tool_call`` (extension-registered tools).
    Each branch must be preceded by the governance hook fire.
    """
    coding_agent_path = PROJECT_ROOT / "src" / "iris_code" / "agent.py"
    if not coding_agent_path.exists():
        pytest.skip("the coding agent is not in this tree (OSS plan decision 2)")
    tree, source = _parse(coding_agent_path)
    method = _method(tree, "CodingAgent", "_execute_developer_tool_calls")
    hook_lines = _call_lines(source, method, "self._governance_pre_tool")
    local_dispatch_lines = _call_lines(source, method, "self._execute_local_tool_call")
    extension_dispatch_lines = _call_lines(source, method, "self._execute_extension_tool_call")

    assert hook_lines, (
        "expected self._governance_pre_tool calls in " "CodingAgent._execute_developer_tool_calls"
    )
    assert local_dispatch_lines, "expected local tool dispatch in the developer loop"
    assert extension_dispatch_lines, "expected extension tool dispatch in the developer loop"

    for dispatch_line in (*local_dispatch_lines, *extension_dispatch_lines):
        assert any(hook_line < dispatch_line for hook_line in hook_lines), (
            f"developer-loop dispatch at line {dispatch_line} must be preceded "
            "by self._governance_pre_tool"
        )


def test_coding_llm_client_turn_and_stream_calls_go_through_pre_llm_hook() -> None:
    tree, source = _parse(LLM_CLIENT_PATH)

    turn = _method(tree, "CodingLLMClient", "invoke_turn")
    turn_llm_lines = _call_lines(source, turn, "model.invoke")
    turn_hook_lines = _call_lines(source, turn, "self._governance_pre_llm")
    assert turn_llm_lines
    assert turn_hook_lines
    assert _hook_dominates(
        turn,
        hook_expression="self._governance_pre_llm",
        call_expression="model.invoke",
        source=source,
    ), "model.invoke in invoke_turn must be preceded by a _governance_pre_llm that runs"

    # The structured call sends nothing itself: each attempt is a governed invoke_turn.
    structured = _method(tree, "CodingLLMClient", "invoke_json")
    assert _call_lines(source, structured, "self.invoke_turn")
    assert not _call_lines(source, structured, "model.invoke")

    stream = _method(tree, "CodingLLMClient", "invoke_stream")
    stream_llm_lines = _call_lines(source, stream, "model.stream")
    stream_hook_lines = _call_lines(source, stream, "self._governance_pre_llm")
    assert stream_llm_lines
    assert stream_hook_lines
    assert _hook_dominates(
        stream,
        hook_expression="self._governance_pre_llm",
        call_expression="model.stream",
        source=source,
    ), "model.stream in invoke_stream must be preceded by a _governance_pre_llm that runs"


def test_module_local_llm_helpers_go_through_governance_wrapper() -> None:
    module_checks: tuple[tuple[Path, str, str], ...] = (
        (INTENT_ROUTER_PATH, "LLMClassifier", "_invoke_with_governance"),
        (TASK_PLANNER_PATH, "TaskPlanner", "_invoke_with_governance"),
        (ENTITY_EXTRACTOR_PATH, "EntityExtractor", "_invoke_with_governance"),
        (COMPACTOR_PATH, "ConversationCompactor", "_invoke_with_governance"),
    )
    for path, class_name, method_name in module_checks:
        tree, source = _parse(path)
        method = _method(tree, class_name, method_name)
        llm_lines = _call_lines(source, method, "self._llm")
        classify_hook_lines = _call_lines(source, method, "self._kernel.fire_sync")
        assert llm_lines, f"expected self._llm call in {class_name}.{method_name}"
        assert (
            classify_hook_lines
        ), f"expected governance fire_sync call in {class_name}.{method_name}"
        assert any(
            any(hook_line < llm_line for hook_line in classify_hook_lines) for llm_line in llm_lines
        ), (f"expected at least one governed self._llm path in " f"{class_name}.{method_name}")


def test_no_new_direct_llm_calls_without_governance_baseline() -> None:
    discovered = _discover_direct_calls().llm_calls

    unexpected = set(discovered) - KNOWN_DIRECT_LLM_CALLS
    # A baseline entry whose file is not in the tree is absent, not stale: the coding
    # agent is not part of the harness release (OSS plan decision 2), and its entries
    # must still be reviewed here, where the files exist.
    removed = {
        entry
        for entry in KNOWN_DIRECT_LLM_CALLS - set(discovered)
        if (PROJECT_ROOT / entry.split(":", 1)[0]).exists()
    }

    assert not unexpected, (
        "new direct LLM call sites must be routed through governance and added to the "
        f"reviewed baseline: {sorted(unexpected)}"
    )
    assert not removed, f"remove stale no-bypass LLM baseline entries: {sorted(removed)}"


def test_no_extra_llm_call_sites_inside_reviewed_methods() -> None:
    """A reviewed method must not grow a second call site unnoticed.

    The gap this closes: the baseline is keyed by ``file:scope:expression``, so adding
    another ``self._llm(...)`` to a method already in it produced a label that was
    already there. With the sites in a set that was invisible — an ungoverned call could
    be dropped into the middle of ``_loop`` and every assertion here still passed.
    """
    discovered = _discover_direct_calls().llm_calls

    surprises = {
        label: (count, EXPECTED_CALL_COUNTS.get(label, 1))
        for label, count in discovered.items()
        if count != EXPECTED_CALL_COUNTS.get(label, 1)
    }

    assert not surprises, (
        "direct LLM call-site counts changed inside reviewed methods "
        "(label: found vs expected): "
        + "; ".join(
            f"{label}: {found} vs {expected}"
            for label, (found, expected) in sorted(surprises.items())
        )
        + ". Route the new call through governance, then record the new count in "
        "EXPECTED_CALL_COUNTS with a reason."
    )


def test_expected_call_counts_has_no_stale_entries() -> None:
    """Every override must still describe a real method, or it is silently permitting
    a count nothing checks."""
    discovered = _discover_direct_calls().llm_calls
    stale = {
        label
        for label in EXPECTED_CALL_COUNTS
        if label not in discovered and (PROJECT_ROOT / label.split(":", 1)[0]).exists()
    }
    assert not stale, f"remove stale EXPECTED_CALL_COUNTS entries: {sorted(stale)}"


def test_no_new_direct_tool_calls_without_pre_tool_use() -> None:
    discovered = _discover_direct_calls().tool_calls

    # Reviewed baseline of deliberate direct ``tool.call`` sites. New entries must be
    # security-reviewed before being added here.
    #   - GovernedToolRunner.execute: THE governed tool path, the loop's and (next) code's.
    # The planner's brief-config intercept used to call configure_brief directly (ADR-0103);
    # it now goes through api.tools, the governed runner (plugin-capabilities step 2), so the
    # runner is the only direct tool.call left in the tree.
    # Counts, not just names: one of these methods growing a second `tool.call` is
    # exactly the change that must not pass unreviewed.
    baseline = {
        "src/iris_harness/agent/tool_runner.py:GovernedToolRunner.execute:tool.call": 1,
        # KNOWN BYPASSES, not reviewed as safe: surfaced when this check stopped keying on
        # the literal `tool.call` (2026-09-30), listed so they stay visible, not so they
        # pass. Each should move to the governed runner (a BoundTools) and leave this list.
        # The general lane's two (research and any plugin tool) left it: they run through
        # GovernedToolRunner now (runtime/handlers/general_tools.py `_run_plugin_tool`).
        "src/iris_personal/plugins/finance_workflows/handlers.py:"
        "FinanceHandlers._handle_statement_email_details_turn:search.call": 1,
    }
    # As for the LLM baseline above: an entry whose file is not in the tree (the public
    # export carries no src/iris_personal) is absent, not stale. Where the file exists
    # its count must match exactly.
    present = {
        label: count
        for label, count in baseline.items()
        if (PROJECT_ROOT / label.split(":", 1)[0]).exists()
    }
    assert dict(discovered) == present


def _discover_direct_calls() -> _DirectCallVisitor:
    aggregate = _DirectCallVisitor(source="", relative_path="")
    for path in sorted(path for root in SRC_ROOTS for path in root.rglob("*.py")):
        tree, source = _parse(path)
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        visitor = _DirectCallVisitor(source=source, relative_path=relative_path)
        visitor.visit(tree)
        # `Counter.update` adds counts; on a set it would have merged names and lost
        # them, which is the defect this whole file was mutated to find.
        aggregate.llm_calls.update(visitor.llm_calls)
        aggregate.tool_calls.update(visitor.tool_calls)
    return aggregate


def _parse(path: Path) -> tuple[ast.Module, str]:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(path)), source


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method_name:
                    return member
    raise AssertionError(f"method not found: {class_name}.{method_name}")


def _statement_paths(root: ast.AST) -> dict[int, tuple[tuple[int, str, int], ...]]:
    """Every statement's position as a chain of ``(owner, field, index)`` steps.

    The chain says which blocks a statement is nested inside and where in each. Two
    statements in the same block share a chain except for the final index; one nested
    inside an ``if`` the other is not has an extra step. That difference is what
    separates "the hook runs before this call" from "the hook is printed above it".
    """
    paths: dict[int, tuple[tuple[int, str, int], ...]] = {}

    def walk(node: ast.AST, prefix: tuple[tuple[int, str, int], ...]) -> None:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.stmt):
                        step = (*prefix, (id(node), field, index))
                        paths[id(item)] = step
                        walk(item, step)
                    elif isinstance(item, ast.AST):
                        walk(item, prefix)
            elif isinstance(value, ast.AST):
                walk(value, prefix)

    walk(root, ())
    return paths


def _enclosing_statement(root: ast.AST, target: ast.AST) -> ast.stmt | None:
    """The innermost statement containing ``target`` (the statement it belongs to)."""
    found: list[ast.stmt] = []

    def walk(node: ast.AST, current: ast.stmt | None) -> None:
        for child in ast.iter_child_nodes(node):
            here = child if isinstance(child, ast.stmt) else current
            if child is target:
                if here is not None:
                    found.append(here)
                return
            walk(child, here)

    walk(root, root if isinstance(root, ast.stmt) else None)
    return found[0] if found else None


def _hook_dominates(
    method: ast.AST, *, hook_expression: str, call_expression: str, source: str
) -> bool:
    """True when every ``call_expression`` is preceded by a ``hook_expression`` that runs.

    "Runs" is the part line numbers cannot express. A hook dominates a call when it is a
    statement of a block that *encloses* the call — same block and earlier, or an
    enclosing one and earlier — so a hook tucked inside a branch the call does not take
    no longer counts. The hook may be in an outer block than the call (the real code has
    the hook in a loop body and the call inside a ``try`` within it); what it may not be
    is *deeper*, because anything deeper is conditional on something the call is not.
    """
    paths = _statement_paths(method)
    hooks = _find_calls(method, hook_expression, source)
    calls = _find_calls(method, call_expression, source)
    if not calls:
        return False

    def path_of(node: ast.AST) -> tuple[tuple[int, str, int], ...] | None:
        statement = _enclosing_statement(method, node)
        return None if statement is None else paths.get(id(statement))

    for call in calls:
        call_path = path_of(call)
        if call_path is None:
            return False
        if not any(_dominates(path_of(hook), call_path) for hook in hooks):
            return False
    return True


def _dominates(
    hook_path: tuple[tuple[int, str, int], ...] | None,
    call_path: tuple[tuple[int, str, int], ...],
) -> bool:
    """``hook_path`` is a statement of a block enclosing ``call_path``, and earlier in it."""
    if not hook_path:
        return False
    common = 0
    while (
        common < len(hook_path)
        and common < len(call_path)
        and hook_path[common] == call_path[common]
    ):
        common += 1
    # The hook must sit directly in the block where the two paths part company. One step
    # beyond the common prefix means "a statement of that block"; more means it is nested
    # inside something — an `if`, a `for`, an `except` — that the call does not enter.
    if len(hook_path) != common + 1 or len(call_path) <= common:
        return False
    hook_owner, hook_field, hook_index = hook_path[common]
    call_owner, call_field, call_index = call_path[common]
    return (hook_owner, hook_field) == (call_owner, call_field) and hook_index < call_index


def _find_calls(node: ast.AST, call_expression: str, source: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_expression(source, child.func) == call_expression
    ]


def _call_lines(source: str, node: ast.AST, call_expression: str) -> list[int]:
    return [
        child.lineno
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_expression(source, child.func) == call_expression
    ]


def _is_direct_llm_call(source: str, func: ast.expr) -> bool:
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr == "_llm":
        return True
    if func.attr not in {"invoke", "invoke_turn", "invoke_stream", "invoke_json"}:
        return False
    receiver = _call_expression(source, func.value)
    return receiver == "client" or "llm_client" in receiver


def _call_expression(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or "<unknown>"
