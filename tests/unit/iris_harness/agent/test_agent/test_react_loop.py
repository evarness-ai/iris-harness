"""Behavioral tests for the ReAct loop engine."""

from __future__ import annotations

from iris_harness.agent.agent_executor import ActivityChunk, TraceChunk
from iris_harness.agent.agentic_core import (
    AgenticCore,
    AgenticCoreConfig,
    ToolSpec,
    _parse_react_step,
)


def _make_echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo", description="Echo the input back.", call=lambda args: args.get("text", "")
    )


def test_parse_react_step_final_answer() -> None:
    text = "Thought: I know the answer.\nFinal Answer: Paris is the capital of France."
    step = _parse_react_step(text)

    assert step.is_terminal
    assert step.final_answer == "Paris is the capital of France."
    assert step.thought == "I know the answer."


def test_parse_react_step_tool_call() -> None:
    text = 'Thought: I need to echo something.\nAction: echo\nAction Input: {"text": "hello"}'
    step = _parse_react_step(text)

    assert not step.is_terminal
    assert step.action == "echo"
    assert step.action_input == {"text": "hello"}


def test_parse_react_step_trailing_text_after_json_object() -> None:
    """Salvage trailing chatter the model appends after valid JSON."""
    text = (
        "Thought: search.\n"
        "Action: research\n"
        'Action Input: {"query": "x", "time_range": "d"}\n\nWaiting for response...'
    )
    step = _parse_react_step(text)

    assert step.action == "research"
    assert step.action_input == {"query": "x", "time_range": "d"}


def test_parse_react_step_unwraps_nested_input_pattern() -> None:
    """qwen2.5-coder:7b sometimes nests real args under an 'input' key."""
    text = (
        "Thought: search.\n"
        "Action: research\n"
        'Action Input: {"input": "{\\"query\\": \\"x\\", \\"time_range\\": \\"d\\"}"}'
    )
    step = _parse_react_step(text)

    assert step.action == "research"
    assert step.action_input == {"query": "x", "time_range": "d"}


def test_parse_react_step_falls_back_to_input_for_non_json() -> None:
    """Plain non-JSON action input still goes under the 'input' key."""
    text = "Thought: t.\nAction: echo\nAction Input: hello world"
    step = _parse_react_step(text)

    assert step.action_input == {"input": "hello world"}


def test_react_loop_reaches_final_answer_on_first_turn() -> None:
    def llm(prompt: str) -> str:
        return "Thought: Simple question.\nFinal Answer: 42"

    core = AgenticCore(AgenticCoreConfig(max_iterations=3), llm_call=llm)
    trace = core.run("What is 6 times 7?")

    assert trace.success
    assert trace.final_answer == "42"
    assert trace.iterations == 1


def test_react_loop_executes_tool_and_returns_answer() -> None:
    call_count = [0]

    def llm(prompt: str) -> str:
        del prompt
        call_count[0] += 1
        if call_count[0] == 1:
            return 'Thought: I will echo.\nAction: echo\nAction Input: {"text": "pong"}'
        return "Thought: Got the observation.\nFinal Answer: pong"

    core = AgenticCore(AgenticCoreConfig(max_iterations=5), llm_call=llm, tools=[_make_echo_tool()])
    trace = core.run("ping")

    assert trace.success
    assert trace.final_answer == "pong"
    assert any(s.observation is not None for s in trace.steps)


def test_react_loop_unknown_tool_returns_error_observation() -> None:
    call_count = [0]

    def llm(prompt: str) -> str:
        del prompt
        call_count[0] += 1
        if call_count[0] == 1:
            return "Thought: Try unknown tool.\nAction: nonexistent\nAction Input: {}"
        return "Thought: Got error.\nFinal Answer: tool missing"

    core = AgenticCore(AgenticCoreConfig(max_iterations=3), llm_call=llm)
    trace = core.run("test")

    assert any("unknown tool" in (s.observation or "") for s in trace.steps)


def test_react_loop_dedup_falls_back_to_observation_when_model_loops() -> None:
    """A repeated action triggers a forced-synthesis pass on the SECOND occurrence
    (no waiting for the stall counter). If the model still won't finalize, the
    loop surfaces the last observation instead of a 'repeated action' dead-end."""

    def llm(prompt: str) -> str:
        return 'Thought: Looping.\nAction: echo\nAction Input: {"text": "x"}'

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=10, stall_limit=2),
        llm_call=llm,
        tools=[_make_echo_tool()],
    )
    trace = core.run("test")

    assert trace.success  # we surface the data we already have
    assert trace.final_answer == "x"  # last good observation
    assert "Stopped" not in trace.final_answer
    assert trace.iterations <= 2  # short-circuited on the repeat, not after 10 turns


def test_react_loop_dedup_forces_clean_final_answer() -> None:
    """On the repeated action, the forced-synthesis nudge gets a clean Final
    Answer and the tool is NOT executed a second time."""
    calls = [0]
    tool_runs = [0]

    def llm(prompt: str) -> str:
        calls[0] += 1
        if calls[0] <= 2:
            return 'Thought: fetch it.\nAction: echo\nAction Input: {"text": "42"}'
        return "Thought: done.\nFinal Answer: The answer is 42."

    def echo(args: dict) -> str:
        tool_runs[0] += 1
        return str(args.get("text", ""))

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=10, stall_limit=2),
        llm_call=llm,
        tools=[ToolSpec(name="echo", description="Echo.", call=echo)],
    )
    trace = core.run("test")

    assert trace.success
    assert "42" in trace.final_answer
    assert "Final Answer" not in trace.final_answer
    assert tool_runs[0] == 1  # repeat short-circuited before re-executing the tool


def test_react_loop_same_tool_different_inputs_does_not_stall() -> None:
    """Regression for the web-search drill-down case: repeated calls to
    the same tool with DIFFERENT inputs are progress, not a stall. The
    pre-2026-05-20 detector compared only the tool name and killed
    legitimate drill-down runs (BBC headline -> follow-up query)."""

    call_count = [0]

    def llm(prompt: str) -> str:
        del prompt
        call_count[0] += 1
        if call_count[0] == 1:
            return 'Thought: Look up the topic.\nAction: echo\nAction Input: {"text": "first"}'
        if call_count[0] == 2:
            return 'Thought: Drill in.\nAction: echo\nAction Input: {"text": "second"}'
        if call_count[0] == 3:
            return 'Thought: Drill further.\nAction: echo\nAction Input: {"text": "third"}'
        return "Thought: Got it.\nFinal Answer: done"

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=10, stall_limit=2),
        llm_call=llm,
        tools=[_make_echo_tool()],
    )
    trace = core.run("test")

    # Three distinct tool calls + a final answer — must not stall.
    assert trace.success, f"unexpected stall: {trace.final_answer!r}"
    assert trace.stall_count < 2
    assert trace.final_answer == "done"


def test_react_loop_no_llm_returns_graceful_error() -> None:
    core = AgenticCore()
    trace = core.run("anything")

    assert not trace.success
    assert "No LLM" in trace.final_answer


def test_react_loop_respects_max_iterations() -> None:
    call_count = [0]

    def llm(prompt: str) -> str:
        call_count[0] += 1
        return "Thought: thinking"

    core = AgenticCore(AgenticCoreConfig(max_iterations=3), llm_call=llm)
    trace = core.run("test")

    assert call_count[0] <= 3
    assert trace.iterations <= 3


def test_react_loop_retries_thought_only_step_before_answering() -> None:
    call_count = [0]

    def llm(prompt: str) -> str:
        del prompt
        call_count[0] += 1
        if call_count[0] == 1:
            return "Thought: I should check the inbox first."
        return "Thought: done.\nFinal Answer: I found the relevant email."

    core = AgenticCore(AgenticCoreConfig(max_iterations=3), llm_call=llm)
    trace = core.run("read that email")

    assert trace.success is True
    assert trace.final_answer == "I found the relevant email."
    assert trace.iterations == 2


def test_react_loop_retries_internal_planning_plaintext_before_answering() -> None:
    call_count = [0]

    def llm(prompt: str) -> str:
        del prompt
        call_count[0] += 1
        if call_count[0] == 1:
            return "The user wants specific emails from Robinhood. I need to fetch them directly."
        return "Final Answer: I found the Robinhood reminder email."

    core = AgenticCore(AgenticCoreConfig(max_iterations=3), llm_call=llm)
    trace = core.run("read the Robinhood payment reminder email")

    assert trace.success is True
    assert trace.final_answer == "I found the Robinhood reminder email."
    assert trace.iterations == 2


def test_react_loop_requires_retrieval_tool_for_fresh_news_queries() -> None:
    call_count = [0]
    tool_calls = [0]
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        call_count[0] += 1
        if call_count[0] == 1:
            return (
                "Thought: I can answer from general knowledge.\n"
                "Final Answer: AI news includes progress in NLP and vision."
            )
        if call_count[0] == 2:
            return (
                "Thought: I should fetch fresh news.\n"
                "Action: research\n"
                'Action Input: {"query": "top AI news today"}'
            )
        return "Thought: I have the results.\nFinal Answer: 1. Story A\n2. Story B"

    def research_tool(args: dict[str, object]) -> str:
        tool_calls[0] += 1
        assert args["query"] == "top AI news today"
        return "1. Story A\n2. Story B"

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=5),
        llm_call=llm,
        tools=[
            ToolSpec(
                name="research",
                description="Search the web for current information.",
                call=research_tool,
            )
        ],
    )
    trace = core.run("what's the top 2 AI news today?")

    assert trace.success is True
    assert trace.final_answer == "1. Story A\n2. Story B"
    assert call_count[0] == 3
    assert tool_calls[0] == 1
    assert "Error: Fresh-news query requires a retrieval tool call" in (
        trace.steps[0].observation or ""
    )
    assert "Return exactly 2 items as a numbered list." in prompts[0]


def test_react_loop_allows_news_final_answer_when_retrieval_tools_unavailable() -> None:
    def llm(prompt: str) -> str:
        del prompt
        return "Thought: no tools available.\nFinal Answer: Here's a high-level summary."

    core = AgenticCore(AgenticCoreConfig(max_iterations=3), llm_call=llm, tools=[_make_echo_tool()])
    trace = core.run("what's the top AI news today?")

    assert trace.success is True
    assert trace.iterations == 1
    assert trace.final_answer == "Here's a high-level summary."


def test_config_supports_max_tokens_and_streaming_flags() -> None:
    cfg = AgenticCoreConfig(max_tokens=512, streaming=True)

    assert cfg.max_tokens == 512
    assert cfg.streaming is True


def test_react_loop_governance_blocks_secret_to_cloud_before_llm() -> None:
    from iris_harness.kernel.governance import build_default_kernel

    llm_called = False

    def llm(prompt: str) -> str:
        nonlocal llm_called
        del prompt
        llm_called = True
        return "Thought: should not run.\nFinal Answer: nope"

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=3),
        llm_call=llm,
        kernel=build_default_kernel(),
        target_tier="tier_3",
    )

    trace = core.run("here is my key sk-ABC1234567890abcdef1234")

    assert llm_called is False
    assert not trace.success
    assert trace.iterations == 0
    assert "blocked by governance" in trace.final_answer


def test_react_loop_governance_classification_is_per_run() -> None:
    from iris_harness.kernel.governance import build_default_kernel

    call_count = 0

    def llm(prompt: str) -> str:
        nonlocal call_count
        del prompt
        call_count += 1
        return "Thought: ok.\nFinal Answer: allowed"

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=3),
        llm_call=llm,
        kernel=build_default_kernel(),
        target_tier="tier_3",
    )

    first_trace = core.run("explain how transformers work")
    second_trace = core.run("here is my key sk-ABC1234567890abcdef1234")

    assert first_trace.success is True
    assert second_trace.success is False
    assert "blocked by governance" in second_trace.final_answer
    assert call_count == 1


def test_react_loop_governance_blocks_tool_before_call() -> None:
    from iris_harness.kernel.governance import build_default_kernel

    llm_count = 0
    tool_called = False

    def llm(prompt: str) -> str:
        nonlocal llm_count
        del prompt
        llm_count += 1
        if llm_count == 1:
            return 'Thought: use a tool.\nAction: echo\nAction Input: {"text": "secret"}'
        return "Thought: blocked.\nFinal Answer: tool was blocked"

    def tool(args: dict[str, object]) -> str:
        nonlocal tool_called
        del args
        tool_called = True
        return "should not run"

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=3),
        llm_call=llm,
        tools=[ToolSpec(name="echo", description="Echo.", call=tool)],
        kernel=build_default_kernel(blocked_tools=frozenset({"echo"})),
        target_tier="tier_1",
    )

    trace = core.run("please echo this")

    assert tool_called is False
    assert trace.success is True
    assert "blocked by governance" in (trace.steps[0].observation or "")


def test_run_stream_emits_activity_trace_and_final_chunks() -> None:
    def llm(prompt: str) -> str:
        return "Thought: I know.\nFinal Answer: hi"

    core = AgenticCore(AgenticCoreConfig(max_iterations=3, streaming=True), llm_call=llm)
    items = list(core.run_stream("hello"))

    activities = [i for i in items if isinstance(i, ActivityChunk)]
    traces = [i for i in items if isinstance(i, TraceChunk)]
    texts = [i for i in items if isinstance(i, str)]
    metas = [i for i in items if isinstance(i, dict)]

    assert activities and "I know" in activities[0].text
    assert traces, "expected at least one TraceChunk"
    assert texts == ["hi"]
    assert metas and metas[-1]["success"] is True
    assert metas[-1]["reason"] == "final_answer"
    assert metas[-1]["iterations"] == 1


def test_run_stream_governance_blocks_secret_to_cloud_before_llm() -> None:
    from iris_harness.kernel.governance import build_default_kernel

    llm_called = False

    def llm(prompt: str) -> str:
        nonlocal llm_called
        del prompt
        llm_called = True
        return "Thought: should not run.\nFinal Answer: nope"

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=3, streaming=True),
        llm_call=llm,
        kernel=build_default_kernel(),
        target_tier="tier_3",
    )
    items = list(core.run_stream("here is my key sk-ABC1234567890abcdef1234"))
    texts = [i for i in items if isinstance(i, str)]
    metas = [i for i in items if isinstance(i, dict)]

    assert llm_called is False
    assert texts and "blocked by governance" in texts[0]
    assert metas[-1]["success"] is False
    assert metas[-1]["reason"] == "governance_denied"
    assert metas[-1]["iterations"] == 0


def test_run_stream_governance_blocks_tool_before_call() -> None:
    from iris_harness.kernel.governance import build_default_kernel

    llm_count = 0
    tool_called = False

    def llm(prompt: str) -> str:
        nonlocal llm_count
        del prompt
        llm_count += 1
        if llm_count == 1:
            return 'Thought: use a tool.\nAction: echo\nAction Input: {"text": "secret"}'
        return "Thought: blocked.\nFinal Answer: tool was blocked"

    def tool(args: dict[str, object]) -> str:
        nonlocal tool_called
        del args
        tool_called = True
        return "should not run"

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=3, streaming=True),
        llm_call=llm,
        tools=[ToolSpec(name="echo", description="Echo.", call=tool)],
        kernel=build_default_kernel(blocked_tools=frozenset({"echo"})),
        target_tier="tier_1",
    )
    items = list(core.run_stream("please echo this"))
    traces = [i for i in items if isinstance(i, TraceChunk)]

    assert tool_called is False
    assert any("blocked by governance" in trace.text for trace in traces)


def test_run_stream_tool_loop_then_final() -> None:
    call_count = [0]

    def llm(prompt: str) -> str:
        del prompt
        call_count[0] += 1
        if call_count[0] == 1:
            return 'Thought: echo.\nAction: echo\nAction Input: {"text": "pong"}'
        return "Thought: done.\nFinal Answer: pong"

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=3, streaming=True),
        llm_call=llm,
        tools=[_make_echo_tool()],
    )
    items = list(core.run_stream("ping"))
    metas = [i for i in items if isinstance(i, dict)]

    assert metas[-1]["success"] is True
    assert metas[-1]["iterations"] == 2


def test_run_stream_retries_internal_planning_plaintext() -> None:
    call_count = [0]

    def llm(prompt: str) -> str:
        del prompt
        call_count[0] += 1
        if call_count[0] == 1:
            return "The user wants specific emails from Robinhood. I need to fetch them directly."
        return "Final Answer: I found the Robinhood reminder email."

    core = AgenticCore(AgenticCoreConfig(max_iterations=3, streaming=True), llm_call=llm)
    items = list(core.run_stream("read the Robinhood payment reminder email"))
    texts = [i for i in items if isinstance(i, str)]
    metas = [i for i in items if isinstance(i, dict)]

    assert texts == ["I found the Robinhood reminder email."]
    assert metas[-1]["success"] is True
    assert metas[-1]["iterations"] == 2


def test_run_stream_stall_recovers_last_observation() -> None:
    """When the model loops instead of finalizing but a tool produced grounded
    data, the stall recovers that observation as the answer rather than emitting
    the useless 'Stopped' sentinel (issue 0001 D / 0003)."""

    def llm(prompt: str) -> str:
        return 'Thought: loop.\nAction: echo\nAction Input: {"text": "the answer is 42"}'

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=10, stall_limit=2, streaming=True),
        llm_call=llm,
        tools=[_make_echo_tool()],
    )
    items = list(core.run_stream("test"))
    metas = [i for i in items if isinstance(i, dict)]
    text = "".join(i for i in items if isinstance(i, str))

    assert metas[-1]["success"] is True
    assert metas[-1]["reason"] == "recovered_observation"
    assert metas[-1]["stall_count"] >= 2  # the stall is still recorded
    assert "the answer is 42" in text  # grounded observation surfaced


def test_run_stream_stall_without_observation_reports_failure() -> None:
    """A stall with NO usable observation still fails honestly with the sentinel."""

    def llm(prompt: str) -> str:
        return 'Thought: loop.\nAction: echo\nAction Input: {"text": ""}'  # empty observation

    core = AgenticCore(
        AgenticCoreConfig(max_iterations=10, stall_limit=2, streaming=True),
        llm_call=llm,
        tools=[_make_echo_tool()],
    )
    items = list(core.run_stream("test"))
    metas = [i for i in items if isinstance(i, dict)]

    assert metas[-1]["success"] is False
    assert metas[-1]["reason"] == "stalled"
    assert metas[-1]["stall_count"] >= 2


def test_react_loop_emits_tool_invoke_session_events(tmp_path, monkeypatch) -> None:
    """ReAct tool calls must surface as tool.invoke.start/end session events so
    they appear in the trace graph + Reasoning tab."""
    import json

    from iris_harness.foundation.observability import session_log

    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)

    calls = [0]

    def llm(prompt: str) -> str:
        calls[0] += 1
        if calls[0] == 1:
            return 'Thought: go.\nAction: echo\nAction Input: {"text": "hi"}'
        return "Thought: done.\nFinal Answer: hi"

    core = AgenticCore(AgenticCoreConfig(max_iterations=5), llm_call=llm, tools=[_make_echo_tool()])
    with session_log.session_scope("toolsess"):
        core.run("ping")

    events = [
        json.loads(line)
        for line in (tmp_path / "session-toolsess.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kinds = [e["kind"] for e in events]
    assert "tool.invoke.start" in kinds
    assert "tool.invoke.end" in kinds
    end = next(e for e in events if e["kind"] == "tool.invoke.end")
    assert end["payload"]["tool"] == "echo"
    assert end["payload"]["ok"] is True
    assert "hi" in str(end["payload"]["result_preview"])


# --- ADR-0110 answers_directly: a finished answer is not restated by another call -------


def _counting_llm(replies: list[str]) -> tuple[list[str], object]:
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return replies[min(len(prompts), len(replies)) - 1]

    return prompts, llm


def _digest_tool(runs: list[int], *, answers_directly: bool = True) -> ToolSpec:
    def call(_args: dict) -> str:
        runs.append(1)
        return "[Current date: 2026-09-16]\nInbox today: 73 new, mostly promotions."

    return ToolSpec(
        name="digest", description="Inbox digest.", call=call, answers_directly=answers_directly
    )


_PICK_DIGEST = "Thought: get the digest.\nAction: digest\nAction Input: {}"


def test_first_tool_that_answers_directly_ends_the_run_without_another_call() -> None:
    runs: list[int] = []
    prompts, llm = _counting_llm(
        [_PICK_DIGEST, "Thought: again.\nAction: digest\nAction Input: {}"]
    )
    core = AgenticCore(
        AgenticCoreConfig(max_iterations=5), llm_call=llm, tools=[_digest_tool(runs)]
    )

    trace = core.run("summarize my inbox")

    assert trace.success
    assert trace.final_answer == "Inbox today: 73 new, mostly promotions."  # header stripped
    assert len(prompts) == 1 and runs == [1]


def test_undeclared_tool_still_goes_back_to_the_model() -> None:
    runs: list[int] = []
    prompts, llm = _counting_llm([_PICK_DIGEST, "Thought: done.\nFinal Answer: 73 new."])
    core = AgenticCore(
        AgenticCoreConfig(max_iterations=5),
        llm_call=llm,
        tools=[_digest_tool(runs, answers_directly=False)],
    )

    trace = core.run("summarize my inbox")

    assert trace.final_answer == "73 new." and len(prompts) == 2


def test_answers_directly_tool_after_another_tool_leaves_the_answer_to_the_model() -> None:
    runs: list[int] = []
    prompts, llm = _counting_llm(
        [
            'Thought: echo first.\nAction: echo\nAction Input: {"text": "x"}',
            _PICK_DIGEST,
            "Thought: combine.\nFinal Answer: x and the digest.",
        ]
    )
    core = AgenticCore(
        AgenticCoreConfig(max_iterations=5),
        llm_call=llm,
        tools=[_make_echo_tool(), _digest_tool(runs)],
    )

    trace = core.run("echo x, then summarize my inbox")

    assert trace.final_answer == "x and the digest." and len(prompts) == 3


def test_repeat_of_answers_directly_tool_returns_result_in_hand_without_synthesis() -> None:
    runs: list[int] = []
    prompts, llm = _counting_llm(
        [
            'Thought: echo first.\nAction: echo\nAction Input: {"text": "x"}',
            _PICK_DIGEST,
            _PICK_DIGEST,  # the repeat the live email turn made
            "Thought: forced.\nFinal Answer: should never be asked for",
        ]
    )
    core = AgenticCore(
        AgenticCoreConfig(max_iterations=6),
        llm_call=llm,
        tools=[_make_echo_tool(), _digest_tool(runs)],
    )

    trace = core.run("echo x, then summarize my inbox")

    assert trace.success
    assert trace.final_answer == "Inbox today: 73 new, mostly promotions."
    assert len(prompts) == 3  # no forced-synthesis call
    assert runs == [1]  # and the tool did not run again


def test_error_from_answers_directly_tool_goes_back_to_the_model() -> None:
    prompts, llm = _counting_llm(
        [_PICK_DIGEST, "Thought: explain.\nFinal Answer: inbox is offline."]
    )
    broken = ToolSpec(
        name="digest",
        description="Inbox digest.",
        call=lambda _a: "inbox_digest unavailable: store locked",
        answers_directly=True,
    )
    core = AgenticCore(AgenticCoreConfig(max_iterations=5), llm_call=llm, tools=[broken])

    trace = core.run("summarize my inbox")

    assert trace.final_answer == "inbox is offline." and len(prompts) == 2


def test_stream_ends_on_answers_directly_tool_and_skips_its_repeat() -> None:
    runs: list[int] = []
    prompts, llm = _counting_llm([_PICK_DIGEST, "Thought: x.\nFinal Answer: never"])
    core = AgenticCore(
        AgenticCoreConfig(max_iterations=5), llm_call=llm, tools=[_digest_tool(runs)]
    )

    chunks = list(core.run_stream("summarize my inbox"))
    texts = [c for c in chunks if isinstance(c, str)]
    meta = chunks[-1]

    assert texts == ["Inbox today: 73 new, mostly promotions."]
    assert isinstance(meta, dict) and meta["reason"] == "answered_by_tool" and meta["success"]
    assert len(prompts) == 1 and runs == [1]

    runs.clear()
    prompts2, llm2 = _counting_llm(
        [
            'Thought: echo first.\nAction: echo\nAction Input: {"text": "x"}',
            _PICK_DIGEST,
            _PICK_DIGEST,
            "Thought: x.\nFinal Answer: never",
        ]
    )
    core2 = AgenticCore(
        AgenticCoreConfig(max_iterations=6, stall_limit=5),
        llm_call=llm2,
        tools=[_make_echo_tool(), _digest_tool(runs)],
    )
    chunks2 = list(core2.run_stream("echo x, then summarize my inbox"))

    assert [c for c in chunks2 if isinstance(c, str)] == ["Inbox today: 73 new, mostly promotions."]
    assert runs == [1] and len(prompts2) == 3
