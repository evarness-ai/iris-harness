"""Tests for the code_exec planner protocol loop."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from iris_harness.agent.agent_executor import ActivityChunk, AgentTask, TraceChunk
from iris_harness.llm.client import CodingLLMConfig
from iris_harness.plugins_builtin.code_exec.handler import _make_code_exec_handler
from iris_harness.tools.sandbox.models import ExecResult


class _TierRouter:
    def __init__(self, provider: str = "github", model: str = "fake-model") -> None:
        self._provider = provider
        self._model = model

    def get_llm_config(self, _intent: str) -> CodingLLMConfig:
        base_url = (
            "http://localhost:11434/v1" if self._provider == "ollama" else "https://example.test/v1"
        )
        return CodingLLMConfig(
            provider=self._provider,
            model=self._model,
            base_url=base_url,
            api_key_env=None,
        )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[str | Sequence[str]],
    shell_results: Sequence[ExecResult] = (),
) -> tuple[list[Any], list[Any]]:
    client_instances: list[Any] = []
    host_instances: list[Any] = []
    pending_results = list(shell_results)

    class _FakeClient:
        def __init__(self, _config: CodingLLMConfig) -> None:
            self.responses = list(responses)
            self.user_prompts: list[str] = []
            self.system_prompts: list[str] = []
            client_instances.append(self)

        def invoke_stream(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
        ) -> Iterator[str]:
            self.user_prompts.append(user_prompt)
            self.system_prompts.append(system_prompt)
            response = self.responses.pop(0)
            if isinstance(response, str):
                yield response
            else:
                yield from response

    class _FakeHost:
        workspace_path = "/tmp/iris-fake-workspace"

        def __init__(self, session_id: str, **_kwargs: object) -> None:
            self.session_id = session_id
            self.calls: list[tuple[str, int]] = []
            host_instances.append(self)

        def run_shell(self, cmd: str, *, timeout: int = 30) -> ExecResult:
            self.calls.append((cmd, timeout))
            return pending_results.pop(0)

    # Patch the module the handler imports FROM, which is the SDK facade, not the module
    # that defines the class. A re-export binds the name a second time at import, so a
    # patch on `iris_harness.llm.client` leaves `iris_harness.sdk.llm` pointing at the
    # real client -- the test then makes live API calls and fails on a proxy 403 rather
    # than on anything it meant to assert. (M6.3 gate-2 burn-down.)
    monkeypatch.setattr("iris_harness.sdk.llm.CodingLLMClient", _FakeClient)
    monkeypatch.setattr("iris_harness.sdk.sandbox.SandboxToolHost", _FakeHost)
    return client_instances, host_instances


def test_code_exec_progress_field_becomes_activity_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        (
            '{"tool":"run_shell","args":{"cmd":"echo ok","timeout":5},'
            '"progress":"building report and verifying output"}'
        ),
        "Done.",
    ]
    result = ExecResult(
        stdout="ok\n",
        stderr="",
        exit_code=0,
        duration_ms=12.0,
        artifacts=("/tmp/iris-fake-workspace/report.pdf",),
    )
    _clients, hosts = _install_fake_runtime(monkeypatch, responses, [result])

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(AgentTask(query="create a report", agent_type="code_exec", session_id="s1"))
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    trace_texts = [chunk.text for chunk in chunks if isinstance(chunk, TraceChunk)]
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert activity_texts[0] == "building report and verifying output"
    assert any("progress: building report and verifying output" in text for text in trace_texts)
    assert hosts[0].calls == [("echo ok", 5)]
    assert "Done." in response_text


def test_code_exec_compute_result_reaches_user_without_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """exp-006 GAP-11: the computed answer (55) must reach the user even when
    the planner emits no clean final prose and there is no file artifact.

    Repro: the planner runs the same successful compute command twice (printing
    55, no artifact). Two identical clean outcomes trip the completion checkpoint
    (identical_outcomes >= 2) → the sandbox handoff path, with no planner prose.
    Pre-fix, the handoff dropped the result behind a generic "handing off the
    result directly" notice and the value 55 never surfaced.
    """
    tool_call = (
        '{"tool":"run_shell","args":{"cmd":"python -c \\"print(55)\\"",'
        '"timeout":10},"progress":"computing the 10th fibonacci number"}'
    )
    responses = [tool_call, tool_call]
    result = ExecResult(stdout="55\n", stderr="", exit_code=0, duration_ms=9.0, artifacts=())
    _clients, hosts = _install_fake_runtime(monkeypatch, responses, [result, result])

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(
            AgentTask(
                query="what is the 10th fibonacci number",
                agent_type="code_exec",
                session_id="s-gap11",
            )
        )
    )
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    # The real result reaches the user, presented as the answer...
    assert "55" in response_text
    assert "Result: 55" in response_text
    # ...not buried in a raw `exit_code=.../stdout:` summary dump, and the
    # misleading "handing off" placeholder no longer fronts a compute result.
    assert "exit_code=" not in response_text
    assert "stdout:" not in response_text
    assert "handing off the result directly" not in response_text


def test_code_exec_bails_after_two_invalid_tool_call_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both responses must sanitize to empty prose so the prose-accept fallback
    # in bootstrap.py (see comment near `final_answer_candidate and last_exit_code is None`)
    # does not short-circuit before the bail counter hits the limit. The
    # ```python and ```bash fences are drop-all markers in _PROSE_DROP_ALL_MARKERS.
    responses = [
        "Sure, here is a script:\n```python\nprint('hello')\n```",
        "Try this instead:\n```bash\necho hello\n```",
    ]
    clients, hosts = _install_fake_runtime(monkeypatch, responses)

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(
            AgentTask(query="run this in sandbox", agent_type="code_exec", session_id="s2")
        )
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert any("(1/2)" in text for text in activity_texts)
    assert any("(2/2)" in text for text in activity_texts)
    assert "Aborted: the planner produced no valid run_shell JSON tool call" in response_text
    assert hosts[0].calls == []
    assert len(clients[0].user_prompts) == 2
    assert "PLANNER GUIDANCE" in clients[0].user_prompts[1]


def test_code_exec_allows_ask_user_for_mid_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        (
            '{"tool":"ask_user","args":{"question":"Which date range should I use?"},'
            '"progress":"checking one missing requirement"}'
        )
    ]
    _clients, hosts = _install_fake_runtime(monkeypatch, responses)

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(AgentTask(query="make a report", agent_type="code_exec", session_id="s2a"))
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert activity_texts == ["checking one missing requirement"]
    assert "I need one detail before I can continue" in response_text
    assert "Which date range should I use?" in response_text
    assert hosts[0].calls == []


def test_code_exec_blocks_ask_user_for_small_tier_and_uses_small_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ask_user_call = (
        '{"tool":"ask_user","args":{"question":"Which filename should I use?"},'
        '"progress":"checking filename"}'
    )
    run_shell_call = '{"tool":"run_shell","args":{"cmd":"echo 4","timeout":5}}'
    result = ExecResult(
        stdout="4\n",
        stderr="",
        exit_code=0,
        duration_ms=8.0,
        artifacts=(),
    )
    clients, hosts = _install_fake_runtime(
        monkeypatch,
        [ask_user_call, run_shell_call, "Done."],
        [result],
    )

    _handler, stream_handler = _make_code_exec_handler(
        _TierRouter(provider="ollama", model="tiny-local")
    )
    chunks = list(
        stream_handler(
            AgentTask(
                query="what is 2 + 2",
                agent_type="code_exec",
                session_id="s2b",
            )
        )
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert "ask_user blocked by tier policy (1/2)" in activity_texts
    assert hosts[0].calls == [("echo 4", 5)]
    assert "Done." in response_text
    assert "ask_user is disabled" in clients[0].system_prompts[0]
    assert "TOOL — ask_user" not in clients[0].system_prompts[0]


def test_code_exec_finalizes_after_repeated_successful_artifact_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = (
        '{"tool":"run_shell","args":{"cmd":"python /workspace/write_report.py",'
        '"timeout":30},"progress":"writing one-page research summary"}'
    )
    result = ExecResult(
        stdout="wrote report\n",
        stderr="",
        exit_code=0,
        duration_ms=18.0,
        artifacts=("/tmp/iris-fake-workspace/agent_harness_summary.md",),
    )
    _clients, hosts = _install_fake_runtime(
        monkeypatch,
        [tool_call, tool_call],
        [result, result],
    )

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(
            AgentTask(
                query="generate a one pager document about agent harness",
                agent_type="code_exec",
                session_id="s3",
            )
        )
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert len(hosts[0].calls) == 2
    assert any("completion checkpoint" in text for text in activity_texts)
    assert "Completed the sandbox work" in response_text
    assert "Asked: generate a one pager document about agent harness" in response_text
    assert "Reached the maximum" not in response_text


def test_code_exec_max_iterations_after_clean_success_returns_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        (
            '{"tool":"run_shell","args":{"cmd":"echo step '
            f'{index}","timeout":30}},"progress":"step {index}"}}'
        )
        for index in range(8)
    ]
    results = [
        ExecResult(
            stdout=f"step {index}\n",
            stderr="",
            exit_code=0,
            duration_ms=10.0,
            artifacts=("/tmp/iris-fake-workspace/agent_harness_summary.md",),
        )
        for index in range(8)
    ]
    _clients, hosts = _install_fake_runtime(monkeypatch, responses, results)

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(
            AgentTask(
                query="generate a one pager document about agent harness",
                agent_type="code_exec",
                session_id="s4",
            )
        )
    )

    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert len(hosts[0].calls) == 8
    assert "Completed the sandbox work" in response_text
    assert "Tool loop: 8 iteration(s)." in response_text
    assert "Reached the maximum" not in response_text


def test_code_exec_repeated_exit_zero_warning_is_not_clean_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = (
        '{"tool":"run_shell","args":{"cmd":"cat > /workspace/report.md '
        '<< \'PYEOF\'","timeout":30},"progress":"writing markdown report"}'
    )
    warning = "bash: line 9: warning: here-document at line 1 delimited by end-of-file"
    result = ExecResult(
        stdout="",
        stderr=warning,
        exit_code=0,
        duration_ms=16.0,
        artifacts=("/tmp/iris-fake-workspace/report.md",),
    )
    _clients, hosts = _install_fake_runtime(
        monkeypatch,
        [tool_call, tool_call],
        [result, result],
    )

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(
            AgentTask(
                query="generate a one pager document about agent harness",
                agent_type="code_exec",
                session_id="s5",
            )
        )
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert len(hosts[0].calls) == 2
    assert any("repeated warning result" in text for text in activity_texts)
    assert "not treating this as a clean completion" in response_text
    assert "here-document" in response_text
    assert "Reached the maximum" not in response_text


def test_code_exec_invalid_abort_does_not_leak_raw_planner_json_to_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-visible abort must not embed raw planner JSON.

    Raw planner output still flows to /trace via TraceChunk, but the chat
    response only carries the friendly abort + a /trace hint.
    """
    secret_marker = "SECRET_INTERNAL_PAYLOAD_42"
    # Both responses must produce empty `final_answer_candidate` to reach the
    # bail path. The first leaks via a `{"tool":` drop-all marker; the second
    # leaks the marker inside a ```python fence, which is also drop-all.
    responses = [
        f'{{"tool":"do_evil","args":{{"x":"{secret_marker}"}}}}',
        f"```python\n# {secret_marker}\nprint('leak')\n```",
    ]
    _clients, _hosts = _install_fake_runtime(monkeypatch, responses)

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(AgentTask(query="do something", agent_type="code_exec", session_id="s6"))
    )

    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))
    trace_texts = [chunk.text for chunk in chunks if isinstance(chunk, TraceChunk)]

    assert "Aborted: the planner produced no valid run_shell JSON tool call" in response_text
    assert "Use /trace" in response_text
    assert "Last planner response" not in response_text
    assert secret_marker not in response_text
    # Raw payload must still be available via /trace.
    assert any(secret_marker in text for text in trace_texts)


def test_code_exec_executes_multiline_heredoc_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run_shell call with raw newlines inside ``cmd`` must be parsed and run."""
    raw_tool_call = (
        '{"tool":"run_shell","args":{"cmd":"cat > /workspace/summary.md << \'PYEOF\'\n'
        "# Agent Harness\n"
        "A one-page summary.\n"
        'PYEOF\ncat /workspace/summary.md","timeout":15},'
        '"progress":"writing one-page summary"}'
    )
    result = ExecResult(
        stdout="# Agent Harness\nA one-page summary.\n",
        stderr="",
        exit_code=0,
        duration_ms=20.0,
        artifacts=("/tmp/iris-fake-workspace/summary.md",),
    )
    _clients, hosts = _install_fake_runtime(
        monkeypatch,
        [raw_tool_call, "Done."],
        [result],
    )

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(
            AgentTask(query="write a one pager", agent_type="code_exec", session_id="s7")
        )
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert len(hosts[0].calls) == 1
    cmd, timeout = hosts[0].calls[0]
    assert "# Agent Harness" in cmd
    assert "PYEOF" in cmd
    assert timeout == 15
    assert activity_texts[0] == "writing one-page summary"
    assert "Aborted" not in response_text
    assert "Done." in response_text
    assert "summary.md" in response_text


def test_code_exec_rejects_script_only_artifact_for_document_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A helper script alone must not satisfy a document-generation request."""
    script_only_call = (
        '{"tool":"run_shell","args":{"cmd":"cat > /workspace/script.py << \'PYEOF\'\n'
        "# Agent Harness Research Summary\n"
        "print('this is only a helper script')\n"
        'PYEOF","timeout":15},"progress":"drafting summary"}'
    )
    document_call = (
        '{"tool":"run_shell","args":{"cmd":"cat > /workspace/agent_harness_summary.md '
        "<< 'PYEOF'\n"
        "# Agent Harness\n\n"
        "Agent Harness is a controlled runtime pattern for building, testing, "
        "and observing agents.\n"
        'PYEOF","timeout":15},"progress":"creating the actual document artifact"}'
    )
    results = [
        ExecResult(
            stdout="",
            stderr="",
            exit_code=0,
            duration_ms=12.0,
            artifacts=("/tmp/iris-fake-workspace/script.py",),
        ),
        ExecResult(
            stdout="wrote markdown\n",
            stderr="",
            exit_code=0,
            duration_ms=14.0,
            artifacts=("/tmp/iris-fake-workspace/agent_harness_summary.md",),
        ),
    ]
    _clients, hosts = _install_fake_runtime(
        monkeypatch,
        [
            script_only_call,
            "The script generated the requested document.",
            document_call,
            "Done.",
        ],
        results,
    )

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(
            AgentTask(
                query="generate a one pager document about agent harness",
                agent_type="code_exec",
                session_id="s8",
            )
        )
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert len(hosts[0].calls) == 2
    assert any("document artifact" in text and "missing" in text for text in activity_texts)
    assert "PLANNER GUIDANCE" in _clients[0].user_prompts[2]
    assert "script.py" not in response_text
    assert "agent_harness_summary.md" in response_text
    assert "Done." in response_text


def test_code_exec_retries_truncated_tool_call_without_burning_invalid_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A tool call cut off mid-JSON is not a protocol failure: the planner tried
    # to call run_shell but ran out of output budget. It must get its own
    # counter and "shorten the command" guidance, not the protocol restatement.
    responses = [
        '{"tool":"run_shell","args":{"cmd":"cat > /workspace/r.py << \'PYEOF\'\\nimport os',
        '{"tool":"run_shell","args":{"cmd":"echo ok","timeout":5}}',
        "Done.",
    ]
    result = ExecResult(
        stdout="ok\n",
        stderr="",
        exit_code=0,
        duration_ms=12.0,
        artifacts=("/tmp/iris-fake-workspace/report.pdf",),
    )
    clients, hosts = _install_fake_runtime(monkeypatch, responses, [result])

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(
            AgentTask(query="create a report", agent_type="code_exec", session_id="s-trunc")
        )
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert any("cut off" in text and "(1/3)" in text for text in activity_texts)
    # Never reported as a protocol failure, and never aborted.
    assert not any("no valid run_shell JSON" in text for text in activity_texts)
    assert "Aborted" not in response_text
    assert "Done." in response_text
    assert hosts[0].calls == [("echo ok", 5)]
    guidance = clients[0].user_prompts[1]
    assert "cut off before the JSON closed" in guidance
    # The truncated garbage must not be echoed back: it is what overflowed the
    # context window in the first place.
    assert "PYEOF" not in guidance


def test_code_exec_truncated_tool_call_abort_names_the_real_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truncated = '{"tool":"run_shell","args":{"cmd":"cat > /workspace/r.py << \'PYEOF\'\\nimport os'
    clients, hosts = _install_fake_runtime(monkeypatch, [truncated] * 3)

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(
            AgentTask(query="create a report", agent_type="code_exec", session_id="s-trunc-abort")
        )
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert any("(3/3)" in text for text in activity_texts)
    assert "too long to finish inside its output budget" in response_text
    # The old message blamed the planner for never calling the tool, which sent
    # debugging in the wrong direction. It must not come back.
    assert "produced no valid run_shell JSON tool call" not in response_text
    assert hosts[0].calls == []
    assert len(clients[0].user_prompts) == 3


def test_code_exec_abandons_stream_on_degenerate_repetition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # llama3.2:3b collapses into a repeated-import loop inside a heredoc and
    # burns the whole generation budget. The guard must cut the stream short.
    unit = "import dash\\nimport dash_table\\n"
    degenerate = ['{"tool":"run_shell","args":{"cmd":"', *([unit] * 200)]
    full_length = len("".join(degenerate))
    responses: list[str | Sequence[str]] = [
        degenerate,
        '{"tool":"run_shell","args":{"cmd":"echo ok","timeout":5}}',
        "Done.",
    ]
    result = ExecResult(
        stdout="ok\n",
        stderr="",
        exit_code=0,
        duration_ms=12.0,
        artifacts=("/tmp/iris-fake-workspace/report.pdf",),
    )
    _clients, hosts = _install_fake_runtime(monkeypatch, responses, [result])

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(
            AgentTask(query="create a report", agent_type="code_exec", session_id="s-rep")
        )
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    trace_texts = [chunk.text for chunk in chunks if isinstance(chunk, TraceChunk)]
    response_text = "".join(chunk for chunk in chunks if isinstance(chunk, str))

    assert any("repeated itself" in text for text in activity_texts)
    # The captured planner output is a fraction of what the model would have
    # emitted — proof the stream was abandoned rather than drained.
    first_trace = next(text for text in trace_texts if "[planner iter=1]" in text)
    assert len(first_trace) < full_length / 2
    assert "Done." in response_text
    assert hosts[0].calls == [("echo ok", 5)]


def test_code_exec_repetition_guard_leaves_healthy_stream_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A long but non-degenerate heredoc must stream to completion.
    body = "".join(f"rows.append(build_row({n}))\\n" for n in range(120))
    healthy = '{"tool":"run_shell","args":{"cmd":"' + body + 'echo done","timeout":5}}'
    result = ExecResult(
        stdout="done\n",
        stderr="",
        exit_code=0,
        duration_ms=12.0,
        artifacts=("/tmp/iris-fake-workspace/report.pdf",),
    )
    _clients, hosts = _install_fake_runtime(monkeypatch, [healthy, "Done."], [result])

    _handler, stream_handler = _make_code_exec_handler(_TierRouter())
    chunks = list(
        stream_handler(
            AgentTask(query="create a report", agent_type="code_exec", session_id="s-healthy")
        )
    )

    activity_texts = [chunk.text for chunk in chunks if isinstance(chunk, ActivityChunk)]
    assert not any("cut off" in text for text in activity_texts)
    assert hosts[0].calls and hosts[0].calls[0][0].endswith("echo done")
