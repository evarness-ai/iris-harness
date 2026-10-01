"""Decision B: a handler whose answer repeats third-party text opts into the output guard.

Every answer passes the model-free response check. A deterministic handler that echoes
text someone else wrote (email subjects, senders, statement text) declares
``guard_output``; when the model-based output guard is enabled (Llama Guard,
``IRIS_CURATOR_OUTPUT_SAFETY``) it also runs on that handler's answers, with the halt
and the warning banner a generated answer gets (deterministic-path parity).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris_harness.agent.response_curator import GOVERNANCE_BLOCKED_TEXT, OutputSafetyVerdict
from iris_harness.runtime import build_runtime
from iris_harness.runtime.intercept_dispatch import InterceptDispatch
from iris_harness.runtime.intercepts import InterceptHit, InterceptSpec, _spec_from_dict
from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.runtime.types import ChatResult

ECHOED = "From: a sender. Subject: something a stranger wrote"


# -- the declaration ------------------------------------------------------------------


def test_intercepts_yaml_can_declare_it() -> None:
    assert _spec_from_dict({"name": "x", "guard_output": True}).guard_output is True
    assert _spec_from_dict({"name": "x"}).guard_output is False


def _api(registry: PluginRegistry) -> PluginAPI:
    registry.add_plugin(PluginRecord(name="p", source="test", status=PluginStatus.LOADED))
    services = HarnessServices(
        config_dir=Path("/nonexistent"),
        data_dir=Path("/nonexistent"),
        tier_router=None,
        agent_executor=None,
        heartbeats=None,
        channels=None,
        deterministic_reply=lambda **kw: None,
    )
    return PluginAPI(plugin="p", services=services, registry=registry)


def test_a_plugin_can_declare_it_when_registering() -> None:
    registry = PluginRegistry()
    _api(registry).register_intercept("echoes", lambda *_a, **_k: None, guard_output=True)
    registration = registry.intercept("echoes")
    assert registration is not None and registration.spec.guard_output is True


@pytest.mark.parametrize(("in_yaml", "in_plugin"), [(True, False), (False, True)])
def test_either_side_saying_yes_is_enough(in_yaml: bool, in_plugin: bool) -> None:
    registry = PluginRegistry()
    _api(registry).register_intercept("echoes", lambda *_a, **_k: None, guard_output=in_plugin)
    host = SimpleNamespace(
        intercept_chain=(InterceptSpec(name="echoes", handler="plugin:p", guard_output=in_yaml),),
        plugin_registry=registry,
        profile=None,
    )
    chain = InterceptDispatch(host).effective_chain()  # type: ignore[arg-type]
    assert [spec.guard_output for spec, _ in chain] == [True]


# -- the guard stage --------------------------------------------------------------------


class _Judge:
    """An output guard that records calls and returns one fixed outcome."""

    def __init__(self, verdict: OutputSafetyVerdict | None = None, *, fail: bool = False):
        self.calls = 0
        self._verdict = verdict or OutputSafetyVerdict(unsafe=False)
        self._fail = fail

    async def judge(self, *, response: str) -> OutputSafetyVerdict:
        self.calls += 1
        if self._fail:
            raise TimeoutError("cold model")
        return self._verdict


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    return build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )


def _answer(runtime: Any, monkeypatch: pytest.MonkeyPatch, *, guard_output: bool) -> None:
    spec = InterceptSpec(name="echo_handler", handler="test", guard_output=guard_output)
    result = ChatResult(ECHOED, "email", "email", ("email",), False, None, {})
    monkeypatch.setattr(
        runtime.intercepts, "dispatch", lambda *_a, **_k: InterceptHit(spec=spec, result=result)
    )


def _with_judge(runtime: Any, judge: _Judge | None) -> None:
    curator = runtime.response_curator
    curator._output_safety_judge = judge
    curator._output_safety_enforce = frozenset({"self_harm"})
    curator._output_safety_log_only = frozenset({"hate"})


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_an_unsafe_echo_is_blocked_when_the_handler_opted_in(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    _answer(runtime, monkeypatch, guard_output=True)
    _with_judge(runtime, _Judge(OutputSafetyVerdict(unsafe=True, categories=("self_harm",))))
    if entry == "chat":
        result = runtime.chat("anything", session_id=f"go-{entry}")
    else:
        result = list(runtime.chat_stream("anything", session_id=f"go-{entry}"))[-1].result
    assert result.response == GOVERNANCE_BLOCKED_TEXT
    assert result.metadata.get("halted_by") == "output_safety"


def test_a_handler_that_did_not_opt_in_never_reaches_the_output_guard(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _answer(runtime, monkeypatch, guard_output=False)
    judge = _Judge(OutputSafetyVerdict(unsafe=True, categories=("self_harm",)))
    _with_judge(runtime, judge)
    result = runtime.chat("anything", session_id="go-off")
    assert judge.calls == 0 and result.response == ECHOED


def test_a_cold_output_guard_fails_open_with_the_banner(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _answer(runtime, monkeypatch, guard_output=True)
    _with_judge(runtime, _Judge(fail=True))
    result = runtime.chat("anything", session_id="go-cold")
    assert result.response.startswith("[Governance warning]")
    assert result.response.endswith(ECHOED)


def test_with_the_output_guard_off_the_answer_is_untouched(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _answer(runtime, monkeypatch, guard_output=True)
    _with_judge(runtime, None)
    assert runtime.chat("anything", session_id="go-disabled").response == ECHOED


def test_a_safe_echo_ships_unchanged(runtime: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _answer(runtime, monkeypatch, guard_output=True)
    judge = _Judge()
    _with_judge(runtime, judge)
    assert runtime.chat("anything", session_id="go-safe").response == ECHOED
    assert judge.calls == 1
