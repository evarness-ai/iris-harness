"""The governance judge, wired (§9.2): who is judged, on which route, and what a verdict does.

The executor runs each review inline so every assertion sees the finished review; the
model is a fake that records the route it was asked on and returns a scripted verdict.
The audit log is the real one, in a temp file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.run_review import CompletedRun
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.runtime.governance_judge import (
    ENABLED_ENV,
    TIER_ENV,
    TIMEOUT_ENV,
    GovernanceJudge,
    build_governance_judge,
    judge_steps,
    judge_timeout_s,
)


class _Inline:
    """An executor that runs the review now, so the test sees its outcome."""

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:
        future: Future[Any] = Future()
        future.set_result(fn(*args))
        return future


class _Parked:
    """An executor that never runs anything, so reviews stay pending."""

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:
        return Future()


def _verdict(recommend: str) -> str:
    return json.dumps(
        {
            "hallucination": 0.1,
            "goal_alignment": 0.9,
            "tool_misuse": 0.7 if recommend != "allow" else 0.0,
            "danger": 0.8 if recommend == "halt_next" else 0.1,
            "recommend": recommend,
            "rationale": "It deleted mail the user did not name.",
        }
    )


_RUN = CompletedRun(
    run_id="run-1",
    query="clean up my promo mail",
    steps=(
        {
            "thought": "find them",
            "action": "trash_email",
            "action_input": {"ids": ["a", "b"]},
            "observation": "trashed 2",
        },
    ),
    final_answer="Trashed 2 promo emails.",
    success=True,
)
_PLAIN = CompletedRun(run_id="run-2", query="hi", steps=(), final_answer="hello", success=True)


class _Harness:
    def __init__(self, tmp_path: Path, reply: str | Exception, **kwargs: Any) -> None:
        self.routes: list[tuple[str, str]] = []
        self.prompts: list[str] = []
        self.alerts: list[tuple[str, str]] = []
        self.audit = AuditLog(db_path=tmp_path / "audit.db")

        def invoke_for(target: tuple[str, str]) -> Callable[[str, str], str]:
            self.routes.append(target)

            def invoke(system: str, user: str) -> str:
                self.prompts.append(user)
                if isinstance(reply, Exception):
                    raise reply
                return reply

            return invoke

        self.judge = GovernanceJudge(
            invoke_for=invoke_for,
            tier_for=lambda target: "tier_1" if target[1] == "private" else "tier_2",
            known_tier=lambda name: name in {"tier2", "private", "tier3"},
            audit_log=lambda: self.audit,
            notify=lambda subject, body: self.alerts.append((subject, body)),
            executor=kwargs.pop("executor", _Inline()),  # type: ignore[arg-type]
            **kwargs,
        )

    def judge_rows(self) -> list[Any]:
        return [r for r in self.audit.query(run_id="run-1") if r.plugin == "llm_judge"]


@pytest.fixture(autouse=True)
def _judge_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENABLED_ENV, "1")


def test_off_by_default_and_off_means_nothing_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENABLED_ENV)
    h = _Harness(tmp_path, _verdict("halt_next"))
    assert h.judge.submit(_RUN, route="private", agent_type="email") is False
    assert h.routes == []


def test_a_run_that_used_no_tools_is_not_judged(tmp_path: Path) -> None:
    h = _Harness(tmp_path, _verdict("allow"))
    assert h.judge.submit(_PLAIN, route="general", agent_type="system") is False
    assert h.routes == []


def test_by_default_the_judge_runs_on_the_same_route_as_the_run(tmp_path: Path) -> None:
    h = _Harness(tmp_path, _verdict("allow"))

    assert h.judge.submit(_RUN, route="private", agent_type="email") is True

    assert h.routes == [("intent", "private")]
    assert "trash_email" in h.prompts[0]
    assert "Trashed 2 promo emails." in h.prompts[0]


def test_halt_is_recorded_and_alerts_the_owner(tmp_path: Path) -> None:
    h = _Harness(tmp_path, _verdict("halt_next"))

    h.judge.submit(_RUN, route="private", agent_type="email")

    (row,) = h.judge_rows()
    assert row.severity == "warn"
    assert "recommend=halt_next" in row.reason
    assert row.tier == "tier_1"
    ((subject, body),) = h.alerts
    assert "halt recommended" in subject
    assert "iris run inspect run-1" in body
    assert "clean up my promo mail" in body


def test_warn_is_recorded_without_an_alert(tmp_path: Path) -> None:
    h = _Harness(tmp_path, _verdict("warn"))
    h.judge.submit(_RUN, route="general", agent_type="system")
    (row,) = h.judge_rows()
    assert row.severity == "warn"
    assert h.alerts == []


def test_allow_is_recorded_quietly(tmp_path: Path) -> None:
    h = _Harness(tmp_path, _verdict("allow"))
    h.judge.submit(_RUN, route="general", agent_type="system")
    (row,) = h.judge_rows()
    assert row.severity == "info"
    assert h.alerts == []


def test_a_failing_model_writes_a_warning_and_raises_nothing(tmp_path: Path) -> None:
    h = _Harness(tmp_path, RuntimeError("ollama down"))
    assert h.judge.submit(_RUN, route="private", agent_type="email") is True
    (row,) = h.judge_rows()
    assert "client error" in row.reason
    assert h.alerts == []


def test_a_full_queue_skips_instead_of_piling_up(tmp_path: Path) -> None:
    h = _Harness(tmp_path, _verdict("allow"), executor=_Parked(), max_pending=2)
    assert h.judge.submit(_RUN, route="general", agent_type="system") is True
    assert h.judge.submit(_RUN, route="general", agent_type="system") is True
    assert h.judge.submit(_RUN, route="general", agent_type="system") is False


def test_the_timeout_setting_is_read_and_bad_values_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TIMEOUT_ENV, "12.5")
    assert judge_timeout_s() == 12.5
    for bad in ("zero", "-3", "0"):
        monkeypatch.setenv(TIMEOUT_ENV, bad)
        assert judge_timeout_s() == 30.0


def test_the_trace_is_rendered_as_text_with_the_final_answer_last() -> None:
    steps = judge_steps(_RUN)
    assert steps[0]["action_input"] == '{"ids": ["a", "b"]}'
    assert steps[-1]["observation"] == "Final answer given to the user: Trashed 2 promo emails."


def test_the_built_judge_asks_the_router_for_the_runs_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_governance_judge resolves the model from the SAME routing intent."""
    asked: list[str] = []

    class _Cfg:
        def model_copy(self, update: dict[str, Any]) -> _Cfg:
            assert update["temperature"] == 0.0
            return self

    class _Router:
        _intent_to_tier: dict[str, str] = {"finance": "private"}

        def get_llm_config(self, intent: str) -> _Cfg:
            asked.append(f"intent:{intent}")
            return _Cfg()

        def get_llm_config_for_tier(self, name: str) -> _Cfg:
            asked.append(f"tier:{name}")
            return _Cfg()

        def get_tier_by_name(self, name: str) -> object | None:
            return object() if name == "tier3" else None

    class _Client:
        def __init__(self, cfg: Any, governance_agent_type: str) -> None:
            assert governance_agent_type == "chat"

        def invoke(self, *, system_prompt: str, user_prompt: str) -> str:
            return _verdict("allow")

    monkeypatch.setattr("iris_harness.llm.client.CodingLLMClient", _Client)
    judge = build_governance_judge(tier_router=_Router(), channels=None)

    assert judge._invoke_for(("intent", "finance"))("s", "u") == _verdict("allow")
    judge._invoke_for(("tier", "tier3"))("s", "u")
    assert asked == ["intent:finance", "tier:tier3"]
    monkeypatch.setenv(TIER_ENV, "tier3")
    assert judge.target_for("finance") == ("tier", "tier3")
    monkeypatch.setenv(TIER_ENV, "nope")
    assert judge.target_for("finance") == ("intent", "finance")


def test_the_owner_picks_the_judges_tier_local_or_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing locks the judge to local or cloud: the setting names any tier."""
    h = _Harness(tmp_path, _verdict("allow"))
    for choice in ("tier3", "private"):
        monkeypatch.setenv(TIER_ENV, choice)
        h.judge.submit(_RUN, route="general", agent_type="system")
    assert h.routes == [("tier", "tier3"), ("tier", "private")]


def test_an_unknown_tier_falls_back_to_the_runs_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(TIER_ENV, "tier9")
    h = _Harness(tmp_path, _verdict("allow"))
    h.judge.submit(_RUN, route="communication", agent_type="email")
    assert h.routes == [("intent", "communication")]
    assert "tier9" in caplog.text
