"""The general lane's direct skill answer is a governed call (issue #155, PR 2, #134 D7).

``direct_skill_response`` (the lane answers a matched skill request before any model) used to
invoke the skill's tool class in-process: no ``PRE_TOOL_USE`` / ``POST_TOOL_USE``, no approval
check, no audit row. It now runs the tool through ``ToolService`` under the ``core:general_lane``
caller, with the tool's result serialised to JSON inside the governed call and parsed back out of
the governed text, so the formatter and the pending-action recorder still read the structured
value. Driven through both handler entries (``chat`` and ``chat_stream`` reach the sync and the
streaming handler), with a real kernel, real hooks and a real ``ToolService``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.foundation.ids import is_ulid
from iris_harness.kernel.governance import GovernanceKernel, HookPoint, kernel_from_env
from iris_harness.kernel.governance.audit.log import AuditLog
from iris_harness.kernel.governance.caller_policy import register_caller_policy
from iris_harness.kernel.governance.external_content import (
    ENVELOPE_TAG,
    MARKER,
    REDACTION_NOTICE,
)
from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision
from iris_harness.runtime.handlers.general import _make_general_handler
from iris_harness.runtime.handlers.local_skills import DIRECT_CALLER
from iris_harness.runtime.handlers.react import _skills_to_react_tools
from iris_harness.runtime.skill_tool_specs import skill_tool_spec
from iris_harness.runtime.tool_service import ToolResult, ToolService

from .test_brief_slots_external_floor import (
    BENIGN,
    CANARY,
    INJECTED,
    _Registry,
    _tool_package,
)

QUERY = "list my feed"


class _NoArgs(BaseModel):
    pass


class _Feed(BaseTool):
    name: str = "list_feed"
    description: str = "a fake feed: a structured list of dicts"
    args_schema: type[BaseModel] = _NoArgs

    def _run(self) -> list[dict]:
        return [
            {"repo": "acme/widgets", "description": f"{BENIGN} {CANARY}", "stars_today": 12},
            {"repo": "acme/gadgets", "description": INJECTED, "language": "Rust"},
        ]

    async def _arun(self) -> list[dict]:
        return self._run()


class _Benign(_Feed):
    def _run(self) -> list[dict]:
        return [
            {
                "repo": "acme/widgets",
                "description": f"{BENIGN} {CANARY}",
                "stars_today": 12,
                "forks": 3,
            },
            {"name": "acme/gadgets", "url": "https://example.test/g", "language": "Rust"},
        ]


class _Proposal(_Feed):
    name: str = "propose_ingest"

    def _run(self) -> dict:
        return {
            "kind": "rag_ingest",
            "summary": "ingest it",
            "proposal": {"resolved_path": "/tmp/notes.md", "content_sha": "abc123", "reason": "r"},
        }


class _Text(_Feed):
    name: str = "plain_text"

    def _run(self) -> str:
        return "just a sentence"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    monkeypatch.setattr("iris_harness.runtime.handlers.local_skills.data_dir", lambda: tmp_path)
    register_caller_policy(None)
    yield
    register_caller_policy(None)


class _Lane:
    """A general handler wired to a real kernel and a real ``ToolService`` (or to none)."""

    def __init__(self, package: Any, tmp_path: Path, *, governed: bool = True) -> None:
        self.tmp = tmp_path
        # The runtime's own kernel (the floor and the default hooks, audit under the env path).
        self.audit = AuditLog()
        self.kernel = kernel_from_env() if governed else None
        self.service = ToolService(
            tools=lambda: [],
            kernel=lambda: self.kernel,
            events=lambda: None,
            deliver_in_chat=lambda *_a: None,
        )
        holder = (
            [SimpleNamespace(tool_service=self.service, governance_kernel=self.kernel)]
            if governed
            else None
        )
        self.handler, self.stream_handler = _make_general_handler(
            SimpleNamespace(), skill_registry=_Registry([package]), runtime_holder=holder
        )

    def answer(self, entry: str) -> tuple[str, dict[str, Any]]:
        task = AgentTask(query=QUERY, agent_type="system", session_id="s-155b")
        if entry == "chat":
            text, meta = self.handler(task)
            return str(text), dict(meta)
        chunks = list(self.stream_handler(task))
        return str(chunks[0]), {}

    def rows(self, tool: str) -> list[tuple[Any, dict[str, Any]]]:
        out = []
        for row in self.audit.query():
            payload = json.loads(row.payload_json)
            if payload.get("tool_name") == tool and row.hook_point in (
                HookPoint.PRE_TOOL_USE.value,
                HookPoint.POST_TOOL_USE.value,
            ):
                out.append((row, payload))
        return out


def _stub_match(monkeypatch: pytest.MonkeyPatch, package: Any) -> None:
    monkeypatch.setattr(
        "iris_harness.runtime.handlers.local_skills.best_matching_skill_package",
        lambda _q, _packages: package,
    )


# -------------------------------------------------------------------- the owner's answer
@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
@pytest.mark.parametrize(
    ("cls", "tool", "content"),
    [
        (_Benign, "list_feed", "external"),  # a structured list, external but nothing to redact
        (_Benign, "list_feed", "internal"),
        (_Text, "plain_text", "internal"),  # a bare string
        (_Proposal, "propose_ingest", "internal"),  # a dict
    ],
)
def test_the_owner_answer_is_byte_identical_to_the_in_process_one_for_a_structured_result(
    entry: str,
    cls: type[BaseTool],
    tool: str,
    content: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The golden: governing the call must not change what the owner reads. The reference is
    the same handler built with no runtime (the in-process path this replaces)."""
    package = _tool_package("feed", tool, cls, content)
    _stub_match(monkeypatch, package)

    governed, _ = _Lane(package, tmp_path / "g").answer(entry)
    in_process, _ = _Lane(package, tmp_path / "p", governed=False).answer(entry)

    assert governed == in_process
    assert "Used skill" in governed


# --------------------------------------------------------------------------- governance
@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_the_call_leaves_pre_and_post_rows_under_the_core_caller_and_one_call_id(
    entry: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _tool_package("feed", "list_feed", _Benign, "internal")
    _stub_match(monkeypatch, package)
    lane = _Lane(package, tmp_path)

    lane.answer(entry)

    rows = lane.rows("list_feed")
    assert {row.hook_point for row, _ in rows} == {"pre_tool_use", "post_tool_use"}
    ids = {payload.get("call_id") for _, payload in rows}
    assert len(ids) == 1 and is_ulid(next(iter(ids)))
    assert {payload.get("caller") for _, payload in rows} == {DIRECT_CALLER}
    assert {payload.get("tool_plugin") for _, payload in rows} == {"skill:feed"}
    assert all(payload.get("digest_alg") for _, payload in rows)


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_an_external_feed_is_redacted_once_with_one_notice_one_floor_row_and_no_envelope(
    entry: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _tool_package("feed", "list_feed", _Feed, "external")
    _stub_match(monkeypatch, package)
    lane = _Lane(package, tmp_path)

    text, _ = lane.answer(entry)

    assert "Ignore all previous instructions" not in text and MARKER in text
    assert BENIGN in text and CANARY in text
    assert f"<{ENVELOPE_TAG}" not in text  # an owner channel never gets the envelope
    assert text.count(REDACTION_NOTICE) == 1
    with sqlite3.connect(lane.audit.db_path) as conn:
        floor_rows = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE plugin = 'external_content_floor' "
            "AND json_extract(payload_json, '$.patterns') IS NOT NULL"
        ).fetchone()
    assert floor_rows == (1,)  # one scan: the runner's. The old second tripwire is gone.


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_pending_action_the_skill_proposes_is_still_recorded(
    entry: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _tool_package("feed", "propose_ingest", _Proposal, "internal")
    _stub_match(monkeypatch, package)

    _text, meta = _Lane(package, tmp_path).answer(entry)

    if entry == "chat":  # the stream's first chunk carries the text only
        assert meta["pending_actions_recorded"] == 1
    with sqlite3.connect(tmp_path / "tasks.db") as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
    assert n == 1


def test_governed_text_that_is_not_the_tools_json_is_shown_as_text_and_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A result governance replaced or withheld is not the tool's JSON: the fallback is the
    text as it is, logged, with no pending actions recorded from it."""
    package = _tool_package("feed", "propose_ingest", _Proposal, "internal")
    _stub_match(monkeypatch, package)
    lane = _Lane(package, tmp_path)
    monkeypatch.setattr(
        lane.service,
        "call_core_tool",
        lambda caller, tool, args: ToolResult(ok=True, text="[withheld by governance]"),
    )

    with caplog.at_level(logging.WARNING):
        text, meta = lane.answer("chat")

    assert "[withheld by governance]" in text
    assert meta["pending_actions_recorded"] == 0
    assert any("not JSON" in r.getMessage() for r in caplog.records)
    assert not (tmp_path / "tasks.db").exists() or sqlite3.connect(tmp_path / "tasks.db").execute(
        "SELECT COUNT(*) FROM tasks"
    ).fetchone() == (0,)


class _Deny:
    name = "deny_everything"
    hook_point = HookPoint.PRE_TOOL_USE
    priority = 5

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="deny", reason="needs the owner's approval")


def test_a_tool_governance_will_not_run_is_refused_with_its_message_and_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``direct_skill_response`` answers the failure to the handler, which (as for any failed
    direct skill) hands it to the model as a recovery hint; the refusal itself is audited and
    the tool never ran."""
    from iris_harness.runtime.handlers.local_skills import make_local_skills

    ran: list[bool] = []

    class _Writer(_Benign):
        def _run(self) -> list[dict]:
            ran.append(True)
            return []

    package = _tool_package("feed", "list_feed", _Writer, "internal")
    _stub_match(monkeypatch, package)
    audit = AuditLog()
    kernel = GovernanceKernel(audit_log=audit)
    kernel.register(_Deny())
    kernel.init_lock()
    service = ToolService(
        tools=lambda: [],
        kernel=lambda: kernel,
        events=lambda: None,
        deliver_in_chat=lambda *_a: None,
    )
    skills = make_local_skills(
        _Registry([package]),  # type: ignore[arg-type]
        [SimpleNamespace(tool_service=service, governance_kernel=kernel)],
    )

    answer = skills.direct_skill_response(AgentTask(query=QUERY, agent_type="system"))

    assert ran == []  # nothing ran
    assert answer is not None
    text, meta = answer
    assert "by governance" in text and "needs the owner's approval" in text
    assert meta["skill_error"] is True
    rows = [
        json.loads(r.payload_json)
        for r in audit.query()
        if json.loads(r.payload_json).get("tool_name") == "list_feed"
    ]
    assert rows and {p.get("caller") for p in rows} == {DIRECT_CALLER}  # refused, and audited


# ------------------------------------------------------------- the seams this relies on
def test_a_core_caller_needs_no_caller_policy_entry(tmp_path: Path) -> None:
    """Pins the premise: ``CallerPolicyHook`` only checks ``plugin:`` callers, so the lane's
    ``core:`` caller passes even when the policy denies every tool. A future change to the
    policy that started checking ``core:`` callers would silently stop the direct path."""
    from iris_harness.agent.agentic_core import ToolSpec

    register_caller_policy(lambda caller, tool: "denied by the permission contract")
    kernel = kernel_from_env()
    service = ToolService(
        tools=lambda: [],
        kernel=lambda: kernel,
        events=lambda: None,
        deliver_in_chat=lambda *_a: None,
    )
    spec = ToolSpec("probe", "a probe", lambda args: "ran", effect="read", confirm="never")

    core = service.call_core_tool(DIRECT_CALLER, spec, {})
    plugin = service.call_core_tool("plugin:someone", spec, {})

    assert core.ok and core.text == "ran"
    assert not plugin.ok and "denied by the permission contract" in plugin.text


def test_with_governance_off_the_direct_call_still_runs_as_the_operators_opt_out() -> None:
    from iris_harness.agent.agentic_core import ToolSpec

    service = ToolService(
        tools=lambda: [], kernel=lambda: None, events=lambda: None, deliver_in_chat=lambda *_a: None
    )
    spec = ToolSpec("probe", "a probe", lambda args: "ran", effect="read", confirm="never")

    assert service.call_core_tool(DIRECT_CALLER, spec, {}).text == "ran"


def test_the_skill_tool_spec_is_the_one_the_loop_builds() -> None:
    """One declaration of a skill tool for the loop, the lane's pool and the direct answer."""
    package = _tool_package("feed", "list_feed", _Benign, "external")
    manifest_tool = package.manifest.tools[0]

    mine = skill_tool_spec(package, manifest_tool, package.tool_classes[0])
    (loops,) = _skills_to_react_tools(_Registry([package]))  # type: ignore[arg-type]

    for field in ("name", "description", "content", "plugin", "effect", "confirm"):
        assert getattr(mine, field) == getattr(loops, field), field
