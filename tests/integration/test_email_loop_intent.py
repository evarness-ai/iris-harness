"""The ``email`` intent on the harness's governed loop (core/SDK boundary plan, email slice 5).

The email plugin claims its intent with ``api.register_loop_intent("email", fallback=...)``
instead of the core naming it: the loop answers email turns, over the tools the plugin
registers, and the plugin's fallback is the degrade path. Driven through a real runtime
with the email plugin mounted, on both surfaces (``chat`` and ``chat_stream``, which the
REPL and the web use), with only the model faked.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from iris_harness.runtime import build_runtime
from iris_harness.runtime.types import ChatResult
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.judgments import JudgmentStore

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("offline_llm")]

ACCT = "gmail:owner@example.com"
_DIGEST = "Thought: read the inbox\nAction: inbox_digest\nAction Input: {}"
# An answer with no read behind it: `email` is a read-first intent, so the loop refuses it.
_UNREAD = "Thought: I know this\nFinal Answer: No bills arrived."


def _msg(msg_id: str, sender: str, subject: str) -> EmailMessage:
    return EmailMessage(
        id=msg_id,
        provider="gmail",  # type: ignore[arg-type]
        account_id=ACCT,
        thread_id=None,
        from_address=sender,
        to=("owner@example.com",),
        subject=subject,
        received_at=datetime.now(UTC),
        snippet=f"{subject} (snippet)",
        attachments=(),
    )


def _seed(data_dir: Path) -> None:
    store = EmailStore(db_path=data_dir / "email.db")
    store.ensure_schema()
    judgments = JudgmentStore(db_path=store.db_path)
    judgments.ensure_schema()
    for msg_id, sender, subject, bucket in (
        ("b1", "alerts@bank.example", "Card statement for Jul-2026", "bill"),
        ("n1", "news@example.com", "AI newsletter", "fyi"),
    ):
        store.upsert(_msg(msg_id, sender, subject))
        judgments.record(message_id=msg_id, account_id=ACCT, bucket=bucket, confidence=0.9)


@pytest.fixture()
def world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[Any, list[str]]]:
    """A real runtime, the email plugin mounted, and a model scripted per turn."""
    from iris_harness.llm.client import CodingLLMClient

    script: list[str] = [_DIGEST]

    def model(self: Any, *, system_prompt: str, user_prompt: str, **_kw: Any) -> str:
        # Only the loop's ReAct steps are scripted; anything else (a narrator) says nothing.
        return script[0] if "Action Input" in user_prompt else ""

    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.delenv("IRIS_AGENTIC_CORE_ENABLED", raising=False)  # the default: on
    monkeypatch.setattr(CodingLLMClient, "invoke", model)
    config_dir, data_dir = tmp_path / "config", tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    _seed(data_dir)
    rt = build_runtime(config_dir=config_dir, data_dir=data_dir, use_background_scheduler=False)
    rt.startup()
    try:
        yield rt, script
    finally:
        rt.shutdown()


def _both(rt: Any, message: str) -> list[ChatResult]:
    """The same turn on ``chat`` and on ``chat_stream`` (each in its own session)."""
    sync = rt.chat(message, session_id="s-sync")
    done = [e for e in rt.chat_stream(message, session_id="s-stream") if e.kind == "done"]
    assert done, "chat_stream ended without a result"
    return [sync, done[-1].result]


def test_the_email_plugin_claims_its_intent_and_the_core_names_it_nowhere(world: Any) -> None:
    rt, _script = world
    claimed = rt.plugin_registry.loop_intents()
    assert "email" in claimed
    assert ("email_workflows", "loop_intent", "email") in rt.plugin_registry.seams()
    # The loop answers `email`; the plugin's declared fallback is its degrade path, not
    # the digest lane the plugin registered as the intent's handler.
    assert "email" in rt._loop_intents
    assert rt.agent_executor._handlers["email"] is rt.react_handler
    assert rt.agent_executor._stream_handlers["email"] is rt.react_stream_handler
    assert rt._react_fallbacks["email"] is claimed["email"]
    assert rt._react_fallbacks["email"] is not rt.plugin_registry.intent_handlers()["email"][0]


def test_an_email_question_is_answered_by_the_loop_on_both_surfaces(world: Any) -> None:
    rt, _script = world
    for result in _both(rt, "summarise my email inbox"):
        assert result.agent_type == "email", result
        assert result.metadata.get("agentic_core") is True, result.metadata
        assert "Card statement for Jul-2026" in result.response


def test_a_turn_the_loop_cannot_ground_degrades_to_the_fallback_on_both_surfaces(
    world: Any,
) -> None:
    rt, script = world
    script[0] = _UNREAD
    for result in _both(rt, "summarise my email inbox"):
        assert result.agent_type == "email", result
        # The fallback's digest, not the loop's refusal and not the unread answer.
        assert "No bills arrived" not in result.response
        assert "Card statement for Jul-2026" in result.response
        assert "AI newsletter" in result.response


def test_the_degrade_path_is_the_declared_fallback_not_the_digest_lane(world: Any) -> None:
    """A bill-email question the loop cannot ground: the fallback answers from the
    judge's bill bucket, which the digest lane never does."""
    from iris_harness.agent.agent_executor import AgentTask

    rt, script = world
    script[0] = _UNREAD
    result = rt.agent_executor.execute(
        AgentTask(
            query="any bill emails today?",
            agent_type="email",
            session_id="s-bill",
            params={"intent": "communication"},
        )
    )
    assert result.metadata.get("bill_email_recall") is True, result.metadata
    assert "Card statement for Jul-2026" in result.output
    assert "AI newsletter" not in result.output


def test_the_fallbacks_read_runs_through_the_runtimes_governed_runner(
    world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The degrade path's read is ``api.tools``: the runtime's kernel sees PRE_TOOL_USE
    for it with the plugin as the caller, not a direct call of the tool's function."""
    from iris_harness.agent.agent_executor import AgentTask
    from iris_harness.kernel.governance import HookPoint

    rt, script = world
    script[0] = _UNREAD
    kernel = rt.governance_kernel
    assert kernel is not None
    seen: list[tuple[str, str]] = []
    real_fire = type(kernel).fire_sync

    def spy(self: Any, point: Any, ctx: Any) -> Any:
        if point is HookPoint.PRE_TOOL_USE:
            seen.append((ctx.payload.get("tool_name"), ctx.metadata.get("caller")))
        return real_fire(self, point, ctx)

    monkeypatch.setattr(type(kernel), "fire_sync", spy)
    rt.agent_executor.execute(
        AgentTask(
            query='read the email with subject containing "Card statement"',
            agent_type="email",
            session_id="s-read",
            params={"intent": "communication"},
        )
    )
    assert ("read_email", "plugin:email_workflows") in seen, seen


def test_with_the_loop_off_the_digest_lane_stands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IRIS_AGENTIC_CORE_ENABLED", "0")
    config_dir, data_dir = tmp_path / "config", tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    rt = build_runtime(config_dir=config_dir, data_dir=data_dir, use_background_scheduler=False)
    assert "email" not in rt._loop_intents
    assert "email" not in rt._react_fallbacks
    lane = rt.plugin_registry.intent_handlers()["email"][0]
    assert rt.agent_executor._handlers["email"] is lane
